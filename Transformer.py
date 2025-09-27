# transformer_algoperf.py (최종 수정본)

import os
import math
import argparse
import functools
from typing import Optional, List
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed.checkpoint as dist_checkpoint

# Hugging Face 라이브러리
from datasets import load_dataset
from transformers import AutoTokenizer

# Shampoo optimizer
from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    DDPShampooConfig,
    CommunicationDType
)

# 평가 메트릭
import sacrebleu
from tqdm import tqdm


# ============= Transformer 모델 구현 (AlgoPerf 사양) =============

class PositionalEncoding(nn.Module):
    """Positional Encoding with sinusoidal functions"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 완전히 새로운 텐서로 만들기
        pe = pe.unsqueeze(0).transpose(0, 1).clone().detach()
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, batch_size, d_model]"""
        # pe도 clone하여 사용
        return x + self.pe[:x.size(0)].clone()


class CustomMultiheadAttention(nn.Module):
    """Multi-Head Attention with separate Q,K,V projections for Shampoo tracking"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len, batch_size = query.size(0), query.size(1)
        Q = self.q_proj(query).view(seq_len, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(-1, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(-1, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        Q, K, V = Q.transpose(0, 2), K.transpose(0, 2), V.transpose(0, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        context = context.transpose(0, 1)
        return self.out_proj(context)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ffn(x2))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, n_heads, dropout)
        self.cross_attn = CustomMultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, tgt_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(x2, encoder_output, encoder_output, src_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.ffn(x2))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_encoder_layers: int,
                 n_decoder_layers: int, d_ff: int, max_seq_len: int, dropout: float, pad_idx: int):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # Encoder와 Decoder용 별도의 PositionalEncoding 생성
        self.encoder_pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.decoder_pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_decoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return ~mask

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.shared_embedding(src) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.encoder_pos_encoding(x)  # encoder용 사용
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.shared_embedding(tgt) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.decoder_pos_encoding(x)  # decoder용 사용
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return self.decoder_norm(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.create_padding_mask(src)
        tgt_seq_len = tgt.size(1)
        tgt_pad_mask = self.create_padding_mask(tgt)
        tgt_la_mask = self.create_look_ahead_mask(tgt_seq_len, tgt.device)
        tgt_mask = tgt_pad_mask & tgt_la_mask.unsqueeze(0).unsqueeze(0)
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        return F.linear(decoder_output.transpose(0, 1), self.shared_embedding.weight)

# ============= 데이터 처리 =============

class WMT17Dataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]['translation']
        src_text, tgt_text = item['de'], item['en']
        src_encoding = self.tokenizer(src_text, max_length=self.max_seq_len, padding='max_length', truncation=True, return_tensors='pt')
        tgt_encoding = self.tokenizer(tgt_text, max_length=self.max_seq_len, padding='max_length', truncation=True, return_tensors='pt')
        return {'src': src_encoding['input_ids'].squeeze(0), 'tgt': tgt_encoding['input_ids'].squeeze(0), 'src_text': src_text, 'tgt_text': tgt_text}


def create_data_loaders(args, tokenizer, global_rank, world_size):
    if global_rank == 0: print("Loading WMT datasets...")
    dataset = load_dataset("wmt17", "de-en", cache_dir=args.data_path)
    val_dataset_raw = load_dataset("wmt14", "de-en", cache_dir=args.data_path)
    train_data = dataset['train']
    if args.max_train_samples:
        train_data = train_data.select(range(min(args.max_train_samples, len(train_data))))
    train_dataset = WMT17Dataset(train_data, tokenizer, args.max_seq_len)
    val_dataset = WMT17Dataset(val_dataset_raw['validation'], tokenizer, args.max_seq_len)
    if global_rank == 0: print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


# ============= 학습 함수 =============

def label_smoothed_cross_entropy(logits, targets, epsilon=0.1, ignore_index=-100):
    vocab_size = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    smooth_target = torch.full_like(log_probs, epsilon / (vocab_size - 1)) # -1 for ignore_index
    smooth_target.scatter_(-1, targets.unsqueeze(-1), 1.0 - epsilon)
    mask = targets != ignore_index
    smooth_target.masked_fill_(~mask.unsqueeze(-1), 0)
    loss = -(smooth_target * log_probs).sum(dim=-1)
    return loss.masked_select(mask).mean()


def train_epoch(model, dataloader, optimizer, device, args, global_step):
    model.train()
    total_loss, total_tokens = 0, 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {args.current_epoch+1} Training", disable=(dist.get_rank() != 0))

    for batch in progress_bar:
        global_step += 1
        lr = get_lr_schedule(global_step, args.warmup_steps, args.d_model)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        src, tgt = batch['src'].to(device), batch['tgt'].to(device)
        tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        output = model(src, tgt_input)
        loss = label_smoothed_cross_entropy(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1), epsilon=args.label_smoothing, ignore_index=tokenizer.pad_token_id)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        num_tokens = (tgt_output != tokenizer.pad_token_id).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        if dist.get_rank() == 0:
            progress_bar.set_postfix({'loss': loss.item(), 'lr': lr})

    return total_loss / total_tokens if total_tokens > 0 else 0, global_step


def evaluate(model, dataloader, device, tokenizer):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0)):
            src, tgt = batch['src'].to(device), batch['tgt'].to(device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            output = model(src, tgt_input)
            loss = F.cross_entropy(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1), ignore_index=tokenizer.pad_token_id)
            num_tokens = (tgt_output != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    return total_loss / total_tokens if total_tokens > 0 else 0


def greedy_decode(model, src, max_len, device, tokenizer):
    model.eval()
    with torch.no_grad():
        src_mask = model.module.create_padding_mask(src)
        encoder_output = model.module.encode(src, src_mask)
        tgt = torch.full((src.size(0), 1), tokenizer.cls_token_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_seq_len = tgt.size(1)
            tgt_pad_mask = model.module.create_padding_mask(tgt)
            tgt_la_mask = model.module.create_look_ahead_mask(tgt_seq_len, device)
            tgt_mask = tgt_pad_mask & tgt_la_mask.unsqueeze(0).unsqueeze(0)
            decoder_output = model.module.decode(tgt, encoder_output, tgt_mask, src_mask)
            logits = F.linear(decoder_output[-1, :, :], model.module.shared_embedding.weight)
            next_token = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if (next_token == tokenizer.sep_token_id).all(): break
    return tgt


def compute_bleu(model, dataloader, device, tokenizer, max_decode_len):
    model.eval()
    hypotheses, references = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing BLEU", disable=(dist.get_rank() != 0)):
            src, tgt_texts = batch['src'].to(device), batch['tgt_text']
            translations = greedy_decode(model, src, max_decode_len, device, tokenizer)
            for i in range(translations.size(0)):
                hyp_ids = [tid for tid in translations[i].cpu().tolist() if tid not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]]
                hypotheses.append(tokenizer.decode(hyp_ids))
                references.append(tgt_texts[i])
    return sacrebleu.corpus_bleu(hypotheses, [references]).score if hypotheses else 0.0


def get_lr_schedule(step, warmup_steps, d_model):
    step = max(1, step)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, epoch, step, best_bleu, checkpoint_path, global_rank):
    """Save checkpoint using regular torch.save instead of distributed checkpoint."""
    if global_rank == 0:  # rank 0에서만 저장
        # DDP wrapper를 벗겨낸 모델 사용
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        # optimizer state를 표준 형식으로 변환
        optimizer_state = optimizer.distributed_state_dict(
            key_to_param=dict(model.module.named_parameters())
        )
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'best_bleu': best_bleu
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

def save_checkpoint_for_condition_number(model, optimizer, epoch, checkpoint_path, global_rank):
    """Save checkpoint in standard format for condition-number.py analysis."""
    if global_rank == 0:
        actual_model = model.module if hasattr(model, 'module') else model
        
        optimizer_state_dict = optimizer.distributed_state_dict(
            key_to_param=dict(actual_model.named_parameters())
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': actual_model.state_dict(),
            'optimizer_state_dict': optimizer_state_dict,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Standard checkpoint saved to {checkpoint_path} for condition number analysis")

def main(args):
    local_rank = setup()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Running DDP on rank {global_rank}/{world_size}, local rank {local_rank}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir) if global_rank == 0 else None

    global tokenizer  # global로 선언하여 다른 함수에서도 사용 가능
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    train_loader, val_loader = create_data_loaders(args, tokenizer, global_rank, world_size)

    model = Transformer(
        vocab_size=len(tokenizer), d_model=args.d_model, n_heads=args.n_heads,
        n_encoder_layers=args.n_layers, n_decoder_layers=args.n_layers, d_ff=args.d_ff,
        max_seq_len=args.max_seq_len, dropout=args.dropout,
        pad_idx=tokenizer.pad_token_id
    ).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, 
                find_unused_parameters=False, broadcast_buffers=False)  # broadcast_buffers 비활성화

    optimizer = DistributedShampoo(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
        epsilon=1e-8, weight_decay=args.weight_decay,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        use_decoupled_weight_decay=True,
        grafting_config=AdamGraftingConfig(beta2=args.beta2, epsilon=1e-8),
        distributed_config=DDPShampooConfig(communication_dtype=CommunicationDType.FP32, num_trainers_per_group=-1, communicate_params=False)
    )

    global_step, best_bleu = 0, 0
    for epoch in range(args.epochs):
        args.current_epoch = epoch
        if global_rank == 0:
            print(f"\n{'='*50}\nEpoch {epoch+1}/{args.epochs}\n{'='*50}")
        train_loader.sampler.set_epoch(epoch)

        start_time = time.time()
        train_loss, global_step = train_epoch(model, train_loader, optimizer, local_rank, args, global_step)
        train_time = time.time() - start_time

        val_loss = evaluate(model, val_loader, device=local_rank, tokenizer=tokenizer)
        bleu_score = 0.0
        if (epoch + 1) % args.bleu_interval == 0:
            bleu_score = compute_bleu(model, val_loader, local_rank, tokenizer, args.max_seq_len)

        metrics_tensor = torch.tensor([train_loss, val_loss, bleu_score], device=local_rank)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        train_loss, val_loss, bleu_score = metrics_tensor.tolist()

        if global_rank == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu_score:.2f} | LR: {lr:.6f} | Time: {train_time:.1f}s")
            if writer:
                writer.add_scalar('Train/Loss', train_loss, global_step)
                writer.add_scalar('Val/Loss', val_loss, global_step)
                if (epoch + 1) % args.bleu_interval == 0: writer.add_scalar('Val/BLEU', bleu_score, global_step)
                writer.add_scalar('Train/LearningRate', lr, global_step)

        if bleu_score > best_bleu:
            best_bleu = bleu_score
            if global_rank == 0: print(f"*** New best BLEU: {best_bleu:.2f} ***")
            save_checkpoint(model, optimizer, epoch, global_step, best_bleu, 
                os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"), global_rank)
            # condition-number.py를 위한 표준 체크포인트 저장
            save_checkpoint_for_condition_number(model, optimizer, epoch, os.path.join(args.checkpoint_dir, f"best_model_epoch_{epoch+1}.pth"), global_rank)

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}")
            save_checkpoint(model, optimizer, epoch, global_step, best_bleu, checkpoint_dir, global_rank)
            # condition-number.py를 위한 표준 체크포인트 저장
            save_checkpoint_for_condition_number(model, optimizer, epoch, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"), global_rank)

    if global_rank == 0:
        print(f"\n{'='*50}\nTraining completed! Best BLEU score: {best_bleu:.2f}\n{'='*50}")
    if writer: writer.close()
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlgoPerf Transformer + WMT with Distributed Shampoo')

    # Model arguments
    parser.add_argument('--d-model', type=int, default=1024, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=6, help='Number of encoder/decoder layers')
    parser.add_argument('--d-ff', type=int, default=4096, help='Feed-forward dimension')
    parser.add_argument('--max-seq-len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1.0, help='Base learning rate')
    parser.add_argument('--warmup-steps', type=int, default=4000, help='Warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='Adam beta2')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')

    # Shampoo arguments
    parser.add_argument('--max-preconditioner-dim', type=int, default=1024, help='Max preconditioner dimension')
    parser.add_argument('--precondition-frequency', type=int, default=50, help='Preconditioning frequency')
    parser.add_argument('--start-preconditioning-step', type=int, default=50, help='Step to start preconditioning')

    # Data arguments
    parser.add_argument('--data-path', type=str, default='./wmt_data', help='Path to cache datasets')
    parser.add_argument('--workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--max-train-samples', type=int, default=None, help='Max training samples for debugging')

    # Logging arguments
    parser.add_argument('--log-dir', type=str, default='logs/transformer_shampoo', help='TensorBoard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/transformer_shampoo', help='Checkpoint directory')
    parser.add_argument('--save-interval', type=int, default=10, help='Save checkpoint interval')
    parser.add_argument('--bleu-interval', type=int, default=5, help='BLEU evaluation interval')

    args = parser.parse_args()
    main(args)
