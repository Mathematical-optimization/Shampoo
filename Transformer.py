# transformer_algoperf.py

import os
import math
import argparse
import functools
from typing import Optional, Tuple, List
from dataclasses import dataclass
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

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
import numpy as np
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
        # clone()을 사용하여 메모리 공유 문제 해결
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1).clone())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, batch_size, d_model]"""
        # clone()을 사용하여 새로운 텐서 생성
        return x + self.pe[:x.size(0)].clone()


class CustomMultiheadAttention(nn.Module):
    """Multi-Head Attention with separate Q,K,V projections for Shampoo tracking"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Separate Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len, batch_size = query.size(0), query.size(1)
        
        # Linear projections in batch from [seq_len, batch, d_model] => [seq_len, batch, h, d_k]
        Q = self.q_proj(query).view(seq_len, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(-1, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(-1, batch_size, self.n_heads, self.d_k).transpose(1, 2)
        
        # Q: [seq_len, n_heads, batch_size, d_k]
        # K, V: [src_len, n_heads, batch_size, d_k]
        
        Q = Q.transpose(0, 2)  # [batch_size, n_heads, seq_len, d_k]
        K = K.transpose(0, 2)  # [batch_size, n_heads, src_len, d_k]
        V = V.transpose(0, 2)  # [batch_size, n_heads, src_len, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        # context: [batch_size, n_heads, seq_len, d_k]
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # context: [batch_size, seq_len, d_model]
        
        context = context.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        output = self.out_proj(context)
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer (Pre-LN as in AlgoPerf)"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN architecture
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, mask))
        
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ffn(x2))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer (Pre-LN as in AlgoPerf)"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, n_heads, dropout)
        self.cross_attn = CustomMultiheadAttention(d_model, n_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN architecture
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, tgt_mask))
        
        x2 = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(x2, encoder_output, encoder_output, src_mask))
        
        x2 = self.norm3(x)
        x = x + self.dropout3(self.ffn(x2))
        
        return x


class Transformer(nn.Module):
    """Full Transformer model for translation (AlgoPerf configuration)"""
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 1024,
                 n_heads: int = 16,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 d_ff: int = 4096,
                 max_seq_len: int = 256,
                 dropout: float = 0.1,
                 label_smoothing: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing
        
        # Shared embeddings
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Final layer norm
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # Output projection - weight sharing
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Xavier uniform initialization"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask [batch_size, 1, 1, seq_len]"""
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_look_ahead_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for decoder"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return ~mask
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encoder forward pass"""
        # src: [batch_size, src_seq_len]
        x = self.shared_embedding(src) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [src_seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        x = self.encoder_norm(x)
        return x
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decoder forward pass"""
        # tgt: [batch_size, tgt_seq_len]
        x = self.shared_embedding(tgt) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [tgt_seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        x = self.decoder_norm(x)
        return x
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Full forward pass"""
        # Create masks
        src_mask = self.create_padding_mask(src)
        
        # Decoder mask (padding + look-ahead)
        tgt_seq_len = tgt.size(1)
        tgt_pad_mask = self.create_padding_mask(tgt)
        tgt_la_mask = self.create_look_ahead_mask(tgt_seq_len, tgt.device)
        tgt_mask = tgt_pad_mask & tgt_la_mask.unsqueeze(0).unsqueeze(0)
        
        # Encode and decode
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # Project to vocab - weight sharing
        decoder_output_transposed = decoder_output.transpose(0, 1)
        output = F.linear(decoder_output_transposed, self.shared_embedding.weight)
        
        return output


# ============= 데이터 처리 =============

class WMT17Dataset(Dataset):
    """WMT17 DE-EN Dataset"""
    def __init__(self, data, tokenizer, max_seq_len=256, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # WMT17 데이터셋 구조: {'translation': {'de': ..., 'en': ...}}
        src_text = item['translation']['de']
        tgt_text = item['translation']['en']
        
        # Tokenize with truncation
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'src': src_encoding['input_ids'].squeeze(0),
            'tgt': tgt_encoding['input_ids'].squeeze(0),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def create_data_loaders(args, tokenizer, global_rank, world_size):
    """Create data loaders for training and validation"""
    
    # Load WMT17 DE-EN dataset
    print(f"[Rank {global_rank}] Loading WMT17 DE-EN dataset...")
    dataset = load_dataset("wmt17", "de-en", cache_dir=args.data_path)
    
    # 데이터셋 정보 출력
    if global_rank == 0:
        print(f"Dataset keys: {dataset.keys()}")
        print(f"Train dataset size: {len(dataset['train'])}")
    
    # Create dataset objects
    train_data = dataset['train']
    if args.max_train_samples:
        train_data = train_data.select(range(min(args.max_train_samples, len(train_data))))
    
    train_dataset = WMT17Dataset(
        train_data,
        tokenizer, 
        args.max_seq_len, 
        is_train=True
    )
    
    # WMT14 validation set 로드 (AlgoPerf 설정)
    val_dataset_raw = load_dataset("wmt14", "de-en", cache_dir=args.data_path)
    val_dataset = WMT17Dataset(
        val_dataset_raw['validation'],
        tokenizer, 
        args.max_seq_len, 
        is_train=False
    )
    
    if global_rank == 0:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create samplers for DDP
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=global_rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=global_rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============= 학습 함수 =============

def label_smoothed_cross_entropy(logits, targets, epsilon=0.1, ignore_index=-100):
    """Label smoothed cross entropy loss"""
    vocab_size = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Create smoothed target distribution
    smooth_target = torch.full_like(log_probs, epsilon / (vocab_size - 1))
    smooth_target.scatter_(-1, targets.unsqueeze(-1), 1.0 - epsilon)
    
    # Mask padding tokens
    mask = targets != ignore_index
    smooth_target[~mask] = 0
    
    # Compute loss
    loss = -(smooth_target * log_probs).sum(dim=-1)
    loss = loss.masked_select(mask).mean()
    
    return loss


def train_epoch(model, dataloader, optimizer, device, label_smoothing=0.1, 
                grad_clip=1.0, pad_idx=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc="Training", disable=(dist.get_rank() != 0))
    
    for batch in progress_bar:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        # Prepare decoder input and target
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        # Compute loss with label smoothing
        loss = label_smoothed_cross_entropy(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1),
            epsilon=label_smoothing,
            ignore_index=pad_idx
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        # Track loss
        num_tokens = (tgt_output != pad_idx).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        
        # Update progress bar
        if dist.get_rank() == 0:
            progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / total_tokens if total_tokens > 0 else 0


def evaluate(model, dataloader, device, label_smoothing=0.1, pad_idx=0):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0)):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            
            loss = label_smoothed_cross_entropy(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1),
                epsilon=label_smoothing,
                ignore_index=pad_idx
            )
            
            num_tokens = (tgt_output != pad_idx).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    return total_loss / total_tokens if total_tokens > 0 else 0


def greedy_decode(model, src, max_len, device, tokenizer):
    """Greedy decoding for translation"""
    model.eval()
    
    with torch.no_grad():
        # Encode source
        src_mask = model.module.create_padding_mask(src)
        encoder_output = model.module.encode(src, src_mask)
        
        # Start with BOS token
        tgt = torch.full((src.size(0), 1), tokenizer.cls_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # Decode
            tgt_mask = model.module.create_padding_mask(tgt)
            tgt_seq_len = tgt.size(1)
            tgt_la_mask = model.module.create_look_ahead_mask(tgt_seq_len, device)
            tgt_mask = tgt_mask & tgt_la_mask.unsqueeze(0).unsqueeze(0)
            
            decoder_output = model.module.decode(tgt, encoder_output, tgt_mask, src_mask)
            
            # Get next token
            logits = F.linear(decoder_output[-1, :, :], model.module.shared_embedding.weight)
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_token == tokenizer.sep_token_id).all():
                break
    
    return tgt


def compute_bleu(model, dataloader, device, tokenizer, max_decode_len=256):
    """Compute BLEU score on validation set"""
    model.eval()
    
    hypotheses = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing BLEU", disable=(dist.get_rank() != 0)):
            src = batch['src'].to(device)
            tgt_texts = batch['tgt_text']
            
            # Generate translations
            translations = greedy_decode(model, src, max_decode_len, device, tokenizer)
            
            # Decode to text
            for i in range(translations.size(0)):
                # Get hypothesis
                hyp_ids = translations[i].cpu().tolist()
                # Remove special tokens
                hyp_ids = [tid for tid in hyp_ids if tid not in [tokenizer.cls_token_id, 
                                                                   tokenizer.sep_token_id, 
                                                                   tokenizer.pad_token_id]]
                hyp_text = tokenizer.decode(hyp_ids, skip_special_tokens=True)
                hypotheses.append(hyp_text)
                
                # Get reference
                references.append(tgt_texts[i])
    
    # Compute BLEU
    if len(hypotheses) > 0:
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return bleu.score
    return 0.0


def get_lr_schedule(step, warmup_steps, d_model):
    """Transformer learning rate schedule"""
    if step < warmup_steps:
        return (d_model ** -0.5) * step * (warmup_steps ** -1.5)
    else:
        return (d_model ** -0.5) * (step ** -0.5)


def setup():
    """Initialize DDP"""
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    """Clean up DDP"""
    dist.destroy_process_group()


def save_checkpoint(model, optimizer, epoch, step, best_bleu, checkpoint_path, global_rank):
    """Save checkpoint"""
    if global_rank == 0:
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.module.state_dict(),
            'best_bleu': best_bleu
        }
        
        # Shampoo optimizer state 저장
        try:
            checkpoint['optimizer_state_dict'] = optimizer.distributed_state_dict(
                key_to_param=model.module.named_parameters()
            )
        except:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def main(args):
    """Main training loop"""
    # Setup DDP
    local_rank = setup()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Running DDP on rank {global_rank}/{world_size}, local rank {local_rank}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir) if global_rank == 0 else None
    
    # Load tokenizer
    print(f"[Rank {global_rank}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    vocab_size = len(tokenizer)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args, tokenizer, global_rank, world_size)
    
    # Create model (AlgoPerf configuration)
    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        pad_idx=tokenizer.pad_token_id
    ).to(local_rank)
    
    # Wrap in DDP with broadcast_buffers=False to avoid buffer sync issues
    model = DDP(
        model, 
        device_ids=[local_rank],
        output_device = local_rank,
        broadcast_buffers=False,  # 버퍼 동기화 비활성화
        find_unused_parameters=True
    )
    
    # Initialize Distributed Shampoo optimizer
    optimizer = DistributedShampoo(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        epsilon=1e-8,
        weight_decay=args.weight_decay,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        use_decoupled_weight_decay=True,
        grafting_config=AdamGraftingConfig(
            beta2=args.beta2,
            epsilon=1e-8
        ),
        distributed_config=DDPShampooConfig(
            communication_dtype=CommunicationDType.FP32,
            num_trainers_per_group=-1,
            communicate_params=False
        )
    )
    
    # Training variables
    global_step = 0
    best_bleu = 0
    
    # Training loop
    for epoch in range(args.epochs):
        if global_rank == 0:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"{'='*50}")
        
        # Set epoch for sampler
        train_loader.sampler.set_epoch(epoch)
        
        # Train
        start_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, local_rank, 
            args.label_smoothing, args.grad_clip, tokenizer.pad_token_id
        )
        train_time = time.time() - start_time
        
        # Update learning rate (per step)
        for _ in range(len(train_loader)):
            global_step += 1
            lr = get_lr_schedule(global_step, args.warmup_steps, args.d_model)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Evaluate
        val_loss = evaluate(model, val_loader, local_rank, args.label_smoothing, tokenizer.pad_token_id)
        
        # Compute BLEU periodically
        bleu_score = 0
        if (epoch + 1) % args.bleu_interval == 0:
            bleu_score = compute_bleu(model, val_loader, local_rank, tokenizer, args.max_seq_len)
            
            # Save best model
            if global_rank == 0 and bleu_score > best_bleu:
                best_bleu = bleu_score
                checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model.pth")
                save_checkpoint(model, optimizer, epoch, global_step, best_bleu, 
                              checkpoint_path, global_rank)
        
        # Gather metrics from all ranks
        metrics_tensor = torch.tensor([train_loss, val_loss, bleu_score], device=local_rank)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        train_loss, val_loss, bleu_score = metrics_tensor.tolist()
        
        # Logging
        if global_rank == 0:
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"BLEU: {bleu_score:.2f} | LR: {lr:.6f} | Time: {train_time:.1f}s")
            
            if writer:
                writer.add_scalar('Train/Loss', train_loss, epoch)
                writer.add_scalar('Val/Loss', val_loss, epoch)
                writer.add_scalar('Val/BLEU', bleu_score, epoch)
                writer.add_scalar('Train/LearningRate', lr, epoch)
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, global_step, best_bleu, 
                          checkpoint_path, global_rank)
    
    # Final evaluation
    if global_rank == 0:
        print(f"\n{'='*50}")
        print("Training completed!")
        print(f"Best BLEU score: {best_bleu:.2f}")
        print(f"{'='*50}")
    
    if writer:
        writer.close()
    
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlgoPerf Transformer + WMT with Distributed Shampoo')
    
    # Model arguments (AlgoPerf Transformer-big configuration)
    parser.add_argument('--d-model', type=int, default=256,
                      help='Model dimension (AlgoPerf uses 1024)')
    parser.add_argument('--n-heads', type=int, default=4,
                      help='Number of attention heads (AlgoPerf uses 16)')
    parser.add_argument('--n-layers', type=int, default=4,
                      help='Number of encoder/decoder layers')
    parser.add_argument('--d-ff', type=int, default=1024,
                      help='Feed-forward dimension (AlgoPerf uses 4096)')
    parser.add_argument('--max-seq-len', type=int, default=128,
                      help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                      help='Label smoothing (AlgoPerf uses 0.1)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=90,
                      help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1.0,
                      help='Base learning rate (scaled by schedule)')
    parser.add_argument('--warmup-steps', type=int, default=103700,
                      help='Warmup steps for learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                      help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9,
                      help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98,
                      help='Adam beta2 (AlgoPerf uses 0.98)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                      help='Gradient clipping')
    
    # Shampoo arguments
    parser.add_argument('--max-preconditioner-dim', type=int, default=1024,
                      help='Maximum preconditioner dimension')
    parser.add_argument('--precondition-frequency', type=int, default=50,
                      help='Preconditioning frequency')
    parser.add_argument('--start-preconditioning-step', type=int, default=50,
                      help='Step to start preconditioning')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='./wmt_data',
                      help='Path to cache datasets')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of data workers')
    parser.add_argument('--max-train-samples', type=int, default=None,
                      help='Maximum training samples (for debugging)')
    
    # Logging arguments
    parser.add_argument('--log-dir', type=str, default='logs/transformer_shampoo',
                      help='TensorBoard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/transformer_shampoo',
                      help='Checkpoint directory')
    parser.add_argument('--save-interval', type=int, default=10,
                      help='Save checkpoint interval')
    parser.add_argument('--bleu-interval', type=int, default=5,
                      help='BLEU evaluation interval')
    
    args = parser.parse_args()
    
    main(args)