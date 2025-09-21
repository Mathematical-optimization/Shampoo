import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from typing import Dict, List
import math
import argparse
from torch.utils.tensorboard import SummaryWriter
import functools
# Hugging Face datasets 라이브러리 import
from datasets import load_dataset
from PIL import Image

# timm 라이브러리 import
try:
    from timm.data import create_transform, Mixup
except ImportError:
    print("ERROR: timm library not found. Please install it using 'pip install timm'")
    exit(1)

# Shampoo 옵티마이저 라이브러리 import
from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    DDPShampooConfig,
)

# --- ViT 모델 코드 수정 ---

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))

# [신규] Q, K, V가 분리된 커스텀 Multi-Head Attention 모듈
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim: int = 384, num_heads: int = 6, attn_dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"

        # Q, K, V를 위한 별도의 Linear 레이어
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, need_weights=False):
        batch_size = query.shape[0]

        # 1. Linear 프로젝션
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Multi-head를 위해 reshape 및 전치
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 4. Attention 적용
        attn_output = torch.matmul(attn_probs, v)

        # 5. Concat 및 Output 프로젝션
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embedding_dim)
        output = self.out_proj(attn_output)
        
        return output, None

# [수정] TransformerEncoderBlock에서 CustomMultiheadAttention 사용
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int = 384, num_heads: int = 6, mlp_dim: int = 1536, attn_dropout: float = 0.0, mlp_dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=embedding_dim)
        # 기존 nn.MultiheadAttention을 CustomMultiheadAttention으로 교체
        self.attn = CustomMultiheadAttention(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = MLPBlock(embedding_dim=embedding_dim, mlp_dim=mlp_dim, dropout=mlp_dropout)
        
    def forward(self, x):
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(query=norm_x, key=norm_x, value=norm_x, need_weights=False)
        x = x + attn_output
        norm_x_mlp = self.norm2(x)
        mlp_output = self.mlp(norm_x_mlp)
        x = x + mlp_output
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size: int = 224, in_channels: int = 3, patch_size: int = 16, num_classes: int = 1000, embedding_dim: int = 384, depth: int = 12, num_heads: int = 6, mlp_dim: int = 1536, attn_dropout: float = 0.0, mlp_dropout: float = 0.1, embedding_dropout: float = 0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.encoder_blocks = nn.ModuleList([TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim, attn_dropout=attn_dropout, mlp_dropout=mlp_dropout) for _ in range(depth)])
        self.classifier_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x).flatten(2, 3).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.classifier_norm(x)
        cls_token_final = x[:, 0]
        logits = self.classifier_head(cls_token_final)
        return logits

# --- 나머지 코드는 변경 없음 ---

def get_warmup_cosine_decay_lr(current_step: int, base_lr: float, num_steps: int, warmup_steps: int) -> float:
    if current_step < warmup_steps:
        return base_lr * (current_step / warmup_steps)
    else:
        progress = (current_step - warmup_steps) / (num_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return base_lr * cosine_decay

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
def cleanup():
    dist.destroy_process_group()

def apply_transforms(examples: Dict[str, List[Image.Image]], transform) -> Dict[str, List[torch.Tensor]]:
    examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['image']]
    return examples
        
def train(args: argparse.Namespace):
    setup()
    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    
    print(f"Running DDP training. Global Rank: {global_rank}, Local Rank: {local_rank}, World Size: {world_size}")

    writer = SummaryWriter(log_dir=args.log_dir) if global_rank == 0 else None

    train_transform = create_transform(
        input_size=224,
        is_training=True,
        auto_augment='rand-m15-n2-mstd0.5',
        interpolation='bicubic',
    )

    val_transform = create_transform(
        input_size=224,
        is_training=False,
        interpolation='bicubic',
    )

    mixup_fn = None
    if args.mixup > 0 or args.label_smoothing > 0:
        mixup_args = {
            'mixup_alpha': args.mixup,
            'cutmix_alpha': 0.0,
            'label_smoothing': args.label_smoothing,
            'num_classes': 1000
        }
        mixup_fn = Mixup(**mixup_args)

    if global_rank == 0:
        print("Hugging Face Hub에서 ImageNet-1k 데이터셋을 다운로드 및 캐싱합니다...")
        load_dataset("imagenet-1k", cache_dir=args.data_path)
    dist.barrier()

    print(f"Rank {global_rank}에서 캐시된 ImageNet-1k 데이터셋을 로딩합니다...")
    dataset = load_dataset("imagenet-1k", cache_dir=args.data_path)

    train_dataset = dataset['train']
    train_dataset.set_transform(functools.partial(apply_transforms, transform=train_transform))
    
    val_dataset = dataset['validation']
    val_dataset.set_transform(functools.partial(apply_transforms, transform=val_transform))
    
    def collate_fn(batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'label': torch.tensor([x['label'] for x in batch], dtype=torch.long)
        }

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    vit_s16_params = {
        'img_size': 224, 'patch_size': 16, 'embedding_dim': 384, 'depth': 12,
        'num_heads': 6, 'mlp_dim': 1536, 'num_classes': 1000
    }
    model = VisionTransformer(**vit_s16_params).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().to(local_rank)

    optimizer = DistributedShampoo(
        model.parameters(),
        lr=args.base_lr,
        betas=(args.beta1, 0.99),
        epsilon=1e-8,
        momentum=False,
        weight_decay=args.weight_decay,
        max_preconditioner_dim=1024,
        precondition_frequency=20,
        use_normalized_grafting=False,
        inv_root_override=2,
        exponent_multiplier=1,
        start_preconditioning_step=20,
        use_nadam=False,
        use_decoupled_weight_decay=True,
        grafting_config=AdamGraftingConfig(beta2=0.995, epsilon=1e-8),
        distributed_config=DDPShampooConfig()
    )
    
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')
            if 'model_state_dict' in checkpoint:
                 model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                 model.module.load_state_dict(checkpoint)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("=> loaded optimizer state")
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    total_steps = len(train_loader) * args.epochs
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        for i, batch in enumerate(train_loader):
            current_step = epoch * len(train_loader) + i
            images = batch['pixel_values'].to(local_rank, non_blocking=True)
            labels = batch['label'].to(local_rank, non_blocking=True)
            
            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            new_lr = get_warmup_cosine_decay_lr(current_step, args.base_lr, total_steps, args.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if global_rank == 0 and (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], LR: {new_lr:.6f}, Loss: {loss.item():.4f}")
                if writer:
                    writer.add_scalar('training_loss', loss.item(), current_step)
                    writer.add_scalar('learning_rate', new_lr, current_step)
        
        model.eval()
        correct = 0 
        total = 0
        val_loss_sum = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                images = batch['pixel_values'].to(local_rank, non_blocking=True)
                labels = batch['label'].to(local_rank, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        total_tensor = torch.tensor(total).to(local_rank)
        correct_tensor = torch.tensor(correct).to(local_rank)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        
        local_avg_loss = val_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_loss_tensor = torch.tensor(local_avg_loss).to(local_rank)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        
        avg_val_loss = avg_loss_tensor.item() / world_size
        accuracy = 100 * correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0

        if global_rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Validation Accuracy: {accuracy:.2f}%, Validation Loss: {avg_val_loss:.4f}")
            if writer:
                writer.add_scalar('validation_accuracy', accuracy, epoch)
                writer.add_scalar('validation_loss', avg_val_loss, epoch)

            if (epoch + 1) % args.save_interval == 0:
                save_path = os.path.join(args.save_dir, f"vit_checkpoint_epoch_{epoch+1}.pth")
                print(f"Saving checkpoint to {save_path}")
                optimizer_state_dict = optimizer.distributed_state_dict(
                    key_to_param=model.module.named_parameters()
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer_state_dict,
                }, save_path)
    if writer:
        writer.close()
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision Transformer on ImageNet with DDP, Shampoo, and Algoperf Augmentations')
    parser.add_argument('--data-path', type=str, required=True, help='Path to cache Hugging Face datasets')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for TensorBoard logs')
    parser.add_argument('--epochs', type=int, default=90, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--log-interval', type=int, default=300, help='Logging frequency')
    parser.add_argument('--save-interval', type=int, default=10, help='Checkpoint saving frequency')
    parser.add_argument('--base-lr', type=float, default=0.0013, help='Base learning rate')
    parser.add_argument('--warmup-steps', type=int, default=7500, help='Number of warmup steps')
    parser.add_argument('--mixup', type=float, default=0.2, help='Mixup alpha (default: 0.2). Set 0 to disable.')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay (default: 0.1)')
    parser.add_argument('--beta1', type=float, default=0.95, help='Beta1/Momentum (default: 0.9)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()
    train(args)