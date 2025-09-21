import torch
import torch.nn as nn
import argparse
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.distributed as dist
from vit import VisionTransformer, DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    DDPShampooConfig,
)
from torch.nn.parallel import DistributedDataParallel as DDP
# vit.py에서 모델과 옵티마이저 정의를 가져옵니다.
from vit import VisionTransformer, DistributedShampoo

def plot_condition_number_trends(checkpoint_dir: str):
    """
    지정된 디렉토리의 모든 체크포인트를 읽어
    Shampoo Preconditioner의 Condition Number 변화 추이를 그래프로 저장합니다.
    """
    if not os.path.isdir(checkpoint_dir):
        print(f"오류: 디렉토리를 찾을 수 없습니다 -> {checkpoint_dir}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"분석을 위해 {device} 장치를 사용합니다.")

    # 1. 체크포인트 파일 목록을 epoch 순서대로 정렬
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    # 'epoch_XX.pth' 또는 'vit_checkpoint_epoch_XX.pth' 패턴에서 숫자(epoch)를 추출하여 정렬
    try:
        checkpoint_files.sort(key=lambda f: int(re.search(r'epoch_(\d+)\.pth', f).group(1)))
    except (TypeError, AttributeError):
        print("오류: '...epoch_XX.pth' 형식의 파일을 찾을 수 없습니다.")
        return
        
    if not checkpoint_files:
        print(f"오류: '{checkpoint_dir}' 디렉토리에서 체크포인트 파일(.pth)을 찾을 수 없습니다.")
        return

    print(f"총 {len(checkpoint_files)}개의 체크포인트 파일을 분석합니다.")
    print(checkpoint_files)

    # 2. Condition Number 데이터를 저장할 딕셔너리
    # 구조: results['파라미터명']['팩터 ID'] = [(epoch1, cond_num1), (epoch2, cond_num2), ...]
    results = defaultdict(lambda: defaultdict(list))

    # 3. 각 체크포인트를 순회하며 데이터 수집
    for filename in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        epoch_match = re.search(r'epoch_(\d+)\.pth', filename)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))
        
        print(f"\n--- Epoch {epoch} 체크포인트 분석 중 ---")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Epoch {epoch} 체크포인트 로딩 실패: {e}")
            continue

        # 모델과 옵티마이저 초기화
        model = VisionTransformer(img_size=224, patch_size=16, embedding_dim=384, depth=12, num_heads=6, mlp_dim=1536, num_classes=1000)
        model.to(device)
        if device.type == 'cuda':
            model = DDP(model, device_ids=[device.index])
        else:
            model = DDP(model)
            
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=0.0013,
            betas=(0.95, 0.99),
            epsilon=1e-8,
            momentum=False,
            weight_decay=0.0005,
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
        try:
            dummy_input = torch.randn(2,3,224,224, device = device)
            dummy_labels = torch.randint(0, 1000, (2,), device = device)
            optimizer.zero_grad()
            outputs = model(dummy_input)
            loss = nn.CrossEntropyLoss()(outputs, dummy_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("옵티마이저 상태 초기화를 위한 더미 스텝 완료.")
        except Exception as e:
            print(f"더미 스텝 중 오류 발생: {e}")
            

        if 'model_state_dict' in checkpoint:
            # DDP로 학습된 모델은 'module.' 접두사가 붙어있을 수 있으므로 제거
            model_state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(model_state, strict=False)
        
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            try:
                optimizer.load_distributed_state_dict(
                    state_dict=checkpoint['optimizer_state_dict'], 
                    key_to_param=model.named_parameters()
                )
                print(f"Epoch {epoch} 옵티마이저 상태 로딩 성공!")
            except Exception as e:
                print(f"주의: Epoch {epoch} 옵티마이저 상태 로딩 중 오류 발생. 일부 Preconditioner 정보가 누락될 수 있습니다. (오류: {e})")
                pass
        else:
            print(f"Epoch {epoch} 체크포인트에 옵티마이저 상태가 없어 건너뜁니다.")
            continue
            
        param_map = {p: name for name, p in model.named_parameters()}

        for param, state_dict in optimizer.state.items():
            if param not in param_map:
                continue
            param_name = param_map.get(param)
            
            for key, value in state_dict.items():
                if isinstance(key, str) and 'block' in key and 'shampoo' in value:
                    shampoo_state = value['shampoo']
                    if hasattr(shampoo_state, 'factor_matrices'):
                        for i, factor_matrix in enumerate(shampoo_state.factor_matrices):
                            # 컨디션 넘버 계산은 2D 정방 행렬에 대해서만 가능
                            if factor_matrix.ndim == 2 and factor_matrix.shape[0] == factor_matrix.shape[1] and factor_matrix.numel() > 1:
                                matrix_to_eval = factor_matrix.detach().to(torch.float64)
                                # torch.linalg.cond가 0으로 된 행렬에 대해 발산하는 것을 방지
                                matrix_to_eval += torch.eye(matrix_to_eval.shape[0], device=matrix_to_eval.device) * 1e-8
                                cond_num = torch.linalg.cond(matrix_to_eval).item()
                                
                                # 결과 저장
                                factor_id = f"{param_name}_{key}_factor_{i}"
                                results[param_name][factor_id].append((epoch, cond_num))
    
    # 4. 수집된 데이터로 그래프 시각화 (개선된 버전)
    print("\n--- 모든 체크포인트 분석 완료. 그래프 생성 중... ---")

    # Q, K, V 파라미터만 필터링
    qkv_results = defaultdict(lambda: defaultdict(list))
    for param_name, factors in results.items():
        if 'q_proj' in param_name or 'k_proj' in param_name or 'v_proj' in param_name:
            qkv_results[param_name] = factors

    if not qkv_results:
        print("분석할 Query, Key, Value 파라미터 데이터를 찾을 수 없습니다.")
        return

    # Attention 블록별로 그래프를 그룹화하기 위한 딕셔너리
    # 구조: block_plots['encoder_blocks.0.attn'] = {'q_proj_...': [...], 'k_proj_...': [...]}
    block_plots = defaultdict(dict)
    for param_name, factors in qkv_results.items():
        # 'encoder_blocks.0.attn.q_proj.weight' -> 'encoder_blocks.0.attn'
        block_name = '.'.join(param_name.split('.')[:-2]) 
        block_plots[block_name].update(factors)

    num_blocks = len(block_plots)
    if num_blocks == 0:
        print("Attention 블록 데이터를 찾을 수 없습니다.")
        return
        
    fig, axes = plt.subplots(num_blocks, 1, figsize=(15, 8 * num_blocks), sharex=True)
    if num_blocks == 1:
        axes = [axes]

    # 각 Attention 블록별로 subplot 생성
    for ax, (block_name, factors) in zip(axes, sorted(block_plots.items())):
        for factor_id, data_points in sorted(factors.items()):
            if data_points:
                epochs, cond_nums = zip(*sorted(data_points))
                # '...q_proj.weight_block_0_factor_0' -> 'q_proj.weight_..._factor_0'
                label_suffix = factor_id.split(block_name + '.')[1]
                
                # Q, K, V에 따라 색상 지정
                color = 'r' if 'q_proj' in label_suffix else 'g' if 'k_proj' in label_suffix else 'b'
                linestyle = '--' if 'factor_1' in label_suffix else '-'
                
                ax.plot(epochs, cond_nums, marker='o', linestyle=linestyle, label=label_suffix, color=color)
        
        ax.set_yscale('log')
        ax.set_title(f"Condition Number Trend for '{block_name}'")
        ax.set_ylabel("Condition Number (log scale)")
        ax.legend(fontsize='small', loc='upper left')
        ax.grid(True, which="both", ls="--")

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    
    save_path = "qkv_condition_number_trends.png"
    plt.savefig(save_path)
    print(f"\nQ/K/V 그래프가 '{save_path}' 파일로 저장되었습니다.")


if __name__ == '__main__':
    # 분산 환경에서 저장된 체크포인트를 로드하기 위해 가짜 분산 환경 설정
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        # 'nccl' 백엔드는 CUDA가 필요하므로, CPU만 있는 환경에서는 'gloo' 사용
        backend = 'gloo' if not torch.cuda.is_available() else 'nccl'
        dist.init_process_group(backend=backend, rank=0, world_size=1)
        
    parser = argparse.ArgumentParser(description='Plot Shampoo Preconditioner Condition Number trends from checkpoints.')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory containing the .pth checkpoint files.')
    args = parser.parse_args()
    
    plot_condition_number_trends(args.checkpoint_dir)

    if dist.is_initialized():
        dist.destroy_process_group()
