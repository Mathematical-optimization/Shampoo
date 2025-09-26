import torch
import torch.nn as nn
import argparse
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import json

def inspect_checkpoint_structure(checkpoint_path: str, verbose: bool = False):
    """체크포인트 구조를 확인하고 디버깅 정보를 출력합니다."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if verbose:
        print("\n=== Checkpoint Structure Inspection ===")
        print(f"Top-level keys: {checkpoint.keys()}")
    
    structure_info = {
        'has_optimizer_state': False,
        'has_state_dict': False,
        'param_count': 0,
        'factor_matrix_count': 0,
        'sample_keys': []
    }
    
    if 'optimizer_state_dict' in checkpoint:
        structure_info['has_optimizer_state'] = True
        opt_state = checkpoint['optimizer_state_dict']
        
        if 'state' in opt_state:
            structure_info['has_state_dict'] = True
            param_states = opt_state['state']
            structure_info['param_count'] = len(param_states)
            
            # 첫 번째 Q/K/V 파라미터의 구조 확인
            for param_name, param_state in param_states.items():
                if any(proj in param_name for proj in ['q_proj', 'k_proj', 'v_proj']):
                    # Factor matrix 키 수집
                    factor_keys = [k for k in param_state.keys() 
                                  if isinstance(k, str) and 'factor_matrices' in k]
                    structure_info['factor_matrix_count'] = len(factor_keys)
                    structure_info['sample_keys'] = factor_keys[:2]
                    
                    if verbose and factor_keys:
                        print(f"\nSample parameter: {param_name}")
                        print(f"Number of factor matrices: {len(factor_keys)}")
                        print(f"Sample keys: {factor_keys[:2]}")
                    break
    
    return structure_info

def parse_state_key(state_key: str):
    """State key를 파싱하여 구성 요소를 추출합니다."""
    try:
        # JSON 형식의 키 파싱
        if isinstance(state_key, str) and state_key.startswith('['):
            key_parts = json.loads(state_key)
            
            # 예상 형식: ["block_X", "shampoo", "factor_matrices", idx]
            if (isinstance(key_parts, list) and len(key_parts) >= 4 and
                'shampoo' in key_parts and 'factor_matrices' in key_parts):
                
                block_id = key_parts[0] if 'block' in key_parts[0] else None
                factor_idx = key_parts[-1] if isinstance(key_parts[-1], int) else None
                
                return {
                    'is_factor_matrix': True,
                    'block_id': block_id,
                    'factor_idx': factor_idx
                }
    except (json.JSONDecodeError, IndexError):
        pass
    
    return {'is_factor_matrix': False}

def compute_condition_number(matrix: torch.Tensor, epsilon: float = 1e-10) -> float:
    """행렬의 condition number를 계산합니다."""
    try:
        # Double precision으로 변환
        matrix = matrix.detach().double()
        
        # 수치 안정성을 위한 작은 값 추가
        if matrix.shape[0] == matrix.shape[1]:
            matrix = matrix + torch.eye(matrix.shape[0], dtype=torch.float64) * epsilon
        
        # Condition number 계산
        cond_num = torch.linalg.cond(matrix).item()
        
        # NaN이나 Inf 체크
        if np.isnan(cond_num) or np.isinf(cond_num):
            return float('inf')
            
        return cond_num
        
    except Exception as e:
        print(f"  Condition number 계산 실패: {e}")
        return float('inf')

def apply_bias_correction(matrix: torch.Tensor, beta2: float, step: int) -> torch.Tensor:
    """Bias correction을 적용합니다."""
    if beta2 < 1.0 and step > 0:
        bias_correction = 1.0 - (beta2 ** step)
        return matrix / bias_correction
    return matrix

def plot_condition_number_trends(checkpoint_dir: str, beta2: float = 0.99):
    """
    지정된 디렉토리의 모든 체크포인트를 읽어
    Shampoo Preconditioner의 Condition Number 변화 추이를 그래프로 저장합니다.
    
    Args:
        checkpoint_dir: 체크포인트 디렉토리 경로
        beta2: Shampoo의 beta2 값 (bias correction용)
    """
    if not os.path.isdir(checkpoint_dir):
        print(f"오류: 디렉토리를 찾을 수 없습니다 -> {checkpoint_dir}")
        return
    
    device = torch.device("cpu")
    print(f"분석을 위해 {device} 장치를 사용합니다.")

    # 체크포인트 파일 목록을 epoch 순서대로 정렬
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    try:
        checkpoint_files.sort(key=lambda f: int(re.search(r'epoch_(\d+)\.pth', f).group(1)))
    except (TypeError, AttributeError):
        print("오류: '...epoch_XX.pth' 형식의 파일을 찾을 수 없습니다.")
        return
        
    if not checkpoint_files:
        print(f"오류: '{checkpoint_dir}' 디렉토리에서 체크포인트 파일(.pth)을 찾을 수 없습니다.")
        return

    print(f"총 {len(checkpoint_files)}개의 체크포인트 파일을 분석합니다.")
    
    # 첫 번째 체크포인트 구조 확인
    if checkpoint_files:
        first_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        structure_info = inspect_checkpoint_structure(first_checkpoint_path, verbose=True)
        
        if not structure_info['has_state_dict']:
            print("오류: 체크포인트에서 optimizer state를 찾을 수 없습니다.")
            return

    # Condition Number 데이터를 저장할 딕셔너리
    results = defaultdict(lambda: defaultdict(list))
    factor_matrix_stats = defaultdict(lambda: {'count': 0, 'non_empty': 0})

    # 각 체크포인트를 순회하며 데이터 수집
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

        # optimizer_state_dict에서 정보 추출
        if 'optimizer_state_dict' not in checkpoint:
            print(f"Epoch {epoch}: optimizer_state_dict가 없습니다.")
            continue
            
        opt_state = checkpoint['optimizer_state_dict']
        
        if 'state' not in opt_state:
            print(f"Epoch {epoch}: optimizer state가 없습니다.")
            continue
            
        param_states = opt_state['state']
        
        # 각 파라미터의 상태 확인
        for param_name, param_state in param_states.items():
            # encoder_blocks의 attention 파라미터만 필터링
            if not ('encoder_blocks' in param_name and 'attn' in param_name):
                continue
            
            # Q, K, V projection weights만 처리
            if not any(proj in param_name for proj in ['q_proj', 'k_proj', 'v_proj']):
                continue
                
            if 'weight' not in param_name:
                continue
            
            # 파라미터 이름 파싱
            match = re.search(r'encoder_blocks\.(\d+)\.attn\.(q_proj|k_proj|v_proj)\.weight', param_name)
            if not match:
                continue
                
            block_idx = int(match.group(1))
            proj_type = match.group(2)
            
            # Projection 타입 이름 매핑
            proj_name_map = {'q_proj': 'Query', 'k_proj': 'Key', 'v_proj': 'Value'}
            proj_name = proj_name_map.get(proj_type, proj_type)
            
            # param_state의 각 키 확인
            for state_key, state_value in param_state.items():
                parsed = parse_state_key(state_key)
                
                if parsed['is_factor_matrix'] and parsed['factor_idx'] is not None:
                    factor_idx = parsed['factor_idx']
                    
                    # Tensor 확인 및 처리
                    if isinstance(state_value, torch.Tensor):
                        factor_matrix_stats[param_name]['count'] += 1
                        
                        # 2D 정방 행렬이고 빈 텐서가 아닌지 확인
                        if (state_value.ndim == 2 and 
                            state_value.shape[0] == state_value.shape[1] and 
                            state_value.shape[0] > 1 and
                            state_value.numel() > 0):
                            
                            factor_matrix_stats[param_name]['non_empty'] += 1
                            
                            # Bias correction 적용
                            corrected_matrix = apply_bias_correction(
                                state_value, beta2, epoch
                            )
                            
                            # Condition number 계산
                            cond_num = compute_condition_number(corrected_matrix)
                            
                            if cond_num != float('inf'):
                                # Factor 0 = Left (L), Factor 1 = Right (R)
                                factor_name = 'L' if factor_idx == 0 else 'R'
                                key = f"Block_{block_idx}_{proj_name}_{factor_name}"
                                
                                results[key]['epochs'].append(epoch)
                                results[key]['cond_nums'].append(cond_num)
                                
                                print(f"  {key}: {cond_num:.2e}")

    # 수집된 데이터로 그래프 시각화
    print("\n--- 모든 체크포인트 분석 완료. 그래프 생성 중... ---")

    if not results:
        print("분석할 데이터가 없습니다. Q/K/V projection weights의 factor matrices를 찾을 수 없습니다.")
        return

    # 12개 블록에 대한 서브플롯 생성
    num_blocks = 12  # ViT-S/16은 12개 블록
    rows = 4
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()
    
    # 색상 및 스타일 매핑
    color_map = {'Query': 'red', 'Key': 'green', 'Value': 'blue'}
    marker_map = {'L': 'o', 'R': 's'}  # L은 원, R은 사각형
    
    for block_idx in range(num_blocks):
        ax = axes[block_idx]
        has_data = False
        
        # 각 projection type과 factor에 대해 플롯
        for proj_type in ['Query', 'Key', 'Value']:
            for factor_type in ['L', 'R']:
                key = f"Block_{block_idx}_{proj_type}_{factor_type}"
                
                if key in results and results[key]['epochs']:
                    epochs = results[key]['epochs']
                    cond_nums = results[key]['cond_nums']
                    
                    label = f"{proj_type} ({factor_type})"
                    ax.semilogy(epochs, cond_nums, 
                               marker=marker_map[factor_type],
                               linestyle='-' if factor_type == 'L' else '--',
                               label=label,
                               color=color_map[proj_type],
                               linewidth=2,
                               markersize=5,
                               alpha=0.8)
                    has_data = True
        
        if has_data:
            ax.set_title(f"Block {block_idx}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Condition Number (log scale)", fontsize=10)
            ax.legend(loc='best', fontsize=8, ncol=2)
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.set_ylim(bottom=1e0)
        else:
            ax.set_title(f"Block {block_idx} (No Data)", fontsize=12)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
    
    plt.suptitle("Shampoo Preconditioner Condition Numbers (L and R) for All Transformer Blocks", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(checkpoint_dir), "qkv_condition_numbers_all_blocks.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n그래프가 '{save_path}' 파일로 저장되었습니다.")
    
    # 통계 정보 출력
    print("\n=== Condition Number 통계 ===")
    for key in sorted(results.keys()):
        if results[key]['cond_nums']:
            cond_nums = results[key]['cond_nums']
            print(f"{key}:")
            print(f"  최소: {min(cond_nums):.2e}, 최대: {max(cond_nums):.2e}, 평균: {np.mean(cond_nums):.2e}")
    
    # 완전성 검증
    print("\n=== Factor Matrix 완전성 검증 ===")
    for param_name, stats in factor_matrix_stats.items():
        if 'q_proj' in param_name or 'k_proj' in param_name or 'v_proj' in param_name:
            expected = 2  # L과 R 각각 하나씩
            actual = stats['non_empty'] // len(checkpoint_files)  # 평균
            if actual < expected:
                print(f"⚠️  {param_name}: 평균 {actual}개 factor matrices (예상: {expected})")
            else:
                print(f"✅ {param_name}: Factor matrices 정상")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Shampoo Preconditioner Condition Number trends from checkpoints.')
    parser.add_argument('--checkpoint-dir', type=str, required=True, 
                       help='Directory containing the .pth checkpoint files.')
    parser.add_argument('--beta2', type=float, default=0.99,
                       help='Beta2 value for bias correction (default: 0.99)')
    args = parser.parse_args()
    
    plot_condition_number_trends(args.checkpoint_dir, args.beta2)
