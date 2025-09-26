# condition-number.py (최종 수정본)
import torch
import torch.nn as nn
import argparse
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import json
from typing import Dict, List, Tuple, Any

def inspect_checkpoint_structure(checkpoint_path: str, verbose: bool = False):
    """체크포인트 구조를 확인하고 디버깅 정보를 출력합니다."""
    print("\n=== Checkpoint Structure Inspection ===")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

    if verbose:
        print(f"Top-level keys: {list(checkpoint.keys())}")

    structure_info = {
        'has_optimizer_state': False,
        'has_state_dict': False,
        'param_count': 0,
        'factor_matrix_count': 0,
        'sample_keys': [],
        'attention_params': []
    }

    if 'optimizer_state_dict' in checkpoint:
        structure_info['has_optimizer_state'] = True
        opt_state = checkpoint['optimizer_state_dict']

        if 'state' in opt_state:
            structure_info['has_state_dict'] = True
            param_states = opt_state['state']
            structure_info['param_count'] = len(param_states)

            # attention weight 파라미터만 찾기 (bias 제외)
            for param_name, param_state in param_states.items():
                if ('attn' in param_name and 
                    any(p in param_name for p in ['q_proj', 'k_proj', 'v_proj']) and
                    'weight' in param_name):  # weight만 처리
                    
                    factor_keys = [k for k in param_state.keys()
                                   if isinstance(k, str) and 'factor_matrices' in k]
                    structure_info['attention_params'].append({
                        'name': param_name,
                        'factor_keys': factor_keys[:4] if factor_keys else []
                    })
                    
                    if verbose and factor_keys:
                        print(f"\nFound attention weight parameter: {param_name}")
                        print(f"  Number of factor matrices: {len(factor_keys)}")
                        if len(factor_keys) > 0:
                            print(f"  Sample keys: {factor_keys[:2]}")
    
    if verbose:
        print(f"\nTotal attention weight parameters found: {len(structure_info['attention_params'])}")
        for param_info in structure_info['attention_params'][:5]:
            print(f"  - {param_info['name']}")
        print("="*50)
    
    return structure_info

def parse_state_key(state_key: str):
    """State key를 파싱하여 구성 요소를 추출합니다."""
    try:
        if isinstance(state_key, str) and state_key.startswith('['):
            key_parts = json.loads(state_key)
            if (isinstance(key_parts, list) and len(key_parts) >= 4 and
                'shampoo' in key_parts and 'factor_matrices' in key_parts):
                block_id = key_parts[0] if 'block' in key_parts[0] else None
                factor_idx = key_parts[-1] if isinstance(key_parts[-1], int) else None
                return {'is_factor_matrix': True, 'block_id': block_id, 'factor_idx': factor_idx}
    except (json.JSONDecodeError, IndexError):
        pass
    return {'is_factor_matrix': False}

def compute_condition_number(matrix: torch.Tensor, epsilon: float = 1e-10) -> float:
    """행렬의 condition number를 계산합니다."""
    try:
        matrix = matrix.detach().double()
        
        if matrix.shape[0] != matrix.shape[1]:
            return float('inf')
        
        matrix = matrix + torch.eye(matrix.shape[0], dtype=torch.float64, device=matrix.device) * epsilon
        
        try:
            cond_num = torch.linalg.cond(matrix).item()
        except:
            try:
                U, S, V = torch.linalg.svd(matrix)
                cond_num = (S[0] / S[-1]).item() if S[-1] > epsilon else float('inf')
            except:
                return float('inf')
        
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
        if bias_correction > 1e-9:
            return matrix / bias_correction
    return matrix

def plot_condition_number_trends(checkpoint_dir: str, beta2: float = 0.98):
    """
    지정된 디렉토리의 모든 체크포인트를 읽어
    Shampoo Preconditioner의 Condition Number 변화 추이를 그래프로 저장합니다.
    """
    if not os.path.isdir(checkpoint_dir):
        print(f"오류: 디렉토리를 찾을 수 없습니다 -> {checkpoint_dir}")
        return
    
    device = torch.device("cpu")
    print(f"분석을 위해 {device} 장치를 사용합니다.")

    # 체크포인트 파일 목록을 epoch 순서대로 정렬
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pth'):
            epoch_match = re.search(r'epoch[_\s]+(\d+)', f)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                checkpoint_files.append((epoch, f))
            elif 'best' not in f.lower():
                checkpoint_files.append((999, f))
    
    checkpoint_files.sort(key=lambda x: x[0])
    checkpoint_files = [f for _, f in checkpoint_files]
        
    if not checkpoint_files:
        print(f"오류: '{checkpoint_dir}' 디렉토리에서 체크포인트 파일(.pth)을 찾을 수 없습니다.")
        return

    print(f"총 {len(checkpoint_files)}개의 체크포인트 파일을 분석합니다.")
    
    # 첫 번째 체크포인트 구조 확인
    structure_info = inspect_checkpoint_structure(
        os.path.join(checkpoint_dir, checkpoint_files[0]), 
        verbose=True
    )

    results = defaultdict(lambda: defaultdict(list))
    debug_info = defaultdict(int)

    # 각 체크포인트를 순회하며 데이터 수집
    for filename in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        epoch_match = re.search(r'epoch[_\s]+(\d+)', filename)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))
        
        print(f"\n--- Epoch {epoch} 체크포인트 분석 중 ({filename}) ---")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Epoch {epoch} 체크포인트 로딩 실패: {e}")
            continue

        if 'optimizer_state_dict' not in checkpoint or 'state' not in checkpoint['optimizer_state_dict']:
            print(f"Epoch {epoch}: optimizer state가 없습니다.")
            continue
            
        param_states = checkpoint['optimizer_state_dict']['state']
        print(f"  총 파라미터 수: {len(param_states)}")
        
        # 파라미터별 처리
        processed_params = set()
        for param_name, param_state in param_states.items():
            # weight 파라미터만 처리 (bias 제외)
            if not ('attn' in param_name and 
                   any(p in param_name for p in ['q_proj', 'k_proj', 'v_proj']) and
                   'weight' in param_name):
                continue
            
            # 정규표현식으로 파라미터 정보 추출
            match = re.search(r'(encoder_layers|decoder_layers)\.(\d+)\.(self_attn|cross_attn)\.(q_proj|k_proj|v_proj)\.weight', param_name)
            if not match:
                debug_info['unmatched'] += 1
                continue
            
            layer_type = match.group(1)
            block_idx = int(match.group(2))
            attn_type = match.group(3)
            proj_type = match.group(4)
            
            proj_name = {'q_proj': 'Query', 'k_proj': 'Key', 'v_proj': 'Value'}[proj_type]
            
            # 이미 처리한 파라미터인지 확인
            param_key = f"{layer_type}_{block_idx}_{attn_type}_{proj_type}"
            if param_key in processed_params:
                continue
            processed_params.add(param_key)
            
            # Factor matrices 처리
            for state_key, state_value in param_state.items():
                parsed = parse_state_key(state_key)
                
                if parsed['is_factor_matrix'] and parsed['factor_idx'] is not None:
                    factor_idx = parsed['factor_idx']
                    
                    if (isinstance(state_value, torch.Tensor) and 
                        state_value.ndim == 2 and 
                        state_value.shape[0] == state_value.shape[1] and 
                        state_value.numel() > 1):
                        
                        corrected_matrix = apply_bias_correction(state_value, beta2, epoch)
                        cond_num = compute_condition_number(corrected_matrix)
                        
                        if cond_num != float('inf'):
                            factor_name = 'L' if factor_idx == 0 else 'R'
                            key = f"{layer_type}_Block_{block_idx}_{attn_type}_{proj_name}_{factor_name}"
                            
                            results[key]['epochs'].append(epoch)
                            results[key]['cond_nums'].append(cond_num)
                            
                            # 각 projection type별로 출력
                            print(f"  ✓ {proj_name} {factor_name}: {key} = {cond_num:.2e}")
    
    print("\n--- 모든 체크포인트 분석 완료. 그래프 생성 중... ---")

    if not results:
        print("경고: 분석할 데이터가 없습니다.")
        return

    print(f"\n수집된 고유 키 수: {len(results)}")
    print(f"총 데이터 포인트 수: {sum(len(v['epochs']) for v in results.values())}")
    
    # 수집된 데이터 요약 출력
    print("\n=== 수집된 데이터 요약 ===")
    for layer_type in ['encoder_layers', 'decoder_layers']:
        layer_keys = [k for k in results.keys() if k.startswith(layer_type)]
        if layer_keys:
            print(f"\n{layer_type}:")
            for block_idx in range(4):  # 4개 블록
                block_keys = [k for k in layer_keys if f"Block_{block_idx}" in k]
                if block_keys:
                    print(f"  Block {block_idx}: {len(block_keys)} keys")
                    for proj in ['Query', 'Key', 'Value']:
                        proj_keys = [k for k in block_keys if proj in k]
                        if proj_keys:
                            print(f"    {proj}: {len(proj_keys)} factors")
    
    # 그래프 시각화
    for layer_type_str in ['encoder_layers', 'decoder_layers']:
        layer_results = {k: v for k, v in results.items() if k.startswith(layer_type_str)}
        if not layer_results:
            continue

        num_layers = 4  # 4개 레이어로 고정
        rows, cols = 2, 2  # 2x2 그리드
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12), squeeze=False)
        axes = axes.flatten()
        
        color_map = {'Query': 'red', 'Key': 'green', 'Value': 'blue'}
        marker_map = {'L': 'o', 'R': 's'}
        
        for block_idx in range(num_layers):
            ax = axes[block_idx]
            has_data = False
            
            attn_types = ['self_attn', 'cross_attn'] if layer_type_str == 'decoder_layers' else ['self_attn']
            
            for attn_type in attn_types:
                for proj_type in ['Query', 'Key', 'Value']:
                    for factor_type in ['L', 'R']:
                        key = f"{layer_type_str}_Block_{block_idx}_{attn_type}_{proj_type}_{factor_type}"
                        
                        if key in results and results[key]['epochs']:
                            has_data = True
                            epochs, cond_nums = results[key]['epochs'], results[key]['cond_nums']
                            
                            label_prefix = f"{attn_type.replace('_attn', '')} " if layer_type_str == 'decoder_layers' else ""
                            label = f"{label_prefix}{proj_type} ({factor_type})"
                            
                            ax.semilogy(epochs, cond_nums, 
                                       marker=marker_map[factor_type],
                                       linestyle='-' if factor_type == 'L' else '--',
                                       label=label,
                                       color=color_map[proj_type],
                                       linewidth=1.5,
                                       markersize=5,
                                       alpha=0.8)
            
            if has_data:
                ax.set_title(f"Layer {block_idx}", fontsize=12, fontweight='bold')
                ax.grid(True, which="both", ls="--", alpha=0.3)
                ax.legend(loc='best', fontsize=8, ncol=2 if layer_type_str == 'decoder_layers' else 1)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Condition Number (log scale)")
            else:
                ax.set_title(f"Layer {block_idx} (No Data)", fontsize=12, color='gray')
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, color='gray')
        
        fig.suptitle(f"Shampoo Preconditioner Condition Numbers ({layer_type_str.replace('_', ' ').title()})", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(os.path.dirname(checkpoint_dir), f"{layer_type_str}_condition_numbers.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n그래프가 '{save_path}' 파일로 저장되었습니다.")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot Shampoo Preconditioner Condition Number trends from checkpoints.')
    parser.add_argument('--checkpoint-dir', type=str, required=True, 
                       help='Directory containing the .pth checkpoint files.')
    parser.add_argument('--beta2', type=float, default=0.98,
                       help='Beta2 value for bias correction (default: 0.98)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for verbose output')
    args = parser.parse_args()
    
    plot_condition_number_trends(args.checkpoint_dir, args.beta2)

if __name__ == '__main__':
    main()
