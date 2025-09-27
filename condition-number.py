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
from typing import Dict, List, Tuple, Any, Optional

def inspect_checkpoint_structure(checkpoint_path: str, verbose: bool = False) -> Optional[Dict]:
    """체크포인트 구조를 확인하고 디버깅 정보를 출력합니다."""
    print("\n=== Checkpoint Structure Inspection ===")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

    # 최상위 키 확인
    if verbose:
        print(f"Top-level keys: {list(checkpoint.keys())}")
    
    # Shampoo 설정 정보 확인
    if 'shampoo_config' in checkpoint:
        print(f"Shampoo config found: {checkpoint['shampoo_config']}")
        beta2 = checkpoint['shampoo_config'].get('beta2', 0.98)
    else:
        beta2 = 0.98  # 기본값
        print(f"No shampoo_config found, using default beta2={beta2}")

    structure_info = {
        'has_optimizer_state': False,
        'has_state_dict': False,
        'param_count': 0,
        'factor_matrix_count': 0,
        'sample_keys': [],
        'attention_params': [],
        'beta2': beta2
    }

    if 'optimizer_state_dict' not in checkpoint:
        print("No optimizer_state_dict found in checkpoint")
        return structure_info

    structure_info['has_optimizer_state'] = True
    opt_state = checkpoint['optimizer_state_dict']

    # state 키 확인
    if 'state' in opt_state:
        structure_info['has_state_dict'] = True
        param_states = opt_state['state']
        structure_info['param_count'] = len(param_states)
        
        print(f"Total parameters in optimizer state: {len(param_states)}")

        # Attention 파라미터 찾기
        attention_count = 0
        for param_name, param_state in param_states.items():
            # self_attn 또는 cross_attn 패턴 확인
            if ('self_attn' in param_name or 'cross_attn' in param_name):
                # q_proj, k_proj, v_proj weight 파라미터만 처리
                if any(p in param_name for p in ['q_proj', 'k_proj', 'v_proj']) and 'weight' in param_name:
                    
                    # Factor matrices 찾기
                    factor_keys = []
                    for key in param_state.keys():
                        if isinstance(key, str):
                            try:
                                # JSON 형식 키 파싱
                                if key.startswith('[') and key.endswith(']'):
                                    key_parts = json.loads(key)
                                    if ('shampoo' in key_parts and 'factor_matrices' in key_parts):
                                        factor_keys.append(key)
                            except:
                                pass
                    
                    if factor_keys:
                        attention_count += 1
                        structure_info['attention_params'].append({
                            'name': param_name,
                            'factor_keys': factor_keys[:4]  # 샘플로 처음 4개만
                        })
                        
                        if verbose:
                            print(f"\nFound attention parameter #{attention_count}: {param_name}")
                            print(f"  Number of factor matrices: {len(factor_keys)}")

    print(f"Total attention parameters found: {len(structure_info['attention_params'])}")
    
    if verbose and structure_info['attention_params']:
        print("\nSample attention parameters:")
        for param_info in structure_info['attention_params'][:3]:
            print(f"  - {param_info['name']}")
    
    print("="*50)
    return structure_info

def parse_state_key(state_key: str) -> Dict:
    """State key를 파싱하여 구성 요소를 추출합니다."""
    try:
        if isinstance(state_key, str) and state_key.startswith('['):
            key_parts = json.loads(state_key)
            if isinstance(key_parts, list) and len(key_parts) >= 4:
                if 'shampoo' in key_parts and 'factor_matrices' in key_parts:
                    # 마지막 요소가 factor index
                    factor_idx = key_parts[-1] if isinstance(key_parts[-1], int) else None
                    return {
                        'is_factor_matrix': True,
                        'factor_idx': factor_idx
                    }
    except:
        pass
    return {'is_factor_matrix': False}

def compute_condition_number(matrix: torch.Tensor, epsilon: float = 1e-10) -> float:
    """행렬의 condition number를 계산합니다."""
    try:
        matrix = matrix.detach().double()
        
        # 정사각 행렬 확인
        if matrix.shape[0] != matrix.shape[1]:
            return float('inf')
        
        # 작은 regularization 추가
        matrix = matrix + torch.eye(matrix.shape[0], dtype=torch.float64, device=matrix.device) * epsilon
        
        # Condition number 계산
        try:
            cond_num = torch.linalg.cond(matrix).item()
        except:
            # SVD를 사용한 대체 방법
            try:
                U, S, V = torch.linalg.svd(matrix)
                cond_num = (S[0] / S[-1]).item() if S[-1] > epsilon else float('inf')
            except:
                return float('inf')
        
        # NaN/Inf 체크
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

def plot_condition_number_trends(checkpoint_dir: str, output_dir: str = None):
    """체크포인트 디렉토리의 모든 파일을 분석하여 condition number 추이를 그래프로 저장합니다."""
    
    if not os.path.isdir(checkpoint_dir):
        print(f"오류: 디렉토리를 찾을 수 없습니다 -> {checkpoint_dir}")
        return
    
    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_dir)
    
    # 체크포인트 파일 찾기 및 정렬
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pth'):
            # condition_epoch_N.pth 패턴 우선
            if 'condition_epoch' in f:
                match = re.search(r'condition_epoch_(\d+)', f)
                if match:
                    epoch = int(match.group(1))
                    checkpoint_files.append((epoch, f))
            elif 'checkpoint_epoch' in f:
                match = re.search(r'checkpoint_epoch_(\d+)', f)
                if match:
                    epoch = int(match.group(1))
                    checkpoint_files.append((epoch, f))
    
    checkpoint_files.sort(key=lambda x: x[0])
    
    if not checkpoint_files:
        print(f"오류: '{checkpoint_dir}' 디렉토리에서 체크포인트 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(checkpoint_files)}개의 체크포인트 파일을 분석합니다.")
    
    # 첫 번째 체크포인트 구조 확인
    first_path = os.path.join(checkpoint_dir, checkpoint_files[0][1])
    structure_info = inspect_checkpoint_structure(first_path, verbose=True)
    
    if not structure_info or not structure_info['attention_params']:
        print("Attention 파라미터를 찾을 수 없습니다.")
        return
    
    beta2 = structure_info.get('beta2', 0.98)
    print(f"Using beta2={beta2} for bias correction")
    
    # 결과 저장용 딕셔너리
    results = defaultdict(lambda: {'epochs': [], 'cond_nums': []})
    
    # 각 체크포인트 분석
    for epoch, filename in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, filename)
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
        analyzed_params = 0
        
        # 각 파라미터 처리
        for param_name, param_state in param_states.items():
            # Attention weight만 처리
            if not (('self_attn' in param_name or 'cross_attn' in param_name) and 
                   any(p in param_name for p in ['q_proj', 'k_proj', 'v_proj']) and 
                   'weight' in param_name):
                continue
            
            # 파라미터 정보 추출
            match = re.search(
                r'(encoder_layers|decoder_layers)\.(\d+)\.(self_attn|cross_attn)\.(q_proj|k_proj|v_proj)\.weight',
                param_name
            )
            if not match:
                continue
            
            layer_type = match.group(1)
            block_idx = int(match.group(2))
            attn_type = match.group(3)
            proj_type = match.group(4)
            
            proj_name = {'q_proj': 'Query', 'k_proj': 'Key', 'v_proj': 'Value'}[proj_type]
            
            # Factor matrices 처리
            factor_count = 0
            for state_key, state_value in param_state.items():
                parsed = parse_state_key(state_key)
                
                if parsed['is_factor_matrix'] and parsed['factor_idx'] is not None:
                    factor_idx = parsed['factor_idx']
                    
                    if (isinstance(state_value, torch.Tensor) and 
                        state_value.ndim == 2 and 
                        state_value.shape[0] == state_value.shape[1] and 
                        state_value.shape[0] > 1):
                        
                        # Bias correction 적용
                        corrected_matrix = apply_bias_correction(state_value, beta2, epoch)
                        cond_num = compute_condition_number(corrected_matrix)
                        
                        if cond_num != float('inf'):
                            factor_name = 'L' if factor_idx == 0 else 'R'
                            key = f"{layer_type}_Block{block_idx}_{attn_type}_{proj_name}_{factor_name}"
                            
                            results[key]['epochs'].append(epoch)
                            results[key]['cond_nums'].append(cond_num)
                            factor_count += 1
            
            if factor_count > 0:
                analyzed_params += 1
        
        print(f"  분석된 attention 파라미터 수: {analyzed_params}")
    
    if not results:
        print("경고: 분석할 데이터가 없습니다.")
        return
    
    print(f"\n수집된 고유 키 수: {len(results)}")
    print(f"총 데이터 포인트 수: {sum(len(v['epochs']) for v in results.values())}")
    
    # 그래프 생성
    for layer_type_str in ['encoder_layers', 'decoder_layers']:
        layer_results = {k: v for k, v in results.items() if k.startswith(layer_type_str)}
        if not layer_results:
            continue
        
        # 블록 수 자동 감지
        max_block = max([int(re.search(r'Block(\d+)', k).group(1)) 
                        for k in layer_results.keys()]) + 1
        
        # 그리드 크기 결정
        rows = (max_block + 1) // 2
        cols = 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows), squeeze=False)
        axes = axes.flatten()
        
        color_map = {'Query': 'red', 'Key': 'green', 'Value': 'blue'}
        marker_map = {'L': 'o', 'R': 's'}
        
        for block_idx in range(max_block):
            ax = axes[block_idx]
            has_data = False
            
            # self_attn과 cross_attn 처리
            attn_types = ['self_attn', 'cross_attn'] if layer_type_str == 'decoder_layers' else ['self_attn']
            
            for attn_type in attn_types:
                for proj_type in ['Query', 'Key', 'Value']:
                    for factor_type in ['L', 'R']:
                        key = f"{layer_type_str}_Block{block_idx}_{attn_type}_{proj_type}_{factor_type}"
                        
                        if key in results and results[key]['epochs']:
                            has_data = True
                            epochs = results[key]['epochs']
                            cond_nums = results[key]['cond_nums']
                            
                            label_prefix = f"{attn_type.replace('_attn', '')} " if len(attn_types) > 1 else ""
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
                ax.legend(loc='best', fontsize=8, ncol=2 if len(attn_types) > 1 else 1)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Condition Number (log scale)")
            else:
                ax.set_title(f"Layer {block_idx} (No Data)", fontsize=12, color='gray')
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
        
        # 사용하지 않는 축 숨기기
        for idx in range(max_block, len(axes)):
            axes[idx].set_visible(False)
        
        title = layer_type_str.replace('_', ' ').title()
        fig.suptitle(f"Shampoo Preconditioner Condition Numbers ({title})",
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 그래프 저장
        save_path = os.path.join(output_dir, f"{layer_type_str}_condition_numbers.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n그래프가 '{save_path}' 파일로 저장되었습니다.")
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Plot Shampoo Preconditioner Condition Number trends from checkpoints.'
    )
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing the .pth checkpoint files.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save output plots (default: parent of checkpoint-dir)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for verbose output')
    
    args = parser.parse_args()
    
    plot_condition_number_trends(
        args.checkpoint_dir, 
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
