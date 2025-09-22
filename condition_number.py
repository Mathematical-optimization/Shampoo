import torch
import torch.nn as nn
import argparse
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import json

def plot_condition_number_trends(checkpoint_dir: str):
    """
    지정된 디렉토리의 모든 체크포인트를 읽어
    Shampoo Preconditioner의 Condition Number 변화 추이를 그래프로 저장합니다.
    """
    if not os.path.isdir(checkpoint_dir):
        print(f"오류: 디렉토리를 찾을 수 없습니다 -> {checkpoint_dir}")
        return
    
    device = torch.device("cpu")  # CPU에서 분석
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

    # Condition Number 데이터를 저장할 딕셔너리
    results = defaultdict(lambda: defaultdict(list))

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

        # optimizer_state_dict에서 직접 정보 추출
        if 'optimizer_state_dict' not in checkpoint:
            print(f"Epoch {epoch}: optimizer_state_dict가 없습니다.")
            continue
            
        optimizer_state = checkpoint['optimizer_state_dict']
        
        if 'state' not in optimizer_state:
            print(f"Epoch {epoch}: optimizer state가 없습니다.")
            continue
            
        state_dict = optimizer_state['state']
        
        # 각 파라미터의 상태 확인
        for param_name, param_state in state_dict.items():
            # encoder_blocks의 attention 파라미터만 필터링 (CustomMultiheadAttention의 q_proj, k_proj, v_proj)
            if 'encoder_blocks' in param_name and 'attn' in param_name and any(proj in param_name for proj in ['q_proj', 'k_proj', 'v_proj']):
                if 'weight' not in param_name:
                    continue
                    
                # param_state의 구조를 파싱
                for state_key, state_value in param_state.items():
                    # state_key가 JSON 형식인지 확인
                    if isinstance(state_key, str) and state_key.startswith('['):
                        try:
                            # JSON 파싱
                            key_parts = json.loads(state_key)
                            
                            # Shampoo factor matrices 찾기
                            if (isinstance(key_parts, list) and len(key_parts) >= 3 and 
                                'shampoo' in key_parts and 'factor_matrices' in key_parts):
                                
                                # Factor index 추출
                                factor_idx = key_parts[-1] if isinstance(key_parts[-1], int) else None
                                
                                if factor_idx is not None and isinstance(state_value, torch.Tensor):
                                    # 2D 정방 행렬이고 빈 텐서가 아닌지 확인
                                    if (state_value.ndim == 2 and 
                                        state_value.shape[0] == state_value.shape[1] and 
                                        state_value.shape[0] > 1 and
                                        state_value.numel() > 0):
                                        
                                        try:
                                            # 조건수 계산
                                            matrix = state_value.detach().double()
                                            # 수치 안정성을 위해 작은 값 추가
                                            matrix = matrix + torch.eye(matrix.shape[0], dtype=torch.float64) * 1e-10
                                            cond_num = torch.linalg.cond(matrix).item()
                                            
                                            if not (np.isnan(cond_num) or np.isinf(cond_num)):
                                                # 파라미터 이름 파싱
                                                match = re.search(r'encoder_blocks\.(\d+)\.attn\.(q_proj|k_proj|v_proj)\.weight', param_name)
                                                if match:
                                                    block_idx = int(match.group(1))
                                                    proj_type = match.group(2)
                                                    
                                                    if proj_type == 'q_proj':
                                                        proj_name = 'Query'
                                                    elif proj_type == 'k_proj':
                                                        proj_name = 'Key'
                                                    else:
                                                        proj_name = 'Value'
                                                    
                                                    key = f"Block_{block_idx}_{proj_name}_Factor_{factor_idx}"
                                                    results[key]['epochs'].append(epoch)
                                                    results[key]['cond_nums'].append(cond_num)
                                                    print(f"  {key}: {cond_num:.2e}")
                                                
                                        except Exception as e:
                                            print(f"  조건수 계산 실패: {e}")
                                            
                        except json.JSONDecodeError:
                            pass
    
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
    
    # 색상 매핑
    color_map = {'Query': 'red', 'Key': 'green', 'Value': 'blue'}
    linestyle_map = {0: '-', 1: '--'}  # Factor 0은 실선, Factor 1은 점선
    
    for block_idx in range(num_blocks):
        ax = axes[block_idx]
        has_data = False
        
        # 각 projection type과 factor에 대해 플롯
        for proj_type in ['Query', 'Key', 'Value']:
            for factor_idx in [0, 1]:
                key = f"Block_{block_idx}_{proj_type}_Factor_{factor_idx}"
                
                if key in results and results[key]['epochs']:
                    epochs = results[key]['epochs']
                    cond_nums = results[key]['cond_nums']
                    
                    ax.semilogy(epochs, cond_nums, 
                               marker='o', 
                               linestyle=linestyle_map[factor_idx],
                               label=f"{proj_type} (Factor {factor_idx})",
                               color=color_map[proj_type],
                               linewidth=2,
                               markersize=5,
                               alpha=0.8)
                    has_data = True
        
        if has_data:
            ax.set_title(f"Block {block_idx}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Condition Number", fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.set_ylim(bottom=1e0)
        else:
            ax.set_title(f"Block {block_idx} (No Data)", fontsize=12)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
    
    plt.suptitle("Shampoo Preconditioner Condition Numbers for All Transformer Blocks", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(checkpoint_dir), "qkv_condition_numbers_all_blocks.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n그래프가 '{save_path}' 파일로 저장되었습니다.")
    
    # 통계 정보 출력
    print("\n=== 조건수 통계 ===")
    for key in sorted(results.keys()):
        if results[key]['cond_nums']:
            cond_nums = results[key]['cond_nums']
            print(f"{key}:")
            print(f"  최소: {min(cond_nums):.2e}, 최대: {max(cond_nums):.2e}, 평균: {np.mean(cond_nums):.2e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Shampoo Preconditioner Condition Number trends from checkpoints.')
    parser.add_argument('--checkpoint-dir', type=str, required=True, 
                       help='Directory containing the .pth checkpoint files.')
    args = parser.parse_args()
    
    plot_condition_number_trends(args.checkpoint_dir)
