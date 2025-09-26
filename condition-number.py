# condition_number.py
# Transformer + WMT2017 dataset

import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
import json

def get_condition_numbers_from_shampoo(model, optimizer):
    """
    Shampoo optimizer의 preconditioner condition number 추출
    """
    condition_numbers = {}
    
    # optimizer state 확인
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param in optimizer.state:
                state = optimizer.state[param]
                
                # Shampoo는 preconditioner matrices를 저장함
                if 'preconditioner_left' in state or 'preconditioner_right' in state:
                    param_name = None
                    
                    # 파라미터 이름 찾기
                    for name, module_param in model.named_parameters():
                        if module_param is param:
                            param_name = name
                            break
                    
                    if param_name and any(key in param_name for key in ['q_proj', 'k_proj', 'v_proj']):
                        cond_info = {}
                        
                        # Left preconditioner (row space)
                        if 'preconditioner_left' in state:
                            P_left = state['preconditioner_left']
                            if P_left is not None:
                                try:
                                    eigenvalues = torch.linalg.eigvalsh(P_left)
                                    max_eig = eigenvalues.max().item()
                                    min_eig = eigenvalues[eigenvalues > 1e-10].min().item()  # 0이 아닌 최소값
                                    cond_left = max_eig / min_eig if min_eig > 0 else float('inf')
                                    cond_info['left_preconditioner'] = {
                                        'condition_number': cond_left,
                                        'max_eigenvalue': max_eig,
                                        'min_eigenvalue': min_eig,
                                        'shape': list(P_left.shape)
                                    }
                                except:
                                    cond_info['left_preconditioner'] = 'Error computing'
                        
                        # Right preconditioner (column space)
                        if 'preconditioner_right' in state:
                            P_right = state['preconditioner_right']
                            if P_right is not None:
                                try:
                                    eigenvalues = torch.linalg.eigvalsh(P_right)
                                    max_eig = eigenvalues.max().item()
                                    min_eig = eigenvalues[eigenvalues > 1e-10].min().item()
                                    cond_right = max_eig / min_eig if min_eig > 0 else float('inf')
                                    cond_info['right_preconditioner'] = {
                                        'condition_number': cond_right,
                                        'max_eigenvalue': max_eig,
                                        'min_eigenvalue': min_eig,
                                        'shape': list(P_right.shape)
                                    }
                                except:
                                    cond_info['right_preconditioner'] = 'Error computing'
                        
                        # Kronecker product의 전체 condition number 추정
                        if 'left_preconditioner' in cond_info and 'right_preconditioner' in cond_info:
                            if isinstance(cond_info['left_preconditioner'], dict) and \
                               isinstance(cond_info['right_preconditioner'], dict):
                                total_cond = cond_info['left_preconditioner']['condition_number'] * \
                                           cond_info['right_preconditioner']['condition_number']
                                cond_info['total_condition_number'] = total_cond
                        
                        condition_numbers[param_name] = cond_info
    
    return condition_numbers


def analyze_attention_layers(checkpoint_path, output_file='condition_analysis.json'):
    """
    체크포인트에서 attention layer의 condition number 분석
    """
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 모델 재구성 (필요한 경우)
    from Transformer import Transformer
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    vocab_size = len(tokenizer)
    
    # 모델 생성
    model = Transformer(
        vocab_size=vocab_size,
        d_model=256,  # 또는 체크포인트에서 로드
        n_heads=4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        d_ff=1024,
        max_seq_len=128,
        dropout=0.1,
        label_smoothing=0.1,
        pad_idx=tokenizer.pad_token_id
    )
    
    # 모델 state dict 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimizer state 로드 및 분석
    results = {
        'encoder_layers': {},
        'decoder_layers': {},
        'summary': {}
    }
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer_state = checkpoint['optimizer_state_dict']
        
        # 각 레이어별로 condition number 추출
        for layer_idx in range(6):  # 6 layers
            # Encoder layers
            encoder_prefix = f'encoder_layers.{layer_idx}.self_attn'
            encoder_conds = {}
            
            for proj_type in ['q_proj', 'k_proj', 'v_proj']:
                weight_key = f'{encoder_prefix}.{proj_type}.weight'
                if weight_key in optimizer_state['state']:
                    state = optimizer_state['state'][weight_key]
                    cond_info = extract_condition_from_state(state)
                    encoder_conds[proj_type] = cond_info
            
            if encoder_conds:
                results['encoder_layers'][f'layer_{layer_idx}'] = encoder_conds
            
            # Decoder layers
            decoder_self_prefix = f'decoder_layers.{layer_idx}.self_attn'
            decoder_cross_prefix = f'decoder_layers.{layer_idx}.cross_attn'
            decoder_conds = {'self_attn': {}, 'cross_attn': {}}
            
            for proj_type in ['q_proj', 'k_proj', 'v_proj']:
                # Self attention
                weight_key = f'{decoder_self_prefix}.{proj_type}.weight'
                if weight_key in optimizer_state['state']:
                    state = optimizer_state['state'][weight_key]
                    cond_info = extract_condition_from_state(state)
                    decoder_conds['self_attn'][proj_type] = cond_info
                
                # Cross attention
                weight_key = f'{decoder_cross_prefix}.{proj_type}.weight'
                if weight_key in optimizer_state['state']:
                    state = optimizer_state['state'][weight_key]
                    cond_info = extract_condition_from_state(state)
                    decoder_conds['cross_attn'][proj_type] = cond_info
            
            if decoder_conds['self_attn'] or decoder_conds['cross_attn']:
                results['decoder_layers'][f'layer_{layer_idx}'] = decoder_conds
    
    # 통계 요약
    all_condition_numbers = []
    for layer_type in ['encoder_layers', 'decoder_layers']:
        for layer_name, layer_data in results[layer_type].items():
            if layer_type == 'encoder_layers':
                for proj_type, cond_info in layer_data.items():
                    if 'total_condition_number' in cond_info:
                        all_condition_numbers.append(cond_info['total_condition_number'])
            else:  # decoder
                for attn_type in ['self_attn', 'cross_attn']:
                    if attn_type in layer_data:
                        for proj_type, cond_info in layer_data[attn_type].items():
                            if 'total_condition_number' in cond_info:
                                all_condition_numbers.append(cond_info['total_condition_number'])
    
    if all_condition_numbers:
        results['summary'] = {
            'mean_condition_number': np.mean(all_condition_numbers),
            'std_condition_number': np.std(all_condition_numbers),
            'max_condition_number': np.max(all_condition_numbers),
            'min_condition_number': np.min(all_condition_numbers),
            'median_condition_number': np.median(all_condition_numbers)
        }
    
    # 결과 저장
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 콘솔 출력
    print_analysis_results(results)
    
    return results


def extract_condition_from_state(state):
    """
    Optimizer state에서 condition number 추출
    """
    cond_info = {}
    
    # Shampoo의 경우 Q matrices 확인
    if 'Q_left' in state or 'Q_right' in state:
        # Left Q matrix
        if 'Q_left' in state and state['Q_left'] is not None:
            Q_left = state['Q_left']
            try:
                eigenvalues = torch.linalg.eigvalsh(Q_left)
                max_eig = eigenvalues.max().item()
                min_eig = eigenvalues[eigenvalues > 1e-10].min().item()
                cond_left = max_eig / min_eig if min_eig > 0 else float('inf')
                cond_info['left_preconditioner'] = {
                    'condition_number': cond_left,
                    'max_eigenvalue': max_eig,
                    'min_eigenvalue': min_eig,
                    'shape': list(Q_left.shape)
                }
            except Exception as e:
                cond_info['left_preconditioner'] = f'Error: {str(e)}'
        
        # Right Q matrix
        if 'Q_right' in state and state['Q_right'] is not None:
            Q_right = state['Q_right']
            try:
                eigenvalues = torch.linalg.eigvalsh(Q_right)
                max_eig = eigenvalues.max().item()
                min_eig = eigenvalues[eigenvalues > 1e-10].min().item()
                cond_right = max_eig / min_eig if min_eig > 0 else float('inf')
                cond_info['right_preconditioner'] = {
                    'condition_number': cond_right,
                    'max_eigenvalue': max_eig,
                    'min_eigenvalue': min_eig,
                    'shape': list(Q_right.shape)
                }
            except Exception as e:
                cond_info['right_preconditioner'] = f'Error: {str(e)}'
        
        # Total condition number
        if 'left_preconditioner' in cond_info and 'right_preconditioner' in cond_info:
            if isinstance(cond_info['left_preconditioner'], dict) and \
               isinstance(cond_info['right_preconditioner'], dict):
                total_cond = cond_info['left_preconditioner']['condition_number'] * \
                           cond_info['right_preconditioner']['condition_number']
                cond_info['total_condition_number'] = total_cond
    
    return cond_info


def print_analysis_results(results):
    """
    분석 결과를 보기 좋게 출력
    """
    print("\n" + "="*80)
    print("SHAMPOO PRECONDITIONER CONDITION NUMBER ANALYSIS")
    print("="*80)
    
    # Encoder layers
    print("\n--- ENCODER LAYERS ---")
    for layer_idx in range(6):
        layer_key = f'layer_{layer_idx}'
        if layer_key in results['encoder_layers']:
            print(f"\nLayer {layer_idx}:")
            layer_data = results['encoder_layers'][layer_key]
            for proj_type in ['q_proj', 'k_proj', 'v_proj']:
                if proj_type in layer_data and 'total_condition_number' in layer_data[proj_type]:
                    cond = layer_data[proj_type]['total_condition_number']
                    print(f"  {proj_type}: {cond:.2e}")
    
    # Decoder layers
    print("\n--- DECODER LAYERS ---")
    for layer_idx in range(6):
        layer_key = f'layer_{layer_idx}'
        if layer_key in results['decoder_layers']:
            print(f"\nLayer {layer_idx}:")
            layer_data = results['decoder_layers'][layer_key]
            
            # Self attention
            if 'self_attn' in layer_data:
                print("  Self-Attention:")
                for proj_type in ['q_proj', 'k_proj', 'v_proj']:
                    if proj_type in layer_data['self_attn'] and \
                       'total_condition_number' in layer_data['self_attn'][proj_type]:
                        cond = layer_data['self_attn'][proj_type]['total_condition_number']
                        print(f"    {proj_type}: {cond:.2e}")
            
            # Cross attention
            if 'cross_attn' in layer_data:
                print("  Cross-Attention:")
                for proj_type in ['q_proj', 'k_proj', 'v_proj']:
                    if proj_type in layer_data['cross_attn'] and \
                       'total_condition_number' in layer_data['cross_attn'][proj_type]:
                        cond = layer_data['cross_attn'][proj_type]['total_condition_number']
                        print(f"    {proj_type}: {cond:.2e}")
    
    # Summary
    if 'summary' in results:
        print("\n--- SUMMARY STATISTICS ---")
        summary = results['summary']
        print(f"Mean Condition Number: {summary['mean_condition_number']:.2e}")
        print(f"Std Condition Number: {summary['std_condition_number']:.2e}")
        print(f"Max Condition Number: {summary['max_condition_number']:.2e}")
        print(f"Min Condition Number: {summary['min_condition_number']:.2e}")
        print(f"Median Condition Number: {summary['median_condition_number']:.2e}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Shampoo preconditioner condition numbers')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='condition_analysis.json',
                      help='Output JSON file')
    
    args = parser.parse_args()
    
    results = analyze_attention_layers(args.checkpoint, args.output)
    print(f"\nResults saved to: {args.output}")
