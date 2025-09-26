import torch
import torch.nn as nn
import argparse
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import json
import math 

def inspect_checkpoint_structure(checkpoint_path: str, verbose: bool = False):
    """Inspects the checkpoint structure and prints debugging information."""
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
            
            # Find a sample parameter from the transformer attention layers
            for param_name, param_state in param_states.items():
                if any(proj in param_name for proj in ['q_proj', 'k_proj', 'v_proj']) and 'attn' in param_name:
                    factor_keys = [k for k in param_state.keys() 
                                  if isinstance(k, str) and 'factor_matrices' in k]
                    structure_info['factor_matrix_count'] = len(factor_keys)
                    structure_info['sample_keys'] = factor_keys[:2]
                    
                    if verbose and factor_keys:
                        print(f"\nSample parameter: {param_name}")
                        print(f"Number of factor matrices found: {len(factor_keys)}")
                        print(f"Sample state keys: {factor_keys[:2]}")
                    break
    
    return structure_info

def parse_state_key(state_key: str):
    """Parses the state key to extract its components."""
    try:
        if isinstance(state_key, str) and state_key.startswith('['):
            key_parts = json.loads(state_key)
            
            if (isinstance(key_parts, list) and len(key_parts) >= 4 and
                'shampoo' in key_parts and 'factor_matrices' in key_parts):
                
                block_id = key_parts[0]
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
    """Calculates the condition number of a matrix with high precision."""
    try:
        matrix = matrix.detach().double()
        
        if matrix.shape[0] == matrix.shape[1]:
            matrix = matrix + torch.eye(matrix.shape[0], dtype=torch.float64, device=matrix.device) * epsilon
        
        cond_num = torch.linalg.cond(matrix).item()
        
        if np.isnan(cond_num) or np.isinf(cond_num):
            return float('inf')
            
        return cond_num
        
    except Exception as e:
        print(f"  Failed to compute condition number: {e}")
        return float('inf')

def apply_bias_correction(matrix: torch.Tensor, beta2: float, step: int) -> torch.Tensor:
    """Applies bias correction to the matrix."""
    if beta2 < 1.0 and step > 0:
        bias_correction = 1.0 - (beta2 ** step)
        return matrix / bias_correction
    return matrix

def plot_condition_number_trends(checkpoint_dir: str, beta2: float = 0.99):
    """
    Reads all checkpoints in a directory and plots the condition number trends
    of the Shampoo preconditioners for Transformer attention layers.
    """
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: Directory not found -> {checkpoint_dir}")
        return
    
    device = torch.device("cpu")
    print(f"Using device: {device} for analysis.")

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    try:
        # Sort files by epoch number
        checkpoint_files.sort(key=lambda f: int(re.search(r'epoch_(\d+)\.pth', f).group(1)))
    except (TypeError, AttributeError):
        print("Error: Could not find files named in the format '...epoch_XX.pth'.")
        return
        
    if not checkpoint_files:
        print(f"Error: No checkpoint files (.pth) found in '{checkpoint_dir}'.")
        return

    print(f"Found {len(checkpoint_files)} checkpoint files to analyze.")
    
    # Inspect the first checkpoint to verify structure
    if checkpoint_files:
        first_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        structure_info = inspect_checkpoint_structure(first_checkpoint_path, verbose=True)
        
        if not structure_info['has_state_dict']:
            print("Error: Optimizer state could not be found in the checkpoint.")
            return

    results = defaultdict(lambda: defaultdict(list))

    for filename in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        epoch_match = re.search(r'epoch_(\d+)\.pth', filename)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))
        
        print(f"\n--- Analyzing Checkpoint for Epoch {epoch} ---")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Failed to load checkpoint for epoch {epoch}: {e}")
            continue

        if 'optimizer_state_dict' not in checkpoint or 'state' not in checkpoint['optimizer_state_dict']:
            print(f"Epoch {epoch}: Optimizer state not found.")
            continue
            
        param_states = checkpoint['optimizer_state_dict']['state']
        
        for param_name, param_state in param_states.items():
            # Regex to match Transformer attention layer weights
            match = re.search(r'(encoder_layers|decoder_layers)\.(\d+)\.(self_attn|cross_attn)\.(q_proj|k_proj|v_proj)\.weight', param_name)
            if not match:
                continue
                
            layer_type, layer_idx, attn_type, proj_type = match.groups()
            layer_idx = int(layer_idx)
            
            proj_name_map = {'q_proj': 'Query', 'k_proj': 'Key', 'v_proj': 'Value'}
            proj_name = proj_name_map.get(proj_type, proj_type)
            
            for state_key, state_value in param_state.items():
                parsed = parse_state_key(state_key)
                
                if parsed['is_factor_matrix'] and parsed['factor_idx'] is not None:
                    if isinstance(state_value, torch.Tensor) and state_value.ndim == 2 and state_value.shape[0] > 1:
                        
                        corrected_matrix = apply_bias_correction(state_value, beta2, epoch)
                        cond_num = compute_condition_number(corrected_matrix)
                        
                        if cond_num != float('inf'):
                            factor_name = 'L' if parsed['factor_idx'] == 0 else 'R'
                            key_prefix = f"{layer_type.replace('_layers', '').capitalize()}_{layer_idx}_{attn_type.replace('_attn', '').capitalize()}"
                            key = f"{key_prefix}_{proj_name}_{factor_name}"
                            
                            results[key]['epochs'].append(epoch)
                            results[key]['cond_nums'].append(cond_num)
                            print(f"  {key}: {cond_num:.2e}")

    print("\n--- Analysis complete. Generating plots... ---")

    if not results:
        print("No data to plot. Could not find factor matrices for attention weights.")
        return

    # Plotting
    color_map = {'Query': 'red', 'Key': 'green', 'Value': 'blue'}
    marker_map = {'L': 'o', 'R': 's'}
    
    # Plot for Encoder
    num_encoder_layers = max([int(re.search(r'Encoder_(\d+)', key).group(1)) for key in results if 'Encoder' in key] + [-1]) + 1
    if num_encoder_layers > 0:
        fig_enc, axes_enc = plt.subplots(math.ceil(num_encoder_layers / 3), 3, figsize=(20, 5 * math.ceil(num_encoder_layers / 3)), squeeze=False)
        axes_enc = axes_enc.flatten()
        
        for layer_idx in range(num_encoder_layers):
            ax = axes_enc[layer_idx]
            has_data = False
            for proj_type in ['Query', 'Key', 'Value']:
                for factor_type in ['L', 'R']:
                    key = f"Encoder_{layer_idx}_Self_{proj_type}_{factor_type}"
                    if key in results and results[key]['epochs']:
                        ax.semilogy(results[key]['epochs'], results[key]['cond_nums'], 
                                   marker=marker_map[factor_type], linestyle='-',
                                   label=f"{proj_type} ({factor_type})", color=color_map[proj_type], alpha=0.8)
                        has_data = True
            
            ax.set_title(f"Encoder Layer {layer_idx} (Self-Attention)", fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Condition Number (log)")
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, which="both", ls="--", alpha=0.5)
            if not has_data:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        
        plt.suptitle("Encoder Preconditioner Condition Numbers", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path_enc = os.path.join(os.path.dirname(checkpoint_dir) or '.', "transformer_encoder_condition_numbers.png")
        plt.savefig(save_path_enc, dpi=150)
        print(f"Encoder plot saved to '{save_path_enc}'")

    # Plot for Decoder
    num_decoder_layers = max([int(re.search(r'Decoder_(\d+)', key).group(1)) for key in results if 'Decoder' in key] + [-1]) + 1
    if num_decoder_layers > 0:
        fig_dec, axes_dec = plt.subplots(num_decoder_layers, 2, figsize=(20, 5 * num_decoder_layers), squeeze=False)
        
        for layer_idx in range(num_decoder_layers):
            # Self-Attention
            ax_self = axes_dec[layer_idx, 0]
            has_data_self = False
            for proj_type in ['Query', 'Key', 'Value']:
                for factor_type in ['L', 'R']:
                    key = f"Decoder_{layer_idx}_Self_{proj_type}_{factor_type}"
                    if key in results and results[key]['epochs']:
                        ax_self.semilogy(results[key]['epochs'], results[key]['cond_nums'],
                                         marker=marker_map[factor_type], linestyle='-',
                                         label=f"{proj_type} ({factor_type})", color=color_map[proj_type], alpha=0.8)
                        has_data_self = True

            ax_self.set_title(f"Decoder Layer {layer_idx} (Self-Attention)", fontweight='bold')
            ax_self.set_xlabel("Epoch")
            ax_self.set_ylabel("Condition Number (log)")
            ax_self.legend(loc='best', fontsize=9)
            ax_self.grid(True, which="both", ls="--", alpha=0.5)
            if not has_data_self:
                ax_self.text(0.5, 0.5, "No Data", ha='center', va='center')

            # Cross-Attention
            ax_cross = axes_dec[layer_idx, 1]
            has_data_cross = False
            for proj_type in ['Query', 'Key', 'Value']:
                for factor_type in ['L', 'R']:
                    key = f"Decoder_{layer_idx}_Cross_{proj_type}_{factor_type}"
                    if key in results and results[key]['epochs']:
                        ax_cross.semilogy(results[key]['epochs'], results[key]['cond_nums'],
                                          marker=marker_map[factor_type], linestyle='-',
                                          label=f"{proj_type} ({factor_type})", color=color_map[proj_type], alpha=0.8)
                        has_data_cross = True
            
            ax_cross.set_title(f"Decoder Layer {layer_idx} (Cross-Attention)", fontweight='bold')
            ax_cross.set_xlabel("Epoch")
            ax_cross.set_ylabel("Condition Number (log)")
            ax_cross.legend(loc='best', fontsize=9)
            ax_cross.grid(True, which="both", ls="--", alpha=0.5)
            if not has_data_cross:
                ax_cross.text(0.5, 0.5, "No Data", ha='center', va='center')

        plt.suptitle("Decoder Preconditioner Condition Numbers", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path_dec = os.path.join(os.path.dirname(checkpoint_dir) or '.', "transformer_decoder_condition_numbers.png")
        plt.savefig(save_path_dec, dpi=150)
        print(f"Decoder plot saved to '{save_path_dec}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Shampoo Preconditioner Condition Number trends for Transformer.')
    parser.add_argument('--checkpoint-dir', type=str, required=True, 
                       help='Directory containing the .pth checkpoint files.')
    parser.add_argument('--beta2', type=float, default=0.99,
                       help='Beta2 value for bias correction (default: 0.99)')
    args = parser.parse_args()
    
    plot_condition_number_trends(args.checkpoint_dir, args.beta2)
