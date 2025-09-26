# condition_number_analysis.py

import torch
import numpy as np
import json
import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime


class ShampooConditionAnalyzer:
    """Analyze Shampoo optimizer's preconditioner condition numbers"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def extract_condition_numbers(self, checkpoint_path: str) -> Dict:
        """Extract condition numbers from a single checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        results = {
            'checkpoint_path': checkpoint_path,
            'epoch': checkpoint.get('epoch', -1),
            'step': checkpoint.get('step', -1),
            'encoder_layers': {},
            'decoder_layers': {},
            'summary': {},
            'detailed_conditions': []
        }
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer_state = checkpoint['optimizer_state_dict']
            
            # Handle both regular state_dict and distributed_state_dict formats
            if 'state' in optimizer_state:
                state_dict = optimizer_state['state']
                self._process_distributed_state(state_dict, results)
            else:
                # Handle regular optimizer state_dict format
                self._process_regular_state(optimizer_state, results)
        
        # Calculate summary statistics
        self._calculate_summary(results)
        
        return results
    
    def _process_distributed_state(self, state_dict: Dict, results: Dict) -> None:
        """Process distributed Shampoo state dictionary"""
        
        all_condition_numbers = []
        
        for param_name, param_state in state_dict.items():
            # Check if this is a Q, K, V projection parameter
            if not any(proj in param_name for proj in ['q_proj', 'k_proj', 'v_proj']):
                continue
            
            # Process each state entry
            for key, value in param_state.items():
                condition_info = self._extract_condition_from_state_entry(
                    key, value, param_name
                )
                
                if condition_info:
                    self._store_condition_info(param_name, condition_info, results)
                    if condition_info['condition_number'] != float('inf'):
                        all_condition_numbers.append(condition_info['condition_number'])
                        results['detailed_conditions'].append({
                            'param': param_name,
                            'key': key,
                            **condition_info
                        })
    
    def _process_regular_state(self, optimizer_state: Dict, results: Dict) -> None:
        """Process regular optimizer state dictionary"""
        
        all_condition_numbers = []
        
        for param_idx, param_state in optimizer_state.items():
            if not isinstance(param_state, dict):
                continue
            
            # Look for Shampoo-specific state
            for block_key, block_state in param_state.items():
                if not isinstance(block_state, dict):
                    continue
                
                if 'shampoo' in block_state:
                    shampoo_state = block_state['shampoo']
                    condition_info = self._extract_shampoo_conditions(shampoo_state)
                    
                    if condition_info:
                        param_name = f"param_{param_idx}_{block_key}"
                        self._store_condition_info(param_name, condition_info, results)
                        
                        if condition_info['condition_number'] != float('inf'):
                            all_condition_numbers.append(condition_info['condition_number'])
                            results['detailed_conditions'].append({
                                'param_idx': param_idx,
                                'block': block_key,
                                **condition_info
                            })
    
    def _extract_condition_from_state_entry(self, key: str, value: torch.Tensor, 
                                           param_name: str) -> Optional[Dict]:
        """Extract condition number from a state entry"""
        
        # Parse JSON-formatted keys from distributed state
        if isinstance(key, str) and 'shampoo' in key and 'factor_matrices' in key:
            try:
                parsed_key = json.loads(key)
                
                # Only process factor matrices (not inverse matrices)
                if 'factor_matrices' in parsed_key and 'inv' not in key:
                    return self._compute_condition_number(value, f"{param_name}_{key}")
            except:
                pass
        
        return None
    
    def _extract_shampoo_conditions(self, shampoo_state: Dict) -> Optional[Dict]:
        """Extract condition numbers from Shampoo state"""
        
        condition_numbers = []
        
        # Look for factor matrices
        if hasattr(shampoo_state, 'factor_matrices'):
            factor_matrices = shampoo_state.factor_matrices
        elif 'factor_matrices' in shampoo_state:
            factor_matrices = shampoo_state['factor_matrices']
        else:
            return None
        
        # Compute condition numbers for each factor matrix
        for i, matrix in enumerate(factor_matrices):
            if matrix is not None:
                cond_info = self._compute_condition_number(matrix, f"factor_{i}")
                if cond_info and cond_info['condition_number'] != float('inf'):
                    condition_numbers.append(cond_info['condition_number'])
        
        if condition_numbers:
            return {
                'condition_number': np.mean(condition_numbers),
                'individual_conditions': condition_numbers,
                'num_factors': len(condition_numbers)
            }
        
        return None
    
    def _compute_condition_number(self, matrix: torch.Tensor, 
                                 identifier: str = "") -> Optional[Dict]:
        """Compute condition number of a matrix"""
        
        if matrix is None or matrix.numel() == 0:
            return None
        
        # Handle 1D tensors (diagonal matrices)
        if len(matrix.shape) == 1:
            max_val = matrix.max().item()
            min_val = matrix[matrix > 1e-10].min().item() if (matrix > 1e-10).any() else 1e-10
            cond_num = max_val / min_val if min_val > 0 else float('inf')
            
            return {
                'condition_number': cond_num,
                'max_value': max_val,
                'min_value': min_val,
                'shape': list(matrix.shape),
                'is_diagonal': True,
                'identifier': identifier
            }
        
        # Handle 2D matrices
        if len(matrix.shape) == 2:
            try:
                eigenvalues = torch.linalg.eigvalsh(matrix)
                max_eig = eigenvalues.max().item()
                min_eig = eigenvalues[eigenvalues > 1e-10].min().item() if (eigenvalues > 1e-10).any() else 1e-10
                cond_num = max_eig / min_eig if min_eig > 0 else float('inf')
                
                return {
                    'condition_number': cond_num,
                    'max_eigenvalue': max_eig,
                    'min_eigenvalue': min_eig,
                    'shape': list(matrix.shape),
                    'rank': torch.linalg.matrix_rank(matrix).item(),
                    'is_diagonal': False,
                    'identifier': identifier
                }
            except Exception as e:
                if self.verbose:
                    print(f"Error computing eigenvalues for {identifier}: {e}")
                return None
        
        return None
    
    def _store_condition_info(self, param_name: str, condition_info: Dict, 
                             results: Dict) -> None:
        """Store condition information in results structure"""
        
        layer_type, layer_idx, proj_type = self._parse_parameter_name(param_name)
        
        if layer_type not in results:
            results[layer_type] = {}
        if layer_idx not in results[layer_type]:
            results[layer_type][layer_idx] = {}
        if proj_type not in results[layer_type][layer_idx]:
            results[layer_type][layer_idx][proj_type] = []
        
        results[layer_type][layer_idx][proj_type].append(condition_info)
    
    def _parse_parameter_name(self, param_name: str) -> Tuple[str, str, str]:
        """Parse parameter name to extract layer information"""
        
        if 'encoder_layers' in param_name:
            layer_type = 'encoder_layers'
            parts = param_name.split('.')
            layer_idx = f"layer_{parts[1]}" if len(parts) > 1 else "layer_0"
        elif 'decoder_layers' in param_name:
            layer_type = 'decoder_layers'
            parts = param_name.split('.')
            layer_idx = f"layer_{parts[1]}" if len(parts) > 1 else "layer_0"
        else:
            layer_type = 'other'
            layer_idx = 'layer_0'
        
        # Extract projection type
        if 'q_proj' in param_name:
            proj_type = 'q_proj'
        elif 'k_proj' in param_name:
            proj_type = 'k_proj'
        elif 'v_proj' in param_name:
            proj_type = 'v_proj'
        else:
            proj_type = 'other'
        
        return layer_type, layer_idx, proj_type
    
    def _calculate_summary(self, results: Dict) -> None:
        """Calculate summary statistics"""
        
        all_conditions = []
        
        # Collect all condition numbers
        for layer_type in ['encoder_layers', 'decoder_layers']:
            if layer_type not in results:
                continue
            
            for layer_idx, layer_data in results[layer_type].items():
                for proj_type, conditions_list in layer_data.items():
                    for cond_info in conditions_list:
                        if 'condition_number' in cond_info:
                            cond_num = cond_info['condition_number']
                            if cond_num != float('inf'):
                                all_conditions.append(cond_num)
        
        # Calculate statistics
        if all_conditions:
            results['summary'] = {
                'mean_condition_number': float(np.mean(all_conditions)),
                'std_condition_number': float(np.std(all_conditions)),
                'max_condition_number': float(np.max(all_conditions)),
                'min_condition_number': float(np.min(all_conditions)),
                'median_condition_number': float(np.median(all_conditions)),
                'percentile_25': float(np.percentile(all_conditions, 25)),
                'percentile_75': float(np.percentile(all_conditions, 75)),
                'total_conditions': len(all_conditions)
            }
        else:
            results['summary'] = {
                'error': 'No valid condition numbers found'
            }
    
    def analyze_checkpoint_sequence(self, checkpoint_dir: str, 
                                  pattern: str = "checkpoint_epoch_*.pth") -> Dict:
        """Analyze condition numbers across multiple checkpoints"""
        
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
        
        if not checkpoint_files:
            print(f"No checkpoint files found matching {pattern} in {checkpoint_dir}")
            return {}
        
        print(f"Found {len(checkpoint_files)} checkpoints to analyze")
        
        sequence_results = {
            'checkpoint_dir': checkpoint_dir,
            'num_checkpoints': len(checkpoint_files),
            'checkpoints': [],
            'trends': {}
        }
        
        # Process each checkpoint
        for ckpt_path in checkpoint_files:
            if self.verbose:
                print(f"Processing {os.path.basename(ckpt_path)}...")
            
            try:
                results = self.extract_condition_numbers(ckpt_path)
                sequence_results['checkpoints'].append(results)
            except Exception as e:
                print(f"Error processing {ckpt_path}: {e}")
                continue
        
        # Calculate trends
        self._calculate_trends(sequence_results)
        
        return sequence_results
    
    def _calculate_trends(self, sequence_results: Dict) -> None:
        """Calculate trends across checkpoints"""
        
        epochs = []
        steps = []
        mean_conditions = []
        max_conditions = []
        min_conditions = []
        std_conditions = []
        
        for ckpt_result in sequence_results['checkpoints']:
            if 'summary' in ckpt_result and 'mean_condition_number' in ckpt_result['summary']:
                epochs.append(ckpt_result.get('epoch', -1))
                steps.append(ckpt_result.get('step', -1))
                mean_conditions.append(ckpt_result['summary']['mean_condition_number'])
                max_conditions.append(ckpt_result['summary']['max_condition_number'])
                min_conditions.append(ckpt_result['summary']['min_condition_number'])
                std_conditions.append(ckpt_result['summary']['std_condition_number'])
        
        sequence_results['trends'] = {
            'epochs': epochs,
            'steps': steps,
            'mean_conditions': mean_conditions,
            'max_conditions': max_conditions,
            'min_conditions': min_conditions,
            'std_conditions': std_conditions
        }
    
    def plot_trends(self, sequence_results: Dict, save_path: Optional[str] = None) -> None:
        """Plot condition number trends"""
        
        if 'trends' not in sequence_results or not sequence_results['trends']['epochs']:
            print("No trend data available to plot")
            return
        
        trends = sequence_results['trends']
        epochs = trends['epochs']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Mean condition number over time
        axes[0, 0].plot(epochs, trends['mean_conditions'], 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Mean Condition Number', fontsize=12)
        axes[0, 0].set_title('Mean Condition Number Trend', fontsize=14, fontweight='bold')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Min/Max condition numbers
        axes[0, 1].fill_between(epochs, trends['min_conditions'], trends['max_conditions'], 
                               alpha=0.3, label='Range')
        axes[0, 1].plot(epochs, trends['mean_conditions'], 'r-', linewidth=2, label='Mean')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Condition Number', fontsize=12)
        axes[0, 1].set_title('Condition Number Range', fontsize=14, fontweight='bold')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Standard deviation
        axes[1, 0].plot(epochs, trends['std_conditions'], 'g-s', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Std Dev of Condition Numbers', fontsize=12)
        axes[1, 0].set_title('Condition Number Variability', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Box plot for distribution at each epoch
        if len(epochs) <= 20:  # Only show box plot for reasonable number of epochs
            condition_data = []
            epoch_labels = []
            
            for ckpt_result in sequence_results['checkpoints']:
                if 'detailed_conditions' in ckpt_result:
                    conditions = [c['condition_number'] for c in ckpt_result['detailed_conditions']
                                if c['condition_number'] != float('inf')]
                    if conditions:
                        condition_data.append(conditions)
                        epoch_labels.append(str(ckpt_result.get('epoch', -1)))
            
            if condition_data:
                bp = axes[1, 1].boxplot(condition_data, labels=epoch_labels)
                axes[1, 1].set_xlabel('Epoch', fontsize=12)
                axes[1, 1].set_ylabel('Condition Number', fontsize=12)
                axes[1, 1].set_title('Condition Number Distribution', fontsize=14, fontweight='bold')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)
                plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        else:
            # Alternative plot for many epochs: correlation
            axes[1, 1].scatter(trends['mean_conditions'], trends['std_conditions'], 
                             c=epochs, cmap='viridis', s=50)
            axes[1, 1].set_xlabel('Mean Condition Number', fontsize=12)
            axes[1, 1].set_ylabel('Std Dev', fontsize=12)
            axes[1, 1].set_title('Mean vs Variability', fontsize=14, fontweight='bold')
            axes[1, 1].set_xscale('log')
            axes[1, 1].set_yscale('log')
            cbar = plt.colorbar(axes[1, 1].scatter(trends['mean_conditions'], 
                                                   trends['std_conditions'], 
                                                   c=epochs, cmap='viridis', s=50), 
                              ax=axes[1, 1])
            cbar.set_label('Epoch')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Shampoo Preconditioner Condition Number Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self, results: Dict) -> None:
        """Print formatted summary of results"""
        
        print("\n" + "="*80)
        print("SHAMPOO PRECONDITIONER CONDITION NUMBER ANALYSIS")
        print("="*80)
        
        if 'checkpoint_path' in results:
            print(f"\nCheckpoint: {results['checkpoint_path']}")
            print(f"Epoch: {results.get('epoch', 'N/A')}")
            print(f"Step: {results.get('step', 'N/A')}")
        
        # Print layer-wise analysis
        for layer_type in ['encoder_layers', 'decoder_layers']:
            if layer_type not in results or not results[layer_type]:
                continue
            
            print(f"\n--- {layer_type.upper().replace('_', ' ')} ---")
            
            for layer_idx in sorted(results[layer_type].keys()):
                layer_data = results[layer_type][layer_idx]
                print(f"\n{layer_idx}:")
                
                for proj_type in ['q_proj', 'k_proj', 'v_proj']:
                    if proj_type not in layer_data:
                        continue
                    
                    conditions = []
                    for cond_info in layer_data[proj_type]:
                        if 'condition_number' in cond_info and cond_info['condition_number'] != float('inf'):
                            conditions.append(cond_info['condition_number'])
                    
                    if conditions:
                        avg_cond = np.mean(conditions)
                        print(f"  {proj_type}: {avg_cond:.2e} (n={len(conditions)})")
        
        # Print summary statistics
        if 'summary' in results and 'mean_condition_number' in results['summary']:
            print("\n--- SUMMARY STATISTICS ---")
            summary = results['summary']
            print(f"Mean:       {summary['mean_condition_number']:.2e}")
            print(f"Std Dev:    {summary['std_condition_number']:.2e}")
            print(f"Min:        {summary['min_condition_number']:.2e}")
            print(f"Max:        {summary['max_condition_number']:.2e}")
            print(f"Median:     {summary['median_condition_number']:.2e}")
            print(f"25th %ile:  {summary['percentile_25']:.2e}")
            print(f"75th %ile:  {summary['percentile_75']:.2e}")
            print(f"Total:      {summary['total_conditions']}")
        
        print("\n" + "="*80)
    
    def save_results(self, results: Dict, output_file: str) -> None:
        """Save results to JSON file"""
        
        # Convert numpy values to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        # Recursively convert numpy values
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(item) for item in d]
            else:
                return convert_numpy(d)
        
        converted_results = convert_dict(results)
        
        with open(output_file, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Shampoo preconditioner condition numbers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single checkpoint
  python condition_number_analysis.py --checkpoint path/to/checkpoint.pth
  
  # Analyze all checkpoints in a directory
  python condition_number_analysis.py --checkpoint-dir path/to/checkpoints/
  
  # Analyze with custom pattern and save plots
  python condition_number_analysis.py --checkpoint-dir path/to/checkpoints/ \\
                                      --pattern "best_model*.pth" \\
                                      --plot --save-plot condition_trends.png
        """
    )
    
    parser.add_argument('--checkpoint', type=str, help='Path to single checkpoint file')
    parser.add_argument('--checkpoint-dir', type=str, help='Directory containing checkpoints')
    parser.add_argument('--pattern', type=str, default='checkpoint_epoch_*.pth',
                       help='Pattern for checkpoint files (default: checkpoint_epoch_*.pth)')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--plot', action='store_true', help='Generate plots for trends')
    parser.add_argument('--save-plot', type=str, help='Path to save plot image')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-print', action='store_true', help='Suppress console output')
    
    args = parser.parse_args()
    
    if not args.checkpoint and not args.checkpoint_dir:
        parser.error('Either --checkpoint or --checkpoint-dir must be specified')
    
    # Initialize analyzer
    analyzer = ShampooConditionAnalyzer(verbose=args.verbose)
    
    # Single checkpoint analysis
    if args.checkpoint:
        print(f"Analyzing checkpoint: {args.checkpoint}")
        results = analyzer.extract_condition_numbers(args.checkpoint)
        
        if not args.no_print:
            analyzer.print_summary(results)
        
        if args.output:
            analyzer.save_results(results, args.output)
        else:
            # Auto-generate output filename
            output_file = args.checkpoint.replace('.pth', '_condition_analysis.json')
            analyzer.save_results(results, output_file)
    
    # Multiple checkpoint analysis
    elif args.checkpoint_dir:
        print(f"Analyzing checkpoints in: {args.checkpoint_dir}")
        sequence_results = analyzer.analyze_checkpoint_sequence(args.checkpoint_dir, args.pattern)
        
        if sequence_results:
            # Print summary of trends
            if not args.no_print and 'trends' in sequence_results:
                trends = sequence_results['trends']
                print(f"\n{'='*60}")
                print("TREND ANALYSIS SUMMARY")
                print(f"{'='*60}")
                print(f"Checkpoints analyzed: {sequence_results['num_checkpoints']}")
                print(f"Epochs: {trends['epochs'][0]} to {trends['epochs'][-1]}")
                print(f"Mean condition number: {np.mean(trends['mean_conditions']):.2e}")
                print(f"Trend: {'Increasing' if trends['mean_conditions'][-1] > trends['mean_conditions'][0] else 'Decreasing'}")
                
                # Calculate rate of change
                if len(trends['epochs']) > 1:
                    rate = (trends['mean_conditions'][-1] - trends['mean_conditions'][0]) / \
                           (trends['epochs'][-1] - trends['epochs'][0])
                    print(f"Rate of change: {rate:.2e} per epoch")
            
            # Generate plots
            if args.plot or args.save_plot:
                analyzer.plot_trends(sequence_results, args.save_plot)
            
            # Save results
            if args.output:
                analyzer.save_results(sequence_results, args.output)
            else:
                # Auto-generate output filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"condition_analysis_sequence_{timestamp}.json"
                analyzer.save_results(sequence_results, output_file)


if __name__ == '__main__':
    main()
