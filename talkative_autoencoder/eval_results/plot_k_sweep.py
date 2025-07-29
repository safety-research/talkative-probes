#!/usr/bin/env python3
"""
Plot variance recovery vs K for best-of-K sweep results.

Usage:
    # Plot all gemma3 JSON files in eval_results/
    source scripts/ensure_env.sh && uv run python eval_results/plot_k_sweep.py
    
    # Plot specific files
    source scripts/ensure_env.sh && uv run python eval_results/plot_k_sweep.py file1.json file2.json
    
    # Plot with custom output name
    source scripts/ensure_env.sh && uv run python eval_results/plot_k_sweep.py -o my_plot.png file1.json file2.json

Note: This script requires matplotlib, which is available in the consistency-lens environment.
Always run with 'source scripts/ensure_env.sh && uv run python' to ensure the correct environment.

The script will:
- Find all gemma3_k_sweep_full*.json files if no files are specified
- Create a single plot with all runs on the same graph
- X-axis: Best of N (K values) on log scale
- Y-axis: Variance Recovered (auto-scaled to data range)
- Save as both PNG and PDF formats
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(json_file):
    """Load results from JSON file, handling potential formatting issues."""
    with open(json_file, 'r') as f:
        content = f.read()
    
    # Try to parse as-is first
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error in {json_file}: {e}")
        # Try to fix common issues like trailing commas
        # Remove trailing commas before closing braces/brackets
        import re
        fixed_content = re.sub(r',\s*([\]}])', r'\1', content)
        try:
            data = json.loads(fixed_content)
            print(f"Successfully parsed {json_file} after fixing trailing commas")
            return data
        except:
            # If still fails, try to extract just the results_by_k part
            print(f"Failed to parse {json_file}, skipping...")
            raise


def plot_variance_recovery(results_files, output_file='variance_recovery_plot.png'):
    """Plot variance recovery vs K for multiple result files."""
    
    plt.figure(figsize=(10, 8))
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_files)))
    
    valid_files = []
    all_data = []
    
    # Load all files first, skipping any that fail
    for result_file in results_files:
        try:
            data = load_results(result_file)
            all_data.append((result_file, data))
            valid_files.append(result_file)
        except Exception as e:
            print(f"Skipping {result_file} due to error: {e}")
            continue
    
    if not valid_files:
        print("No valid result files found!")
        return
    
    for idx, (result_file, data) in enumerate(all_data):
        
        # Extract K values and variance recovery
        results = data['results_by_k']
        k_values = [r['k'] for r in results]
        var_recovery = [r['variance_recovery'] for r in results]
        
        # Skip confidence intervals
        has_ci = False
        
        # Extract label from filename
        label = Path(result_file).stem.replace('gemma3_k_sweep_full', '').replace('_', ' ').strip()
        if not label:
            label = f"Run {idx+1}"
        
        # Plot line
        plt.plot(k_values, var_recovery, 'o-', color=colors[idx], 
                linewidth=2.5, markersize=8, label=label)
        
        # No confidence intervals
    
    # Set log scale for x-axis
    plt.xscale('log', base=2)
    
    # Labels and title
    plt.xlabel('Best of N', fontsize=14)
    plt.ylabel('Variance Recovered', fontsize=14)
    plt.title('Gemma3 27B Test Time Interpretability', fontsize=16, pad=20)
    
    # Set x-axis ticks to show actual K values
    # Extract unique k values from already loaded data
    unique_k = sorted(set(k for _, data in all_data for k in [r['k'] for r in data['results_by_k']]))
    plt.xticks(unique_k, [str(k) for k in unique_k])
    
    # Grid
    plt.grid(True, alpha=0.3)
    
    # Legend
    if len(results_files) > 1:
        plt.legend(loc='lower right', fontsize=12)
    
    # Y-axis limits - auto-scale to data
    # Get min and max values from all data
    all_var_recovery = []
    for _, data in all_data:
        results = data['results_by_k']
        all_var_recovery.extend([r['variance_recovery'] for r in results])
    
    if all_var_recovery:
        y_min = min(all_var_recovery) * 0.95  # Add 5% padding
        y_max = max(all_var_recovery) * 1.05  # Add 5% padding
        plt.ylim(y_min, y_max)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Also save as PDF for publication quality
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF version saved to {pdf_file}")
    
    plt.close()


def plot_all_json_files():
    """Find all gemma3 JSON files and create a combined plot."""
    eval_dir = Path(__file__).parent
    
    # Find all gemma3 k_sweep files
    json_files = sorted(eval_dir.glob('gemma3_k_sweep_full*.json'))
    
    if not json_files:
        print("No gemma3_k_sweep_full*.json files found in eval_results/")
        return
    
    print(f"Found {len(json_files)} result files:")
    for f in json_files:
        print(f"  - {f.name}")
    
    # Create plot
    plot_variance_recovery(json_files, output_file='gemma3_variance_recovery.png')
    
    # Don't create individual plots


def main():
    parser = argparse.ArgumentParser(description='Plot variance recovery from K-sweep results')
    parser.add_argument('files', nargs='*', help='JSON result files to plot (if none, plots all gemma3 files)')
    parser.add_argument('-o', '--output', default='variance_recovery_plot.png', 
                       help='Output filename (default: variance_recovery_plot.png)')
    args = parser.parse_args()
    
    if args.files:
        # Plot specific files
        plot_variance_recovery(args.files, args.output)
    else:
        # Plot all gemma3 files
        plot_all_json_files()


if __name__ == '__main__':
    main()