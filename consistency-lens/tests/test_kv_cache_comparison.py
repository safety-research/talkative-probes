#!/usr/bin/env python3
"""Compare memory usage between different generation methods."""

import gc
import torch
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from datetime import datetime
import json

from lens.models.decoder import Decoder, DecoderConfig
from lens.models.kv_cache_experimental import (
    KVCacheExperimental, 
    generate_with_kv_cache_experimental
)


def profile_memory_usage(func, *args, **kwargs):
    """Profile GPU memory usage of a function."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Get baseline
        baseline_allocated = torch.cuda.memory_allocated() / (1024**3)
        baseline_reserved = torch.cuda.memory_reserved() / (1024**3)
        
        # Run function
        result = func(*args, **kwargs)
        
        torch.cuda.synchronize()
        
        # Get peak stats
        peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        
        return {
            'result': result,
            'baseline_allocated_gb': baseline_allocated,
            'baseline_reserved_gb': baseline_reserved,
            'peak_allocated_gb': peak_allocated,
            'peak_reserved_gb': peak_reserved,
            'delta_allocated_gb': peak_allocated - baseline_allocated,
            'delta_reserved_gb': peak_reserved - baseline_reserved,
        }
    else:
        return {
            'result': func(*args, **kwargs),
            'baseline_allocated_gb': 0,
            'baseline_reserved_gb': 0,
            'peak_allocated_gb': 0,
            'peak_reserved_gb': 0,
            'delta_allocated_gb': 0,
            'delta_reserved_gb': 0,
        }


def test_naive_generation(decoder, activation, max_length):
    """Test standard generation without optimizations."""
    with torch.no_grad():
        result = decoder.generate_soft(
            activation_input=activation,
            max_length=max_length,
            gumbel_tau=1.0,
            use_projection=True,
            print_prompt=False
        )
    return result


def test_checkpoint_generation(decoder, activation, max_length):
    """Test generation with gradient checkpointing."""
    with torch.no_grad():
        result = decoder.generate_soft_chkpt(
            activation_input=activation,
            max_length=max_length,
            gumbel_tau=1.0,
            use_projection=True,
            print_prompt=False,
            checkpoint_every_n_tokens=1
        )
    return result


def test_kv_cache_generation(decoder, activation, max_length):
    """Test generation with KV caching."""
    with torch.no_grad():
        result = decoder.generate_soft_kv_cached(
            activation_input=activation,
            max_length=max_length,
            gumbel_tau=1.0,
            use_projection=True,
            print_prompt=False
        )
    return result


def test_experimental_kv_cache(decoder, activation, max_length):
    """Test experimental KV cache implementation."""
    # Get initial embeddings through decoder
    with torch.no_grad():
        # Project activation to embeddings
        if decoder.config.eye_init:
            projected = decoder.proj(activation)
        else:
            projected = activation
        
        # Add prompt embeddings
        batch_size = activation.shape[0]
        prompt_left = decoder.prompt_left_emb.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_right = decoder.prompt_right_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine
        full_embeds = torch.cat([prompt_left, projected.unsqueeze(1), prompt_right], dim=1)
        
        # Use experimental generation
        result = generate_with_kv_cache_experimental(
            decoder.base,
            full_embeds,
            max_length=max_length,
            temperature=1.0,
            track_memory=True
        )
    
    return result


def run_comparison(model_name="gpt2", device="cuda", batch_sizes=[1, 2, 4], lengths=[8, 16, 32, 64]):
    """Run comprehensive comparison of generation methods."""
    
    print(f"Running memory comparison on {device}")
    print("="*80)
    
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        use_checkpointing=False,
    )
    decoder = Decoder(config).to(device)
    decoder.eval()
    
    # Set prompt
    prompt = "a long time ago in a galaxy far far away, <embed> there"
    decoder.set_prompt(prompt, tokenizer)
    
    d_model = decoder.base.config.hidden_size
    
    results = {
        'model_name': model_name,
        'device': str(device),
        'd_model': d_model,
        'timestamp': datetime.now().isoformat(),
        'comparisons': []
    }
    
    for batch_size in batch_sizes:
        for length in lengths:
            print(f"\nTesting batch_size={batch_size}, length={length}")
            print("-"*40)
            
            # Create test activation
            activation = torch.randn(
                batch_size, d_model, 
                device=device, 
                dtype=decoder.proj.weight.dtype
            ) * 0.1
            
            comparison = {
                'batch_size': batch_size,
                'length': length,
                'methods': {}
            }
            
            # Test each method
            methods = [
                ('naive', test_naive_generation),
                ('checkpoint', test_checkpoint_generation),
                ('kv_cache', test_kv_cache_generation),
                # ('experimental', test_experimental_kv_cache),  # Comment out if causing issues
            ]
            
            for method_name, method_func in methods:
                print(f"  Testing {method_name}...")
                try:
                    profile = profile_memory_usage(
                        method_func, 
                        decoder, 
                        activation, 
                        length
                    )
                    
                    comparison['methods'][method_name] = {
                        'peak_memory_gb': profile['peak_allocated_gb'],
                        'delta_memory_gb': profile['delta_allocated_gb'],
                        'success': True
                    }
                    
                    print(f"    Peak memory: {profile['peak_allocated_gb']:.3f} GB")
                    print(f"    Delta memory: {profile['delta_allocated_gb']:.3f} GB")
                    
                except Exception as e:
                    print(f"    ERROR: {str(e)}")
                    comparison['methods'][method_name] = {
                        'peak_memory_gb': None,
                        'delta_memory_gb': None,
                        'success': False,
                        'error': str(e)
                    }
            
            # Calculate savings
            if 'naive' in comparison['methods'] and 'kv_cache' in comparison['methods']:
                naive_mem = comparison['methods']['naive'].get('peak_memory_gb')
                kv_mem = comparison['methods']['kv_cache'].get('peak_memory_gb')
                
                if naive_mem and kv_mem:
                    savings_pct = (naive_mem - kv_mem) / naive_mem * 100
                    ratio = naive_mem / kv_mem
                    comparison['savings'] = {
                        'percentage': savings_pct,
                        'ratio': ratio
                    }
                    print(f"\n  KV Cache Savings: {savings_pct:.1f}% (ratio: {ratio:.2f}x)")
            
            results['comparisons'].append(comparison)
    
    return results


def plot_results(results, save_path="kv_cache_comparison.png"):
    """Create visualization of memory usage comparison."""
    
    # Extract data for plotting
    batch_sizes = sorted(set(c['batch_size'] for c in results['comparisons']))
    lengths = sorted(set(c['length'] for c in results['comparisons']))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot 1: Memory vs Length for different batch sizes
    ax = axes[0]
    for batch_size in batch_sizes[:3]:  # Limit to 3 batch sizes
        for method in ['naive', 'kv_cache']:
            data_points = []
            for comp in results['comparisons']:
                if comp['batch_size'] == batch_size and method in comp['methods']:
                    mem = comp['methods'][method].get('peak_memory_gb')
                    if mem:
                        data_points.append((comp['length'], mem))
            
            if data_points:
                x, y = zip(*sorted(data_points))
                label = f"{method} (B={batch_size})"
                style = '-' if method == 'naive' else '--'
                ax.plot(x, y, style, marker='o', label=label)
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Peak Memory (GB)')
    ax.set_title('Memory Usage vs Sequence Length')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Savings percentage
    ax = axes[1]
    for batch_size in batch_sizes[:3]:
        savings_data = []
        for comp in results['comparisons']:
            if comp['batch_size'] == batch_size and 'savings' in comp:
                savings_data.append((comp['length'], comp['savings']['percentage']))
        
        if savings_data:
            x, y = zip(*sorted(savings_data))
            ax.plot(x, y, marker='o', label=f"B={batch_size}")
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Memory Savings (%)')
    ax.set_title('KV Cache Memory Savings')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Memory ratio (naive/kv_cache)
    ax = axes[2]
    for batch_size in batch_sizes[:3]:
        ratio_data = []
        for comp in results['comparisons']:
            if comp['batch_size'] == batch_size and 'savings' in comp:
                ratio_data.append((comp['length'], comp['savings']['ratio']))
        
        if ratio_data:
            x, y = zip(*sorted(ratio_data))
            ax.plot(x, y, marker='o', label=f"B={batch_size}")
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Memory Ratio (Naive/KV Cache)')
    ax.set_title('Memory Usage Ratio')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Summary statistics
    ax = axes[3]
    ax.axis('off')
    
    # Calculate average savings
    all_savings = []
    for comp in results['comparisons']:
        if 'savings' in comp:
            all_savings.append(comp['savings']['percentage'])
    
    if all_savings:
        avg_savings = np.mean(all_savings)
        min_savings = np.min(all_savings)
        max_savings = np.max(all_savings)
        
        summary_text = f"""
Summary Statistics:
    
Model: {results['model_name']}
Device: {results['device']}
Hidden Size: {results['d_model']}

Average Memory Savings: {avg_savings:.1f}%
Min Savings: {min_savings:.1f}%
Max Savings: {max_savings:.1f}%

Total Comparisons: {len(results['comparisons'])}
Timestamp: {results['timestamp']}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {save_path}")


def analyze_scaling(results):
    """Analyze how memory scales with sequence length."""
    
    print("\n" + "="*80)
    print("Memory Scaling Analysis")
    print("="*80)
    
    # Group by batch size
    batch_data = {}
    for comp in results['comparisons']:
        batch_size = comp['batch_size']
        if batch_size not in batch_data:
            batch_data[batch_size] = {'lengths': [], 'naive': [], 'kv_cache': []}
        
        batch_data[batch_size]['lengths'].append(comp['length'])
        
        if 'naive' in comp['methods'] and comp['methods']['naive']['success']:
            batch_data[batch_size]['naive'].append(comp['methods']['naive']['peak_memory_gb'])
        else:
            batch_data[batch_size]['naive'].append(None)
            
        if 'kv_cache' in comp['methods'] and comp['methods']['kv_cache']['success']:
            batch_data[batch_size]['kv_cache'].append(comp['methods']['kv_cache']['peak_memory_gb'])
        else:
            batch_data[batch_size]['kv_cache'].append(None)
    
    # Analyze scaling for each batch size
    for batch_size, data in sorted(batch_data.items()):
        print(f"\nBatch size {batch_size}:")
        
        # Sort by length
        sorted_indices = np.argsort(data['lengths'])
        lengths = np.array(data['lengths'])[sorted_indices]
        
        # Analyze naive scaling (should be ~O(n²))
        naive_mems = [data['naive'][i] for i in sorted_indices if data['naive'][i] is not None]
        if len(naive_mems) >= 2:
            # Calculate approximate scaling exponent
            # Memory ~ length^k, so log(memory) ~ k * log(length)
            valid_lengths = [lengths[i] for i in range(len(lengths)) if data['naive'][sorted_indices[i]] is not None]
            if len(valid_lengths) >= 2:
                log_lengths = np.log(valid_lengths[1:])
                log_mem_ratios = np.log(np.array(naive_mems[1:]) / np.array(naive_mems[:-1]))
                log_len_ratios = np.log(np.array(valid_lengths[1:]) / np.array(valid_lengths[:-1]))
                
                scaling_exponents = log_mem_ratios / log_len_ratios
                avg_exponent = np.mean(scaling_exponents)
                print(f"  Naive method scaling: ~O(n^{avg_exponent:.2f})")
        
        # Analyze KV cache scaling (should be ~O(n))
        kv_mems = [data['kv_cache'][i] for i in sorted_indices if data['kv_cache'][i] is not None]
        if len(kv_mems) >= 2:
            valid_lengths = [lengths[i] for i in range(len(lengths)) if data['kv_cache'][sorted_indices[i]] is not None]
            if len(valid_lengths) >= 2:
                log_lengths = np.log(valid_lengths[1:])
                log_mem_ratios = np.log(np.array(kv_mems[1:]) / np.array(kv_mems[:-1]))
                log_len_ratios = np.log(np.array(valid_lengths[1:]) / np.array(valid_lengths[:-1]))
                
                scaling_exponents = log_mem_ratios / log_len_ratios
                avg_exponent = np.mean(scaling_exponents)
                print(f"  KV cache scaling: ~O(n^{avg_exponent:.2f})")


def main():
    """Run the complete analysis."""
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cpu":
        print("WARNING: Running on CPU, memory measurements may not be accurate")
        print("For best results, run on a GPU")
    
    # Run comparison
    results = run_comparison(
        model_name="gpt2",
        device=device,
        batch_sizes=[1, 2, 4],
        lengths=[8, 16, 32, 64, 96]
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"kv_cache_comparison_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Create plots
    plot_results(results)
    
    # Analyze scaling behavior
    analyze_scaling(results)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()