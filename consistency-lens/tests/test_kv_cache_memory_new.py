#!/usr/bin/env python3
"""Comprehensive memory testing for KV cache implementation."""

import gc
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

from lens.models.decoder import Decoder, DecoderConfig


def get_gpu_memory_stats():
    """Get current GPU memory usage statistics."""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,
            'max_reserved': torch.cuda.max_memory_reserved() / 1024**3,
        }
    return {}


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def test_memory_usage(model_name="gpt2", device="cuda", batch_size=4, num_lengths=[8, 16, 32, 64]):
    """Test memory usage for different generation methods and lengths."""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize decoder
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
    
    # Test activation
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(batch_size, d_model, device=device, dtype=decoder.proj.weight.dtype) * 0.1
    
    results = {
        'model_name': model_name,
        'batch_size': batch_size,
        'device': str(device),
        'd_model': d_model,
        'prompt': prompt,
        'timestamp': datetime.now().isoformat(),
        'methods': {}
    }
    
    # Test each method
    methods = [
        ('naive', lambda act, max_len: decoder.generate_soft(
            activation_input=act,
            max_length=max_len,
            gumbel_tau=1.0,
            use_projection=True,
            print_prompt=False
        )),
        ('checkpoint', lambda act, max_len: decoder.generate_soft_chkpt(
            activation_input=act,
            max_length=max_len,
            gumbel_tau=1.0,
            use_projection=True,
            print_prompt=False,
            checkpoint_every_n_tokens=1
        )),
        ('kv_cache', lambda act, max_len: decoder.generate_soft_kv_cached(
            activation_input=act,
            max_length=max_len,
            gumbel_tau=1.0,
            use_projection=True,
            print_prompt=False
        ))
    ]
    
    for method_name, method_fn in methods:
        print(f"\n{'='*60}")
        print(f"Testing method: {method_name}")
        print(f"{'='*60}")
        
        method_results = {
            'lengths': [],
            'peak_memory': [],
            'allocated_memory': [],
            'reserved_memory': [],
            'time_taken': [],
        }
        
        for max_length in num_lengths:
            print(f"\n  Testing length: {max_length}")
            
            # Reset memory stats
            reset_gpu_memory_stats()
            
            # Get baseline memory
            baseline = get_gpu_memory_stats()
            print(f"    Baseline - Allocated: {baseline['allocated']:.3f} GB, Reserved: {baseline['reserved']:.3f} GB")
            
            # Run generation
            import time
            start_time = time.time()
            
            try:
                with torch.no_grad():
                    result = method_fn(activation, max_length)
                
                # Force synchronization
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Get peak memory stats
                peak_stats = get_gpu_memory_stats()
                
                print(f"    Peak - Allocated: {peak_stats['max_allocated']:.3f} GB, Reserved: {peak_stats['max_reserved']:.3f} GB")
                print(f"    Delta - Allocated: {peak_stats['max_allocated'] - baseline['allocated']:.3f} GB")
                print(f"    Time taken: {end_time - start_time:.3f} seconds")
                
                # Store results
                method_results['lengths'].append(max_length)
                method_results['peak_memory'].append(peak_stats['max_allocated'])
                method_results['allocated_memory'].append(peak_stats['allocated'])
                method_results['reserved_memory'].append(peak_stats['max_reserved'])
                method_results['time_taken'].append(end_time - start_time)
                
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                method_results['lengths'].append(max_length)
                method_results['peak_memory'].append(None)
                method_results['allocated_memory'].append(None)
                method_results['reserved_memory'].append(None)
                method_results['time_taken'].append(None)
            
            # Clean up
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        results['methods'][method_name] = method_results
    
    return results


def plot_memory_comparison(results, save_path="memory_comparison.png"):
    """Plot memory usage comparison across methods."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot memory usage
    for method_name, method_results in results['methods'].items():
        lengths = method_results['lengths']
        peak_memory = method_results['peak_memory']
        # Filter out None values
        valid_points = [(l, m) for l, m in zip(lengths, peak_memory) if m is not None]
        if valid_points:
            lengths, peak_memory = zip(*valid_points)
            ax1.plot(lengths, peak_memory, marker='o', label=method_name)
    
    ax1.set_xlabel('Generation Length (tokens)')
    ax1.set_ylabel('Peak GPU Memory (GB)')
    ax1.set_title('Peak Memory Usage by Generation Method')
    ax1.legend()
    ax1.grid(True)
    
    # Plot time taken
    for method_name, method_results in results['methods'].items():
        lengths = method_results['lengths']
        time_taken = method_results['time_taken']
        # Filter out None values
        valid_points = [(l, t) for l, t in zip(lengths, time_taken) if t is not None]
        if valid_points:
            lengths, time_taken = zip(*valid_points)
            ax2.plot(lengths, time_taken, marker='o', label=method_name)
    
    ax2.set_xlabel('Generation Length (tokens)')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Generation Time by Method')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\nPlot saved to: {save_path}")


def calculate_theoretical_memory(d_model, num_layers, batch_size, seq_length):
    """Calculate theoretical memory usage for attention computation."""
    
    # Attention scores: batch_size * num_heads * seq_length * seq_length
    # Assuming num_heads = 12 for GPT-2
    num_heads = 12
    bytes_per_float = 4  # float32
    
    # O(n²) attention memory
    attention_memory = batch_size * num_heads * seq_length * seq_length * bytes_per_float
    
    # KV cache memory (O(n))
    # Per layer: 2 * batch_size * num_heads * seq_length * (d_model // num_heads)
    kv_cache_memory = 2 * batch_size * seq_length * d_model * bytes_per_float * num_layers
    
    return {
        'attention_o_n2': attention_memory / 1024**3,  # GB
        'kv_cache_o_n': kv_cache_memory / 1024**3,     # GB
        'ratio': attention_memory / kv_cache_memory if kv_cache_memory > 0 else float('inf')
    }


def main():
    """Run comprehensive memory tests."""
    
    print("KV Cache Memory Usage Analysis")
    print("="*60)
    
    # Test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run tests
    results = test_memory_usage(
        model_name="gpt2",
        device=device,
        batch_size=4,
        num_lengths=[8, 16, 32, 64, 128]
    )
    
    # Save results
    results_path = f"kv_cache_memory_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Plot comparison
    plot_memory_comparison(results)
    
    # Print theoretical analysis
    print("\n" + "="*60)
    print("Theoretical Memory Analysis")
    print("="*60)
    
    d_model = results['d_model']
    batch_size = results['batch_size']
    num_layers = 12  # GPT-2 has 12 layers
    
    for seq_length in [16, 32, 64, 128]:
        theory = calculate_theoretical_memory(d_model, num_layers, batch_size, seq_length)
        print(f"\nSequence length: {seq_length}")
        print(f"  O(n²) attention memory: {theory['attention_o_n2']:.3f} GB")
        print(f"  O(n) KV cache memory: {theory['kv_cache_o_n']:.3f} GB")
        print(f"  Theoretical ratio: {theory['ratio']:.1f}x")
    
    # Analyze actual results
    print("\n" + "="*60)
    print("Actual Memory Savings Analysis")
    print("="*60)
    
    naive_results = results['methods'].get('naive', {})
    kv_cache_results = results['methods'].get('kv_cache', {})
    
    if naive_results and kv_cache_results:
        for i, length in enumerate(naive_results['lengths']):
            naive_mem = naive_results['peak_memory'][i]
            kv_mem = kv_cache_results['peak_memory'][i]
            
            if naive_mem and kv_mem:
                savings = (naive_mem - kv_mem) / naive_mem * 100
                ratio = naive_mem / kv_mem
                print(f"\nLength {length}:")
                print(f"  Naive: {naive_mem:.3f} GB")
                print(f"  KV Cache: {kv_mem:.3f} GB")
                print(f"  Savings: {savings:.1f}%")
                print(f"  Ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()