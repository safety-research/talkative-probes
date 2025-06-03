#!/usr/bin/env python3
"""Test attention computation timing specifically."""

import torch
import time
import matplotlib.pyplot as plt
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


def time_attention_computation():
    """Time attention computation with and without KV cache."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # GPT-2 attention parameters
    hidden_size = 768
    num_heads = 12
    
    # Initialize attention layer
    attention = GPT2Attention(torch.nn.ModuleDict({
        'c_attn': torch.nn.Linear(hidden_size, 3 * hidden_size),
        'c_proj': torch.nn.Linear(hidden_size, hidden_size),
        'attn_dropout': torch.nn.Dropout(0.1),
        'resid_dropout': torch.nn.Dropout(0.1),
    }))
    attention.num_heads = num_heads
    attention.head_dim = hidden_size // num_heads
    attention.split_size = hidden_size
    attention.to(device)
    
    print("Pure Attention Computation Timing")
    print("="*60)
    
    batch_size = 4
    sequence_lengths = [8, 16, 32, 64, 128, 256]
    
    naive_times = []
    cached_times = []
    
    for seq_len in sequence_lengths:
        print(f"\nSequence length: {seq_len}")
        
        # Test naive attention (full recomputation)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
        
        # Warm up
        with torch.no_grad():
            _ = attention(hidden_states)[0]
        
        # Time naive approach
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            for _ in range(10):  # Multiple iterations for more stable timing
                output = attention(hidden_states)[0]
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        naive_time = (time.time() - start) / 10
        naive_times.append(naive_time)
        
        # Test with simulated KV cache (incremental attention)
        cached_time_total = 0
        
        # Precompute all keys and values
        with torch.no_grad():
            qkv = attention.c_attn(hidden_states)
            query, key, value = qkv.split(hidden_size, dim=2)
            
            # Reshape for multi-head attention
            def split_heads(tensor):
                return tensor.view(batch_size, -1, num_heads, hidden_size // num_heads).permute(0, 2, 1, 3)
            
            all_keys = split_heads(key)
            all_values = split_heads(value)
            all_queries = split_heads(query)
        
        # Time incremental attention (simulating KV cache)
        for pos in range(seq_len):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            
            with torch.no_grad():
                # Current query
                q = all_queries[:, :, pos:pos+1, :]
                
                # Use cached K,V up to current position
                k = all_keys[:, :, :pos+1, :]
                v = all_values[:, :, :pos+1, :]
                
                # Compute attention for current position only
                attn_weights = torch.matmul(q, k.transpose(-1, -2))
                attn_weights = attn_weights / (hidden_size // num_heads) ** 0.5
                attn_weights = torch.softmax(attn_weights, dim=-1)
                attn_output = torch.matmul(attn_weights, v)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            cached_time_total += time.time() - start
        
        cached_time = cached_time_total / seq_len  # Average per position
        cached_times.append(cached_time)
        
        print(f"  Naive (full): {naive_time:.6f}s")
        print(f"  Cached (avg per pos): {cached_time:.6f}s")
        print(f"  Speedup: {naive_time/cached_time:.2f}x")
        
        # Theoretical computation analysis
        naive_flops = batch_size * seq_len * seq_len * hidden_size * 2  # QK^T computation
        cached_flops_avg = batch_size * seq_len * hidden_size * 2 / 2  # Average position
        print(f"  Theoretical FLOP ratio: {naive_flops/cached_flops_avg:.1f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(sequence_lengths, naive_times, 'o-', label='Naive (O(n²))')
    plt.plot(sequence_lengths, cached_times, 's-', label='KV Cached (O(n))')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.title('Attention Computation Time')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    speedups = [n/c for n, c in zip(naive_times, cached_times)]
    plt.plot(sequence_lengths, speedups, 'o-')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup')
    plt.title('KV Cache Speedup vs Naive')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('attention_timing_comparison.png')
    print(f"\nPlot saved to: attention_timing_comparison.png")
    
    # Scaling analysis
    print("\n" + "="*60)
    print("Scaling Analysis")
    print("="*60)
    
    # Check if times follow expected scaling
    if len(sequence_lengths) >= 3:
        # For O(n²), doubling n should ~4x the time
        # For O(n), doubling n should ~2x the time
        
        print("\nNaive scaling (should be ~4x for 2x length):")
        for i in range(1, len(sequence_lengths)):
            if sequence_lengths[i] == 2 * sequence_lengths[i-1]:
                ratio = naive_times[i] / naive_times[i-1]
                print(f"  {sequence_lengths[i-1]} -> {sequence_lengths[i]}: {ratio:.2f}x")
        
        print("\nCached scaling (should be ~2x for 2x length):")
        for i in range(1, len(sequence_lengths)):
            if sequence_lengths[i] == 2 * sequence_lengths[i-1]:
                ratio = cached_times[i] / cached_times[i-1]
                print(f"  {sequence_lengths[i-1]} -> {sequence_lengths[i]}: {ratio:.2f}x")


if __name__ == "__main__":
    time_attention_computation()