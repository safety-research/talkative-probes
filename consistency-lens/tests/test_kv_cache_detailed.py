#!/usr/bin/env python3
"""Detailed memory profiling for KV cache to understand overhead."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import gc
from lens.models.decoder import Decoder, DecoderConfig


def detailed_memory_profile():
    """Profile memory usage at each stage of generation."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reset memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
    
    def get_memory():
        if torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            }
        return {'allocated_mb': 0, 'reserved_mb': 0}
    
    print("Detailed Memory Profiling")
    print("="*60)
    
    # Stage 1: Empty GPU
    mem_empty = get_memory()
    print(f"1. Empty GPU: {mem_empty['allocated_mb']:.1f} MB")
    
    # Stage 2: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    mem_tokenizer = get_memory()
    print(f"2. After tokenizer: {mem_tokenizer['allocated_mb']:.1f} MB (+{mem_tokenizer['allocated_mb'] - mem_empty['allocated_mb']:.1f} MB)")
    
    # Stage 3: Initialize decoder
    config = DecoderConfig(
        model_name="gpt2",
        use_kv_cache=True,
        use_checkpointing=False,
    )
    decoder = Decoder(config).to(device)
    decoder.eval()
    mem_decoder = get_memory()
    print(f"3. After decoder init: {mem_decoder['allocated_mb']:.1f} MB (+{mem_decoder['allocated_mb'] - mem_tokenizer['allocated_mb']:.1f} MB)")
    
    # Stage 4: Set prompt
    prompt = "a long time ago in a galaxy far far away, <embed> there"
    decoder.set_prompt(prompt, tokenizer)
    mem_prompt = get_memory()
    print(f"4. After set prompt: {mem_prompt['allocated_mb']:.1f} MB (+{mem_prompt['allocated_mb'] - mem_decoder['allocated_mb']:.1f} MB)")
    
    # Stage 5: Create activation
    batch_size = 4
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(batch_size, d_model, device=device, dtype=decoder.proj.weight.dtype) * 0.1
    mem_activation = get_memory()
    print(f"5. After activation: {mem_activation['allocated_mb']:.1f} MB (+{mem_activation['allocated_mb'] - mem_prompt['allocated_mb']:.1f} MB)")
    
    print("\nModel Statistics:")
    print(f"- Model parameters: {sum(p.numel() for p in decoder.parameters()) / 1e6:.1f}M")
    print(f"- Trainable parameters: {sum(p.numel() for p in decoder.parameters() if p.requires_grad) / 1e6:.1f}M")
    print(f"- Hidden size: {d_model}")
    print(f"- Batch size: {batch_size}")
    
    # Test generation with different lengths
    print("\nGeneration Memory Usage:")
    print("-"*60)
    
    for method_name, method_fn in [
        ("naive", lambda act, length: decoder.generate_soft(
            activation_input=act, max_length=length, gumbel_tau=1.0,
            use_projection=True, print_prompt=False
        )),
        ("kv_cache", lambda act, length: decoder.generate_soft_kv_cached(
            activation_input=act, max_length=length, gumbel_tau=1.0,
            use_projection=True, print_prompt=False
        ))
    ]:
        print(f"\nMethod: {method_name}")
        
        for length in [16, 32, 64]:
            # Reset peak memory stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            mem_before = get_memory()
            
            with torch.no_grad():
                result = method_fn(activation, length)
            
            mem_after = get_memory()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            print(f"  Length {length}: Peak {peak_mem:.1f} MB (delta: {peak_mem - mem_before['allocated_mb']:.1f} MB)")
            
            # Clean up
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Theoretical memory calculation
    print("\nTheoretical Memory Analysis:")
    print("-"*60)
    
    num_layers = 12  # GPT-2
    num_heads = 12
    head_dim = d_model // num_heads
    
    for seq_len in [16, 32, 64]:
        # Attention memory (per layer)
        attention_mem_mb = (batch_size * num_heads * seq_len * seq_len * 4) / 1024**2
        total_attention_mb = attention_mem_mb * num_layers
        
        # KV cache memory
        kv_cache_mb = (2 * batch_size * seq_len * d_model * 4 * num_layers) / 1024**2
        
        # Other activations (rough estimate)
        # - Hidden states: batch * seq * d_model * 4 bytes per layer
        # - MLP intermediate: batch * seq * 4*d_model * 4 bytes per layer
        hidden_states_mb = (batch_size * seq_len * d_model * 4 * num_layers) / 1024**2
        mlp_intermediate_mb = (batch_size * seq_len * 4 * d_model * 4 * num_layers) / 1024**2
        
        print(f"\nSequence length {seq_len}:")
        print(f"  Attention only: {total_attention_mb:.1f} MB")
        print(f"  KV cache only: {kv_cache_mb:.1f} MB")
        print(f"  Hidden states: {hidden_states_mb:.1f} MB")
        print(f"  MLP intermediate: {mlp_intermediate_mb:.1f} MB")
        print(f"  Total theoretical: {total_attention_mb + hidden_states_mb + mlp_intermediate_mb:.1f} MB (naive)")
        print(f"  Total theoretical: {kv_cache_mb + hidden_states_mb + mlp_intermediate_mb:.1f} MB (KV cache)")
        print(f"  Expected savings: {((total_attention_mb - kv_cache_mb) / (total_attention_mb + hidden_states_mb + mlp_intermediate_mb)) * 100:.1f}%")


if __name__ == "__main__":
    detailed_memory_profile()