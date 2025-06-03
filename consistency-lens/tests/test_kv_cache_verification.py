#!/usr/bin/env python3
"""Verify KV cache is working correctly by checking computation patterns."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
import time


def verify_kv_cache_behavior():
    """Verify that KV cache is actually being used and working correctly."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = DecoderConfig(
        model_name="gpt2",
        use_kv_cache=True,
        use_checkpointing=False,
    )
    decoder = Decoder(config).to(device)
    decoder.eval()
    
    prompt = "a long time ago in a galaxy far far away, <embed> there"
    decoder.set_prompt(prompt, tokenizer)
    
    # Test parameters
    batch_size = 2
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(batch_size, d_model, device=device, dtype=decoder.proj.weight.dtype) * 0.1
    max_length = 50
    
    print("KV Cache Verification")
    print("="*60)
    
    # Test 1: Timing comparison
    print("\n1. Timing Comparison (50 tokens):")
    
    # Warm up
    with torch.no_grad():
        _ = decoder.generate_soft(activation, max_length=5, gumbel_tau=1.0, use_projection=True, print_prompt=False)
    
    # Time naive method
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        result_naive = decoder.generate_soft(
            activation_input=activation,
            max_length=max_length,
            gumbel_tau=1.0,
            use_projection=True,
            print_prompt=False
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_naive = time.time() - start
    
    # Time KV cache method
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        result_kv = decoder.generate_soft_kv_cached(
            activation_input=activation,
            max_length=max_length,
            gumbel_tau=1.0,
            use_projection=True,
            print_prompt=False
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_kv = time.time() - start
    
    print(f"  Naive method: {time_naive:.3f} seconds")
    print(f"  KV cache method: {time_kv:.3f} seconds")
    print(f"  Speedup: {time_naive/time_kv:.2f}x")
    
    # Test 2: Check outputs are functionally equivalent (with same random seed)
    print("\n2. Output Consistency Check:")
    
    # Set fixed seed for deterministic generation
    torch.manual_seed(42)
    with torch.no_grad():
        result1 = decoder.generate_soft(
            activation_input=activation[:1],  # Just first sample
            max_length=20,
            gumbel_tau=0.01,  # Low temperature for more deterministic
            use_projection=True,
            print_prompt=False
        )
    
    torch.manual_seed(42)
    with torch.no_grad():
        result2 = decoder.generate_soft_kv_cached(
            activation_input=activation[:1],
            max_length=20,
            gumbel_tau=0.01,
            use_projection=True,
            print_prompt=False
        )
    
    # Compare outputs
    # Check if Generated objects have the same structure
    if hasattr(result1, 'soft_logits') and hasattr(result2, 'soft_logits'):
        soft_diff = torch.abs(result1.soft_logits - result2.soft_logits).max().item()
        print(f"  Max soft logits difference: {soft_diff:.6f}")
        print(f"  Outputs are {'consistent' if soft_diff < 1e-4 else 'DIFFERENT!'}")
    else:
        # Compare hard tokens
        hard_diff = (result1.hard_token_ids != result2.hard_token_ids).sum().item()
        print(f"  Hard token differences: {hard_diff}")
        print(f"  Outputs are {'consistent' if hard_diff == 0 else 'DIFFERENT!'}")
    
    # Test 3: Memory scaling pattern
    print("\n3. Memory Scaling Pattern:")
    lengths = [10, 20, 40, 80]
    
    for method_name, method in [("Naive", "generate_soft"), ("KV Cache", "generate_soft_kv_cached")]:
        print(f"\n  {method_name}:")
        memories = []
        
        for length in lengths:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                result = getattr(decoder, method)(
                    activation_input=activation,
                    max_length=length,
                    gumbel_tau=1.0,
                    use_projection=True,
                    print_prompt=False
                )
            
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated()
                delta = (peak_mem - mem_before) / 1024**2
                memories.append(delta)
                print(f"    Length {length}: +{delta:.1f} MB")
            
            del result
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate scaling
        if len(memories) >= 2 and torch.cuda.is_available():
            # Check if doubling length roughly doubles memory (linear) or quadruples it (quadratic)
            ratio_1_2 = memories[1] / memories[0] if memories[0] > 0 else 0
            ratio_2_3 = memories[2] / memories[1] if memories[1] > 0 else 0
            ratio_3_4 = memories[3] / memories[2] if memories[2] > 0 else 0
            avg_ratio = (ratio_1_2 + ratio_2_3 + ratio_3_4) / 3
            print(f"    Average scaling factor when doubling length: {avg_ratio:.2f}")
            print(f"    Suggests O(n^{torch.log2(torch.tensor(avg_ratio)).item():.2f}) scaling")
    
    # Test 4: Verify KV cache accumulation
    print("\n4. KV Cache Accumulation Test:")
    print("  (Checking if KV cache properly accumulates across positions)")
    
    # Hook to capture KV cache sizes
    cache_sizes = []
    def hook_fn(module, input, output):
        if hasattr(module, '_kv_cache') and module._kv_cache is not None:
            if hasattr(module._kv_cache, 'keys') and len(module._kv_cache.keys) > 0:
                if module._kv_cache.keys[0] is not None:
                    cache_sizes.append(module._kv_cache.keys[0].shape[-2])  # sequence dimension
    
    # Register hooks on attention layers
    hooks = []
    for i, layer in enumerate(decoder.base.transformer.h):
        hook = layer.attn.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # Generate with KV cache
    with torch.no_grad():
        _ = decoder.generate_soft_kv_cached(
            activation_input=activation[:1],
            max_length=10,
            gumbel_tau=1.0,
            use_projection=True,
            print_prompt=False
        )
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    if cache_sizes:
        print(f"  Cache sizes observed: {cache_sizes[:5]}... (first 5)")
        print(f"  Cache is {'accumulating' if len(set(cache_sizes)) > 1 else 'NOT accumulating'}")
    else:
        print("  Could not observe cache sizes (hooks may need adjustment)")
    
    print("\n" + "="*60)
    print("Verification complete!")


if __name__ == "__main__":
    verify_kv_cache_behavior()