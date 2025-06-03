#!/usr/bin/env python3
"""Minimal test to isolate decoder Flash issue."""

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from lens.models.kv_cache import compute_with_kv_cache
from lens.models.flash_kv_cache_v2 import compute_with_flash_kv_cache, FLASH_AVAILABLE


def test_decoder_minimal():
    """Test decoder generation with minimal setup."""
    
    if not FLASH_AVAILABLE:
        print("Flash Attention not available")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available")
        return
    
    print("Minimal Decoder Test")
    print("=" * 60)
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Simple prompt
    prompt = "Hello world"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_embeds = model.transformer.wte(inputs.input_ids)
    
    print(f"Prompt: '{prompt}'")
    print(f"Input shape: {input_embeds.shape}")
    
    # Test 1: Standard forward
    with torch.no_grad():
        outputs_std = model(inputs_embeds=input_embeds)
        logits_std = outputs_std.logits[:, -1, :]
    
    print(f"\nStandard forward:")
    print(f"  Logits shape: {logits_std.shape}")
    print(f"  Logits norm: {logits_std.norm():.3f}")
    print(f"  Top 5 logits: {logits_std[0].topk(5).values.tolist()}")
    
    # Test 2: KV cache forward
    with torch.no_grad():
        hidden_kv, _ = compute_with_kv_cache(
            model.transformer,
            input_embeds,
            use_cache=False
        )
        logits_kv = model.lm_head(hidden_kv[:, -1, :])
    
    print(f"\nKV cache forward:")
    print(f"  Logits norm: {logits_kv.norm():.3f}")
    print(f"  Top 5 logits: {logits_kv[0].topk(5).values.tolist()}")
    print(f"  Max diff from standard: {(logits_kv - logits_std).abs().max():.6f}")
    
    # Test 3: Flash forward
    with torch.no_grad():
        hidden_flash, _ = compute_with_flash_kv_cache(
            model.transformer,
            input_embeds,
            use_cache=False
        )
        logits_flash = model.lm_head(hidden_flash[:, -1, :])
    
    print(f"\nFlash forward:")
    print(f"  Logits norm: {logits_flash.norm():.3f}")
    print(f"  Top 5 logits: {logits_flash[0].topk(5).values.tolist()}")
    print(f"  Max diff from standard: {(logits_flash - logits_std).abs().max():.6f}")
    
    # Test the actual issue: incremental generation
    print(f"\n\nIncremental generation test:")
    
    # Process prompt with caching
    from lens.models.kv_cache import KVCache
    from lens.models.flash_kv_cache_v2 import FlashKVCache
    
    # KV cache incremental
    print("\nKV Cache incremental:")
    with torch.no_grad():
        # Initial prompt
        hidden_kv, kv_cache = compute_with_kv_cache(
            model.transformer,
            input_embeds,
            use_cache=True
        )
        logits_kv_1 = model.lm_head(hidden_kv[:, -1:, :])
        print(f"  Step 1 logits norm: {logits_kv_1.norm():.3f}")
        
        # Generate one token
        next_token = logits_kv_1.argmax(dim=-1)
        next_embed = model.transformer.wte(next_token)
        
        # Process next token
        hidden_kv_2, kv_cache = compute_with_kv_cache(
            model.transformer,
            next_embed,
            kv_cache=kv_cache,
            position_offset=input_embeds.size(1),
            use_cache=True
        )
        logits_kv_2 = model.lm_head(hidden_kv_2)
        print(f"  Step 2 logits norm: {logits_kv_2.norm():.3f}")
    
    # Flash incremental
    print("\nFlash incremental:")
    with torch.no_grad():
        # Initial prompt
        hidden_flash, flash_cache = compute_with_flash_kv_cache(
            model.transformer,
            input_embeds,
            use_cache=True
        )
        logits_flash_1 = model.lm_head(hidden_flash[:, -1:, :])
        print(f"  Step 1 logits norm: {logits_flash_1.norm():.3f}")
        print(f"  Step 1 diff from KV: {(logits_flash_1 - logits_kv_1).abs().max():.6f}")
        
        # Generate one token (same as KV)
        next_embed = model.transformer.wte(next_token)
        
        # Process next token
        hidden_flash_2, flash_cache = compute_with_flash_kv_cache(
            model.transformer,
            next_embed,
            kv_cache=flash_cache,
            position_offset=input_embeds.size(1),
            use_cache=True
        )
        logits_flash_2 = model.lm_head(hidden_flash_2)
        print(f"  Step 2 logits norm: {logits_flash_2.norm():.3f}")
        print(f"  Step 2 diff from KV: {(logits_flash_2 - logits_kv_2).abs().max():.6f}")
        
        # Check cache contents
        print(f"\nCache analysis:")
        print(f"  KV cache layers: {len(kv_cache)}")
        print(f"  Flash cache layers: {len(flash_cache)}")
        if len(kv_cache) > 0 and len(flash_cache) > 0:
            print(f"  KV cache seq len: {kv_cache.get_seq_length(0)}")
            print(f"  Flash cache seq len: {flash_cache.get_seq_length(0)}")


if __name__ == "__main__":
    test_decoder_minimal()