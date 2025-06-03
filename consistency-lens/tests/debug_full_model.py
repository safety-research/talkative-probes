#!/usr/bin/env python3
"""Debug full model Flash Attention integration."""

import torch
from transformers import GPT2Model
from lens.models.kv_cache import KVCache, compute_with_kv_cache
from lens.models.flash_kv_cache_v2 import FlashKVCache, compute_with_flash_kv_cache, FLASH_AVAILABLE


def test_full_model_forward():
    """Test full model forward pass with both KV cache implementations."""
    
    if not FLASH_AVAILABLE:
        print("Flash Attention not available")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available")
        return
    
    print("Full Model Forward Pass Comparison")
    print("=" * 60)
    
    # Load GPT-2
    model = GPT2Model.from_pretrained('gpt2').to(device)
    transformer = model
    
    # Test input
    batch_size = 1
    seq_len = 4
    hidden_size = 768
    
    torch.manual_seed(42)
    input_embeds = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Standard forward pass (reference)
    with torch.no_grad():
        outputs_std = model(inputs_embeds=input_embeds)
        hidden_states_std = outputs_std.last_hidden_state
    
    print(f"Standard forward:")
    print(f"  Output shape: {hidden_states_std.shape}")
    print(f"  Output norm: {hidden_states_std.norm():.3f}")
    print(f"  Output mean: {hidden_states_std.mean():.6f}")
    print(f"  Output std: {hidden_states_std.std():.3f}")
    
    # KV cache forward
    with torch.no_grad():
        hidden_states_kv, _ = compute_with_kv_cache(
            transformer,
            input_embeds,
            use_cache=False
        )
    
    print(f"\nKV cache forward:")
    print(f"  Output norm: {hidden_states_kv.norm():.3f}")
    print(f"  Max diff from standard: {(hidden_states_kv - hidden_states_std).abs().max():.6f}")
    
    # Flash attention forward
    with torch.no_grad():
        hidden_states_flash, _ = compute_with_flash_kv_cache(
            transformer,
            input_embeds,
            use_cache=False
        )
    
    print(f"\nFlash attention forward:")
    print(f"  Output norm: {hidden_states_flash.norm():.3f}")
    print(f"  Max diff from standard: {(hidden_states_flash - hidden_states_std).abs().max():.6f}")
    print(f"  Max diff from KV cache: {(hidden_states_flash - hidden_states_kv).abs().max():.6f}")
    
    # Layer-by-layer comparison
    print(f"\n\nLayer-by-layer analysis:")
    
    # Reset and do layer-by-layer
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_embeds = transformer.wpe(position_ids)
    hidden = transformer.drop(input_embeds + position_embeds)
    
    hidden_kv = hidden.clone()
    hidden_flash = hidden.clone()
    
    for layer_idx, layer in enumerate(transformer.h):
        # Standard attention
        residual = hidden
        hidden = layer.ln_1(hidden)
        attn_output, _ = layer.attn(hidden)
        hidden = residual + attn_output
        residual = hidden
        hidden = layer.ln_2(hidden)
        mlp_output = layer.mlp(hidden)
        hidden = residual + mlp_output
        
        # KV cache path (manually)
        residual_kv = hidden_kv
        hidden_kv_ln = layer.ln_1(hidden_kv)
        attn_output_kv, _ = layer.attn(hidden_kv_ln)
        hidden_kv = residual_kv + attn_output_kv
        residual_kv = hidden_kv
        hidden_kv = layer.ln_2(hidden_kv)
        mlp_output_kv = layer.mlp(hidden_kv)
        hidden_kv = residual_kv + mlp_output_kv
        
        # Check difference at this layer
        layer_diff = (hidden - hidden_kv).abs().max().item()
        
        if layer_idx < 3:  # Only check first few layers
            print(f"\n  Layer {layer_idx}:")
            print(f"    Standard vs KV diff: {layer_diff:.6f}")
            
            # Now test Flash at this specific layer
            # We'll use the flash_kv_cache_v2 logic for just this layer
            residual_flash = hidden_flash
            hidden_flash_ln = layer.ln_1(hidden_flash)
            
            # Compute Q, K, V
            qkv = layer.attn.c_attn(hidden_flash_ln)
            query, key, value = qkv.split(layer.attn.embed_dim, dim=2)
            
            # Reshape
            num_heads = layer.attn.num_heads
            head_dim = layer.attn.head_dim
            query = query.view(batch_size, seq_len, num_heads, head_dim)
            key = key.view(batch_size, seq_len, num_heads, head_dim)
            value = value.view(batch_size, seq_len, num_heads, head_dim)
            
            # Flash attention
            from lens.models.flash_kv_cache_v2 import flash_attn_func
            attn_output_flash = flash_attn_func(
                query.to(torch.bfloat16),
                key.to(torch.bfloat16),
                value.to(torch.bfloat16),
                dropout_p=0.0,
                causal=True,
            ).to(query.dtype)
            
            attn_output_flash = attn_output_flash.view(batch_size, seq_len, -1)
            attn_output_flash = layer.attn.c_proj(attn_output_flash)
            attn_output_flash = layer.attn.resid_dropout(attn_output_flash)
            
            # Compare attention outputs
            print(f"    Attention output diff (std vs flash): {(attn_output - attn_output_flash).abs().max():.6f}")
            print(f"    Attention output norms: std={attn_output.norm():.3f}, flash={attn_output_flash.norm():.3f}")
            
            # Complete the layer
            hidden_flash = residual_flash + attn_output_flash
            residual_flash = hidden_flash
            hidden_flash = layer.ln_2(hidden_flash)
            mlp_output_flash = layer.mlp(hidden_flash)
            hidden_flash = residual_flash + mlp_output_flash
            
            print(f"    Layer output diff (std vs flash): {(hidden - hidden_flash).abs().max():.6f}")
    
    # Final layer norm
    hidden = transformer.ln_f(hidden)
    hidden_flash = transformer.ln_f(hidden_flash)
    
    print(f"\n\nFinal comparison:")
    print(f"  Manual standard vs flash diff: {(hidden - hidden_flash).abs().max():.6f}")
    print(f"  Hidden norms: std={hidden.norm():.3f}, flash={hidden_flash.norm():.3f}")


if __name__ == "__main__":
    test_full_model_forward()