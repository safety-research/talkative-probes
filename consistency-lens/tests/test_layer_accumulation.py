#!/usr/bin/env python3
"""Test how errors accumulate across layers."""

import torch
from transformers import GPT2Model
from lens.models.flash_kv_cache_v2 import compute_with_flash_kv_cache, FLASH_AVAILABLE
from lens.models.kv_cache import compute_with_kv_cache


def test_layer_accumulation():
    """Test error accumulation across layers."""
    
    if not FLASH_AVAILABLE:
        print("Flash Attention not available")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available")
        return
    
    # Load model
    model = GPT2Model.from_pretrained('gpt2').to(device)
    
    # Test different sequence lengths
    for seq_len in [4, 8, 16]:
        print(f"\nSequence length: {seq_len}")
        print("=" * 60)
        
        # Test input
        batch_size = 1
        hidden_size = 768
        
        torch.manual_seed(42)
        input_embeds = torch.randn(batch_size, seq_len, hidden_size, device=device)
        
        # Standard forward
        with torch.no_grad():
            output_std = model(inputs_embeds=input_embeds).last_hidden_state
        
        # KV cache forward
        with torch.no_grad():
            output_kv, _ = compute_with_kv_cache(model, input_embeds, use_cache=False)
        
        # Flash forward
        with torch.no_grad():
            output_flash, _ = compute_with_flash_kv_cache(model, input_embeds, use_cache=False)
        
        print(f"Final outputs:")
        print(f"  Standard norm: {output_std.norm():.3f}")
        print(f"  KV cache norm: {output_kv.norm():.3f}")
        print(f"  Flash norm: {output_flash.norm():.3f}")
        print(f"  KV vs Std diff: {(output_kv - output_std).abs().max():.6f}")
        print(f"  Flash vs Std diff: {(output_flash - output_std).abs().max():.6f}")
        
        # Layer-by-layer tracking
        print(f"\nLayer-by-layer error accumulation:")
        
        # Manual layer processing to track errors
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_embeds = model.wpe(position_ids)
        hidden = model.drop(input_embeds + position_embeds)
        
        hidden_flash = hidden.clone()
        
        for layer_idx in range(min(6, len(model.h))):  # First 6 layers
            layer = model.h[layer_idx]
            
            # Standard processing
            residual = hidden
            hidden = layer.ln_1(hidden)
            attn_output = layer.attn(hidden)[0]
            hidden = residual + attn_output
            residual = hidden
            hidden = layer.ln_2(hidden)
            mlp_output = layer.mlp(hidden)
            hidden = residual + mlp_output
            
            # Flash processing (inline)
            residual_flash = hidden_flash
            hidden_flash_ln = layer.ln_1(hidden_flash)
            
            # Flash attention
            qkv = layer.attn.c_attn(hidden_flash_ln)
            query, key, value = qkv.split(layer.attn.embed_dim, dim=2)
            
            num_heads = layer.attn.num_heads
            head_dim = layer.attn.head_dim
            
            query = query.view(batch_size, seq_len, num_heads, head_dim)
            key = key.view(batch_size, seq_len, num_heads, head_dim)
            value = value.view(batch_size, seq_len, num_heads, head_dim)
            
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
            
            hidden_flash = residual_flash + attn_output_flash
            residual_flash = hidden_flash
            hidden_flash = layer.ln_2(hidden_flash)
            mlp_output_flash = layer.mlp(hidden_flash)
            hidden_flash = residual_flash + mlp_output_flash
            
            # Track error
            error = (hidden - hidden_flash).abs().max().item()
            rel_error = error / hidden.norm().item() * 100
            print(f"  Layer {layer_idx}: abs_error={error:.6f}, rel_error={rel_error:.2f}%, norm_ratio={hidden_flash.norm()/hidden.norm():.3f}")


if __name__ == "__main__":
    test_layer_accumulation()