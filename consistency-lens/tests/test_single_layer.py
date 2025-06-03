#!/usr/bin/env python3
"""Test single layer to isolate the issue."""

import torch
from transformers import GPT2Model
from lens.models.flash_kv_cache_v2 import flash_attn_func, FLASH_AVAILABLE


def test_single_layer():
    """Test a single GPT2 layer with different implementations."""
    
    if not FLASH_AVAILABLE:
        print("Flash Attention not available")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available")
        return
    
    # Load model
    model = GPT2Model.from_pretrained('gpt2').to(device)
    layer = model.h[0]
    
    # Test input
    batch_size = 1
    seq_len = 4
    hidden_size = 768
    
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    print("Single Layer Test")
    print("=" * 60)
    print(f"Input norm: {hidden_states.norm():.3f}")
    
    # Method 1: Standard layer forward
    output_std = layer(hidden_states)[0]
    print(f"\nStandard layer forward:")
    print(f"  Output norm: {output_std.norm():.3f}")
    
    # Method 2: Manual implementation (matching what flash_kv_cache_v2 does)
    residual = hidden_states
    hidden_ln = layer.ln_1(hidden_states)
    
    # Standard attention
    attn_output_std = layer.attn(hidden_ln)[0]
    hidden_after_attn = residual + attn_output_std
    
    # MLP
    residual = hidden_after_attn
    hidden_ln2 = layer.ln_2(hidden_after_attn)
    mlp_output = layer.mlp(hidden_ln2)
    output_manual = residual + mlp_output
    
    print(f"\nManual implementation:")
    print(f"  Output norm: {output_manual.norm():.3f}")
    print(f"  Diff from standard: {(output_manual - output_std).abs().max():.6f}")
    
    # Method 3: Manual with Flash Attention
    residual = hidden_states
    hidden_ln = layer.ln_1(hidden_states)
    
    # Flash attention
    qkv = layer.attn.c_attn(hidden_ln)
    query, key, value = qkv.split(layer.attn.embed_dim, dim=2)
    
    num_heads = layer.attn.num_heads
    head_dim = layer.attn.head_dim
    
    query = query.view(batch_size, seq_len, num_heads, head_dim)
    key = key.view(batch_size, seq_len, num_heads, head_dim)
    value = value.view(batch_size, seq_len, num_heads, head_dim)
    
    # Flash attention
    attn_output_flash = flash_attn_func(
        query.to(torch.bfloat16),
        key.to(torch.bfloat16),
        value.to(torch.bfloat16),
        dropout_p=0.0,
        causal=True,
    ).to(query.dtype)
    
    attn_output_flash = attn_output_flash.view(batch_size, seq_len, hidden_size)
    attn_output_flash = layer.attn.c_proj(attn_output_flash)
    attn_output_flash = layer.attn.resid_dropout(attn_output_flash)
    
    hidden_after_attn_flash = residual + attn_output_flash
    
    # MLP (same as before)
    residual = hidden_after_attn_flash
    hidden_ln2 = layer.ln_2(hidden_after_attn_flash)
    mlp_output = layer.mlp(hidden_ln2)
    output_flash = residual + mlp_output
    
    print(f"\nFlash attention implementation:")
    print(f"  Output norm: {output_flash.norm():.3f}")
    print(f"  Diff from standard: {(output_flash - output_std).abs().max():.6f}")
    
    # Debug intermediate values
    print(f"\nIntermediate values:")
    print(f"  Attention output norm (std): {attn_output_std.norm():.3f}")
    print(f"  Attention output norm (flash): {attn_output_flash.norm():.3f}")
    print(f"  Attention diff: {(attn_output_flash - attn_output_std).abs().max():.6f}")


if __name__ == "__main__":
    test_single_layer()