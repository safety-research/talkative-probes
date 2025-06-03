#!/usr/bin/env python3
"""Debug Flash Attention scaling issue."""

import torch
from transformers import GPT2Model
from lens.models.flash_kv_cache_v2 import flash_attn_func, FLASH_AVAILABLE


def test_direct_attention():
    """Test Flash Attention vs standard attention directly."""
    
    if not FLASH_AVAILABLE:
        print("Flash Attention not available")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available")
        return
    
    print("Direct Attention Comparison")
    print("=" * 60)
    
    # Load GPT-2 to get actual parameters
    model = GPT2Model.from_pretrained('gpt2').to(device)
    layer = model.h[0]
    
    # Test input
    batch_size = 1
    seq_len = 4
    hidden_size = 768
    num_heads = 12
    head_dim = 64
    
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Layer norm
    ln_output = layer.ln_1(hidden_states)
    
    # Compute Q, K, V
    qkv = layer.attn.c_attn(ln_output)
    query, key, value = qkv.split(layer.attn.embed_dim, dim=2)
    
    print(f"\nBefore reshaping:")
    print(f"  Query shape: {query.shape}, norm: {query.norm():.3f}")
    
    # Reshape for attention
    query = query.view(batch_size, seq_len, num_heads, head_dim)
    key = key.view(batch_size, seq_len, num_heads, head_dim)
    value = value.view(batch_size, seq_len, num_heads, head_dim)
    
    print(f"\nAfter reshaping:")
    print(f"  Query shape: {query.shape}, norm: {query.norm():.3f}")
    
    # Standard GPT-2 attention
    query_std = query.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
    key_std = key.permute(0, 2, 1, 3)
    value_std = value.permute(0, 2, 1, 3)
    
    # Scale is applied in the attention computation
    scale = head_dim ** -0.5
    scores = torch.matmul(query_std, key_std.transpose(-2, -1)) * scale
    
    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    attn_output_std = torch.matmul(attn_weights, value_std)
    attn_output_std = attn_output_std.permute(0, 2, 1, 3).contiguous()
    attn_output_std = attn_output_std.view(batch_size, seq_len, hidden_size)
    
    print(f"\nStandard attention:")
    print(f"  Scores (before softmax) range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Output norm: {attn_output_std.norm():.3f}")
    
    # Flash Attention (multiple attempts)
    print(f"\n\nFlash Attention tests:")
    
    # Test 1: Direct Flash (no scaling)
    attn_output_flash1 = flash_attn_func(
        query.to(torch.bfloat16),
        key.to(torch.bfloat16),
        value.to(torch.bfloat16),
        dropout_p=0.0,
        causal=True,
    ).to(query.dtype)
    attn_output_flash1 = attn_output_flash1.view(batch_size, seq_len, hidden_size)
    
    diff1 = (attn_output_flash1 - attn_output_std).abs().max().item()
    print(f"\n  Test 1 (no scaling): max diff = {diff1:.3f}, output norm = {attn_output_flash1.norm():.3f}")
    
    # Test 2: Pre-scale query
    query_scaled = query * scale
    attn_output_flash2 = flash_attn_func(
        query_scaled.to(torch.bfloat16),
        key.to(torch.bfloat16),
        value.to(torch.bfloat16),
        dropout_p=0.0,
        causal=True,
    ).to(query.dtype)
    attn_output_flash2 = attn_output_flash2.view(batch_size, seq_len, hidden_size)
    
    diff2 = (attn_output_flash2 - attn_output_std).abs().max().item()
    print(f"\n  Test 2 (pre-scale query): max diff = {diff2:.3f}, output norm = {attn_output_flash2.norm():.3f}")
    
    # Test 3: Post-scale output
    attn_output_flash3 = flash_attn_func(
        query.to(torch.bfloat16),
        key.to(torch.bfloat16),
        value.to(torch.bfloat16),
        dropout_p=0.0,
        causal=True,
    ).to(query.dtype)
    attn_output_flash3 = attn_output_flash3 * scale
    attn_output_flash3 = attn_output_flash3.view(batch_size, seq_len, hidden_size)
    
    diff3 = (attn_output_flash3 - attn_output_std).abs().max().item()
    print(f"\n  Test 3 (post-scale output): max diff = {diff3:.3f}, output norm = {attn_output_flash3.norm():.3f}")
    
    # Apply output projection to see final effect
    print(f"\n\nAfter output projection:")
    final_std = layer.attn.c_proj(attn_output_std)
    final_flash1 = layer.attn.c_proj(attn_output_flash1)
    
    print(f"  Standard: norm = {final_std.norm():.3f}")
    print(f"  Flash (no scale): norm = {final_flash1.norm():.3f}")
    print(f"  Ratio: {final_flash1.norm() / final_std.norm():.3f}")


if __name__ == "__main__":
    test_direct_attention()