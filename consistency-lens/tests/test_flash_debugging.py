#!/usr/bin/env python3
"""Debug why Flash Attention produces different outputs."""

import torch
from transformers import AutoTokenizer, AutoModel
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.flash_kv_cache import FLASH_AVAILABLE, flash_attn_func


def test_flash_attention_precision():
    """Test Flash Attention with controlled inputs."""
    
    if not FLASH_AVAILABLE:
        print("Flash Attention not available")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Flash Attention Debugging")
    print("=" * 60)
    
    # Test 1: Simple attention computation
    print("\n1. Testing basic attention computation:")
    
    batch_size = 2
    seq_len = 4
    num_heads = 12
    head_dim = 64
    
    # Create simple test tensors in different dtypes
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print(f"\n  Testing dtype: {dtype}")
        
        # Create Q, K, V
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        
        # Normalize to prevent overflow
        q = q / (head_dim ** 0.5)
        
        try:
            # Flash attention
            if dtype == torch.float32:
                # Convert to bf16 for Flash
                q_flash = q.to(torch.bfloat16)
                k_flash = k.to(torch.bfloat16)
                v_flash = v.to(torch.bfloat16)
                
                flash_out = flash_attn_func(
                    q_flash, k_flash, v_flash,
                    dropout_p=0.0,
                    causal=True,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=False,
                ).to(dtype)
            else:
                flash_out = flash_attn_func(
                    q, k, v,
                    dropout_p=0.0,
                    causal=True,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=False,
                )
            
            # Standard attention (for comparison)
            # (batch, heads, seq, head_dim) format for standard attention
            q_std = q.permute(0, 2, 1, 3)
            k_std = k.permute(0, 2, 1, 3)
            v_std = v.permute(0, 2, 1, 3)
            
            # Compute attention scores
            scores = torch.matmul(q_std, k_std.transpose(-2, -1))
            
            # Apply causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
            
            # Softmax
            attn_weights = torch.softmax(scores, dim=-1)
            
            # Apply to values
            std_out = torch.matmul(attn_weights, v_std)
            std_out = std_out.permute(0, 2, 1, 3)  # Back to (batch, seq, heads, head_dim)
            
            # Compare
            diff = (flash_out - std_out).abs().max().item()
            print(f"    Max difference: {diff:.6f}")
            print(f"    Flash output range: [{flash_out.min():.3f}, {flash_out.max():.3f}]")
            print(f"    Standard output range: [{std_out.min():.3f}, {std_out.max():.3f}]")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    # Test 2: GPT-2 model comparison
    print("\n\n2. Testing with actual GPT-2 model:")
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create two decoders
    decoder_kv = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        use_flash_attention=False,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    decoder_flash = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=False,
        use_flash_attention=True,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    # Ensure same weights
    decoder_flash.load_state_dict(decoder_kv.state_dict())
    
    # Set same prompt
    decoder_kv.set_prompt("explain <embed>:", tokenizer)
    decoder_flash.set_prompt("explain <embed>:", tokenizer)
    
    # Test with small input
    batch_size = 1
    d_model = 768
    seq_length = 8
    
    # Create activation
    torch.manual_seed(42)
    activation = torch.randn(batch_size, d_model, device=device)
    
    # Generate with both methods (tau=0 for deterministic)
    with torch.no_grad():
        gen_kv = decoder_kv.generate_soft_kv_cached(
            activation.clone(), max_length=seq_length, gumbel_tau=0.0
        )
        
        gen_flash = decoder_flash.generate_soft_kv_flash(
            activation.clone(), max_length=seq_length, gumbel_tau=0.0
        )
    
    # Compare outputs
    print(f"\n  Comparing generation outputs:")
    print(f"  Token IDs match: {torch.allclose(gen_flash.hard_token_ids, gen_kv.hard_token_ids)}")
    
    # Check logits
    logit_diff = (gen_flash.raw_lm_logits - gen_kv.raw_lm_logits).abs()
    print(f"  Max logit difference: {logit_diff.max().item():.6f}")
    print(f"  Mean logit difference: {logit_diff.mean().item():.6f}")
    
    # Check which positions have large differences
    large_diffs = logit_diff.max(dim=-1)[0] > 1.0  # Positions with >1.0 max diff
    if large_diffs.any():
        print(f"  Positions with large differences: {large_diffs.nonzero(as_tuple=True)}")
        
        # Check first position with large difference
        pos_idx = large_diffs.nonzero(as_tuple=True)[1][0].item()
        print(f"\n  Analyzing position {pos_idx}:")
        
        # Get top-5 predictions from each method
        kv_top5 = gen_kv.raw_lm_logits[0, pos_idx].topk(5)
        flash_top5 = gen_flash.raw_lm_logits[0, pos_idx].topk(5)
        
        print(f"    KV Cache top-5 logits: {kv_top5.values.tolist()}")
        print(f"    Flash top-5 logits: {flash_top5.values.tolist()}")
        
        # Decode tokens
        kv_tokens = tokenizer.convert_ids_to_tokens(kv_top5.indices.tolist())
        flash_tokens = tokenizer.convert_ids_to_tokens(flash_top5.indices.tolist())
        
        print(f"    KV Cache top-5 tokens: {kv_tokens}")
        print(f"    Flash top-5 tokens: {flash_tokens}")
    
    # Check embeddings
    emb_diff = (gen_flash.generated_text_embeddings - gen_kv.generated_text_embeddings).abs()
    print(f"\n  Max embedding difference: {emb_diff.max().item():.6f}")
    print(f"  Mean embedding difference: {emb_diff.mean().item():.6f}")
    
    # Test if it's a dtype issue
    print(f"\n  Checking dtypes:")
    print(f"  Decoder dtype: {decoder_kv.proj.weight.dtype}")
    print(f"  KV output dtype: {gen_kv.raw_lm_logits.dtype}")
    print(f"  Flash output dtype: {gen_flash.raw_lm_logits.dtype}")


if __name__ == "__main__":
    test_flash_attention_precision()