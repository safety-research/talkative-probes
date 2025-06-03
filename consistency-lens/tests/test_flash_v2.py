#!/usr/bin/env python3
"""Test Flash Attention v2 implementation for correct scaling."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.flash_kv_cache_v2 import FLASH_AVAILABLE


def test_flash_v2_correctness():
    """Test that Flash Attention v2 produces correct outputs."""
    
    if not FLASH_AVAILABLE:
        print("Flash Attention not available, skipping test")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping test")
        return
    
    print("Testing Flash Attention v2 Implementation")
    print("=" * 60)
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create two decoders with same weights
    print("Creating decoders...")
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
    
    # Test configurations
    test_configs = [
        (1, 8),
        (2, 16),
        (1, 32),
    ]
    
    for batch_size, seq_length in test_configs:
        print(f"\nTesting batch_size={batch_size}, seq_length={seq_length}")
        
        # Create test activation
        torch.manual_seed(42)
        activation = torch.randn(batch_size, 768, device=device)
        
        # Generate with both methods
        with torch.no_grad():
            gen_kv = decoder_kv.generate_soft_kv_cached(
                activation.clone(), max_length=seq_length, gumbel_tau=0.0
            )
            
            gen_flash = decoder_flash.generate_soft_kv_flash(
                activation.clone(), max_length=seq_length, gumbel_tau=0.0
            )
        
        # Compare outputs
        print(f"  Token IDs match: {torch.allclose(gen_flash.hard_token_ids, gen_kv.hard_token_ids)}")
        
        # Check logits
        logit_diff = (gen_flash.raw_lm_logits - gen_kv.raw_lm_logits).abs()
        max_diff = logit_diff.max().item()
        mean_diff = logit_diff.mean().item()
        
        print(f"  Max logit difference: {max_diff:.6f}")
        print(f"  Mean logit difference: {mean_diff:.6f}")
        
        # Check if differences are acceptable (should be < 0.1 for functional equivalence)
        if max_diff < 0.1:
            print(f"  ✓ Flash Attention v2 is functionally equivalent")
        else:
            print(f"  ✗ Large differences detected")
            
            # Debug first position with large difference
            large_diffs = logit_diff.max(dim=-1)[0] > 0.1
            if large_diffs.any():
                pos_idx = large_diffs.nonzero(as_tuple=True)[1][0].item()
                print(f"\n  Analyzing position {pos_idx}:")
                
                # Get top-5 predictions
                kv_logits = gen_kv.raw_lm_logits[0, pos_idx]
                flash_logits = gen_flash.raw_lm_logits[0, pos_idx]
                
                kv_top5 = kv_logits.topk(5)
                flash_top5 = flash_logits.topk(5)
                
                print(f"    KV Cache top-5 logits: {[f'{v:.3f}' for v in kv_top5.values.tolist()]}")
                print(f"    Flash top-5 logits: {[f'{v:.3f}' for v in flash_top5.values.tolist()]}")
                
                # Check if the relative order is preserved
                kv_indices = kv_top5.indices.tolist()
                flash_indices = flash_top5.indices.tolist()
                if kv_indices == flash_indices:
                    print(f"    Token order preserved despite magnitude difference")
        
        # Check embeddings
        emb_diff = (gen_flash.generated_text_embeddings - gen_kv.generated_text_embeddings).abs()
        print(f"\n  Max embedding difference: {emb_diff.max().item():.6f}")
        print(f"  Mean embedding difference: {emb_diff.mean().item():.6f}")
    
    print("\n" + "=" * 60)
    print("Testing gradient flow...")
    
    # Test gradient flow with small example
    torch.manual_seed(42)
    activation = torch.randn(1, 768, device=device, requires_grad=True)
    
    # KV cache
    gen_kv = decoder_kv.generate_soft_kv_cached(
        activation.clone(), max_length=8, gumbel_tau=1.0
    )
    loss_kv = gen_kv.raw_lm_logits.sum()
    loss_kv.backward()
    grad_kv = activation.grad.clone()
    
    # Reset grad
    activation.grad = None
    
    # Flash
    gen_flash = decoder_flash.generate_soft_kv_flash(
        activation.clone(), max_length=8, gumbel_tau=1.0
    )
    loss_flash = gen_flash.raw_lm_logits.sum()
    loss_flash.backward()
    grad_flash = activation.grad.clone()
    
    # Compare gradients
    grad_diff = (grad_flash - grad_kv).abs()
    relative_diff = grad_diff.mean() / grad_kv.abs().mean() * 100
    
    print(f"  Gradient relative difference: {relative_diff:.1f}%")
    if relative_diff < 10:
        print(f"  ✓ Gradients are consistent")
    else:
        print(f"  ✗ Large gradient differences")


if __name__ == "__main__":
    test_flash_v2_correctness()