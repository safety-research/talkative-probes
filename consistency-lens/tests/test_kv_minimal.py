#!/usr/bin/env python3
"""Minimal test of KV cache with multi-layer patching."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_single_batch_single_token():
    """Test with single batch, single token."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Minimal KV Cache Test")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("<embed>", tokenizer)  # Minimal prompt
    
    d_model = decoder.base.config.hidden_size
    batch_size = 1
    seq_length = 1
    
    # Create activation
    torch.manual_seed(42)
    activation = torch.randn(batch_size, d_model, device=device)
    
    # Generate with both methods
    print("\nGenerating 1 token with batch size 1...")
    torch.manual_seed(123)
    gen1 = decoder.generate_soft(activation.clone(), max_length=seq_length, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen2 = decoder.generate_soft_kv_cached(activation.clone(), max_length=seq_length, gumbel_tau=1.0)
    
    # Compare
    token1 = gen1.hard_token_ids[0, 0].item()
    token2 = gen2.hard_token_ids[0, 0].item()
    logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
    
    print(f"\nResults:")
    print(f"  Token 1: {token1} ('{tokenizer.decode([token1])}')")
    print(f"  Token 2: {token2} ('{tokenizer.decode([token2])}')")
    print(f"  Logits diff: {logits_diff:.2e}")
    
    if token1 == token2 and logits_diff < 1e-4:
        print("  ✓ Single token generation matches!")
    else:
        print("  ✗ Even single token differs!")
        
        # Debug logits
        print("\n  Debugging logits:")
        logits1 = gen1.raw_lm_logits[0, 0]
        logits2 = gen2.raw_lm_logits[0, 0]
        
        # Top 5 from each
        top1 = logits1.topk(5)
        top2 = logits2.topk(5)
        
        print("  Top 5 tokens from generate_soft:")
        for val, idx in zip(top1.values, top1.indices):
            print(f"    {idx.item()}: {val.item():.2f} ('{tokenizer.decode([idx.item()])}')")
        
        print("  Top 5 tokens from generate_soft_kv_cached:")
        for val, idx in zip(top2.values, top2.indices):
            print(f"    {idx.item()}: {val.item():.2f} ('{tokenizer.decode([idx.item()])}')")


def test_without_multilayer():
    """Test KV cache without multi-layer patching as sanity check."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nSanity Check: KV Cache WITHOUT Multi-Layer Patching")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,  # No multi-layer patching
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("<embed>", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Generate
    torch.manual_seed(123)
    gen1 = decoder.generate_soft(activation.clone(), max_length=4, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen2 = decoder.generate_soft_kv_cached(activation.clone(), max_length=4, gumbel_tau=1.0)
    
    # Compare
    tokens_same = torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)
    logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
    
    print(f"  Tokens identical: {tokens_same}")
    print(f"  Logits diff: {logits_diff:.2e}")
    
    if tokens_same and logits_diff < 1e-4:
        print("  ✓ KV cache works without multi-layer patching")
    else:
        print("  ✗ KV cache broken even without multi-layer patching")


if __name__ == "__main__":
    test_single_batch_single_token()
    test_without_multilayer()