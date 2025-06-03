#!/usr/bin/env python3
"""Simple test for multi-token generation."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_simple():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Testing Multi-Token Generation")
    print("=" * 60)
    
    # Test with multi-layer patching
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device).eval()
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Generate 5 tokens with both methods
    print("\nGenerating 5 tokens...")
    
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    gen1 = decoder.generate_soft(activation.clone(), max_length=5, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen2 = decoder.generate_soft_kv_cached(activation.clone(), max_length=5, gumbel_tau=1.0)
    
    # Compare results
    tokens1 = gen1.hard_token_ids[0].tolist()
    tokens2 = gen2.hard_token_ids[0].tolist()
    
    text1 = tokenizer.decode(tokens1)
    text2 = tokenizer.decode(tokens2)
    
    print(f"\ngenerate_soft:      {tokens1}")
    print(f"                    '{text1}'")
    print(f"\ngenerate_kv_cached: {tokens2}")
    print(f"                    '{text2}'")
    
    print(f"\nTokens match: {tokens1 == tokens2}")
    
    # Check logits
    logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
    print(f"Max logits diff: {logits_diff:.2e}")
    
    # Show where they differ
    if tokens1 != tokens2:
        print("\nDifferences:")
        for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
            if t1 != t2:
                print(f"  Position {i}: {t1} ('{tokenizer.decode([t1])}') vs {t2} ('{tokenizer.decode([t2])}')")


if __name__ == "__main__":
    test_simple()