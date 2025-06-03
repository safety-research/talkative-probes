#!/usr/bin/env python3
"""Test with temperature to see if it's just numerical precision."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.flash_kv_cache_v2 import FLASH_AVAILABLE


def test_with_temperature():
    """Test with different temperatures."""
    
    if not FLASH_AVAILABLE:
        print("Flash Attention not available")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available")
        return
    
    print("Temperature Test")
    print("=" * 60)
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create decoders
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
    decoder_flash.load_state_dict(decoder_kv.state_dict(), strict=False)
    
    # Set same prompt
    decoder_kv.set_prompt("explain <embed>:", tokenizer)
    decoder_flash.set_prompt("explain <embed>:", tokenizer)
    
    # Test activation
    torch.manual_seed(42)
    activation = torch.randn(1, 768, device=device)
    
    # Test different temperatures
    for tau in [0.0, 0.5, 1.0]:
        print(f"\n\nTesting with tau={tau}")
        print("-" * 40)
        
        # Set same random seed for both
        torch.manual_seed(123)
        with torch.no_grad():
            gen_kv = decoder_kv.generate_soft_kv_cached(
                activation.clone(), max_length=8, gumbel_tau=tau
            )
        
        torch.manual_seed(123)
        with torch.no_grad():
            gen_flash = decoder_flash.generate_soft_kv_flash(
                activation.clone(), max_length=8, gumbel_tau=tau
            )
        
        # Check first position in detail
        print(f"\nFirst position analysis:")
        logits_kv = gen_kv.raw_lm_logits[0, 0]
        logits_flash = gen_flash.raw_lm_logits[0, 0]
        
        diff = (logits_flash - logits_kv).abs()
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")
        
        # Get top 10 tokens for each
        kv_top10 = logits_kv.topk(10)
        flash_top10 = logits_flash.topk(10)
        
        print(f"\n  KV top-10 tokens:")
        for i in range(10):
            token_id = kv_top10.indices[i].item()
            value = kv_top10.values[i].item()
            print(f"    {i+1}. {token_id:5d} ({tokenizer.decode(token_id):>10s}): {value:.3f}")
        
        print(f"\n  Flash top-10 tokens:")
        for i in range(10):
            token_id = flash_top10.indices[i].item()
            value = flash_top10.values[i].item()
            print(f"    {i+1}. {token_id:5d} ({tokenizer.decode(token_id):>10s}): {value:.3f}")
        
        # Check if top tokens match
        kv_top_ids = set(kv_top10.indices.tolist())
        flash_top_ids = set(flash_top10.indices.tolist())
        overlap = len(kv_top_ids & flash_top_ids)
        print(f"\n  Top-10 overlap: {overlap}/10")
        
        # Compare generated sequences
        kv_tokens = gen_kv.hard_token_ids[0].tolist()
        flash_tokens = gen_flash.hard_token_ids[0].tolist()
        
        print(f"\n  Generated sequences:")
        print(f"    KV:    '{tokenizer.decode(kv_tokens)}'")
        print(f"    Flash: '{tokenizer.decode(flash_tokens)}'")
        
        # Check how many tokens match
        matches = sum(1 for i in range(len(kv_tokens)) if kv_tokens[i] == flash_tokens[i])
        print(f"    Matching tokens: {matches}/{len(kv_tokens)}")


if __name__ == "__main__":
    test_with_temperature()