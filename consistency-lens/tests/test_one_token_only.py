#!/usr/bin/env python3
"""Test just one token generation to isolate the issue."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_one_token():
    """Test generating just one token."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Testing One Token Generation")
    print("=" * 60)
    
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
    
    # Generate 1 token with both methods
    print("\nGenerating 1 token...")
    
    torch.manual_seed(123)
    gen1 = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen2 = decoder.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    print(f"\nResults:")
    print(f"  generate_soft token: {gen1.hard_token_ids[0, 0].item()} ('{tokenizer.decode([gen1.hard_token_ids[0, 0].item()])}')")
    print(f"  KV cached token: {gen2.hard_token_ids[0, 0].item()} ('{tokenizer.decode([gen2.hard_token_ids[0, 0].item()])}')")
    
    print(f"\nLogits comparison:")
    logits1 = gen1.raw_lm_logits[0, 0]
    logits2 = gen2.raw_lm_logits[0, 0]
    
    print(f"  generate_soft range: [{logits1.min().item():.2f}, {logits1.max().item():.2f}]")
    print(f"  KV cached range: [{logits2.min().item():.2f}, {logits2.max().item():.2f}]")
    print(f"  Difference: {(logits1 - logits2).abs().max().item():.2f}")
    
    # Check top 5
    top1 = logits1.topk(5)
    top2 = logits2.topk(5)
    
    print(f"\n  Top 5 from generate_soft:")
    for i, (val, idx) in enumerate(zip(top1.values, top1.indices)):
        print(f"    {i+1}. {idx.item()}: {val.item():.2f} ('{tokenizer.decode([idx.item()])}')")
    
    print(f"\n  Top 5 from KV cached:")
    for i, (val, idx) in enumerate(zip(top2.values, top2.indices)):
        print(f"    {i+1}. {idx.item()}: {val.item():.2f} ('{tokenizer.decode([idx.item()])}')")
    
    # Now test without multi-layer patching
    print("\n\n" + "="*60)
    print("Testing WITHOUT multi-layer patching:")
    
    config2 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,  # Disabled
        per_layer_projections=False,
    )
    
    decoder2 = Decoder(config2).to(device).eval()
    decoder2.set_prompt("explain <embed>:", tokenizer)
    decoder2.load_state_dict(decoder.state_dict(), strict=False)
    
    torch.manual_seed(123)
    gen3 = decoder2.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen4 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    print(f"\nWithout multi-layer patching:")
    print(f"  generate_soft token: {gen3.hard_token_ids[0, 0].item()}")
    print(f"  KV cached token: {gen4.hard_token_ids[0, 0].item()}")
    print(f"  Match: {gen3.hard_token_ids[0, 0].item() == gen4.hard_token_ids[0, 0].item()}")
    print(f" top 5 logits: {gen3.raw_lm_logits[0, 0].topk(5)}")
    print(f" top 5 logits: {gen4.raw_lm_logits[0, 0].topk(5)}")
    print(f"  Logits diff: {(gen3.raw_lm_logits - gen4.raw_lm_logits).abs().max().item():.2e}")


if __name__ == "__main__":
    test_one_token()