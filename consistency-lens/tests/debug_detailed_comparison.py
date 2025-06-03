#!/usr/bin/env python3
"""Detailed comparison of generate_soft vs generate_soft_kv_cached."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def compare_methods_step_by_step():
    """Compare the two methods step by step."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Step-by-Step Method Comparison")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    # Create two decoders
    decoder1 = Decoder(config).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    decoder2 = Decoder(config).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    decoder2.load_state_dict(decoder1.state_dict())
    
    d_model = decoder1.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Generate just 1 token and compare
    print("\nGenerating 1 token...")
    torch.manual_seed(123)
    gen1 = decoder1.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen2 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    print(f"\nFirst token:")
    t1 = gen1.hard_token_ids[0, 0].item()
    t2 = gen2.hard_token_ids[0, 0].item()
    print(f"  generate_soft: {t1} ('{tokenizer.decode([t1])}')")
    print(f"  generate_soft_kv_cached: {t2} ('{tokenizer.decode([t2])}')")
    
    # Compare logits
    logits1 = gen1.raw_lm_logits[0, 0]
    logits2 = gen2.raw_lm_logits[0, 0]
    logits_diff = (logits1 - logits2).abs().max().item()
    print(f"\nLogits max diff: {logits_diff:.2e}")
    
    # Show top 5 from each
    top1 = logits1.topk(5)
    top2 = logits2.topk(5)
    
    print("\nTop 5 logits from generate_soft:")
    for i, (val, idx) in enumerate(zip(top1.values, top1.indices)):
        print(f"  {i+1}. Token {idx.item()}: {val.item():.2f} ('{tokenizer.decode([idx.item()])}')")
    
    print("\nTop 5 logits from generate_soft_kv_cached:")
    for i, (val, idx) in enumerate(zip(top2.values, top2.indices)):
        print(f"  {i+1}. Token {idx.item()}: {val.item():.2f} ('{tokenizer.decode([idx.item()])}')")
    
    # Check the actual sequence being processed
    print("\n\nAnalyzing sequences:")
    
    # Build initial sequence
    parts = []
    if decoder1.prompt_left_emb is not None:
        parts.append(decoder1.prompt_left_emb.expand(1, -1, -1))
    a_proj = decoder1._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    if decoder1.prompt_right_emb is not None:
        parts.append(decoder1.prompt_right_emb.expand(1, -1, -1))
    seq_embs = torch.cat(parts, dim=1)
    
    print(f"Sequence shape: {seq_embs.shape}")
    print(f"Embed position (where activation is inserted): {decoder1.prompt_left_emb.size(0) if decoder1.prompt_left_emb is not None else 0}")
    
    # The key difference: in generate_soft, the activation is re-inserted at each layer
    # for every token generation. In KV cache, it's only inserted once during the initial pass.
    print("\nKey difference:")
    print("- generate_soft: Re-inserts activation at all layers for EVERY generated token")
    print("- generate_soft_kv_cached: Inserts activation only during initial pass, then caches")


if __name__ == "__main__":
    compare_methods_step_by_step()