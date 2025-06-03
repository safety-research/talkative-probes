#!/usr/bin/env python3
"""Debug position handling in KV cache with multi-layer patching."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_positions():
    """Debug how positions are handled in both methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Debugging Position Handling")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Trace what happens in generate_soft
    print("\nPrompt analysis:")
    print(f"  Prompt: '{decoder.prompt_text}'")
    print(f"  Left prompt: {decoder.prompt_left_emb.shape if decoder.prompt_left_emb is not None else None}")
    print(f"  Right prompt: {decoder.prompt_right_emb.shape if decoder.prompt_right_emb is not None else None}")
    
    # Check the sequence construction
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
        print(f"  Left tokens: {decoder.prompt_left_emb.size(0)}")
    
    a_proj = decoder._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    print(f"  Activation position: {parts[0].size(1) if len(parts) > 1 else 0}")
    
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
        print(f"  Right tokens: {decoder.prompt_right_emb.size(0)}")
    
    seq_embs = torch.cat(parts, dim=1)
    print(f"  Total sequence length: {seq_embs.size(1)}")
    
    # Generate one token with both methods
    print("\nGenerating 1 token...")
    torch.manual_seed(123)
    gen1 = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen2 = decoder.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    # Check generated tokens
    print(f"\nGenerated tokens:")
    print(f"  generate_soft: {gen1.hard_token_ids[0, 0].item()} ('{tokenizer.decode([gen1.hard_token_ids[0, 0].item()])}')")
    print(f"  generate_soft_kv_cached: {gen2.hard_token_ids[0, 0].item()} ('{tokenizer.decode([gen2.hard_token_ids[0, 0].item()])}')")
    
    # Check the logits distribution
    print(f"\nLogits analysis:")
    logits1 = gen1.raw_lm_logits[0, 0]
    logits2 = gen2.raw_lm_logits[0, 0]
    
    # Top 5 from each
    top1 = logits1.topk(5)
    top2 = logits2.topk(5)
    
    print("  Top 5 from generate_soft:")
    for val, idx in zip(top1.values, top1.indices):
        print(f"    {idx.item()}: {val.item():.2f} ('{tokenizer.decode([idx.item()])}')")
    
    print("  Top 5 from generate_soft_kv_cached:")
    for val, idx in zip(top2.values, top2.indices):
        print(f"    {idx.item()}: {val.item():.2f} ('{tokenizer.decode([idx.item()])}')")
    
    # Check if position embeddings are the issue
    print("\nChecking position embeddings...")
    transformer = decoder.base.transformer
    seq_length = seq_embs.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    position_embeds = transformer.wpe(position_ids[0])
    print(f"  Position IDs: {position_ids}")
    print(f"  Position embeddings shape: {position_embeds.shape}")


if __name__ == "__main__":
    debug_positions()