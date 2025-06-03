#!/usr/bin/env python3
"""Debug embed position calculation in both methods."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def trace_embed_positions():
    """Trace embed position calculation in both methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Tracing Embed Position Calculation")
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
    
    print("\nPrompt analysis:")
    print(f"  Prompt text: '{decoder.prompt_text}'")
    print(f"  Prompt left emb shape: {decoder.prompt_left_emb.shape if decoder.prompt_left_emb is not None else None}")
    print(f"  Prompt right emb shape: {decoder.prompt_right_emb.shape if decoder.prompt_right_emb is not None else None}")
    
    # Build sequence as generate_soft does
    print("\n1. generate_soft sequence building:")
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
        print(f"  Added left prompt: shape {parts[-1].shape}")
    
    # Always insert activation
    a_proj = decoder._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    print(f"  Added activation: shape {parts[-1].shape}")
    
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
        print(f"  Added right prompt: shape {parts[-1].shape}")
    
    seq_embs = torch.cat(parts, dim=1)
    print(f"  Final sequence shape: {seq_embs.shape}")
    
    # Calculate embed position
    embed_pos = decoder.prompt_left_emb.size(0) if decoder.prompt_left_emb is not None else 0
    print(f"  Embed position: {embed_pos}")
    
    # Verify by checking norms at each position
    print("\n  Position norms:")
    for i in range(seq_embs.size(1)):
        norm = seq_embs[0, i].norm().item()
        print(f"    Position {i}: norm = {norm:.4f}")
    
    # Now check generate_soft_kv_cached
    print("\n2. generate_soft_kv_cached sequence building:")
    parts2 = []
    if decoder.prompt_left_emb is not None:
        parts2.append(decoder.prompt_left_emb.expand(1, -1, -1))
        print(f"  Added left prompt: shape {parts2[-1].shape}")
    
    # Check the condition
    if not decoder.config.patch_all_layers:
        print("  Would add activation (patch_all_layers=False)")
    else:
        print("  NOT adding activation (patch_all_layers=True)")
    
    if decoder.prompt_right_emb is not None:
        parts2.append(decoder.prompt_right_emb.expand(1, -1, -1))
        print(f"  Added right prompt: shape {parts2[-1].shape}")
    
    seq_embs2 = torch.cat(parts2, dim=1)
    print(f"  Final sequence shape: {seq_embs2.shape}")
    
    # Calculate embed position
    embed_pos2 = decoder.prompt_left_emb.size(0) if decoder.prompt_left_emb is not None else 0
    print(f"  Embed position: {embed_pos2}")
    
    print("\n3. Comparison:")
    print(f"  generate_soft seq length: {seq_embs.shape[1]}")
    print(f"  generate_soft_kv_cached seq length: {seq_embs2.shape[1]}")
    print(f"  Difference: {seq_embs.shape[1] - seq_embs2.shape[1]}")
    
    if seq_embs.shape[1] != seq_embs2.shape[1]:
        print("\n  ✗ SEQUENCE LENGTHS DIFFER!")
        print("  This explains the divergence - the methods are processing different sequences!")


if __name__ == "__main__":
    trace_embed_positions()