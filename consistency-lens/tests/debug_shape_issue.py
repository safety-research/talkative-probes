#!/usr/bin/env python3
"""Debug shape issue in KV cache."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_shape():
    """Debug the shape issue."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
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
    
    # Build initial sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
    a_proj = decoder._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
    seq_embs = torch.cat(parts, dim=1)
    
    print(f"Initial sequence shape: {seq_embs.shape}")
    
    # Add position embeddings
    transformer = decoder.base.transformer
    seq_length = seq_embs.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    position_embeds = transformer.wpe(position_ids.squeeze(0))
    
    print(f"Position IDs shape: {position_ids.shape}")
    print(f"Position embeds shape before: {position_embeds.shape}")
    print(f"Position embeds shape after unsqueeze: {position_embeds.unsqueeze(0).shape}")
    
    hidden_states = transformer.drop(seq_embs + position_embeds.unsqueeze(0))
    print(f"Hidden states shape: {hidden_states.shape}")


if __name__ == "__main__":
    debug_shape()