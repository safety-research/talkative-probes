#!/usr/bin/env python3
"""Debug wrapper input shape."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.kv_cache import KVCache, GPT2AttentionWithCache


def debug_wrapper():
    """Debug what's being passed to the wrapper."""
    
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
    
    # Process as in generate_soft_kv_cached
    transformer = decoder.base.transformer
    hidden_states = seq_embs
    seq_length = hidden_states.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(1, -1)
    
    print(f"Hidden states shape before position: {hidden_states.shape}")
    print(f"Position IDs shape: {position_ids.shape}")
    
    # Add position embeddings
    position_embeds = transformer.wpe(position_ids.squeeze(0))
    print(f"Position embeds shape: {position_embeds.shape}")
    print(f"Position embeds unsqueezed shape: {position_embeds.unsqueeze(0).shape}")
    
    # This is the problematic line
    hidden_states_with_pos = hidden_states + position_embeds.unsqueeze(0)
    print(f"Hidden states + pos shape: {hidden_states_with_pos.shape}")
    
    hidden_states = transformer.drop(hidden_states_with_pos)
    print(f"Hidden states after drop shape: {hidden_states.shape}")
    
    # Try the wrapper
    kv_cache = KVCache()
    layer_0 = transformer.h[0]
    wrapper = GPT2AttentionWithCache(layer_0)
    
    print(f"\nCalling wrapper with hidden_states shape: {hidden_states.shape}")
    try:
        output, cache = wrapper(hidden_states, kv_cache, 0, use_cache=True)
        print(f"Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"hidden_states.dim(): {hidden_states.dim()}")


if __name__ == "__main__":
    debug_wrapper()