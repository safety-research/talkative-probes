#!/usr/bin/env python3
"""Debug why logits are so different."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.kv_cache import KVCache, GPT2AttentionWithCache


def debug_logits():
    """Debug the logits computation."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Debugging Logits Computation")
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
    
    # Manually trace through KV cached generation
    transformer = decoder.base.transformer
    layers = transformer.h
    final_norm = transformer.ln_f
    
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
    
    # Process with KV cache
    kv_cache = KVCache()
    hidden_states = seq_embs
    seq_length = hidden_states.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    
    # Add position embeddings
    position_embeds = transformer.wpe(position_ids)
    hidden_states = transformer.drop(hidden_states + position_embeds)
    
    print(f"Hidden states after position: {hidden_states.shape}")
    
    # Process through one layer to check
    embed_pos = 3  # Where activation is patched
    single_proj = decoder._apply_projection(activation)
    
    # First layer
    layer_0 = layers[0]
    wrapper = GPT2AttentionWithCache(layer_0)
    hidden_states_0, kv_cache = wrapper(hidden_states, kv_cache, 0, use_cache=True)
    
    # Replace activation at embed position
    hidden_states_0[:, embed_pos] = single_proj
    
    print(f"\nAfter first layer:")
    print(f"  Hidden states shape: {hidden_states_0.shape}")
    print(f"  Hidden state at position -1 (last): norm = {hidden_states_0[0, -1].norm().item():.2f}")
    print(f"  Hidden state at embed_pos: norm = {hidden_states_0[0, embed_pos].norm().item():.2f}")
    
    # Continue through all layers (simplified)
    hidden_states = hidden_states_0
    for layer_idx in range(1, len(layers)):
        wrapper = GPT2AttentionWithCache(layers[layer_idx])
        hidden_states, kv_cache = wrapper(hidden_states, kv_cache, layer_idx, use_cache=True)
        # Replace activation
        hidden_states[:, embed_pos] = single_proj
    
    # Final norm
    hidden_states = final_norm(hidden_states)
    
    print(f"\nAfter all layers and final norm:")
    print(f"  Hidden states shape: {hidden_states.shape}")
    print(f"  Last hidden state norm: {hidden_states[0, -1].norm().item():.2f}")
    
    # Get logits
    logits = decoder.out(hidden_states[:, -1])
    print(f"\nLogits:")
    print(f"  Shape: {logits.shape}")
    print(f"  Min: {logits.min().item():.2f}")
    print(f"  Max: {logits.max().item():.2f}")
    print(f"  Mean: {logits.mean().item():.2f}")
    
    # Compare with generate_soft
    print("\n\nComparing with generate_soft:")
    torch.manual_seed(123)
    gen = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    logits_gen = gen.raw_lm_logits[0, 0]
    
    print(f"  Logits min: {logits_gen.min().item():.2f}")
    print(f"  Logits max: {logits_gen.max().item():.2f}")
    print(f"  Logits mean: {logits_gen.mean().item():.2f}")


if __name__ == "__main__":
    debug_logits()