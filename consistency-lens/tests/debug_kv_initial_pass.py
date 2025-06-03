#!/usr/bin/env python3
"""Debug the initial pass in KV cache generation."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.kv_cache import KVCache, GPT2AttentionWithCache


def debug_initial_pass():
    """Debug what happens in the initial KV cache pass."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("KV Cache Initial Pass Debug")
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
    
    # Setup
    transformer = decoder.base.transformer
    layers = transformer.h
    final_norm = transformer.ln_f
    kv_cache = KVCache()
    
    # Process initial sequence as in generate_soft_kv_cached
    hidden_states = seq_embs
    seq_length = hidden_states.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    
    # Pre-compute single projection
    single_proj = decoder._apply_projection(activation)
    
    # Calculate embed position
    embed_pos = 3
    
    # Add position embeddings
    position_embeds = transformer.wpe(position_ids)
    hidden_states = transformer.drop(hidden_states + position_embeds)
    
    print(f"\nProcessing through layers:")
    print(f"  Embed position: {embed_pos}")
    print(f"  Initial hidden norm: {hidden_states[0, -1].norm().item():.2f}")
    
    # Process each layer with activation patching
    for layer_idx, layer_module in enumerate(layers):
        # Use wrapper for KV caching
        wrapper = GPT2AttentionWithCache(layer_module)
        hidden_states, kv_cache = wrapper(
            hidden_states, kv_cache, layer_idx, use_cache=True
        )
        
        # Replace activation at the embed position for this layer
        if layer_idx > 0:  # Skip layer 0 since activation is already there
            hidden_states[:, embed_pos] = single_proj
            
        if layer_idx < 3:  # Print first few layers
            print(f"  After layer {layer_idx}: hidden[-1] norm = {hidden_states[0, -1].norm().item():.2f}")
    
    # Final layer norm
    hidden_states = final_norm(hidden_states)
    
    print(f"\nAfter all layers:")
    print(f"  Final hidden shape: {hidden_states.shape}")
    print(f"  Last position norm: {hidden_states[0, -1].norm().item():.2f}")
    
    # Get logits
    logits = decoder.out(hidden_states[:, -1])
    print(f"\nLogits:")
    print(f"  Range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    print(f"  Top 5:")
    top5 = logits[0].topk(5)
    for val, idx in zip(top5.values, top5.indices):
        print(f"    {idx.item()}: {val.item():.2f} ('{tokenizer.decode([idx.item()])}')")
    
    # Now compare with what generate_soft would produce
    print("\n\nComparing with generate_soft:")
    torch.manual_seed(123)
    gen = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    gen_logits = gen.raw_lm_logits[0, 0]
    
    print(f"  Logits range: [{gen_logits.min().item():.2f}, {gen_logits.max().item():.2f}]")
    print(f"  Top 5:")
    top5_gen = gen_logits.topk(5)
    for val, idx in zip(top5_gen.values, top5_gen.indices):
        print(f"    {idx.item()}: {val.item():.2f} ('{tokenizer.decode([idx.item()])}')")
    
    print(f"\nLogits difference: {(logits[0] - gen_logits).abs().max().item():.2f}")


if __name__ == "__main__":
    debug_initial_pass()