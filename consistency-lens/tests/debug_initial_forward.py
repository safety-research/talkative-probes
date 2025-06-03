#!/usr/bin/env python3
"""Debug the initial forward pass differences."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_initial_forward():
    """Compare initial forward pass in both methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Debugging Initial Forward Pass")
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
    decoder.eval()
    
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
    
    print(f"Sequence shape: {seq_embs.shape}")
    print(f"Embed position: 3")
    
    # Get components
    transformer = decoder.base.transformer
    layers = transformer.h
    final_norm = transformer.ln_f
    
    # Method 1: How generate_soft does it (custom forward with patching)
    print("\nMethod 1: generate_soft style (custom forward)")
    
    hidden_states_1 = seq_embs
    seq_length = hidden_states_1.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    
    # Pre-compute single projection
    single_proj = decoder._apply_projection(activation)
    embed_pos = 3
    
    # NO position embeddings added here - layers handle it
    # Run through transformer layers with activation patching
    for layer_idx, layer_module in enumerate(layers):
        # Apply layer with position_ids
        layer_outputs = layer_module(hidden_states_1, position_ids=position_ids)
        hidden_states_1 = layer_outputs[0]
        
        # Replace activation at the embed position
        if layer_idx > 0:
            hidden_states_1[:, embed_pos] = single_proj
        
        if layer_idx < 2:
            print(f"  After layer {layer_idx}: hidden[-1] norm = {hidden_states_1[0, -1].norm().item():.2f}")
    
    # Final layer norm
    hidden_states_1 = final_norm(hidden_states_1)
    
    # Get logits
    logits_1 = decoder.out(hidden_states_1[:, -1])
    
    print(f"  Final hidden norm: {hidden_states_1[0, -1].norm().item():.2f}")
    print(f"  Logits range: [{logits_1.min().item():.2f}, {logits_1.max().item():.2f}]")
    print(f"  Top logit: {logits_1.argmax().item()}")
    
    # Method 2: How KV cache does it (with position embeddings added manually)
    print("\nMethod 2: KV cache style (manual position embeddings)")
    
    hidden_states_2 = seq_embs
    
    # Add position embeddings manually (this is the difference!)
    position_embeds = transformer.wpe(position_ids)
    hidden_states_2 = transformer.drop(hidden_states_2 + position_embeds)
    
    # Process through layers
    for layer_idx, layer_module in enumerate(layers):
        # Just pass hidden states, NOT position_ids (already embedded)
        layer_outputs = layer_module(hidden_states_2, use_cache=True)
        hidden_states_2 = layer_outputs[0]
        
        # Replace activation
        if layer_idx > 0:
            hidden_states_2[:, embed_pos] = single_proj
            
        if layer_idx < 2:
            print(f"  After layer {layer_idx}: hidden[-1] norm = {hidden_states_2[0, -1].norm().item():.2f}")
    
    # Final layer norm
    hidden_states_2 = final_norm(hidden_states_2)
    
    # Get logits
    logits_2 = decoder.out(hidden_states_2[:, -1])
    
    print(f"  Final hidden norm: {hidden_states_2[0, -1].norm().item():.2f}")
    print(f"  Logits range: [{logits_2.min().item():.2f}, {logits_2.max().item():.2f}]")
    print(f"  Top logit: {logits_2.argmax().item()}")
    
    # Compare
    print(f"\nDifferences:")
    print(f"  Hidden states: {(hidden_states_1 - hidden_states_2).abs().max().item():.2e}")
    print(f"  Logits: {(logits_1 - logits_2).abs().max().item():.2e}")
    
    # The issue: In generate_soft, we pass position_ids to layers
    # In KV cache, we add position embeddings manually then don't pass position_ids
    print("\n\nThe key difference:")
    print("- generate_soft: Pass position_ids to layers (they add embeddings internally)")
    print("- KV cache: Add position embeddings manually, then don't pass position_ids")
    print("This causes DOUBLE position embedding in some cases!")


if __name__ == "__main__":
    debug_initial_forward()