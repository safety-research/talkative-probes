#!/usr/bin/env python3
"""Debug hidden states after initial pass."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def compare_hidden_states():
    """Compare hidden states from both methods after initial pass."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Hidden States Comparison")
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
    
    print(f"Sequence: [left_prompt(3), activation(1), right_prompt(1)] = {seq_embs.shape}")
    
    # Method 1: generate_soft style (simplified)
    print("\nMethod 1: generate_soft style")
    transformer = decoder.base.transformer
    layers = transformer.h
    final_norm = transformer.ln_f
    
    # Get position IDs
    seq_length = seq_embs.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    
    # Embedding layer (no position embeddings added here in multi-layer patching)
    hidden_states_1 = seq_embs
    
    # Calculate embed position
    embed_pos = 3
    single_proj = decoder._apply_projection(activation)
    
    # Run through transformer layers with activation patching
    for layer_idx, layer_module in enumerate(layers):
        # Apply layer
        layer_outputs = layer_module(hidden_states_1, position_ids=position_ids)
        hidden_states_1 = layer_outputs[0]
        
        # Replace activation at the embed position for this layer
        if layer_idx > 0:  # Skip layer 0 since activation is already there
            hidden_states_1[:, embed_pos] = single_proj
    
    # Final layer norm
    hidden_states_1 = final_norm(hidden_states_1)
    
    print(f"  Final hidden shape: {hidden_states_1.shape}")
    print(f"  Last position norm: {hidden_states_1[0, -1].norm().item():.2f}")
    
    # Get logits
    logits_1 = decoder.out(hidden_states_1[:, -1])
    print(f"  Logits range: [{logits_1.min().item():.2f}, {logits_1.max().item():.2f}]")
    
    # Method 2: What's actually in generate_soft
    print("\nMethod 2: Actual generate_soft")
    torch.manual_seed(123)
    # Manually run one step of generate_soft
    hidden_states_2 = seq_embs
    
    # The actual code does a full forward pass through the model
    out = decoder.base(inputs_embeds=seq_embs, output_hidden_states=True)
    h_last = out.hidden_states[-1]  # Last hidden state
    logits_2 = decoder.out(h_last[:, -1])
    
    print(f"  Final hidden shape: {h_last.shape}")
    print(f"  Last position norm: {h_last[0, -1].norm().item():.2f}")
    print(f"  Logits range: [{logits_2.min().item():.2f}, {logits_2.max().item():.2f}]")
    
    # Compare
    print(f"\nDifference in last hidden: {(hidden_states_1[0, -1] - h_last[0, -1]).norm().item():.2f}")
    print(f"Difference in logits: {(logits_1 - logits_2).abs().max().item():.2f}")
    
    print("\nKey insight: When patch_all_layers=True, generate_soft does a CUSTOM forward pass")
    print("that patches the activation at each layer. But the base model forward pass doesn't!")


if __name__ == "__main__":
    compare_hidden_states()