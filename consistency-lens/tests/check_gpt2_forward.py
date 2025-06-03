#!/usr/bin/env python3
"""Check how GPT2Model handles position embeddings."""

import torch
from transformers import GPT2Model
import inspect


def check_gpt2_forward():
    """Check GPT2Model forward signature and behavior."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2Model.from_pretrained("gpt2").to(device)
    
    print("GPT2Model Forward Analysis")
    print("=" * 60)
    
    # Create test input
    batch_size = 1
    seq_length = 5
    hidden_size = 768
    inputs_embeds = torch.randn(batch_size, seq_length, hidden_size, device=device)
    
    # Test 1: Just inputs_embeds
    print("\nTest 1: model(inputs_embeds=...)")
    outputs1 = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
    print(f"  Last hidden norm: {outputs1.last_hidden_state[0, -1].norm().item():.2f}")
    
    # Test 2: With position_ids
    print("\nTest 2: model(inputs_embeds=..., position_ids=...)")
    position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
    outputs2 = model(inputs_embeds=inputs_embeds, position_ids=position_ids, output_hidden_states=True)
    print(f"  Last hidden norm: {outputs2.last_hidden_state[0, -1].norm().item():.2f}")
    
    print(f"\nDifference: {(outputs1.last_hidden_state - outputs2.last_hidden_state).abs().max().item():.2e}")
    
    # Test 3: Manual forward through transformer layers
    print("\nTest 3: Manual forward (like our custom implementation)")
    
    # Start with embeddings
    hidden_states = inputs_embeds
    
    # Add position embeddings manually
    position_embeds = model.wpe(position_ids)
    hidden_states = hidden_states + position_embeds
    hidden_states = model.drop(hidden_states)
    
    # Process through layers
    for i, layer in enumerate(model.h):
        layer_outputs = layer(hidden_states)
        hidden_states = layer_outputs[0]
        if i == 0:
            print(f"  After layer 0: norm = {hidden_states[0, -1].norm().item():.2f}")
    
    # Final layer norm
    hidden_states = model.ln_f(hidden_states)
    print(f"  Final hidden norm: {hidden_states[0, -1].norm().item():.2f}")
    
    print(f"\nManual vs native difference: {(outputs1.last_hidden_state - hidden_states).abs().max().item():.2e}")
    
    # The key insight: when we pass inputs_embeds to the model, it adds position embeddings
    # But when we process layers manually, we need to add them ourselves!


if __name__ == "__main__":
    check_gpt2_forward()