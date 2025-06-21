#!/usr/bin/env python3
"""
Test to understand GPT2LMHeadModel outputs.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Config

def test_gpt2_outputs():
    """Test what GPT2LMHeadModel returns with and without output_hidden_states."""
    
    # Create a small GPT2 model
    config = GPT2Config(n_embd=768, n_layer=2, n_head=12, vocab_size=50257)
    model = GPT2LMHeadModel(config)
    model.eval()
    
    # Create dummy input
    batch_size = 1
    seq_len = 5
    hidden_size = 768
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
    
    print("Testing GPT2LMHeadModel outputs:")
    print(f"Input shape: {inputs_embeds.shape}")
    print()
    
    # Test 1: Without output_hidden_states
    print("1. Without output_hidden_states=True:")
    with torch.no_grad():
        outputs1 = model(inputs_embeds=inputs_embeds)
    
    print(f"   Type of outputs: {type(outputs1)}")
    print(f"   Has 'last_hidden_state'? {hasattr(outputs1, 'last_hidden_state')}")
    print(f"   Has 'logits'? {hasattr(outputs1, 'logits')}")
    
    if hasattr(outputs1, 'logits'):
        print(f"   outputs.logits shape: {outputs1.logits.shape}")
    if hasattr(outputs1, 'last_hidden_state'):
        print(f"   outputs.last_hidden_state shape: {outputs1.last_hidden_state.shape}")
    else:
        print(f"   outputs[0] shape: {outputs1[0].shape if hasattr(outputs1, '__getitem__') else 'N/A'}")
    
    # Test 2: With output_hidden_states
    print("\n2. With output_hidden_states=True:")
    with torch.no_grad():
        outputs2 = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
    
    print(f"   Type of outputs: {type(outputs2)}")
    print(f"   Has 'last_hidden_state'? {hasattr(outputs2, 'last_hidden_state')}")
    print(f"   Has 'hidden_states'? {hasattr(outputs2, 'hidden_states')}")
    print(f"   Has 'logits'? {hasattr(outputs2, 'logits')}")
    
    if hasattr(outputs2, 'logits'):
        print(f"   outputs.logits shape: {outputs2.logits.shape}")
    if hasattr(outputs2, 'last_hidden_state'):
        print(f"   outputs.last_hidden_state shape: {outputs2.last_hidden_state.shape}")
    if hasattr(outputs2, 'hidden_states'):
        print(f"   outputs.hidden_states length: {len(outputs2.hidden_states)}")
        print(f"   outputs.hidden_states[-1] shape: {outputs2.hidden_states[-1].shape}")
    
    # Test 3: What fwd_tokens is trying to do
    print("\n3. What fwd_tokens code does:")
    outputs3 = model(inputs_embeds=inputs_embeds)
    hidden_states = outputs3.last_hidden_state if hasattr(outputs3, "last_hidden_state") else outputs3[0]
    print(f"   Extracted 'hidden_states' shape: {hidden_states.shape}")
    print(f"   Is this hidden states (768) or logits (50257)? Last dim = {hidden_states.shape[-1]}")


if __name__ == "__main__":
    test_gpt2_outputs()