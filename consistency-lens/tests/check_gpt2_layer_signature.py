#!/usr/bin/env python3
"""Check GPT2 layer signature."""

import torch
from transformers import AutoModelForCausalLM
import inspect


def check_signature():
    """Check what arguments GPT2 layers expect."""
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    layer = model.transformer.h[0]
    
    print("GPT2 Layer (Block) Signature:")
    print("=" * 60)
    
    # Get the forward method signature
    sig = inspect.signature(layer.forward)
    print(f"Forward signature: {sig}")
    
    # Try calling with different arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    hidden_states = torch.randn(1, 5, 768, device=device)
    position_ids = torch.arange(5, device=device).unsqueeze(0)
    
    print("\nTesting different call patterns:")
    
    # Test 1: Just hidden states
    print("\n1. layer(hidden_states)")
    try:
        out1 = layer(hidden_states)
        print(f"   Success! Output shape: {out1[0].shape}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: With position_ids
    print("\n2. layer(hidden_states, position_ids=position_ids)")
    try:
        out2 = layer(hidden_states, position_ids=position_ids)
        print(f"   Success! Output shape: {out2[0].shape}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: With layer_past (for KV caching)
    print("\n3. layer(hidden_states, layer_past=(key, value))")
    # Create dummy past
    key = torch.randn(1, 12, 4, 64, device=device)  # (batch, heads, seq, head_dim)
    value = torch.randn(1, 12, 4, 64, device=device)
    try:
        hidden_new = torch.randn(1, 1, 768, device=device)  # New token
        out3 = layer(hidden_new, layer_past=(key, value))
        print(f"   Success! Output shape: {out3[0].shape}")
        if len(out3) > 1 and out3[1] is not None:
            print(f"   New past key shape: {out3[1][0].shape}")
            print(f"   New past value shape: {out3[1][1].shape}")
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    check_signature()