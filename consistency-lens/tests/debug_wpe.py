#!/usr/bin/env python3
"""Debug wpe (position embeddings) behavior."""

import torch
from transformers import AutoModelForCausalLM


def debug_wpe():
    """Debug how wpe handles different input shapes."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    wpe = model.transformer.wpe
    
    print("Testing wpe behavior")
    print("=" * 60)
    
    # Test 1: 1D input
    print("\nTest 1: 1D input")
    pos1d = torch.tensor([0, 1, 2], device=device)
    emb1d = wpe(pos1d)
    print(f"  Input shape: {pos1d.shape}")
    print(f"  Output shape: {emb1d.shape}")
    
    # Test 2: 2D input with batch size 1
    print("\nTest 2: 2D input (batch=1)")
    pos2d_b1 = torch.tensor([[0, 1, 2]], device=device)
    emb2d_b1 = wpe(pos2d_b1)
    print(f"  Input shape: {pos2d_b1.shape}")
    print(f"  Output shape: {emb2d_b1.shape}")
    
    # Test 3: 2D input with batch size 2
    print("\nTest 3: 2D input (batch=2)")
    pos2d_b2 = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device)
    emb2d_b2 = wpe(pos2d_b2)
    print(f"  Input shape: {pos2d_b2.shape}")
    print(f"  Output shape: {emb2d_b2.shape}")
    
    # Check if they're the same
    print("\nComparison:")
    print(f"  1D vs 2D (batch=1): max diff = {(emb1d - emb2d_b1[0]).abs().max().item():.2e}")
    print(f"  2D batch[0] vs batch[1]: max diff = {(emb2d_b2[0] - emb2d_b2[1]).abs().max().item():.2e}")
    
    # Test 4: What GPT-2 expects
    print("\nTest 4: GPT-2 forward pass")
    input_ids = torch.tensor([[50256, 50256, 50256]], device=device)  # pad tokens
    with torch.no_grad():
        # Method 1: with input_ids
        out1 = model(input_ids=input_ids, output_hidden_states=True)
        
        # Method 2: with inputs_embeds and position_ids
        inputs_embeds = model.get_input_embeddings()(input_ids)
        position_ids = torch.arange(3, device=device).unsqueeze(0)
        out2 = model(inputs_embeds=inputs_embeds, position_ids=position_ids, output_hidden_states=True)
        
    print(f"  Logits diff: {(out1.logits - out2.logits).abs().max().item():.2e}")


if __name__ == "__main__":
    debug_wpe()