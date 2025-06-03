#!/usr/bin/env python3
"""Debug why layer 11 has zero gradient - detailed analysis."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_computation_graph():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = 1
    d_model = 768
    seq_length = 4
    
    print("Debugging Layer 11 Zero Gradient Issue")
    print("=" * 60)
    
    # Create decoder with per-layer projections
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=True,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    # Create activation
    activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    # Let's trace through what happens:
    print("\n1. Understanding the computation flow:")
    print("-" * 40)
    
    # The key insight: Let's think about what happens at layer 11
    # - Layer 11 processes the hidden states and outputs new hidden states
    # - THEN we replace position embed_pos with our projected activation
    # - This replaced value goes through final layer norm and then to output projection
    # - But does this replacement actually affect the final output?
    
    embed_pos = decoder.prompt_left_emb.size(0) if decoder.prompt_left_emb is not None else 0
    print(f"Embed position: {embed_pos}")
    print(f"Prompt structure: left({embed_pos}) + embed(1) + right(1) = total prompt length: {embed_pos + 2}")
    
    # Generate and track what happens
    gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
    
    print(f"\nDuring generation:")
    print(f"- We generate {seq_length} new tokens")
    print(f"- Total sequence length becomes: {embed_pos + 2 + seq_length}")
    print(f"- For next token prediction, we use position -1")
    
    # Let's check if the issue is that we're only using the last position
    print("\n2. Testing different loss functions:")
    print("-" * 40)
    
    # Test 1: Loss on all generated positions
    decoder.zero_grad()
    activation.grad = None
    gen1 = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
    loss1 = gen1.generated_text_embeddings.sum()
    loss1.backward()
    
    print(f"\nLoss = sum(all generated embeddings):")
    for i in range(12):
        grad_norm = decoder.proj_weight.grad[i].norm().item()
        print(f"  Layer {i:2d} grad norm: {grad_norm:.4f}")
    
    # Test 2: Loss that explicitly depends on the embed position's influence
    decoder.zero_grad()
    activation.grad = None
    
    # Generate with a custom forward pass to track hidden states
    print("\n3. Analyzing hidden state flow:")
    print("-" * 40)
    
    # The real issue might be:
    # After layer 11, we have hidden_states of shape (B, seq_len, d_model)
    # We replace hidden_states[:, embed_pos] with our projection
    # Then this goes through final layer norm
    # Then for generation, we only use hidden_states[:, -1] for logits
    # So the question is: does position embed_pos influence position -1 after final layer norm?
    
    # In a transformer with NO additional attention after layer 11,
    # the final layer norm operates position-wise
    # So changing position embed_pos after layer 11 only affects that position's output
    # It doesn't affect other positions!
    
    print("\nKey insight:")
    print("- Layer 11's output has already mixed information via attention")
    print("- After layer 11, we replace position embed_pos")
    print("- Final layer norm is position-wise (no mixing)")
    print("- We generate from position -1, not from embed_pos")
    print("- Therefore, layer 11's replacement doesn't affect generation!")
    print("\nThis explains why layer 11 has zero gradient.")
    
    # Let's verify this by checking if the embed position's final output is used
    print("\n4. Checking which positions contribute to loss:")
    print("-" * 40)
    
    # The generated embeddings are only from newly generated positions
    # They don't include the prompt positions
    print(f"Generated embeddings shape: {gen1.generated_text_embeddings.shape}")
    print(f"These are positions {embed_pos + 2} through {embed_pos + 2 + seq_length - 1}")
    print(f"The embed position ({embed_pos}) is not included in the loss!")
    
    # Create a loss that explicitly uses the embed position
    print("\n5. Testing with a loss that uses embed position:")
    print("-" * 40)
    
    # We need to modify how we compute the loss to actually use the embed position
    # But in the current setup, generated_text_embeddings only contains newly generated tokens
    # Not the prompt tokens where we're doing the replacement


if __name__ == "__main__":
    debug_computation_graph()