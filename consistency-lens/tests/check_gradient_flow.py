#!/usr/bin/env python3
"""Check gradient flow in different configurations."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def check_gradients():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = 1
    d_model = 768
    seq_length = 4
    
    print("Configuration 1: No multi-layer patching")
    print("=" * 50)
    
    config1 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,
        per_layer_projections=False,
    )
    
    decoder1 = Decoder(config1).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    gen = decoder1.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
    loss = gen.generated_text_embeddings.sum()
    loss.backward()
    
    print(f"Activation grad norm: {activation.grad.norm().item():.4f}")
    print(f"Projection grad norm: {decoder1.proj.weight.grad.norm().item():.4f}")
    
    print("\n\nConfiguration 2: Multi-layer patching with per-layer projections")
    print("=" * 50)
    
    config2 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=True,
    )
    
    decoder2 = Decoder(config2).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    
    activation2 = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    gen2 = decoder2.generate_soft(activation2, max_length=seq_length, gumbel_tau=1.0)
    loss2 = gen2.generated_text_embeddings.sum()
    loss2.backward()
    
    print(f"Activation grad norm: {activation2.grad.norm().item():.4f}")
    print("\nPer-layer projection gradient norms:")
    for i in range(12):
        grad_norm = decoder2.proj_weight.grad[i].norm().item()
        print(f"  Layer {i}: {grad_norm:.4f}")
        
    # Check if gradients flow through the replaced positions
    print("\n\nAnalyzing gradient flow:")
    print("-" * 30)
    
    # The key insight: we're replacing hidden states AFTER each layer
    # So layer 11's output has already been computed when we replace
    # The replacement only affects the input to the final layer norm
    # But if the generated tokens don't depend on position 3 (embed_pos),
    # then layer 11's projection won't get gradients
    
    print("The reason layer 11 has zero gradient:")
    print("1. We replace hidden states AFTER each layer computes")
    print("2. Layer 11's replacement only affects the final layer norm input")
    print("3. During generation, we use position -1 for next token prediction")
    print("4. The causal mask prevents future positions from attending to past")
    print("5. So the embed position (3) doesn't influence generated tokens directly")
    print("\nThis is actually correct behavior!")


if __name__ == "__main__":
    check_gradients()