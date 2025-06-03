#!/usr/bin/env python3
"""Simple test for multi-layer patching - comparing generate_soft with and without multi-layer patching."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_multi_layer_patching():
    """Test that multi-layer patching works correctly."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = 2
    d_model = 768  # GPT-2 hidden size
    seq_length = 8
    gumbel_tau = 1.0
    
    print("Testing multi-layer patching functionality...")
    print("=" * 60)
    
    # Test 1: Compare single projection vs per-layer projections
    print("\nTest 1: Single projection vs per-layer projections")
    print("-" * 50)
    
    # Create two decoders
    config1 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,  # Single projection
    )
    
    config2 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=True,  # Per-layer projections
    )
    
    decoder1 = Decoder(config1).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    decoder2 = Decoder(config2).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    
    # Create activation
    torch.manual_seed(42)
    activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    # Generate with both
    gen1 = decoder1.generate_soft(activation, max_length=seq_length, gumbel_tau=gumbel_tau)
    gen2 = decoder2.generate_soft(activation, max_length=seq_length, gumbel_tau=gumbel_tau)
    
    # They should be different (since projections are different)
    emb_diff = torch.max(torch.abs(gen1.generated_text_embeddings - gen2.generated_text_embeddings)).item()
    print(f"Max embedding difference: {emb_diff:.4f}")
    assert emb_diff > 0.1, "Outputs should be different with different projection configurations"
    
    # Test gradients flow for both
    loss1 = gen1.generated_text_embeddings.sum()
    loss1.backward()
    grad1 = activation.grad.clone()
    activation.grad.zero_()
    
    loss2 = gen2.generated_text_embeddings.sum()
    loss2.backward()
    grad2 = activation.grad.clone()
    
    print(f"Activation gradient norm (single proj): {grad1.norm().item():.4f}")
    print(f"Activation gradient norm (per-layer): {grad2.norm().item():.4f}")
    
    assert grad1.norm() > 0, "Gradients should flow with single projection"
    assert grad2.norm() > 0, "Gradients should flow with per-layer projections"
    
    # Check per-layer projection gradients
    assert decoder2.proj_weight.grad is not None, "Per-layer projections should have gradients"
    n_layers = decoder2.base.config.num_hidden_layers
    print(f"\nPer-layer projection gradient norms:")
    layers_with_grad = 0
    for i in range(n_layers):
        grad_norm = decoder2.proj_weight.grad[i].norm().item()
        print(f"  Layer {i}: {grad_norm:.4f}")
        if grad_norm > 0:
            layers_with_grad += 1
    # At least most layers should have gradients (layer 0 might be skipped, last layer might not contribute)
    assert layers_with_grad >= n_layers - 2, f"At least {n_layers-2} layers should have gradients, but only {layers_with_grad} do"
    
    print("\n✓ Test 1 passed!")
    
    # Test 2: Compare with and without multi-layer patching
    print("\n\nTest 2: With vs without multi-layer patching")
    print("-" * 50)
    
    config_no_patch = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,  # No multi-layer patching
        per_layer_projections=False,
    )
    
    config_patch = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,  # Multi-layer patching
        per_layer_projections=False,
    )
    
    decoder_no_patch = Decoder(config_no_patch).to(device)
    decoder_no_patch.set_prompt("explain <embed>:", tokenizer)
    
    decoder_patch = Decoder(config_patch).to(device)
    decoder_patch.set_prompt("explain <embed>:", tokenizer)
    
    # Make sure they have the same projection weights
    decoder_patch.proj.weight.data.copy_(decoder_no_patch.proj.weight.data)
    decoder_patch.proj.bias.data.copy_(decoder_no_patch.proj.bias.data)
    
    # Create activation
    torch.manual_seed(42)
    activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    # Generate with both
    gen_no_patch = decoder_no_patch.generate_soft(activation, max_length=seq_length, gumbel_tau=gumbel_tau)
    activation.grad = None
    gen_patch = decoder_patch.generate_soft(activation, max_length=seq_length, gumbel_tau=gumbel_tau)
    
    # They should be different
    emb_diff = torch.max(torch.abs(gen_no_patch.generated_text_embeddings - gen_patch.generated_text_embeddings)).item()
    print(f"Max embedding difference: {emb_diff:.4f}")
    assert emb_diff > 0.1, "Outputs should be different with/without multi-layer patching"
    
    print("\n✓ Test 2 passed!")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_multi_layer_patching()