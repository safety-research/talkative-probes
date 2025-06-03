#!/usr/bin/env python3
"""Test that gradients align between generate_soft and generate_soft_kv_cached."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_gradients():
    """Test gradient alignment between both generation methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Testing Gradient Alignment")
    print("=" * 60)
    
    # Test 1: Without multi-layer patching
    print("\nTest 1: Without multi-layer patching")
    config1 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,
        per_layer_projections=False,
    )
    
    decoder1 = Decoder(config1).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder1.base.config.hidden_size
    activation1 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation2 = activation1.clone().detach().requires_grad_(True)
    
    # Generate with both methods
    torch.manual_seed(42)
    gen1 = decoder1.generate_soft(activation1, max_length=5, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen2 = decoder1.generate_soft_kv_cached(activation2, max_length=5, gumbel_tau=1.0)
    
    # Compute loss (e.g., sum of logits)
    loss1 = gen1.raw_lm_logits.sum()
    loss2 = gen2.raw_lm_logits.sum()
    
    # Compute gradients
    loss1.backward()
    loss2.backward()
    
    # Compare gradients
    grad_diff = (activation1.grad - activation2.grad).abs().max().item()
    print(f"  Activation gradient max diff: {grad_diff:.2e}")
    print(f"  Gradients match: {grad_diff < 1e-4}")
    
    # Test 2: With multi-layer patching
    print("\n\nTest 2: With multi-layer patching")
    config2 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder2 = Decoder(config2).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    
    activation3 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation4 = activation3.clone().detach().requires_grad_(True)
    
    # Generate with both methods
    torch.manual_seed(42)
    gen3 = decoder2.generate_soft(activation3, max_length=5, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen4 = decoder2.generate_soft_kv_cached(activation4, max_length=5, gumbel_tau=1.0)
    
    # Compute loss
    loss3 = gen3.raw_lm_logits.sum()
    loss4 = gen4.raw_lm_logits.sum()
    
    # Compute gradients
    loss3.backward()
    loss4.backward()
    
    # Compare gradients
    grad_diff2 = (activation3.grad - activation4.grad).abs().max().item()
    print(f"  Activation gradient max diff: {grad_diff2:.2e}")
    print(f"  Gradients match: {grad_diff2 < 1e-4}")
    
    # Test 3: With per-layer projections
    print("\n\nTest 3: With per-layer projections")
    config3 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=True,
    )
    
    decoder3 = Decoder(config3).to(device)
    decoder3.set_prompt("explain <embed>:", tokenizer)
    
    activation5 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation6 = activation5.clone().detach().requires_grad_(True)
    
    # Generate with both methods
    torch.manual_seed(42)
    gen5 = decoder3.generate_soft(activation5, max_length=5, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen6 = decoder3.generate_soft_kv_cached(activation6, max_length=5, gumbel_tau=1.0)
    
    # Compute loss
    loss5 = gen5.raw_lm_logits.sum()
    loss6 = gen6.raw_lm_logits.sum()
    
    # Compute gradients
    loss5.backward()
    loss6.backward()
    
    # Compare gradients
    grad_diff3 = (activation5.grad - activation6.grad).abs().max().item()
    print(f"  Activation gradient max diff: {grad_diff3:.2e}")
    print(f"  Gradients match: {grad_diff3 < 1e-4}")
    
    # Test gradient flow through projection layers
    print("\n\nTest 4: Gradient flow through projection layers")
    
    # Check if projection weights receive gradients
    if config3.per_layer_projections:
        # Clear existing gradients
        decoder3.zero_grad()
        
        activation7 = torch.randn(1, d_model, device=device, requires_grad=True)
        gen7 = decoder3.generate_soft(activation7, max_length=5, gumbel_tau=1.0)
        loss7 = gen7.raw_lm_logits.sum()
        loss7.backward()
        
        # Check projection weight gradients
        proj_grad_norm = decoder3.proj_weight.grad.norm().item()
        print(f"  Projection weight gradient norm: {proj_grad_norm:.2e}")
        print(f"  Projection weights receive gradients: {proj_grad_norm > 0}")
        
        # Now with KV caching
        decoder3.zero_grad()
        activation8 = torch.randn(1, d_model, device=device, requires_grad=True)
        gen8 = decoder3.generate_soft_kv_cached(activation8, max_length=5, gumbel_tau=1.0)
        loss8 = gen8.raw_lm_logits.sum()
        loss8.backward()
        
        proj_grad_norm_kv = decoder3.proj_weight.grad.norm().item()
        print(f"  Projection weight gradient norm (KV cached): {proj_grad_norm_kv:.2e}")
        
        grad_diff_proj = abs(proj_grad_norm - proj_grad_norm_kv) / max(proj_grad_norm, proj_grad_norm_kv)
        print(f"  Relative difference: {grad_diff_proj:.2%}")
    
    print("\n" + "=" * 60)
    print("Gradient alignment test complete!")


if __name__ == "__main__":
    test_gradients()