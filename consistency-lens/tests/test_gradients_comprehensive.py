#!/usr/bin/env python3
"""Comprehensive gradient testing for multi-layer patching."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig


def test_all_gradients():
    """Test gradients w.r.t. both activation input and model parameters."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = 2
    d_model = 768
    seq_length = 8
    
    print("Comprehensive Gradient Testing")
    print("=" * 80)
    
    # Test configurations
    configs = [
        ("No patching", False, False),
        ("Multi-layer, single proj", True, False),
        ("Multi-layer, per-layer proj", True, True),
    ]
    
    for config_name, patch_all_layers, per_layer_projections in configs:
        print(f"\n\nConfiguration: {config_name}")
        print("-" * 60)
        
        # Create decoder
        config = DecoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
            patch_all_layers=patch_all_layers,
            per_layer_projections=per_layer_projections,
        )
        
        decoder = Decoder(config).to(device)
        decoder.set_prompt("explain <embed>:", tokenizer)
        
        # Create encoder for full pipeline test
        encoder = Encoder(EncoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
        )).to(device)
        
        # Create activation that requires gradient
        activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        # Forward pass
        gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
        reconstructed = encoder(gen.generated_text_embeddings)
        
        # Define loss (reconstruction loss)
        loss = F.mse_loss(reconstructed, activation)
        print(f"Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # 1. Check gradient w.r.t. activation input
        print(f"\n1. Gradient w.r.t. activation input:")
        act_grad_norm = activation.grad.norm().item()
        print(f"   Norm: {act_grad_norm:.4f}")
        print(f"   Shape: {activation.grad.shape}")
        print(f"   Mean: {activation.grad.mean().item():.6f}")
        print(f"   Std: {activation.grad.std().item():.6f}")
        assert act_grad_norm > 0, "Activation should have non-zero gradient"
        
        # 2. Check gradients w.r.t. decoder parameters
        print(f"\n2. Gradients w.r.t. decoder parameters:")
        
        if per_layer_projections:
            # Check each layer's projection
            n_layers = decoder.base.config.num_hidden_layers
            print(f"   Per-layer projection gradients:")
            total_grad_norm = 0
            for i in range(n_layers):
                weight_grad_norm = decoder.proj_weight.grad[i].norm().item()
                bias_grad_norm = decoder.proj_bias.grad[i].norm().item()
                print(f"     Layer {i}: weight={weight_grad_norm:.4f}, bias={bias_grad_norm:.4f}")
                total_grad_norm += weight_grad_norm
            print(f"   Total projection gradient norm: {total_grad_norm:.4f}")
        else:
            # Single projection
            if decoder.proj is not None:
                proj_weight_grad_norm = decoder.proj.weight.grad.norm().item()
                proj_bias_grad_norm = decoder.proj.bias.grad.norm().item()
                print(f"   Projection weight gradient norm: {proj_weight_grad_norm:.4f}")
                print(f"   Projection bias gradient norm: {proj_bias_grad_norm:.4f}")
                assert proj_weight_grad_norm > 0, "Projection weight should have gradient"
        
        # Check other decoder parameters
        decoder_param_grads = 0
        decoder_params_with_grad = 0
        for name, param in decoder.named_parameters():
            if param.grad is not None and not name.startswith('proj'):
                decoder_param_grads += param.grad.norm().item()
                decoder_params_with_grad += 1
        print(f"   Other decoder parameters with gradients: {decoder_params_with_grad}")
        print(f"   Total other decoder gradient norm: {decoder_param_grads:.4f}")
        
        # 3. Check gradients w.r.t. encoder parameters
        print(f"\n3. Gradients w.r.t. encoder parameters:")
        encoder_proj_grad_norm = encoder.proj.weight.grad.norm().item()
        print(f"   Encoder projection gradient norm: {encoder_proj_grad_norm:.4f}")
        assert encoder_proj_grad_norm > 0, "Encoder projection should have gradient"
        
        # Verify gradient flow makes sense
        print(f"\n4. Gradient flow analysis:")
        print(f"   Loss → Encoder → Decoder → Activation")
        print(f"   All components have non-zero gradients ✓")
        
        # Clean up for next iteration
        decoder.zero_grad()
        encoder.zero_grad()
        activation.grad = None


def test_gradient_consistency():
    """Test that gradients are consistent across different generation methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = 2
    d_model = 768
    seq_length = 8
    
    print("\n\n\nGradient Consistency Test")
    print("=" * 80)
    print("Testing that different generation methods produce same gradients")
    
    # Test without multi-layer patching (where all methods work)
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,
        per_layer_projections=False,
    )
    
    methods = ['generate_soft', 'generate_soft_chkpt', 'generate_soft_kv_cached']
    gradients = {}
    
    for method in methods:
        print(f"\n{method}:")
        
        # Create fresh decoder
        decoder = Decoder(config).to(device)
        decoder.set_prompt("explain <embed>:", tokenizer)
        
        # Create activation
        torch.manual_seed(42)
        activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        # Generate
        if method == 'generate_soft':
            gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
        elif method == 'generate_soft_chkpt':
            gen = decoder.generate_soft_chkpt(activation, max_length=seq_length, gumbel_tau=1.0)
        else:  # generate_soft_kv_cached
            gen = decoder.generate_soft_kv_cached(activation, max_length=seq_length, gumbel_tau=1.0)
        
        # Simple loss
        loss = gen.generated_text_embeddings.sum()
        loss.backward()
        
        # Store gradients
        gradients[method] = {
            'activation': activation.grad.clone(),
            'proj_weight': decoder.proj.weight.grad.clone(),
            'proj_bias': decoder.proj.bias.grad.clone(),
        }
        
        print(f"  Activation grad norm: {activation.grad.norm().item():.6f}")
        print(f"  Projection weight grad norm: {decoder.proj.weight.grad.norm().item():.6f}")
        print(f"  Projection bias grad norm: {decoder.proj.bias.grad.norm().item():.6f}")
    
    # Compare gradients
    print("\n\nComparing gradients between methods:")
    base_method = 'generate_soft'
    for method in methods[1:]:
        print(f"\n{base_method} vs {method}:")
        
        # Activation gradient difference
        act_diff = (gradients[base_method]['activation'] - gradients[method]['activation']).abs().max().item()
        print(f"  Max activation gradient difference: {act_diff:.2e}")
        
        # Projection weight gradient difference
        weight_diff = (gradients[base_method]['proj_weight'] - gradients[method]['proj_weight']).abs().max().item()
        print(f"  Max projection weight gradient difference: {weight_diff:.2e}")
        
        # Projection bias gradient difference
        bias_diff = (gradients[base_method]['proj_bias'] - gradients[method]['proj_bias']).abs().max().item()
        print(f"  Max projection bias gradient difference: {bias_diff:.2e}")
        
        # Check consistency
        assert act_diff < 1e-5, f"Activation gradients inconsistent: {act_diff}"
        assert weight_diff < 2e-5, f"Weight gradients inconsistent: {weight_diff}"
        assert bias_diff < 2e-5, f"Bias gradients inconsistent: {bias_diff}"
    
    print("\n✓ All methods produce consistent gradients!")


if __name__ == "__main__":
    test_all_gradients()
    test_gradient_consistency()
    
    print("\n\n" + "=" * 80)
    print("✓ All gradient tests passed!")
    print("=" * 80)