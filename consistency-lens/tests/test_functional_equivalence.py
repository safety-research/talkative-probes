#!/usr/bin/env python3
"""Test functional equivalence of generation methods during training."""

import torch
import numpy as np
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from lens.models.flash_kv_cache import FLASH_AVAILABLE


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def test_functional_equivalence():
    """Test that all methods produce similar training losses."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Functional Equivalence Test")
    print("=" * 80)
    print("Testing if different generation methods produce similar losses")
    print()
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    d_model = 768
    
    # Test configurations
    test_configs = [
        (2, 16),
        (4, 32),
        (2, 64),
    ]
    
    methods = ["naive", "kv_cache"]
    if FLASH_AVAILABLE:
        methods.append("flash")
    
    # Natural language prefix
    lm_loss_natural_prefix = "explain something:"
    cached_prefix_ids = tokenizer(lm_loss_natural_prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    
    for batch_size, seq_length in test_configs:
        print(f"\nBatch={batch_size}, Length={seq_length}")
        print("-" * 60)
        
        # Store losses for comparison
        method_losses = {}
        
        for method in methods:
            # Set seed for each method to ensure same initialization
            set_random_seed(42)
            
            # Create models
            if method == "naive":
                decoder_config = DecoderConfig(
                    model_name=model_name,
                    use_kv_cache=False,
                    use_flash_attention=False,
                    base_model=False,
                    projection_layer=True,
                    eye_init=True,
                )
            elif method == "kv_cache":
                decoder_config = DecoderConfig(
                    model_name=model_name,
                    use_kv_cache=True,
                    use_flash_attention=False,
                    base_model=False,
                    projection_layer=True,
                    eye_init=True,
                )
            else:  # flash
                decoder_config = DecoderConfig(
                    model_name=model_name,
                    use_kv_cache=False,
                    use_flash_attention=True,
                    base_model=False,
                    projection_layer=True,
                    eye_init=True,
                )
            
            decoder = Decoder(decoder_config).to(device)
            encoder = Encoder(EncoderConfig(
                model_name=model_name,
                base_model=False,
                projection_layer=True,
                eye_init=True,
            )).to(device)
            orig_model = OrigWrapper(model_name).to(device)
            
            decoder.set_prompt("explain <embed>:", tokenizer)
            
            models = {
                "dec": decoder,
                "enc": encoder,
                "orig": orig_model
            }
            
            # Create identical batch for each method
            set_random_seed(123)  # Different seed for data
            batch = {
                "A": torch.randn(batch_size, d_model, device=device),
                "A_prime": torch.randn(batch_size, d_model, device=device),
                "input_ids_A": torch.randint(0, 50257, (batch_size, 128), device=device),
                "layer_idx": torch.tensor([6] * batch_size, device=device).unsqueeze(1),
                "token_pos_A": torch.tensor([64] * batch_size, device=device).unsqueeze(1),
            }
            
            # Loss parameters
            loss_fns = {
                "t_text": seq_length,
                "tau": 1.0,  # Use deterministic generation
                "alpha": 0.1,
                "kl_base_weight": 1.0,
                "entropy_weight": 0.01,
                "mse_weight": 0.1,
                "lm_weight": 0.1,
            }
            
            # Run training step
            with torch.no_grad():  # Just test forward pass
                losses = train_step(batch, models, loss_fns, tokenizer, cached_prefix_ids)
            
            method_losses[method] = {
                'total': losses['total'].item(),
                'kl': losses['kl'].item(),
                'lm': losses['lm'].item(),
                'mse': losses['mse'].item(),
                'entropy': losses['entropy'].item(),
            }
            
            print(f"\n  {method.upper()}:")
            print(f"    Total loss: {losses['total'].item():.6f}")
            print(f"    KL loss: {losses['kl'].item():.6f}")
            print(f"    LM loss: {losses['lm'].item():.6f}")
            print(f"    MSE loss: {losses['mse'].item():.6f}")
            print(f"    Entropy: {losses['entropy'].item():.6f}")
            
            # Cleanup
            del models
        
        # Compare losses
        if len(method_losses) > 1:
            print("\n  Loss Comparison:")
            baseline = method_losses.get('naive', method_losses.get('kv_cache'))
            
            for method, losses in method_losses.items():
                if method != 'naive':
                    total_diff = abs(losses['total'] - baseline['total']) / baseline['total'] * 100
                    kl_diff = abs(losses['kl'] - baseline['kl']) / baseline['kl'] * 100 if baseline['kl'] > 0 else 0
                    
                    print(f"    {method} vs naive:")
                    print(f"      Total loss difference: {total_diff:.1f}%")
                    print(f"      KL loss difference: {kl_diff:.1f}%")
                    
                    if total_diff < 5:  # Less than 5% difference
                        print(f"      ✓ Functionally equivalent")
                    else:
                        print(f"      ✗ Significant difference")


def test_gradient_consistency():
    """Test that gradients are similar across methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 80)
    print("Gradient Consistency Test")
    print("=" * 80)
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    d_model = 768
    batch_size = 2
    seq_length = 16
    
    methods = ["naive", "kv_cache"]
    if FLASH_AVAILABLE:
        methods.append("flash")
    
    # Store gradients for comparison
    method_gradients = {}
    
    for method in methods:
        set_random_seed(42)
        
        # Create models
        if method == "naive":
            config = DecoderConfig(
                model_name=model_name,
                use_kv_cache=False,
                use_flash_attention=False,
                base_model=False,
                projection_layer=True,
                eye_init=True,
            )
        elif method == "kv_cache":
            config = DecoderConfig(
                model_name=model_name,
                use_kv_cache=True,
                use_flash_attention=False,
                base_model=False,
                projection_layer=True,
                eye_init=True,
            )
        else:  # flash
            config = DecoderConfig(
                model_name=model_name,
                use_kv_cache=False,
                use_flash_attention=True,
                base_model=False,
                projection_layer=True,
                eye_init=True,
            )
        
        decoder = Decoder(config).to(device)
        encoder = Encoder(EncoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
            eye_init=True,
        )).to(device)
        
        decoder.set_prompt("explain <embed>:", tokenizer)
        
        # Create activation
        set_random_seed(123)
        activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        # Forward pass
        if method == "naive":
            gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
        elif method == "kv_cache":
            gen = decoder.generate_soft_kv_cached(activation, max_length=seq_length, gumbel_tau=1.0)
        else:  # flash
            gen = decoder.generate_soft_kv_flash(activation, max_length=seq_length, gumbel_tau=1.0)
        
        # Simple loss
        reconstructed = encoder(gen.generated_text_embeddings)
        loss = torch.nn.functional.mse_loss(reconstructed, activation.detach())
        
        # Backward
        loss.backward()
        
        # Store gradient info
        method_gradients[method] = {
            'activation_grad': activation.grad.clone(),
            'decoder_proj_grad': decoder.proj.weight.grad.clone() if decoder.proj.weight.grad is not None else None,
            'encoder_proj_grad': encoder.proj.weight.grad.clone() if encoder.proj.weight.grad is not None else None,
        }
        
        print(f"\n{method.upper()}:")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Activation grad norm: {activation.grad.norm().item():.6f}")
        
        # Cleanup
        del decoder, encoder
    
    # Compare gradients
    if len(method_gradients) > 1 and 'naive' in method_gradients:
        print("\nGradient Comparison (vs naive):")
        
        for method in ['kv_cache', 'flash']:
            if method in method_gradients:
                # Compare activation gradients
                act_grad_diff = (method_gradients[method]['activation_grad'] - 
                               method_gradients['naive']['activation_grad']).norm()
                act_grad_norm = method_gradients['naive']['activation_grad'].norm()
                relative_diff = (act_grad_diff / act_grad_norm * 100).item()
                
                print(f"\n  {method}:")
                print(f"    Activation gradient difference: {relative_diff:.1f}%")
                
                if relative_diff < 10:  # Less than 10% difference
                    print(f"    ✓ Gradients are consistent")
                else:
                    print(f"    ✗ Significant gradient difference")


if __name__ == "__main__":
    test_functional_equivalence()
    test_gradient_consistency()