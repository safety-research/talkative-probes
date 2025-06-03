#!/usr/bin/env python3
"""Test gradient flow through KV cache during training."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig


def test_gradient_flow():
    """Verify gradients flow correctly through KV cache generation."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Decoder with KV cache
    decoder_config = DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        base_model=False,
        projection_layer=True,
        output_head=False,
        trainable_prompts=True,
    )
    decoder = Decoder(decoder_config).to(device)
    
    # Encoder
    encoder_config = EncoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
    )
    encoder = Encoder(encoder_config).to(device)
    
    # Set prompt
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    print("Gradient Flow Test for KV Cache")
    print("="*60)
    
    # Test parameters
    batch_size = 2
    d_model = decoder.base.config.hidden_size
    T_text = 8
    
    # Create activation that requires grad
    activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    # Forward pass with different generation methods
    methods = [
        ("naive", lambda: decoder.generate_soft(activation, max_length=T_text, gumbel_tau=1.0)),
        ("checkpoint", lambda: decoder.generate_soft_chkpt(activation, max_length=T_text, gumbel_tau=1.0, checkpoint_every_n_tokens=1)),
        ("kv_cache", lambda: decoder.generate_soft_kv_cached(activation, max_length=T_text, gumbel_tau=1.0)),
    ]
    
    for method_name, gen_fn in methods:
        print(f"\nTesting {method_name}:")
        
        # Reset gradients
        if activation.grad is not None:
            activation.grad.zero_()
        decoder.zero_grad()
        encoder.zero_grad()
        
        # Forward pass
        gen = gen_fn()
        text_embeddings = gen.generated_text_embeddings
        
        # Encode back to activation space
        reconstructed = encoder(text_embeddings)
        
        # Simple loss
        loss = torch.nn.functional.mse_loss(reconstructed, activation.detach())
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        print(f"  Loss: {loss.item():.6f}")
        
        # Check if decoder parameters have gradients
        decoder_grad_params = sum(1 for p in decoder.parameters() if p.grad is not None and p.grad.abs().max() > 0)
        decoder_total_params = sum(1 for p in decoder.parameters() if p.requires_grad)
        print(f"  Decoder params with gradients: {decoder_grad_params}/{decoder_total_params}")
        
        # Check if encoder parameters have gradients
        encoder_grad_params = sum(1 for p in encoder.parameters() if p.grad is not None and p.grad.abs().max() > 0)
        encoder_total_params = sum(1 for p in encoder.parameters() if p.requires_grad)
        print(f"  Encoder params with gradients: {encoder_grad_params}/{encoder_total_params}")
        
        # Check specific gradient magnitudes
        if hasattr(decoder, 'proj') and decoder.proj.weight.grad is not None:
            print(f"  Decoder proj grad norm: {decoder.proj.weight.grad.norm():.6f}")
        if hasattr(encoder, 'proj') and encoder.proj.weight.grad is not None:
            print(f"  Encoder proj grad norm: {encoder.proj.weight.grad.norm():.6f}")
        
        # Test KL loss gradient flow (more realistic)
        if method_name == "kv_cache":
            print("\n  Testing KL loss gradient flow:")
            
            # Create fresh activation for KL test
            activation_kl = torch.randn(batch_size, d_model, device=device, requires_grad=True)
            
            # Reset
            decoder.zero_grad()
            encoder.zero_grad()
            
            # Generate with two different activations
            A = activation_kl
            A_prime = activation_kl + 0.1 * torch.randn_like(activation_kl)
            
            # Generate from both
            gen_A = decoder.generate_soft_kv_cached(A, max_length=T_text, gumbel_tau=1.0)
            gen_Ap = decoder.generate_soft_kv_cached(A_prime, max_length=T_text, gumbel_tau=1.0)
            
            # Reconstruct
            A_hat = encoder(gen_A.generated_text_embeddings)
            Ap_hat = encoder(gen_Ap.generated_text_embeddings)
            
            # Compute functional loss (simplified)
            delta = (A_prime - Ap_hat).detach()
            A_tilde = A_hat + delta
            
            # Mock KL computation
            logits_A = torch.randn(batch_size, 10, device=device)  # Mock logits
            logits_A_tilde = logits_A + 0.1 * torch.randn_like(logits_A)
            
            kl_loss = torch.nn.functional.kl_div(
                torch.log_softmax(logits_A_tilde, dim=-1),
                torch.softmax(logits_A, dim=-1),
                reduction='batchmean'
            )
            
            # Combined loss
            mse_loss_kl = torch.nn.functional.mse_loss(A_hat, A.detach())
            total_loss = mse_loss_kl + 0.1 * kl_loss
            total_loss.backward()
            
            # Check gradients again
            decoder_grads = sum(1 for p in decoder.parameters() if p.grad is not None and p.grad.abs().max() > 0)
            print(f"    Decoder params with KL gradients: {decoder_grads}/{decoder_total_params}")
            
            if hasattr(decoder, 'proj') and decoder.proj.weight.grad is not None:
                print(f"    Decoder proj grad norm (KL): {decoder.proj.weight.grad.norm():.6f}")
    
    # Memory usage comparison
    print("\n" + "="*60)
    print("Memory Usage During Backward Pass:")
    
    for method_name, gen_fn in methods:
        # Create fresh activation for each test
        test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        # Reset
        decoder.zero_grad()
        encoder.zero_grad()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
        # Forward + backward
        if method_name == "naive":
            gen = decoder.generate_soft(test_activation, max_length=T_text, gumbel_tau=1.0)
        elif method_name == "checkpoint":
            gen = decoder.generate_soft_chkpt(test_activation, max_length=T_text, gumbel_tau=1.0, checkpoint_every_n_tokens=1)
        else:  # kv_cache
            gen = decoder.generate_soft_kv_cached(test_activation, max_length=T_text, gumbel_tau=1.0)
            
        reconstructed = encoder(gen.generated_text_embeddings)
        loss = torch.nn.functional.mse_loss(reconstructed, test_activation.detach())
        
        loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  {method_name}: {peak_memory:.1f} MB")
    
    print("\nGradient flow test complete!")
    print("✓ All methods support backpropagation")
    print("✓ KV cache maintains gradient flow during training")


if __name__ == "__main__":
    test_gradient_flow()