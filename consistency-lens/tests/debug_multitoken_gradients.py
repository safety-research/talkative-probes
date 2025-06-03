#!/usr/bin/env python3
"""Debug gradient mismatch for multi-token generation."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_multitoken():
    """Debug gradients for multi-token generation."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Debugging Multi-Token Gradient Mismatch")
    print("=" * 60)
    
    # Test with multi-layer patching
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    
    # Test gradient accumulation over tokens
    for num_tokens in [1, 2, 3]:
        print(f"\n\nTest with {num_tokens} token(s):")
        
        activation1 = torch.randn(1, d_model, device=device, requires_grad=True)
        activation2 = activation1.clone().detach().requires_grad_(True)
        
        # Generate tokens
        torch.manual_seed(42)
        gen1 = decoder.generate_soft(activation1, max_length=num_tokens, gumbel_tau=1.0)
        
        torch.manual_seed(42)
        gen2 = decoder.generate_soft_kv_cached(activation2, max_length=num_tokens, gumbel_tau=1.0)
        
        # Check outputs
        print(f"  Outputs match: {torch.allclose(gen1.raw_lm_logits, gen2.raw_lm_logits, atol=1e-4)}")
        print(f"  Token IDs soft: {gen1.hard_token_ids[0].tolist()}")
        print(f"  Token IDs kv:   {gen2.hard_token_ids[0].tolist()}")
        
        # Compute loss on all tokens
        loss1 = gen1.raw_lm_logits.sum()
        loss2 = gen2.raw_lm_logits.sum()
        
        # Backward
        loss1.backward()
        loss2.backward()
        
        grad_diff = (activation1.grad - activation2.grad).abs().max().item()
        grad_norm1 = activation1.grad.norm().item()
        grad_norm2 = activation2.grad.norm().item()
        
        print(f"  Gradient max diff: {grad_diff:.2e}")
        print(f"  Gradient norm 1: {grad_norm1:.2e}")
        print(f"  Gradient norm 2: {grad_norm2:.2e}")
        print(f"  Relative diff: {grad_diff / max(grad_norm1, grad_norm2) * 100:.2f}%")
    
    # Now let's check what happens with the computation graph
    print("\n\nChecking computation graph structure:")
    
    # Fresh activation
    activation = torch.randn(1, d_model, device=device, requires_grad=True)
    
    # Generate 2 tokens and inspect intermediate values
    torch.manual_seed(42)
    gen_soft = decoder.generate_soft(activation.clone(), max_length=2, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen_kv = decoder.generate_soft_kv_cached(activation.clone(), max_length=2, gumbel_tau=1.0)
    
    # Check if gradients are enabled on outputs
    print(f"\nGradient enabled on outputs:")
    print(f"  generate_soft logits: {gen_soft.raw_lm_logits.requires_grad}")
    print(f"  generate_kv logits: {gen_kv.raw_lm_logits.requires_grad}")
    print(f"  generate_soft embeddings: {gen_soft.generated_text_embeddings.requires_grad}")
    print(f"  generate_kv embeddings: {gen_kv.generated_text_embeddings.requires_grad}")
    
    # Test with a more realistic loss function
    print("\n\nTest with cross-entropy loss:")
    
    activation3 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation4 = activation3.clone().detach().requires_grad_(True)
    
    # Generate
    torch.manual_seed(42)
    gen3 = decoder.generate_soft(activation3, max_length=5, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen4 = decoder.generate_soft_kv_cached(activation4, max_length=5, gumbel_tau=1.0)
    
    # Create dummy targets
    target = torch.randint(0, decoder.base.config.vocab_size, (1, 5), device=device)
    
    # Cross entropy loss
    ce_loss = nn.CrossEntropyLoss()
    loss3 = ce_loss(gen3.raw_lm_logits[0], target[0])
    loss4 = ce_loss(gen4.raw_lm_logits[0], target[0])
    
    print(f"  CE Loss 1: {loss3.item():.4f}")
    print(f"  CE Loss 2: {loss4.item():.4f}")
    
    loss3.backward()
    loss4.backward()
    
    grad_diff_ce = (activation3.grad - activation4.grad).abs().max().item()
    print(f"  Gradient diff with CE loss: {grad_diff_ce:.2e}")
    
    # Check if the issue is specific to multi-layer patching
    print("\n\nComparing with/without multi-layer patching (2 tokens):")
    
    # Without patching
    config_no_patch = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,
        per_layer_projections=False,
    )
    
    decoder_no_patch = Decoder(config_no_patch).to(device)
    decoder_no_patch.set_prompt("explain <embed>:", tokenizer)
    
    activation5 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation6 = activation5.clone().detach().requires_grad_(True)
    
    torch.manual_seed(42)
    gen5 = decoder_no_patch.generate_soft(activation5, max_length=2, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen6 = decoder_no_patch.generate_soft_kv_cached(activation6, max_length=2, gumbel_tau=1.0)
    
    loss5 = gen5.raw_lm_logits.sum()
    loss6 = gen6.raw_lm_logits.sum()
    
    loss5.backward()
    loss6.backward()
    
    grad_diff_no_patch = (activation5.grad - activation6.grad).abs().max().item()
    print(f"  Without multi-layer patching: {grad_diff_no_patch:.2e}")


if __name__ == "__main__":
    debug_multitoken()