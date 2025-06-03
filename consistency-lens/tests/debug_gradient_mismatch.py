#!/usr/bin/env python3
"""Debug gradient mismatch between generate_soft and generate_soft_kv_cached."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def trace_gradients():
    """Trace gradient computation through both methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Debugging Gradient Mismatch")
    print("=" * 60)
    
    # Simple test case with single token generation
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
    
    # Test with single token first
    print("\nTest 1: Single token generation")
    activation1 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation2 = activation1.clone().detach().requires_grad_(True)
    
    # Generate single token
    torch.manual_seed(42)
    gen1 = decoder.generate_soft(activation1, max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen2 = decoder.generate_soft_kv_cached(activation2, max_length=1, gumbel_tau=1.0)
    
    # Check outputs match
    print(f"Outputs match: {torch.allclose(gen1.raw_lm_logits, gen2.raw_lm_logits, atol=1e-4)}")
    
    # Simple loss
    loss1 = gen1.raw_lm_logits.sum()
    loss2 = gen2.raw_lm_logits.sum()
    
    print(f"Loss 1: {loss1.item():.4f}")
    print(f"Loss 2: {loss2.item():.4f}")
    
    # Compute gradients
    loss1.backward(retain_graph=True)
    loss2.backward(retain_graph=True)
    
    grad_diff = (activation1.grad - activation2.grad).abs().max().item()
    print(f"Gradient max diff: {grad_diff:.2e}")
    
    # Now let's trace where the difference comes from
    print("\n\nTest 2: Detailed gradient analysis")
    
    # Clear gradients
    activation1.grad = None
    activation2.grad = None
    decoder.zero_grad()
    
    # Let's check the gradient flow through embeddings
    gen1_emb = gen1.generated_text_embeddings
    gen2_emb = gen2.generated_text_embeddings
    
    print(f"Embeddings match: {torch.allclose(gen1_emb, gen2_emb, atol=1e-5)}")
    
    # Gradient through embeddings
    dummy_target = torch.randn_like(gen1_emb)
    loss1_emb = ((gen1_emb - dummy_target) ** 2).sum()
    loss2_emb = ((gen2_emb - dummy_target) ** 2).sum()
    
    loss1_emb.backward()
    loss2_emb.backward()
    
    grad_diff_emb = (activation1.grad - activation2.grad).abs().max().item()
    print(f"Gradient diff through embeddings: {grad_diff_emb:.2e}")
    
    # Test 3: Check if the issue is with multi-layer patching
    print("\n\nTest 3: Without multi-layer patching")
    
    config_simple = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,  # Disabled
        per_layer_projections=False,
    )
    
    decoder_simple = Decoder(config_simple).to(device)
    decoder_simple.set_prompt("explain <embed>:", tokenizer)
    
    activation3 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation4 = activation3.clone().detach().requires_grad_(True)
    
    torch.manual_seed(42)
    gen3 = decoder_simple.generate_soft(activation3, max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen4 = decoder_simple.generate_soft_kv_cached(activation4, max_length=1, gumbel_tau=1.0)
    
    loss3 = gen3.raw_lm_logits.sum()
    loss4 = gen4.raw_lm_logits.sum()
    
    loss3.backward()
    loss4.backward()
    
    grad_diff_simple = (activation3.grad - activation4.grad).abs().max().item()
    print(f"Gradient diff without multi-layer patching: {grad_diff_simple:.2e}")
    
    # Test 4: Check projection layer gradients
    print("\n\nTest 4: Projection layer gradient analysis")
    
    # Reset
    decoder.zero_grad()
    
    # Track projection usage
    activation5 = torch.randn(1, d_model, device=device, requires_grad=True)
    
    # Hook to capture projection gradients
    proj_grads_soft = []
    proj_grads_kv = []
    
    def capture_grad_soft(grad):
        proj_grads_soft.append(grad.clone())
        return grad
    
    def capture_grad_kv(grad):
        proj_grads_kv.append(grad.clone())
        return grad
    
    # First with generate_soft
    if decoder.proj is not None:
        handle1 = decoder.proj.weight.register_hook(capture_grad_soft)
    
    gen5 = decoder.generate_soft(activation5, max_length=2, gumbel_tau=1.0)
    loss5 = gen5.raw_lm_logits.sum()
    loss5.backward()
    
    if decoder.proj is not None:
        handle1.remove()
    
    # Now with KV cached
    decoder.zero_grad()
    activation6 = activation5.clone().detach().requires_grad_(True)
    
    if decoder.proj is not None:
        handle2 = decoder.proj.weight.register_hook(capture_grad_kv)
    
    torch.manual_seed(42)  # Reset seed
    gen6 = decoder.generate_soft_kv_cached(activation6, max_length=2, gumbel_tau=1.0)
    loss6 = gen6.raw_lm_logits.sum()
    loss6.backward()
    
    if decoder.proj is not None:
        handle2.remove()
    
    print(f"Number of projection gradient captures (soft): {len(proj_grads_soft)}")
    print(f"Number of projection gradient captures (kv): {len(proj_grads_kv)}")
    
    if len(proj_grads_soft) > 0 and len(proj_grads_kv) > 0:
        for i, (g1, g2) in enumerate(zip(proj_grads_soft, proj_grads_kv)):
            diff = (g1 - g2).abs().max().item()
            print(f"  Projection gradient diff {i}: {diff:.2e}")


if __name__ == "__main__":
    trace_gradients()