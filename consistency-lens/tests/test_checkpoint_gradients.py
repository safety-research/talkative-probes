#!/usr/bin/env python3
"""Test gradient correctness with checkpointing."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_simple_checkpoint_gradient():
    """Test gradient checkpointing with a simple example."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Simple Gradient Checkpointing Test")
    print("=" * 60)
    
    # Simple test function
    def simple_forward(x, w):
        """Simple forward that can be checkpointed."""
        y = x @ w
        z = F.relu(y)
        return z.sum()
    
    # Test data
    torch.manual_seed(42)
    x = torch.randn(10, 20, device=device, requires_grad=True)
    w = torch.randn(20, 30, device=device, requires_grad=True)
    
    # Normal forward-backward
    x1 = x.clone().detach().requires_grad_(True)
    w1 = w.clone().detach().requires_grad_(True)
    
    loss1 = simple_forward(x1, w1)
    loss1.backward()
    
    # Checkpointed forward-backward
    x2 = x.clone().detach().requires_grad_(True)
    w2 = w.clone().detach().requires_grad_(True)
    
    from torch.utils.checkpoint import checkpoint
    loss2 = checkpoint(simple_forward, x2, w2, use_reentrant=False)
    loss2.backward()
    
    # Compare gradients
    x_grad_diff = (x1.grad - x2.grad).abs().max().item()
    w_grad_diff = (w1.grad - w2.grad).abs().max().item()
    
    print(f"X gradient diff: {x_grad_diff:.2e}")
    print(f"W gradient diff: {w_grad_diff:.2e}")
    
    if x_grad_diff < 1e-6 and w_grad_diff < 1e-6:
        print("✓ Simple checkpointing works correctly!")
    else:
        print("✗ Simple checkpointing has issues!")


def test_decoder_checkpoint_minimal():
    """Minimal test of decoder checkpointing."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nDecoder Checkpoint Minimal Test")
    print("=" * 60)
    
    # Simple config
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,  # Start without multi-layer
    )
    
    # Create decoder
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    
    # Test with single token generation
    print("\nTest 1: Single token, no multi-layer patching")
    
    torch.manual_seed(42)
    activation1 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation2 = activation1.detach().clone().requires_grad_(True)
    
    # Generate
    gen1 = decoder.generate_soft(activation1, max_length=1, gumbel_tau=1.0)
    gen2 = decoder.generate_soft_chkpt(activation2, max_length=1, gumbel_tau=1.0, checkpoint_every_n_tokens=1)
    
    # Check outputs
    emb_same = torch.allclose(gen1.generated_text_embeddings, gen2.generated_text_embeddings, atol=1e-6)
    print(f"  Embeddings match: {emb_same}")
    
    # Backward
    loss1 = gen1.generated_text_embeddings.sum()
    loss2 = gen2.generated_text_embeddings.sum()
    
    loss1.backward()
    loss2.backward()
    
    # Compare gradients
    act_grad_diff = (activation1.grad - activation2.grad).abs().max().item()
    proj_grad_diff = (decoder.proj.weight.grad - decoder.proj.weight.grad).abs().max().item()  # Same decoder
    
    print(f"  Activation grad diff: {act_grad_diff:.2e}")
    
    # Now test with multi-layer patching
    print("\nTest 2: Single token, WITH multi-layer patching")
    
    config_multi = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder_multi = Decoder(config_multi).to(device)
    decoder_multi.set_prompt("explain <embed>:", tokenizer)
    
    # Must zero gradients
    decoder_multi.zero_grad()
    
    activation3 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation4 = activation3.detach().clone().requires_grad_(True)
    
    # Generate
    gen3 = decoder_multi.generate_soft(activation3, max_length=1, gumbel_tau=1.0)
    gen4 = decoder_multi.generate_soft_chkpt(activation4, max_length=1, gumbel_tau=1.0, checkpoint_every_n_tokens=1)
    
    # Check outputs
    emb_same_multi = torch.allclose(gen3.generated_text_embeddings, gen4.generated_text_embeddings, atol=1e-6)
    print(f"  Embeddings match: {emb_same_multi}")
    
    # Backward
    loss3 = gen3.generated_text_embeddings.sum()
    loss4 = gen4.generated_text_embeddings.sum()
    
    loss3.backward()
    loss4.backward()
    
    # Compare gradients
    act_grad_diff_multi = (activation3.grad - activation4.grad).abs().max().item()
    
    # Need to get projection gradients from different places
    gen3_proj_grad = decoder_multi.proj.weight.grad.clone()
    decoder_multi.zero_grad()  # Clear before second backward
    
    # Redo for clean gradient
    activation4_clean = activation3.detach().clone().requires_grad_(True)
    gen4_clean = decoder_multi.generate_soft_chkpt(activation4_clean, max_length=1, gumbel_tau=1.0, checkpoint_every_n_tokens=1)
    loss4_clean = gen4_clean.generated_text_embeddings.sum()
    loss4_clean.backward()
    gen4_proj_grad = decoder_multi.proj.weight.grad.clone()
    
    proj_grad_diff_multi = (gen3_proj_grad - gen4_proj_grad).abs().max().item()
    
    print(f"  Activation grad diff: {act_grad_diff_multi:.2e}")
    print(f"  Projection grad diff: {proj_grad_diff_multi:.2e}")
    
    if act_grad_diff_multi < 1e-5:
        print("  ✓ Activation gradients match well!")
    else:
        print("  ✗ Activation gradients differ significantly!")


if __name__ == "__main__":
    test_simple_checkpoint_gradient()
    test_decoder_checkpoint_minimal()