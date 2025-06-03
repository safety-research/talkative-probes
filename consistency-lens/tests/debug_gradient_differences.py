#!/usr/bin/env python3
"""Debug gradient differences between generate_soft and generate_soft_chkpt."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_gradient_differences():
    """Debug why gradients differ between methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Debugging Gradient Differences")
    print("=" * 60)
    
    # Test with multi-layer patching
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    # Create two identical decoders
    decoder1 = Decoder(config).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    decoder2 = Decoder(config).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    decoder2.load_state_dict(decoder1.state_dict())
    
    d_model = decoder1.base.config.hidden_size
    batch_size = 2
    seq_length = 8
    
    # Fixed seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Create activation
    activation1 = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    activation2 = activation1.detach().clone().requires_grad_(True)
    
    # Generate with both methods
    print("\nGenerating with both methods...")
    gen1 = decoder1.generate_soft(activation1, max_length=seq_length, gumbel_tau=1.0)
    gen2 = decoder2.generate_soft_chkpt(activation2, max_length=seq_length, gumbel_tau=1.0)
    
    # Verify outputs are identical
    emb_diff = (gen1.generated_text_embeddings - gen2.generated_text_embeddings).abs().max().item()
    logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
    ids_same = torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)
    
    print(f"\nOutput comparison:")
    print(f"  Embeddings max diff: {emb_diff:.2e}")
    print(f"  Logits max diff: {logits_diff:.2e}")
    print(f"  Hard IDs identical: {ids_same}")
    
    # Compute identical losses
    loss1 = gen1.generated_text_embeddings.sum()
    loss2 = gen2.generated_text_embeddings.sum()
    
    print(f"\nLoss comparison:")
    print(f"  Loss 1: {loss1.item():.6f}")
    print(f"  Loss 2: {loss2.item():.6f}")
    print(f"  Loss diff: {abs(loss1.item() - loss2.item()):.2e}")
    
    # Backward pass
    loss1.backward()
    loss2.backward()
    
    # Compare activation gradients
    act_grad_diff = (activation1.grad - activation2.grad).abs()
    print(f"\nActivation gradient comparison:")
    print(f"  Max diff: {act_grad_diff.max().item():.2e}")
    print(f"  Mean diff: {act_grad_diff.mean().item():.2e}")
    print(f"  Relative diff: {(act_grad_diff / (activation1.grad.abs() + 1e-8)).mean().item():.2e}")
    
    # Compare projection gradients
    if config.per_layer_projections:
        proj_grad1 = decoder1.proj_weight.grad
        proj_grad2 = decoder2.proj_weight.grad
    else:
        proj_grad1 = decoder1.proj.weight.grad
        proj_grad2 = decoder2.proj.weight.grad
    
    proj_grad_diff = (proj_grad1 - proj_grad2).abs()
    print(f"\nProjection gradient comparison:")
    print(f"  Max diff: {proj_grad_diff.max().item():.2e}")
    print(f"  Mean diff: {proj_grad_diff.mean().item():.2e}")
    print(f"  Relative diff: {(proj_grad_diff / (proj_grad1.abs() + 1e-8)).mean().item():.2e}")
    
    # Check gradient norms
    print(f"\nGradient norms:")
    print(f"  Activation grad norm 1: {activation1.grad.norm().item():.4f}")
    print(f"  Activation grad norm 2: {activation2.grad.norm().item():.4f}")
    print(f"  Projection grad norm 1: {proj_grad1.norm().item():.4f}")
    print(f"  Projection grad norm 2: {proj_grad2.norm().item():.4f}")
    
    # Test without multi-layer patching as baseline
    print("\n\nTesting WITHOUT multi-layer patching (baseline):")
    config_base = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,  # No multi-layer patching
    )
    
    decoder3 = Decoder(config_base).to(device)
    decoder3.set_prompt("explain <embed>:", tokenizer)
    
    decoder4 = Decoder(config_base).to(device)
    decoder4.set_prompt("explain <embed>:", tokenizer)
    decoder4.load_state_dict(decoder3.state_dict())
    
    # Create activation
    activation3 = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    activation4 = activation3.detach().clone().requires_grad_(True)
    
    # Generate
    gen3 = decoder3.generate_soft(activation3, max_length=seq_length, gumbel_tau=1.0)
    gen4 = decoder4.generate_soft_chkpt(activation4, max_length=seq_length, gumbel_tau=1.0)
    
    # Losses
    loss3 = gen3.generated_text_embeddings.sum()
    loss4 = gen4.generated_text_embeddings.sum()
    
    loss3.backward()
    loss4.backward()
    
    # Compare
    act_grad_diff_base = (activation3.grad - activation4.grad).abs().max().item()
    proj_grad_diff_base = (decoder3.proj.weight.grad - decoder4.proj.weight.grad).abs().max().item()
    
    print(f"  Activation grad diff: {act_grad_diff_base:.2e}")
    print(f"  Projection grad diff: {proj_grad_diff_base:.2e}")
    
    if proj_grad_diff_base < 1e-5:
        print(f"  ✓ Baseline gradients match well!")
    else:
        print(f"  ✗ Even baseline has gradient differences!")


def test_gradient_accumulation():
    """Test if gradient accumulation is the issue."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nTesting Gradient Accumulation")
    print("=" * 60)
    
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
    
    # Test 1: Single forward-backward
    print("\nTest 1: Single forward-backward")
    decoder.zero_grad()
    activation1 = torch.randn(1, d_model, device=device, requires_grad=True)
    gen1 = decoder.generate_soft(activation1, max_length=4, gumbel_tau=1.0)
    loss1 = gen1.generated_text_embeddings.sum()
    loss1.backward()
    
    proj_grad1 = decoder.proj.weight.grad.clone()
    print(f"  Projection grad norm: {proj_grad1.norm().item():.4f}")
    
    # Test 2: Checkpointed
    print("\nTest 2: Checkpointed")
    decoder.zero_grad()
    activation2 = activation1.detach().clone().requires_grad_(True)
    gen2 = decoder.generate_soft_chkpt(activation2, max_length=4, gumbel_tau=1.0)
    loss2 = gen2.generated_text_embeddings.sum()
    loss2.backward()
    
    proj_grad2 = decoder.proj.weight.grad.clone()
    print(f"  Projection grad norm: {proj_grad2.norm().item():.4f}")
    
    diff = (proj_grad1 - proj_grad2).abs().max().item()
    print(f"  Gradient diff: {diff:.2e}")


if __name__ == "__main__":
    debug_gradient_differences()
    test_gradient_accumulation()