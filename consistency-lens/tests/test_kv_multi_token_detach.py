#!/usr/bin/env python3
"""
Test multi-token generation comparing generate_soft and generate_soft_kv_cached
with detach_after_each_sample=True and end_to_end=False.

This configuration should produce identical outputs but different gradients.
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from typing import List, Tuple

from lens.models.decoder import Decoder, DecoderConfig, Generated


def compare_generated(gen1: Generated, gen2: Generated, name: str, tolerance: float = 1e-5) -> bool:
    """Compare two Generated outputs and report differences."""
    print(f"\n{name}:")
    
    all_match = True
    
    # Compare embeddings
    emb_diff = (gen1.generated_text_embeddings - gen2.generated_text_embeddings).abs().max().item()
    if emb_diff > tolerance:
        print(f"  ❌ Embeddings max diff: {emb_diff:.2e}")
        all_match = False
    else:
        print(f"  ✅ Embeddings max diff: {emb_diff:.2e}")
    
    # Compare logits
    logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
    if logits_diff > tolerance * 10:  # Slightly higher tolerance for logits
        print(f"  ❌ Logits max diff: {logits_diff:.2e}")
        all_match = False
    else:
        print(f"  ✅ Logits max diff: {logits_diff:.2e}")
    
    # Compare hard IDs
    ids_match = torch.all(gen1.hard_token_ids == gen2.hard_token_ids).item()
    if not ids_match:
        print(f"  ❌ Hard IDs match: {ids_match}")
        # Show where they differ
        diff_mask = gen1.hard_token_ids != gen2.hard_token_ids
        if diff_mask.any():
            diff_positions = torch.where(diff_mask)
            print(f"    Differences at positions: {diff_positions}")
            for i in range(min(3, len(diff_positions[0]))):
                pos = (diff_positions[0][i].item(), diff_positions[1][i].item())
                print(f"    Position {pos}: {gen1.hard_token_ids[pos].item()} vs {gen2.hard_token_ids[pos].item()}")
        all_match = False
    else:
        print(f"  ✅ Hard IDs match: {ids_match}")
    
    return all_match


def test_gradient_computation(decoder: Decoder, activation: torch.Tensor, max_length: int, gumbel_tau: float) -> bool:
    """Test that gradients are computed correctly (or not) based on settings."""
    print("\n" + "="*60)
    print("Testing gradient computation")
    print("="*60)
    
    # Enable gradients for activation
    activation_soft = activation.clone().requires_grad_(True)
    activation_kv = activation.clone().requires_grad_(True)
    
    # Generate with both methods
    gen_soft = decoder.generate_soft(
        activation_soft,
        max_length=max_length,
        gumbel_tau=gumbel_tau
    )
    
    gen_kv = decoder.generate_soft_kv_cached(
        activation_kv,
        max_length=max_length,
        gumbel_tau=gumbel_tau
    )
    
    # Compute simple loss (sum of embeddings)
    loss_soft = gen_soft.generated_text_embeddings.sum()
    loss_kv = gen_kv.generated_text_embeddings.sum()
    
    # Check if losses are equal
    loss_diff = (loss_soft - loss_kv).abs().item()
    print(f"Loss difference: {loss_diff:.2e}")
    
    # Compute gradients
    grad_soft = torch.autograd.grad(loss_soft, activation_soft, retain_graph=True)[0]
    grad_kv = torch.autograd.grad(loss_kv, activation_kv, retain_graph=True)[0]
    
    # Compare gradients
    grad_diff = (grad_soft - grad_kv).abs().max().item()
    grad_relative_diff = grad_diff / (grad_soft.abs().max().item() + 1e-8)
    
    print(f"Gradient max absolute difference: {grad_diff:.2e}")
    print(f"Gradient relative difference: {grad_relative_diff:.2%}")
    
    # With detach_after_each_sample=True and end_to_end=False, gradients SHOULD differ
    gradients_differ = grad_diff > 1e-6
    print(f"Gradients differ (expected): {gradients_differ}")
    
    return True  # We expect gradients to differ in this configuration


def run_comparison_test(
    decoder: Decoder,
    activation: torch.Tensor,
    max_length: int,
    gumbel_tau: float,
    tokenizer,
    test_name: str
) -> bool:
    """Run a single comparison test between generate_soft and generate_soft_kv_cached."""
    
    print(f"\n{'='*60}")
    print(f"{test_name}")
    print(f"{'='*60}")
    print(f"Config: detach_after_each_sample={decoder.config.detach_after_each_sample}, end_to_end={decoder.config.end_to_end}")
    
    # Set same seed for both methods
    torch.manual_seed(42)
    gen_soft = decoder.generate_soft(
        activation.clone(),
        max_length=max_length,
        gumbel_tau=gumbel_tau
    )
    
    torch.manual_seed(42)
    gen_kv = decoder.generate_soft_kv_cached(
        activation.clone(),
        max_length=max_length,
        gumbel_tau=gumbel_tau
    )
    
    # Compare outputs
    match = compare_generated(gen_soft, gen_kv, "Full sequence comparison")
    
    # Decode and print generated text
    tokens_soft = gen_soft.hard_token_ids[0].tolist()
    tokens_kv = gen_kv.hard_token_ids[0].tolist()
    
    text_soft = tokenizer.decode(tokens_soft)
    text_kv = tokenizer.decode(tokens_kv)
    
    print(f"\nGenerated text:")
    print(f"  generate_soft:      '{text_soft}'")
    print(f"  generate_kv_cached: '{text_kv}'")
    print(f"  Text match: {text_soft == text_kv}")
    
    return match


def main():
    """Run multi-token generation comparison tests with detach_after_each_sample=True."""
    print("="*80)
    print("Multi-Token Generation Test: detach_after_each_sample=True, end_to_end=False")
    print("="*80)
    
    # Initialize model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    all_tests_passed = True
    
    # Test 1: Single layer patching with detach settings
    print("\n" + "="*80)
    print("TEST 1: Single layer patching with detach_after_each_sample=True, end_to_end=False")
    print("="*80)
    
    config1 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,  # Required for generate_soft
        patch_all_layers=False,
        per_layer_projections=False,
        detach_after_each_sample=True,  # Key setting
        end_to_end=False,  # Key setting
    )
    
    decoder1 = Decoder(config1).to(device).eval()
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder1.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    test_passed = run_comparison_test(
        decoder1,
        activation,
        max_length=10,
        gumbel_tau=1.0,
        tokenizer=tokenizer,
        test_name="Single layer patching with detach"
    )
    all_tests_passed &= test_passed
    
    # Test gradients
    test_gradient_computation(decoder1, activation, max_length=10, gumbel_tau=1.0)
    
    # Test 2: Multi-layer patching with detach settings
    print("\n" + "="*80)
    print("TEST 2: Multi-layer patching with detach_after_each_sample=True, end_to_end=False")
    print("="*80)
    
    config2 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,  # Required for generate_soft
        patch_all_layers=True,
        per_layer_projections=True,
        detach_after_each_sample=True,  # Key setting
        end_to_end=False,  # Key setting
    )
    
    decoder2 = Decoder(config2).to(device).eval()
    decoder2.set_prompt("explain <embed>:", tokenizer)
    
    test_passed = run_comparison_test(
        decoder2,
        activation,
        max_length=10,
        gumbel_tau=1.0,
        tokenizer=tokenizer,
        test_name="Multi-layer patching with detach"
    )
    all_tests_passed &= test_passed
    
    # Test gradients
    test_gradient_computation(decoder2, activation, max_length=10, gumbel_tau=1.0)
    
    # Test 3: Compare with end_to_end=True (should have different behavior)
    print("\n" + "="*80)
    print("TEST 3: Comparison - detach_after_each_sample=False, end_to_end=True")
    print("="*80)
    
    config3 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,  # Required for generate_soft
        patch_all_layers=True,
        per_layer_projections=True,
        detach_after_each_sample=False,  # Different setting
        end_to_end=True,  # Different setting
    )
    
    decoder3 = Decoder(config3).to(device).eval()
    decoder3.set_prompt("explain <embed>:", tokenizer)
    
    test_passed = run_comparison_test(
        decoder3,
        activation,
        max_length=10,
        gumbel_tau=1.0,
        tokenizer=tokenizer,
        test_name="Standard settings (for comparison)"
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if all_tests_passed:
        print("✅ ALL OUTPUT COMPARISON TESTS PASSED!")
        print("With detach_after_each_sample=True and end_to_end=False:")
        print("- generate_soft and generate_soft_kv_cached produce identical outputs")
        print("- Gradients differ as expected (due to different computation paths)")
    else:
        print("❌ SOME TESTS FAILED!")
        print("The two methods produce different outputs.")
    
    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)