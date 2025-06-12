#!/usr/bin/env python3
"""
Simplified test for detach_after_each_sample=True and end_to_end=False configuration.
Tests that outputs match between generate_soft and generate_soft_kv_cached.
"""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig, Generated


def compare_outputs(gen1: Generated, gen2: Generated) -> tuple[bool, dict]:
    """Compare outputs and return match status and statistics."""
    stats = {}
    
    # Compare embeddings
    emb_diff = (gen1.generated_text_embeddings - gen2.generated_text_embeddings).abs().max().item()
    stats['emb_diff'] = emb_diff
    
    # Compare logits
    logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
    stats['logits_diff'] = logits_diff
    
    # Compare hard IDs
    ids_match = torch.all(gen1.hard_token_ids == gen2.hard_token_ids).item()
    stats['ids_match'] = ids_match
    
    # Overall match (with tolerance)
    match = emb_diff < 1e-5 and logits_diff < 1e-3 and ids_match
    
    return match, stats


def test_configuration(config_name: str, **config_kwargs):
    """Test a specific decoder configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")
    
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create decoder with specified config
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,  # Required for generate_soft
        **config_kwargs
    )
    
    decoder = Decoder(config).to(device).eval()
    decoder.set_prompt("The answer is <embed>:", tokenizer)
    
    # Test parameters
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    max_length = 10
    gumbel_tau = 1.0
    
    # Generate with both methods
    torch.manual_seed(42)
    gen_soft = decoder.generate_soft(activation, max_length, gumbel_tau)
    
    torch.manual_seed(42)
    gen_kv = decoder.generate_soft_kv_cached(activation, max_length, gumbel_tau)
    
    # Compare
    match, stats = compare_outputs(gen_soft, gen_kv)
    
    # Print results
    print(f"Config settings:")
    for key, value in config_kwargs.items():
        print(f"  {key}: {value}")
    
    print(f"\nResults:")
    print(f"  Embeddings diff: {stats['emb_diff']:.2e}")
    print(f"  Logits diff: {stats['logits_diff']:.2e}")
    print(f"  Hard IDs match: {stats['ids_match']}")
    print(f"  Overall match: {'✅' if match else '❌'}")
    
    # Show generated text
    text_soft = tokenizer.decode(gen_soft.hard_token_ids[0].tolist())
    text_kv = tokenizer.decode(gen_kv.hard_token_ids[0].tolist())
    print(f"\nGenerated text:")
    print(f"  generate_soft:      '{text_soft}'")
    print(f"  generate_kv_cached: '{text_kv}'")
    
    # Test gradients if relevant
    if config_kwargs.get('detach_after_each_sample', False) and not config_kwargs.get('end_to_end', True):
        print(f"\nGradient test (expecting difference):")
        activation_soft = activation.clone().requires_grad_(True)
        activation_kv = activation.clone().requires_grad_(True)
        
        torch.manual_seed(42)
        gen_soft = decoder.generate_soft(activation_soft, max_length, gumbel_tau)
        torch.manual_seed(42)
        gen_kv = decoder.generate_soft_kv_cached(activation_kv, max_length, gumbel_tau)
        
        loss_soft = gen_soft.generated_text_embeddings.sum()
        loss_kv = gen_kv.generated_text_embeddings.sum()
        
        grad_soft = torch.autograd.grad(loss_soft, activation_soft)[0]
        grad_kv = torch.autograd.grad(loss_kv, activation_kv)[0]
        
        grad_diff = (grad_soft - grad_kv).abs().max().item()
        print(f"  Gradient difference: {grad_diff:.2e} {'(expected)' if grad_diff > 1e-6 else '(unexpected)'}")
    
    # Cleanup to save memory
    del decoder
    torch.cuda.empty_cache()
    
    return match


def main():
    """Run all test configurations."""
    print("KV Cache Test: detach_after_each_sample and end_to_end configurations")
    
    all_passed = True
    
    # Test 1: Default configuration (baseline)
    all_passed &= test_configuration(
        "Default (end_to_end=True, detach=False)",
        patch_all_layers=False,
        per_layer_projections=False,
        detach_after_each_sample=False,
        end_to_end=True,
    )
    
    # Test 2: Detach configuration
    all_passed &= test_configuration(
        "Detach mode (end_to_end=False, detach=True)",
        patch_all_layers=False,
        per_layer_projections=False,
        detach_after_each_sample=True,
        end_to_end=False,
    )
    
    # Test 3: Multi-layer with detach
    all_passed &= test_configuration(
        "Multi-layer + Detach mode",
        patch_all_layers=True,
        per_layer_projections=True,
        detach_after_each_sample=True,
        end_to_end=False,
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if all_passed:
        print("✅ All tests passed! Outputs match between generate_soft and generate_soft_kv_cached")
    else:
        print("❌ Some tests failed!")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())