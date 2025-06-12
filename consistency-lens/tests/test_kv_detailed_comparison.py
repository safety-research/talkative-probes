#!/usr/bin/env python3
"""
Detailed comparison of generate_soft vs generate_soft_kv_cached 
for multiple tokens with end_to_end=False and detach_after_each_sample=True.
"""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig, Generated


def detailed_token_comparison(decoder, activation, max_length, gumbel_tau, tokenizer):
    """Compare token generation step by step between both methods."""
    
    print("\nDetailed Token-by-Token Comparison")
    print("="*80)
    
    # Set seeds and generate
    torch.manual_seed(42)
    gen_soft = decoder.generate_soft(activation.clone(), max_length, gumbel_tau)
    
    torch.manual_seed(42)
    gen_kv = decoder.generate_soft_kv_cached(activation.clone(), max_length, gumbel_tau)
    
    # Extract token sequences
    tokens_soft = gen_soft.hard_token_ids[0].tolist()
    tokens_kv = gen_kv.hard_token_ids[0].tolist()
    
    # Compare logits at each position
    print("\nLogits comparison at each position:")
    for i in range(max_length):
        logits_soft_i = gen_soft.raw_lm_logits[0, i]
        logits_kv_i = gen_kv.raw_lm_logits[0, i]
        
        # Get top 5 predictions from each
        top5_soft = torch.topk(logits_soft_i, 5)
        top5_kv = torch.topk(logits_kv_i, 5)
        
        logits_diff = (logits_soft_i - logits_kv_i).abs().max().item()
        
        print(f"\nPosition {i}:")
        print(f"  Logits diff: {logits_diff:.2e}")
        print(f"  Selected token (soft): {tokens_soft[i]} = '{tokenizer.decode([tokens_soft[i]])}'")
        print(f"  Selected token (kv):   {tokens_kv[i]} = '{tokenizer.decode([tokens_kv[i]])}'")
        print(f"  Match: {'✅' if tokens_soft[i] == tokens_kv[i] else '❌'}")
        
        # Show top 5 predictions if they differ
        if tokens_soft[i] != tokens_kv[i]:
            print(f"  Top 5 (soft): {top5_soft.indices.tolist()}")
            print(f"  Top 5 (kv):   {top5_kv.indices.tolist()}")
    
    # Full sequence comparison
    text_soft = tokenizer.decode(tokens_soft)
    text_kv = tokenizer.decode(tokens_kv)
    
    print(f"\nFull sequences:")
    print(f"  generate_soft:      '{text_soft}'")
    print(f"  generate_kv_cached: '{text_kv}'")
    print(f"  Identical: {'✅' if text_soft == text_kv else '❌'}")
    
    return tokens_soft == tokens_kv


def test_multiple_seeds(decoder, activation, max_length, gumbel_tau, tokenizer, num_seeds=5):
    """Test with multiple random seeds to ensure consistency."""
    
    print("\nTesting with Multiple Seeds")
    print("="*80)
    
    all_match = True
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        gen_soft = decoder.generate_soft(activation.clone(), max_length, gumbel_tau)
        
        torch.manual_seed(seed)
        gen_kv = decoder.generate_soft_kv_cached(activation.clone(), max_length, gumbel_tau)
        
        tokens_soft = gen_soft.hard_token_ids[0].tolist()
        tokens_kv = gen_kv.hard_token_ids[0].tolist()
        
        match = tokens_soft == tokens_kv
        all_match &= match
        
        text_soft = tokenizer.decode(tokens_soft)
        text_kv = tokenizer.decode(tokens_kv)
        
        print(f"\nSeed {seed}: {'✅' if match else '❌'}")
        if not match:
            print(f"  Soft: '{text_soft}'")
            print(f"  KV:   '{text_kv}'")
    
    print(f"\nAll seeds match: {'✅' if all_match else '❌'}")
    return all_match


def test_different_lengths(decoder, activation, tokenizer, max_lengths=[5, 10, 20]):
    """Test generation with different sequence lengths."""
    
    print("\nTesting Different Sequence Lengths")
    print("="*80)
    
    all_match = True
    
    for length in max_lengths:
        torch.manual_seed(42)
        gen_soft = decoder.generate_soft(activation.clone(), length, gumbel_tau=1.0)
        
        torch.manual_seed(42)
        gen_kv = decoder.generate_soft_kv_cached(activation.clone(), length, gumbel_tau=1.0)
        
        tokens_soft = gen_soft.hard_token_ids[0].tolist()
        tokens_kv = gen_kv.hard_token_ids[0].tolist()
        
        match = tokens_soft == tokens_kv
        all_match &= match
        
        print(f"\nLength {length}: {'✅' if match else '❌'}")
        if not match:
            print(f"  First mismatch at position: {next(i for i in range(length) if tokens_soft[i] != tokens_kv[i])}")
    
    return all_match


def main():
    """Run comprehensive tests."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test configuration with end_to_end=False, detach_after_each_sample=True
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,
        patch_all_layers=False,
        per_layer_projections=False,
        detach_after_each_sample=True,
        end_to_end=False,
    )
    
    decoder = Decoder(config).to(device).eval()
    decoder.set_prompt("The answer is <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device, requires_grad=True)
    
    print("\nConfiguration:")
    print(f"  end_to_end: {config.end_to_end}")
    print(f"  detach_after_each_sample: {config.detach_after_each_sample}")
    
    # Test 1: Detailed token comparison
    detailed_match = detailed_token_comparison(decoder, activation, max_length=10, gumbel_tau=1.0, tokenizer=tokenizer)
    
    # Test 2: Multiple seeds
    seeds_match = test_multiple_seeds(decoder, activation, max_length=10, gumbel_tau=1.0, tokenizer=tokenizer)
    
    # Test 3: Different lengths
    lengths_match = test_different_lengths(decoder, activation, tokenizer=tokenizer)
    
    # Test 4: Check Gumbel noise is truly identical
    print("\nGumbel Noise Verification")
    print("="*80)
    
    # Generate multiple times with same seed
    results = []
    for _ in range(3):
        torch.manual_seed(42)
        gen = decoder.generate_soft(activation.clone(), max_length=5, gumbel_tau=1.0)
        results.append(gen.hard_token_ids[0].tolist())
    
    noise_consistent = all(r == results[0] for r in results)
    print(f"Gumbel noise consistency: {'✅' if noise_consistent else '❌'}")
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    all_tests_passed = detailed_match and seeds_match and lengths_match and noise_consistent
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("With end_to_end=False and detach_after_each_sample=True:")
        print("- Token sequences are IDENTICAL between generate_soft and generate_soft_kv_cached")
        print("- This holds across multiple seeds and sequence lengths")
        print("- Gumbel noise is deterministic with fixed seed")
    else:
        print("❌ SOME TESTS FAILED!")
        print("The methods produce different outputs.")
    
    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    exit(main())