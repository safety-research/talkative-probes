#!/usr/bin/env python3
"""
Test multi-token generation comparing generate_soft and generate_soft_kv_cached.
This test generates multiple tokens and compares outputs at each step.
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
    
    # Step-by-step comparison
    print(f"\nStep-by-step token comparison:")
    for i in range(max_length):
        token_soft = tokens_soft[i] if i < len(tokens_soft) else -1
        token_kv = tokens_kv[i] if i < len(tokens_kv) else -1
        match_symbol = "✅" if token_soft == token_kv else "❌"
        
        token_text_soft = tokenizer.decode([token_soft]) if token_soft != -1 else "N/A"
        token_text_kv = tokenizer.decode([token_kv]) if token_kv != -1 else "N/A"
        
        print(f"  Step {i}: {match_symbol} {token_soft} ('{token_text_soft}') vs {token_kv} ('{token_text_kv}')")
    
    return match


def main():
    """Run multi-token generation comparison tests."""
    print("="*80)
    print("Multi-Token Generation Comparison Test")
    print("="*80)
    
    # Initialize model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    all_tests_passed = True
    
    # Test 1: Without multi-layer patching
    config1 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,
        per_layer_projections=False,
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
        test_name="Test 1: Without multi-layer patching"
    )
    all_tests_passed &= test_passed
    # Test 2: With multi-layer patching
    config2 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=True,
    )
    
    decoder2 = Decoder(config2).to(device).eval()
    decoder2.set_prompt("explain <embed>:", tokenizer)
    
    test_passed = run_comparison_test(
        decoder2,
        activation,
        max_length=10,
        gumbel_tau=1.0,
        tokenizer=tokenizer,
        test_name="Test 2: With multi-layer patching"
    )
    all_tests_passed &= test_passed
    
    # Test 3: Lower temperature
    test_passed = run_comparison_test(
        decoder2,
        activation,
        max_length=10,
        gumbel_tau=0.5,
        tokenizer=tokenizer,
        test_name="Test 3: Lower temperature (tau=0.5)"
    )
    all_tests_passed &= test_passed
    
    # Test 4: Different prompt
    decoder2.set_prompt("The activation represents <embed>:", tokenizer)
    test_passed = run_comparison_test(
        decoder2,
        activation,
        max_length=10,
        gumbel_tau=1.0,
        tokenizer=tokenizer,
        test_name="Test 4: Different prompt"
    )
    all_tests_passed &= test_passed
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("generate_soft and generate_soft_kv_cached produce identical outputs.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("The two methods produce different outputs.")


    test_embed_string = "a b c d e f g h i j<embed> l m n o o p"
    special_token = tokenizer.encode(' k')
    test_embed_string = "a long time ago, in a<embed> far"
    special_token = tokenizer.encode(' galaxy')
    special_token_str = tokenizer.decode([special_token[0]])
    decoder2.set_prompt(test_embed_string, tokenizer)
    print(f"doing multi-layer patching with special token {tokenizer.decode([special_token[0]])}")
    torch.manual_seed(42)
    testing=decoder2.generate_soft(activation, max_length=100, gumbel_tau=1.0, do_patching=False, special_token=special_token[0])
    torch.manual_seed(42)
    testingkv=decoder2.generate_soft_kv_cached(activation, max_length=100, gumbel_tau=1.0, do_patching=False, special_token=special_token[0])
    tokids=(testing.hard_token_ids[0].tolist())
    tokids_kv=testingkv.hard_token_ids[0].tolist()
    textids=tokenizer.decode(tokids).replace("\n", "\\n")
    textids_kv=tokenizer.decode(tokids_kv).replace("\n", "\\n")
    print(f"with multi-layer patching")
    print("orig, no patching",textids)
    print("kv cached, no patching",textids_kv)


    decoder1.set_prompt(test_embed_string, tokenizer)
    print(f"doing single layer patching with special token {special_token_str}")
    torch.manual_seed(42)
    testing=decoder1.generate_soft(activation, max_length=100, gumbel_tau=1.0, do_patching=False, special_token=special_token[0])
    torch.manual_seed(42)
    testingkv=decoder1.generate_soft_kv_cached(activation, max_length=100, gumbel_tau=1.0, do_patching=False, special_token=special_token[0])
    tokids=(testing.hard_token_ids[0].tolist())
    tokids_kv=testingkv.hard_token_ids[0].tolist()
    textids=tokenizer.decode(tokids).replace("\n", "\\n")
    textids_kv=tokenizer.decode(tokids_kv).replace("\n", "\\n")
    print(f"with single layer patching")
    print("orig, no patching",textids)
    print("kv cached, no patching",textids_kv)

    
    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)