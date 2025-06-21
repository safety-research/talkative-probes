#!/usr/bin/env python3
"""
Test if RNG state management is causing mismatches.
"""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
import os
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_with_fresh_rng():
    """Test with fresh RNG state each time."""
    
    device = torch.device("cuda")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create decoder with multi-layer patching
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,
        patch_all_layers=True,
        per_layer_projections=True,
        end_to_end=True,
        detach_after_each_sample=False,
    )
    
    decoder = Decoder(config).to(device).eval()
    decoder.set_prompt("The answer is <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    
    print("Testing RNG state management")
    print("="*80)
    
    # Test multiple times with same seed
    test_cases = [
        (5, 42),
        (10, 123),
        (20, 456),
        (50, 789),
        (50, 999),  # The problematic case
    ]
    
    all_match = True
    
    for max_length, seed in test_cases:
        print(f"\nTesting length={max_length}, seed={seed}")
        
        # Create fresh activation each time
        set_all_seeds(seed)
        activation = torch.randn(1, d_model, device=device)
        
        # Generate with all three methods
        set_all_seeds(seed)
        gen_soft = decoder.generate_soft(activation.clone(), max_length, gumbel_tau=1.0)
        
        set_all_seeds(seed)
        gen_kv = decoder.generate_soft_kv_cached(activation.clone(), max_length, gumbel_tau=1.0)
        
        set_all_seeds(seed)
        gen_kv_nondiff = decoder.generate_soft_kv_cached_nondiff(activation.clone(), max_length, gumbel_tau=1.0)
        
        # Compare tokens
        tokens_soft = gen_soft.hard_token_ids[0].tolist()
        tokens_kv = gen_kv.hard_token_ids[0].tolist()
        tokens_kv_nondiff = gen_kv_nondiff.hard_token_ids[0].tolist()
        
        match_kv = tokens_soft == tokens_kv
        match_nondiff = tokens_soft == tokens_kv_nondiff
        match = match_kv and match_nondiff
        
        if match:
            print(f"  ✅ All methods match")
        else:
            print(f"  ❌ Mismatch detected!")
            if not match_kv:
                print(f"    - generate_soft vs generate_soft_kv_cached")
            if not match_nondiff:
                print(f"    - generate_soft vs generate_soft_kv_cached_nondiff")
            
            # Find first mismatch
            for i in range(max_length):
                if tokens_soft[i] != tokens_kv[i] or tokens_soft[i] != tokens_kv_nondiff[i]:
                    print(f"    - First mismatch at position {i}")
                    print(f"      soft: {tokens_soft[i]} ({tokenizer.decode([tokens_soft[i]])})")
                    print(f"      kv: {tokens_kv[i]} ({tokenizer.decode([tokens_kv[i]])})")
                    print(f"      nondiff: {tokens_kv_nondiff[i]} ({tokenizer.decode([tokens_kv_nondiff[i]])})")
                    break
        
        all_match &= match
    
    print("\n" + "="*80)
    if all_match:
        print("✅ All tests passed with proper RNG state management")
    else:
        print("❌ Some tests failed even with proper RNG state management")


if __name__ == "__main__":
    test_with_fresh_rng()