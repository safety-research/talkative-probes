#!/usr/bin/env python3
"""
Debug multi-layer patching mismatch at length=50.
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def debug_multilayer_mismatch():
    """Debug the specific mismatch case for multi-layer patching."""
    
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
    
    # Test the problematic cases
    test_cases = [
        (50, 123),
        (50, 789),
    ]
    
    for max_length, seed in test_cases:
        print(f"\nTesting length={max_length}, seed={seed}")
        print("="*80)
        
        # Create activation with deterministic seed
        set_all_seeds(seed + 1000)
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
        
        # Find first mismatch
        first_mismatch = None
        for i in range(max_length):
            if tokens_soft[i] != tokens_kv[i] or tokens_soft[i] != tokens_kv_nondiff[i]:
                first_mismatch = i
                break
        
        if first_mismatch is not None:
            print(f"\nFirst mismatch at position {first_mismatch}:")
            
            # Show context around mismatch
            start = max(0, first_mismatch - 3)
            end = min(max_length, first_mismatch + 3)
            
            print("\nToken comparison (position: soft, kv, nondiff):")
            for i in range(start, end):
                t_soft = tokens_soft[i]
                t_kv = tokens_kv[i]
                t_nondiff = tokens_kv_nondiff[i]
                
                text_soft = tokenizer.decode([t_soft])
                text_kv = tokenizer.decode([t_kv])
                text_nondiff = tokenizer.decode([t_nondiff])
                
                match = "✅" if t_soft == t_kv == t_nondiff else "❌"
                print(f"{i:3d}: {t_soft:5d} ({text_soft:>10s}), {t_kv:5d} ({text_kv:>10s}), {t_nondiff:5d} ({text_nondiff:>10s}) {match}")
            
            # Check if it's specifically the KV cache that differs
            if tokens_soft == tokens_kv_nondiff and tokens_soft != tokens_kv:
                print("\n⚠️  generate_soft matches generate_soft_kv_cached_nondiff")
                print("   But generate_soft_kv_cached differs!")
                print("   This suggests an issue with the differentiable KV cache implementation.")
            elif tokens_kv == tokens_kv_nondiff and tokens_soft != tokens_kv:
                print("\n⚠️  Both KV cache methods match each other")
                print("   But generate_soft differs!")
                print("   This suggests the KV cache is working correctly but differently from generate_soft.")
            
            # Show generated text snippets
            print(f"\nGenerated text around mismatch:")
            snippet_start = max(0, first_mismatch - 10)
            snippet_end = min(max_length, first_mismatch + 10)
            
            print(f"Soft:    '...{tokenizer.decode(tokens_soft[snippet_start:snippet_end])}...'")
            print(f"KV:      '...{tokenizer.decode(tokens_kv[snippet_start:snippet_end])}...'")
            print(f"Nondiff: '...{tokenizer.decode(tokens_kv_nondiff[snippet_start:snippet_end])}...'")
            
            # Check logits at position before mismatch
            if first_mismatch > 0:
                print(f"\nLogits analysis at position {first_mismatch-1} (before mismatch):")
                
                logits_soft = gen_soft.raw_lm_logits[0, first_mismatch-1]
                logits_kv = gen_kv.raw_lm_logits[0, first_mismatch-1]
                
                # Get top 5 tokens by probability
                probs_soft = torch.softmax(logits_soft, dim=-1)
                probs_kv = torch.softmax(logits_kv, dim=-1)
                
                top5_soft = torch.topk(probs_soft, 5)
                top5_kv = torch.topk(probs_kv, 5)
                
                print("\nTop 5 predictions (soft):")
                for i in range(5):
                    tok_id = top5_soft.indices[i].item()
                    prob = top5_soft.values[i].item()
                    print(f"  {tok_id}: {tokenizer.decode([tok_id]):>10s} (p={prob:.4f})")
                
                print("\nTop 5 predictions (kv):")
                for i in range(5):
                    tok_id = top5_kv.indices[i].item()
                    prob = top5_kv.values[i].item()
                    print(f"  {tok_id}: {tokenizer.decode([tok_id]):>10s} (p={prob:.4f})")
                
                # Check if the chosen tokens have similar probabilities
                chosen_soft = tokens_soft[first_mismatch]
                chosen_kv = tokens_kv[first_mismatch]
                
                prob_chosen_soft = probs_soft[chosen_soft].item()
                prob_chosen_kv = probs_kv[chosen_kv].item()
                
                print(f"\nProbabilities of chosen tokens:")
                print(f"  Soft chose {chosen_soft} with p={prob_chosen_soft:.4f}")
                print(f"  KV chose {chosen_kv} with p={prob_chosen_kv:.4f}")
                
                # Check logits difference
                logits_diff = (logits_soft - logits_kv).abs()
                print(f"\nLogits difference statistics:")
                print(f"  Max diff: {logits_diff.max().item():.2e}")
                print(f"  Mean diff: {logits_diff.mean().item():.2e}")
                print(f"  Std diff: {logits_diff.std().item():.2e}")
        else:
            print("✅ No mismatch found! All tokens match.")
            print(f"Generated text: '{tokenizer.decode(tokens_soft)}'")


if __name__ == "__main__":
    debug_multilayer_mismatch()