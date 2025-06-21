#!/usr/bin/env python3
"""
Debug the mismatch at length=50, seed=999 for multi-layer patching.
"""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def debug_mismatch():
    """Debug the specific mismatch case."""
    
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
    activation = torch.randn(1, d_model, device=device)
    
    # Test the problematic case
    max_length = 50
    seed = 999
    
    print(f"Testing length={max_length}, seed={seed}")
    print("="*80)
    
    # Generate with all three methods
    torch.manual_seed(seed)
    gen_soft = decoder.generate_soft(activation.clone(), max_length, gumbel_tau=1.0)
    
    torch.manual_seed(seed)
    gen_kv = decoder.generate_soft_kv_cached(activation.clone(), max_length, gumbel_tau=1.0)
    
    torch.manual_seed(seed)
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
        start = max(0, first_mismatch - 5)
        end = min(max_length, first_mismatch + 5)
        
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
        
        # Show logits comparison at mismatch position
        print(f"\nLogits at position {first_mismatch-1} (before mismatch):")
        logits_soft = gen_soft.raw_lm_logits[0, first_mismatch-1]
        logits_kv = gen_kv.raw_lm_logits[0, first_mismatch-1]
        
        # Get top 5 tokens by probability
        top5_soft = torch.topk(logits_soft, 5)
        top5_kv = torch.topk(logits_kv, 5)
        
        print("\nTop 5 predictions (soft):")
        for i in range(5):
            tok_id = top5_soft.indices[i].item()
            prob = torch.softmax(logits_soft, dim=-1)[tok_id].item()
            print(f"  {tok_id}: {tokenizer.decode([tok_id]):>10s} (p={prob:.4f})")
        
        print("\nTop 5 predictions (kv):")
        for i in range(5):
            tok_id = top5_kv.indices[i].item()
            prob = torch.softmax(logits_kv, dim=-1)[tok_id].item()
            print(f"  {tok_id}: {tokenizer.decode([tok_id]):>10s} (p={prob:.4f})")
        
        # Check logits difference
        logits_diff = (logits_soft - logits_kv).abs()
        print(f"\nMax logits difference: {logits_diff.max().item():.2e}")
        print(f"Mean logits difference: {logits_diff.mean().item():.2e}")
        
        # Show full generated texts
        print("\nFull generated texts:")
        print(f"Soft:    '{tokenizer.decode(tokens_soft)}'")
        print(f"KV:      '{tokenizer.decode(tokens_kv)}'")
        print(f"Nondiff: '{tokenizer.decode(tokens_kv_nondiff)}'")
    else:
        print("No mismatch found!")
        print(f"All tokens match for length={max_length}, seed={seed}")


if __name__ == "__main__":
    debug_mismatch()