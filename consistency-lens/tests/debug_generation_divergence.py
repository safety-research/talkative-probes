#!/usr/bin/env python3
"""Find where generation methods diverge with multi-layer patching."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def trace_generation_step_by_step():
    """Trace generation step by step to find divergence."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Tracing Generation Step by Step")
    print("=" * 60)
    
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
    torch.manual_seed(42)
    activation = torch.randn(1, d_model, device=device)
    
    # Generate 1 token with fixed seed
    print("\nGenerating 1 token:")
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    gen1 = decoder1.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    gen2 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    print(f"  Logits diff: {(gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item():.2e}")
    print(f"  IDs same: {torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)}")
    print(f"  Token 1: {gen1.hard_token_ids[0, 0].item()} vs {gen2.hard_token_ids[0, 0].item()}")
    
    # Try with different seeds to see if it's just randomness
    print("\nTrying different random seeds:")
    for seed in [42, 123, 456, 789]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gen1 = decoder1.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gen2 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
        
        logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
        ids_same = torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)
        
        print(f"  Seed {seed}: logits_diff={logits_diff:.2e}, ids_same={ids_same}")
    
    # Check if it's the Gumbel-Softmax causing issues
    print("\nChecking without Gumbel noise (tau=0):")
    gen1 = decoder1.generate_soft(activation.clone(), max_length=1, gumbel_tau=0.0)
    gen2 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=0.0)
    
    logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
    ids_same = torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)
    
    print(f"  Logits diff: {logits_diff:.2e}")
    print(f"  IDs same: {ids_same}")
    
    if logits_diff > 1e-4:
        # Find which tokens have different logits
        diff_per_token = (gen1.raw_lm_logits[0, 0] - gen2.raw_lm_logits[0, 0]).abs()
        top_diffs = diff_per_token.topk(10)
        
        print("\n  Top differing tokens:")
        for i, (d, idx) in enumerate(zip(top_diffs.values, top_diffs.indices)):
            token = tokenizer.decode([idx.item()])
            logit1 = gen1.raw_lm_logits[0, 0, idx].item()
            logit2 = gen2.raw_lm_logits[0, 0, idx].item()
            print(f"    {idx.item():5d} ('{token}'): diff={d.item():.2e}, "
                  f"logit1={logit1:.2f}, logit2={logit2:.2f}")


def check_embed_position_issue():
    """Check if embed position calculation is the issue."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nChecking Embed Position Calculation")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device)
    
    # Test with different prompts
    prompts = [
        "<embed>",
        "explain <embed>",
        "explain <embed>:",
        "The <embed> is",
    ]
    
    for prompt in prompts:
        decoder.set_prompt(prompt, tokenizer)
        
        # Calculate embed position
        embed_pos = decoder.prompt_left_emb.size(0) if decoder.prompt_left_emb is not None else 0
        
        # Get actual token IDs
        left_str, *right = prompt.split("<embed>")
        left_ids = tokenizer(left_str, add_special_tokens=False).input_ids if left_str else []
        
        print(f"\n  Prompt: '{prompt}'")
        print(f"    Left string: '{left_str}'")
        print(f"    Left IDs: {left_ids}")
        print(f"    Embed position: {embed_pos}")
        print(f"    Expected position: {len(left_ids)}")
        
        if embed_pos != len(left_ids):
            print("    ✗ MISMATCH!")
        else:
            print("    ✓ Correct")


if __name__ == "__main__":
    trace_generation_step_by_step()
    check_embed_position_issue()