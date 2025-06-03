#!/usr/bin/env python3
"""Debug random seed issues in KV cache generation."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_deterministic_generation():
    """Test if generation is deterministic with fixed seeds."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Testing Deterministic Generation")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,
    )
    
    # Test multiple seeds
    for seed in [42, 123, 456]:
        print(f"\nSeed = {seed}:")
        
        # Method 1: generate_soft
        decoder1a = Decoder(config).to(device)
        decoder1a.set_prompt("explain <embed>:", tokenizer)
        
        decoder1b = Decoder(config).to(device)
        decoder1b.set_prompt("explain <embed>:", tokenizer)
        decoder1b.load_state_dict(decoder1a.state_dict())
        
        d_model = decoder1a.base.config.hidden_size
        activation = torch.randn(1, d_model, device=device)
        
        # Generate twice with same seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gen1a = decoder1a.generate_soft(activation.clone(), max_length=4, gumbel_tau=1.0)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gen1b = decoder1b.generate_soft(activation.clone(), max_length=4, gumbel_tau=1.0)
        
        ids_same_soft = torch.equal(gen1a.hard_token_ids, gen1b.hard_token_ids)
        print(f"  generate_soft deterministic: {ids_same_soft}")
        if ids_same_soft:
            print(f"    Generated: {gen1a.hard_token_ids[0].tolist()}")
        
        # Method 2: generate_soft_kv_cached
        decoder2a = Decoder(config).to(device)
        decoder2a.set_prompt("explain <embed>:", tokenizer)
        
        decoder2b = Decoder(config).to(device)
        decoder2b.set_prompt("explain <embed>:", tokenizer)
        decoder2b.load_state_dict(decoder2a.state_dict())
        
        # Generate twice with same seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gen2a = decoder2a.generate_soft_kv_cached(activation.clone(), max_length=4, gumbel_tau=1.0)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gen2b = decoder2b.generate_soft_kv_cached(activation.clone(), max_length=4, gumbel_tau=1.0)
        
        ids_same_kv = torch.equal(gen2a.hard_token_ids, gen2b.hard_token_ids)
        print(f"  generate_soft_kv_cached deterministic: {ids_same_kv}")
        if ids_same_kv:
            print(f"    Generated: {gen2a.hard_token_ids[0].tolist()}")
        
        # Compare across methods with same seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gen1 = decoder1a.generate_soft(activation.clone(), max_length=4, gumbel_tau=1.0)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gen2 = decoder2a.generate_soft_kv_cached(activation.clone(), max_length=4, gumbel_tau=1.0)
        
        ids_same_cross = torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)
        logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
        
        print(f"  Cross-method comparison:")
        print(f"    IDs same: {ids_same_cross}")
        print(f"    Logits diff: {logits_diff:.2e}")
        if not ids_same_cross:
            print(f"    Soft IDs: {gen1.hard_token_ids[0].tolist()}")
            print(f"    KV IDs:   {gen2.hard_token_ids[0].tolist()}")


def test_where_divergence_starts():
    """Find exactly where the methods diverge."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nFinding Divergence Point")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Generate incrementally and compare
    for length in range(1, 5):
        print(f"\nGenerating {length} token(s):")
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        gen1 = decoder.generate_soft(activation.clone(), max_length=length, gumbel_tau=1.0)
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        gen2 = decoder.generate_soft_kv_cached(activation.clone(), max_length=length, gumbel_tau=1.0)
        
        # Compare each position
        for t in range(length):
            logit_diff = (gen1.raw_lm_logits[0, t] - gen2.raw_lm_logits[0, t]).abs().max().item()
            id1 = gen1.hard_token_ids[0, t].item()
            id2 = gen2.hard_token_ids[0, t].item()
            
            print(f"  Position {t}: logit_diff = {logit_diff:.2e}, "
                  f"id1 = {id1} ('{tokenizer.decode([id1])}'), "
                  f"id2 = {id2} ('{tokenizer.decode([id2])}')")
            
            if logit_diff > 1e-4:
                print(f"    → Divergence starts at position {t}!")
                break


if __name__ == "__main__":
    test_deterministic_generation()
    test_where_divergence_starts()