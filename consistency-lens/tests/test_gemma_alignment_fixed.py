#!/usr/bin/env python3
"""
Test alignment of generation methods for tiny Gemma model.
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


def test_gemma_alignment():
    """Test that all generation methods align for Gemma."""
    
    device = torch.device("cuda")
    model_name = "ariG23498/tiny-gemma-2-test"
    
    print("="*80)
    print(f"Testing Generation Alignment for: {model_name}")
    print("="*80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # First, let's understand the model architecture
    print("\nChecking model architecture...")
    temp_decoder = Decoder(DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,
    )).to(device)
    
    print(f"Base model type: {temp_decoder.base.__class__.__name__}")
    print(f"Hidden size: {temp_decoder.base.config.hidden_size}")
    
    # Find the transformer layers
    if hasattr(temp_decoder.base, 'model'):
        transformer = temp_decoder.base.model
        print(f"Found transformer at: base.model")
    else:
        transformer = temp_decoder.base
        
    # Check for layers attribute
    if hasattr(transformer, 'layers'):
        print(f"Number of layers: {len(transformer.layers)}")
        layer_attr = 'layers'
    elif hasattr(transformer, 'h'):
        print(f"Number of layers: {len(transformer.h)}")
        layer_attr = 'h'
    else:
        print("Could not find transformer layers!")
        return
    
    del temp_decoder
    torch.cuda.empty_cache()
    
    # Test configurations
    configs = [
        {
            "name": "Single-layer",
            "params": {
                "patch_all_layers": False,
                "per_layer_projections": False,
            }
        },
        # Skip multi-layer for now if it's not supported
    ]
    
    # Check if multi-layer patching is supported
    if hasattr(transformer, layer_attr):
        configs.append({
            "name": "Multi-layer patching",
            "params": {
                "patch_all_layers": True,
                "per_layer_projections": True,
            }
        })
    
    for config in configs:
        print(f"\n{'-'*70}")
        print(f"Configuration: {config['name']}")
        print(f"{'-'*70}")
        
        # Create decoder
        decoder = Decoder(DecoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
            output_head=False,
            end_to_end=True,
            detach_after_each_sample=False,
            **config['params']
        )).to(device).eval()
        
        decoder.set_prompt("The answer is <embed>:", tokenizer)
        
        d_model = decoder.base.config.hidden_size
        
        # Test with small lengths first
        test_cases = [
            (5, 42),
            (10, 123),
        ]
        
        all_match = True
        
        for max_length, seed in test_cases:
            print(f"\nTesting length={max_length}, seed={seed}")
            
            # Create activation with specific seed
            set_all_seeds(seed + 1000)
            activation = torch.randn(1, d_model, device=device)
            
            # Generate with all three methods using same seed
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
            
            # Check matches
            match_kv = tokens_soft == tokens_kv
            match_nondiff = tokens_soft == tokens_kv_nondiff
            match_all = match_kv and match_nondiff
            
            if match_all:
                print(f"✅ All methods match")
                text = tokenizer.decode(tokens_soft)
                print(f"   Text: '{text}'")
            else:
                print(f"❌ Methods differ!")
                
                # Show token by token comparison
                print(f"\n   Token comparison:")
                print(f"   Pos | Soft | KV   | KV-ND | Match")
                print(f"   ----|------|------|-------|------")
                for i in range(max_length):
                    t_soft = tokens_soft[i] if i < len(tokens_soft) else -1
                    t_kv = tokens_kv[i] if i < len(tokens_kv) else -1
                    t_kvnd = tokens_kv_nondiff[i] if i < len(tokens_kv_nondiff) else -1
                    match = "✓" if t_soft == t_kv == t_kvnd else "✗"
                    print(f"   {i:3d} | {t_soft:4d} | {t_kv:4d} | {t_kvnd:5d} | {match}")
                
                all_match = False
        
        if all_match:
            print(f"\n✅ All generation methods align for {config['name']}!")
        else:
            print(f"\n❌ Generation methods do not align for {config['name']}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_match:
        print("✅ Tiny Gemma model works correctly with all generation methods!")
    else:
        print("❌ Tiny Gemma model has alignment issues between generation methods")
        print("\nThis could be due to:")
        print("1. Different handling of attention masks")
        print("2. KV cache implementation differences for Gemma2")
        print("3. Numerical precision issues with the tiny model")


if __name__ == "__main__":
    test_gemma_alignment()