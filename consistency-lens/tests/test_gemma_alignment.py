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
    
    # Test configurations
    configs = [
        {
            "name": "Single-layer",
            "params": {
                "patch_all_layers": False,
                "per_layer_projections": False,
            }
        },
        {
            "name": "Multi-layer patching",
            "params": {
                "patch_all_layers": True,
                "per_layer_projections": True,
            }
        },
    ]
    
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
        print(f"Hidden size: {d_model}")
        print(f"Number of layers: {len(decoder.base.transformer.layers)}")
        
        # Test multiple seeds and lengths
        test_cases = [
            (5, 42),
            (10, 123),
            (20, 456),
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
            
            # Decode texts
            text_soft = tokenizer.decode(tokens_soft)
            text_kv = tokenizer.decode(tokens_kv)
            text_kv_nondiff = tokenizer.decode(tokens_kv_nondiff)
            
            # Check matches
            match_kv = tokens_soft == tokens_kv
            match_nondiff = tokens_soft == tokens_kv_nondiff
            match_all = match_kv and match_nondiff
            
            if match_all:
                print(f"✅ All methods match")
                print(f"   Text: '{text_soft}'")
            else:
                print(f"❌ Methods differ!")
                print(f"   generate_soft:                   '{text_soft}'")
                print(f"   generate_soft_kv_cached:         '{text_kv}'")
                print(f"   generate_soft_kv_cached_nondiff: '{text_kv_nondiff}'")
                
                # Find first difference
                for i in range(max_length):
                    if i < len(tokens_soft) and i < len(tokens_kv):
                        if tokens_soft[i] != tokens_kv[i]:
                            print(f"   First difference at position {i}:")
                            print(f"     soft: {tokens_soft[i]} ('{tokenizer.decode([tokens_soft[i]])}')")
                            print(f"     kv:   {tokens_kv[i]} ('{tokenizer.decode([tokens_kv[i]])}')")
                            break
                
                all_match = False
        
        if all_match:
            print(f"\n✅ All generation methods align for {config['name']}!")
        else:
            print(f"\n❌ Generation methods do not align for {config['name']}")
            
            # Debug: Check if it's a model architecture issue
            print("\nDebug info:")
            print(f"Model type: {decoder.base.__class__.__name__}")
            print(f"Has 'layers' attribute: {hasattr(decoder.base.transformer, 'layers')}")
            print(f"Has 'h' attribute: {hasattr(decoder.base.transformer, 'h')}")
            
            # Check KV cache support
            print("\nChecking KV cache support:")
            try:
                # Try a simple forward pass with past_key_values
                test_input = torch.randint(0, tokenizer.vocab_size, (1, 5), device=device)
                with torch.no_grad():
                    outputs = decoder.base(test_input, use_cache=True)
                    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                        print(f"✅ Model supports KV caching")
                        print(f"   Past key values type: {type(outputs.past_key_values)}")
                        print(f"   Number of layers: {len(outputs.past_key_values)}")
                    else:
                        print(f"❌ Model may not support KV caching properly")
            except Exception as e:
                print(f"❌ Error testing KV cache: {e}")


if __name__ == "__main__":
    test_gemma_alignment()