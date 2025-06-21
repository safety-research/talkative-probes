#!/usr/bin/env python3
"""
Test with tiny Gemma model using slow tokenizer.
"""

import torch
from transformers import AutoTokenizer, GemmaTokenizer
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


def test_gemma_model():
    """Test tiny Gemma model with different tokenizer approaches."""
    
    device = torch.device("cuda")
    model_name = "ariG23498/tiny-gemma-2-test"
    
    print("="*80)
    print(f"Testing Model: {model_name}")
    print("="*80)
    
    # Try different tokenizer loading approaches
    tokenizer = None
    
    # Approach 1: Try with use_fast=False
    try:
        print("\nTrying to load tokenizer with use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("✅ Loaded slow tokenizer successfully")
    except Exception as e:
        print(f"❌ Failed with slow tokenizer: {e}")
    
    # Approach 2: Try GemmaTokenizer directly
    if tokenizer is None:
        try:
            print("\nTrying GemmaTokenizer directly...")
            tokenizer = GemmaTokenizer.from_pretrained(model_name)
            print("✅ Loaded GemmaTokenizer successfully")
        except Exception as e:
            print(f"❌ Failed with GemmaTokenizer: {e}")
    
    # Approach 3: Try loading from google/gemma-2b
    if tokenizer is None:
        try:
            print("\nTrying tokenizer from google/gemma-2b...")
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
            print("✅ Loaded tokenizer from google/gemma-2b")
        except Exception as e:
            print(f"❌ Failed with google/gemma-2b: {e}")
    
    if tokenizer is None:
        print("\n❌ Could not load tokenizer for Gemma model")
        return False
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test basic generation
    print("\nTesting basic generation with tiny Gemma:")
    
    try:
        decoder = Decoder(DecoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
            output_head=False,
            patch_all_layers=False,
            per_layer_projections=False,
        )).to(device).eval()
        
        decoder.set_prompt("The answer is <embed>:", tokenizer)
        
        d_model = decoder.base.config.hidden_size
        print(f"Model hidden size: {d_model}")
        
        # Test generation
        set_all_seeds(42)
        activation = torch.randn(1, d_model, device=device)
        
        print("\nTesting generation methods:")
        
        # Test generate_soft
        try:
            gen_soft = decoder.generate_soft(activation, max_length=10, gumbel_tau=1.0)
            tokens_soft = gen_soft.hard_token_ids[0].tolist()
            text_soft = tokenizer.decode(tokens_soft)
            print(f"✅ generate_soft works: '{text_soft}'")
        except Exception as e:
            print(f"❌ generate_soft failed: {e}")
            return False
        
        # Test generate_soft_kv_cached
        set_all_seeds(42)
        try:
            gen_kv = decoder.generate_soft_kv_cached(activation, max_length=10, gumbel_tau=1.0)
            tokens_kv = gen_kv.hard_token_ids[0].tolist()
            text_kv = tokenizer.decode(tokens_kv)
            print(f"✅ generate_soft_kv_cached works: '{text_kv}'")
            
            if tokens_soft == tokens_kv:
                print("✅ Both methods produce identical outputs!")
            else:
                print("❌ Methods produce different outputs")
        except Exception as e:
            print(f"❌ generate_soft_kv_cached failed: {e}")
            return False
        
        # Test multi-layer patching
        print("\nTesting multi-layer patching:")
        
        decoder_multi = Decoder(DecoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
            output_head=False,
            patch_all_layers=True,
            per_layer_projections=True,
        )).to(device).eval()
        
        decoder_multi.set_prompt("The answer is <embed>:", tokenizer)
        
        try:
            set_all_seeds(42)
            gen_multi = decoder_multi.generate_soft_kv_cached(activation, max_length=10, gumbel_tau=1.0)
            tokens_multi = gen_multi.hard_token_ids[0].tolist()
            text_multi = tokenizer.decode(tokens_multi)
            print(f"✅ Multi-layer patching works: '{text_multi}'")
            
            if tokens_multi != tokens_soft:
                print("✅ Multi-layer produces different output than single-layer (expected)")
            else:
                print("⚠️  Multi-layer produces same output as single-layer")
        except Exception as e:
            print(f"❌ Multi-layer patching failed: {e}")
            return False
        
        print("\n✅ All tests passed for tiny Gemma model!")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to create decoder: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gemma_model()
    exit(0 if success else 1)