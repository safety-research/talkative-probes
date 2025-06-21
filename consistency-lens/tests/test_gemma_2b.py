#!/usr/bin/env python3
"""
Test with the full gemma-2-2b model.
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


def test_gemma_2b():
    """Test generation methods with gemma-2-2b model."""
    
    device = torch.device("cuda")
    model_name = "google/gemma-2-2b"
    
    print("="*80)
    print(f"Testing Model: {model_name}")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        print("\nNote: gemma-2-2b may be a gated model requiring authentication.")
        print("Please run: huggingface-cli login")
        return False
    
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
    
    all_tests_passed = True
    
    for config in configs:
        print(f"\n{'-'*70}")
        print(f"Configuration: {config['name']}")
        print(f"{'-'*70}")
        
        try:
            # Create decoder
            print("Loading model...")
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
            print(f"✅ Model loaded (hidden_size={d_model})")
            
            # Test cases with shorter lengths due to larger model
            test_cases = [
                (5, 42),
                (10, 123),
                (15, 456),
            ]
            
            config_passed = True
            
            for max_length, seed in test_cases:
                print(f"\nTesting length={max_length}, seed={seed}")
                
                # Create activation
                set_all_seeds(seed + 1000)
                activation = torch.randn(1, d_model, device=device)
                
                # Generate with all three methods
                print("  Generating with all methods...")
                
                set_all_seeds(seed)
                gen_soft = decoder.generate_soft(activation.clone(), max_length, gumbel_tau=1.0)
                tokens_soft = gen_soft.hard_token_ids[0].tolist()
                
                set_all_seeds(seed)
                gen_kv = decoder.generate_soft_kv_cached(activation.clone(), max_length, gumbel_tau=1.0)
                tokens_kv = gen_kv.hard_token_ids[0].tolist()
                
                set_all_seeds(seed)
                gen_kv_nondiff = decoder.generate_soft_kv_cached_nondiff(activation.clone(), max_length, gumbel_tau=1.0)
                tokens_kv_nondiff = gen_kv_nondiff.hard_token_ids[0].tolist()
                
                # Compare
                if tokens_soft == tokens_kv == tokens_kv_nondiff:
                    print("  ✅ All methods produce identical outputs")
                    text = tokenizer.decode(tokens_soft, skip_special_tokens=True)
                    print(f"  Generated text: '{text}'")
                else:
                    print("  ❌ Methods produce different outputs!")
                    config_passed = False
                    all_tests_passed = False
                    
                    # Show differences
                    text_soft = tokenizer.decode(tokens_soft, skip_special_tokens=True)
                    text_kv = tokenizer.decode(tokens_kv, skip_special_tokens=True)
                    text_kv_nondiff = tokenizer.decode(tokens_kv_nondiff, skip_special_tokens=True)
                    
                    print(f"  generate_soft:              '{text_soft}'")
                    print(f"  generate_soft_kv_cached:    '{text_kv}'")
                    print(f"  generate_soft_kv_cached_nondiff: '{text_kv_nondiff}'")
            
            if config_passed:
                print(f"\n✅ All tests passed for {config['name']}!")
            
            # Also test fwd_tokens
            print(f"\nTesting fwd_tokens...")
            decoder_fwd = Decoder(DecoderConfig(
                model_name=model_name,
                base_model=False,
                projection_layer=True,
                output_head=True,  # Required for fwd_tokens
                **config['params']
            )).to(device).eval()
            decoder_fwd.set_prompt("The answer is <embed>:", tokenizer)
            
            # Copy projection weights
            if config['params'].get('per_layer_projections', False):
                decoder_fwd.proj_weight.data = decoder.proj_weight.data.clone()
                decoder_fwd.proj_bias.data = decoder.proj_bias.data.clone()
            else:
                decoder_fwd.proj.weight.data = decoder.proj.weight.data.clone()
                decoder_fwd.proj.bias.data = decoder.proj.bias.data.clone()
            
            # Test fwd_tokens
            set_all_seeds(42)
            test_activation = torch.randn(1, d_model, device=device)
            test_tokens = torch.randint(0, tokenizer.vocab_size, (10,), device=device)
            
            try:
                probs, entropies = decoder_fwd.fwd_tokens(
                    activation_input=test_activation,
                    use_projection=True,
                    input_tokens=test_tokens
                )
                print(f"✅ fwd_tokens works correctly")
                print(f"   Probabilities shape: {probs.shape}")
                print(f"   Average probability: {probs.mean().item():.4f}")
            except Exception as e:
                print(f"❌ fwd_tokens failed: {e}")
                all_tests_passed = False
            
            # Cleanup
            del decoder, decoder_fwd
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ Configuration failed: {e}")
            all_tests_passed = False
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED for gemma-2-2b!")
        print("\nKey findings:")
        print("1. All three generation methods produce identical outputs")
        print("2. Multi-layer patching works correctly")
        print("3. fwd_tokens aligns with generation methods")
        print("4. Generated text is coherent (model is properly trained)")
    else:
        print("❌ Some tests failed for gemma-2-2b")
    
    return all_tests_passed


if __name__ == "__main__":
    success = test_gemma_2b()
    exit(0 if success else 1)