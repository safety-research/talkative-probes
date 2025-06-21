#!/usr/bin/env python3
"""
Test fwd_tokens with the simpler fix and validate with tiny Gemma model.
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


def test_model_generation(model_name: str, device: torch.device):
    """Test generation methods for a specific model."""
    
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
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
    
    all_passed = True
    
    for config in configs:
        print(f"\n{'-'*70}")
        print(f"Configuration: {config['name']}")
        print(f"{'-'*70}")
        
        try:
            # Test 1: Three generation methods produce identical outputs
            print("\n1. Testing generation methods alignment:")
            
            decoder_gen = Decoder(DecoderConfig(
                model_name=model_name,
                base_model=False,
                projection_layer=True,
                output_head=False,
                end_to_end=True,
                detach_after_each_sample=False,
                **config['params']
            )).to(device).eval()
            decoder_gen.set_prompt("The answer is <embed>:", tokenizer)
            
            d_model = decoder_gen.base.config.hidden_size
            
            # Test multiple cases
            test_cases = [(10, 42), (20, 123), (50, 456)]
            methods_match = True
            
            for max_length, seed in test_cases:
                set_all_seeds(seed + 1000)
                activation = torch.randn(1, d_model, device=device)
                
                # Generate with all three methods
                set_all_seeds(seed)
                gen_soft = decoder_gen.generate_soft(activation.clone(), max_length, gumbel_tau=1.0)
                
                set_all_seeds(seed)
                gen_kv = decoder_gen.generate_soft_kv_cached(activation.clone(), max_length, gumbel_tau=1.0)
                
                set_all_seeds(seed)
                gen_kv_nondiff = decoder_gen.generate_soft_kv_cached_nondiff(activation.clone(), max_length, gumbel_tau=1.0)
                
                # Compare tokens
                tokens_soft = gen_soft.hard_token_ids[0].tolist()
                tokens_kv = gen_kv.hard_token_ids[0].tolist()
                tokens_kv_nondiff = gen_kv_nondiff.hard_token_ids[0].tolist()
                
                if tokens_soft == tokens_kv == tokens_kv_nondiff:
                    print(f"   ✅ Length={max_length}, seed={seed}: All methods match")
                else:
                    print(f"   ❌ Length={max_length}, seed={seed}: Methods differ!")
                    methods_match = False
                    all_passed = False
            
            if methods_match:
                print("\n   ✅ All three generation methods produce identical outputs!")
            
            # Test 2: fwd_tokens alignment
            print("\n2. Testing fwd_tokens alignment:")
            
            decoder_fwd = Decoder(DecoderConfig(
                model_name=model_name,
                base_model=False,
                projection_layer=True,
                output_head=True,
                **config['params']
            )).to(device).eval()
            decoder_fwd.set_prompt("The answer is <embed>:", tokenizer)
            
            # Copy projection weights
            if config['params'].get('per_layer_projections', False):
                decoder_fwd.proj_weight.data = decoder_gen.proj_weight.data.clone()
                decoder_fwd.proj_bias.data = decoder_gen.proj_bias.data.clone()
            else:
                decoder_fwd.proj.weight.data = decoder_gen.proj.weight.data.clone()
                decoder_fwd.proj.bias.data = decoder_gen.proj.bias.data.clone()
            
            # Test fwd_tokens
            fwd_works = True
            for max_length, seed in [(10, 42), (20, 123)]:
                set_all_seeds(seed + 1000)
                activation = torch.randn(1, d_model, device=device)
                
                # Generate tokens
                set_all_seeds(seed)
                gen = decoder_gen.generate_soft_kv_cached(activation.clone(), max_length, gumbel_tau=1.0)
                generated_tokens = gen.hard_token_ids[0]
                
                try:
                    # Evaluate with fwd_tokens
                    probs_fwd, _ = decoder_fwd.fwd_tokens(
                        activation_input=activation.clone(),
                        use_projection=True,
                        input_tokens=generated_tokens
                    )
                    
                    # Compare probabilities
                    logits_gen = gen.raw_lm_logits[0]
                    probs_gen = torch.softmax(logits_gen, dim=-1)
                    probs_gen_selected = probs_gen.gather(1, generated_tokens.unsqueeze(-1)).squeeze(-1)
                    
                    prob_diff = (probs_fwd[0] - probs_gen_selected).abs().max().item()
                    
                    if prob_diff < 1e-4:
                        print(f"   ✅ Length={max_length}, seed={seed}: fwd_tokens matches (diff={prob_diff:.2e})")
                    else:
                        print(f"   ❌ Length={max_length}, seed={seed}: fwd_tokens differs (diff={prob_diff:.2e})")
                        fwd_works = False
                        all_passed = False
                        
                except Exception as e:
                    print(f"   ❌ fwd_tokens failed: {e}")
                    fwd_works = False
                    all_passed = False
                    break
            
            if fwd_works:
                print("\n   ✅ fwd_tokens aligns with generation methods!")
            
            # Cleanup
            del decoder_gen, decoder_fwd
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ Configuration failed: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
    
    return all_passed


def main():
    """Test multiple models including tiny Gemma."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (GPU 3)")
    
    # Models to test
    models = [
        "gpt2",
        "ariG23498/tiny-gemma-2-test",
    ]
    
    all_models_passed = True
    
    for model_name in models:
        try:
            passed = test_model_generation(model_name, device)
            all_models_passed &= passed
        except Exception as e:
            print(f"\n❌ Failed to test {model_name}: {e}")
            all_models_passed = False
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    if all_models_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nKey findings:")
        print("1. All three generation methods produce identical outputs")
        print("2. fwd_tokens aligns correctly with generation methods")
        print("3. Multi-layer patching works correctly")
        print("4. Both GPT-2 and tiny Gemma models are supported")
    else:
        print("❌ SOME TESTS FAILED!")
    
    return 0 if all_models_passed else 1


if __name__ == "__main__":
    exit(main())