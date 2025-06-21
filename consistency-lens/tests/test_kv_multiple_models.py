#!/usr/bin/env python3
"""
Validate that generate_soft and generate_soft_kv_cached produce identical outputs
across different model architectures: GPT-2 and tiny-gemma-2.
"""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig, Generated
import sys


def test_model_outputs(model_name: str, device: torch.device, max_lengths=[5, 10, 20], num_seeds=3):
    """Test a specific model with various configurations."""
    
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name}")
    print(f"{'='*80}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test different configurations
    test_configs = [
        {
            "name": "Standard (end_to_end=True, detach=False)",
            "end_to_end": True,
            "detach_after_each_sample": False,
            "patch_all_layers": False,
            "per_layer_projections": False,
        },
        {
            "name": "Detach mode (end_to_end=False, detach=True)",
            "end_to_end": False,
            "detach_after_each_sample": True,
            "patch_all_layers": False,
            "per_layer_projections": False,
        },
        {
            "name": "Multi-layer patching",
            "end_to_end": True,
            "detach_after_each_sample": False,
            "patch_all_layers": True,
            "per_layer_projections": True,
        },
    ]
    
    all_tests_passed = True
    
    for test_cfg in test_configs:
        print(f"\n{'-'*60}")
        print(f"Configuration: {test_cfg['name']}")
        print(f"{'-'*60}")
        
        try:
            # Create decoder config
            config = DecoderConfig(
                model_name=model_name,
                base_model=False,
                projection_layer=True,
                output_head=False,  # Required for generate_soft
                end_to_end=test_cfg["end_to_end"],
                detach_after_each_sample=test_cfg["detach_after_each_sample"],
                patch_all_layers=test_cfg["patch_all_layers"],
                per_layer_projections=test_cfg["per_layer_projections"],
            )
            
            # Create decoder
            decoder = Decoder(config).to(device).eval()
            decoder.set_prompt("The answer is <embed>:", tokenizer)
            
            # Get model dimension
            d_model = decoder.base.config.hidden_size
            print(f"Model hidden size: {d_model}")
            
            # Test multiple sequence lengths
            config_passed = True
            for max_length in max_lengths:
                length_passed = True
                
                # Test multiple seeds
                for seed in range(num_seeds):
                    activation = torch.randn(1, d_model, device=device)
                    
                    # Generate with both methods
                    torch.manual_seed(seed)
                    gen_soft = decoder.generate_soft(activation, max_length, gumbel_tau=1.0)
                    
                    torch.manual_seed(seed)
                    gen_kv = decoder.generate_soft_kv_cached(activation, max_length, gumbel_tau=1.0)
                    
                    # Compare outputs
                    tokens_soft = gen_soft.hard_token_ids[0].tolist()
                    tokens_kv = gen_kv.hard_token_ids[0].tolist()
                    
                    tokens_match = tokens_soft == tokens_kv
                    logits_diff = (gen_soft.raw_lm_logits - gen_kv.raw_lm_logits).abs().max().item()
                    emb_diff = (gen_soft.generated_text_embeddings - gen_kv.generated_text_embeddings).abs().max().item()
                    
                    if seed == 0:  # Show details for first seed
                        text_soft = tokenizer.decode(tokens_soft)
                        text_kv = tokenizer.decode(tokens_kv)
                        
                        print(f"\n  Length={max_length}, Seed={seed}:")
                        print(f"    Soft: '{text_soft}'")
                        print(f"    KV:   '{text_kv}'")
                        print(f"    Match: {'✅' if tokens_match else '❌'}")
                        print(f"    Logits diff: {logits_diff:.2e}")
                        print(f"    Emb diff: {emb_diff:.2e}")
                    
                    length_passed &= tokens_match
                    
                    if not tokens_match:
                        print(f"    ❌ MISMATCH at length={max_length}, seed={seed}")
                        # Show token-by-token comparison
                        for i in range(max_length):
                            if i < len(tokens_soft) and i < len(tokens_kv):
                                t_soft = tokens_soft[i]
                                t_kv = tokens_kv[i]
                                if t_soft != t_kv:
                                    print(f"      Position {i}: {t_soft} vs {t_kv}")
                
                if length_passed:
                    print(f"  ✅ All seeds passed for length={max_length}")
                else:
                    config_passed = False
            
            # Test gradients (only for appropriate configs)
            if test_cfg["end_to_end"] and not test_cfg["detach_after_each_sample"]:
                print(f"\n  Gradient test (expecting match):")
                activation_soft = torch.randn(1, d_model, device=device, requires_grad=True)
                activation_kv = activation_soft.clone().detach().requires_grad_(True)
                
                torch.manual_seed(42)
                gen_soft = decoder.generate_soft(activation_soft, max_length=5, gumbel_tau=1.0)
                torch.manual_seed(42)
                gen_kv = decoder.generate_soft_kv_cached(activation_kv, max_length=5, gumbel_tau=1.0)
                
                loss_soft = gen_soft.generated_text_embeddings.sum()
                loss_kv = gen_kv.generated_text_embeddings.sum()
                
                grad_soft = torch.autograd.grad(loss_soft, activation_soft)[0]
                grad_kv = torch.autograd.grad(loss_kv, activation_kv)[0]
                
                grad_diff = (grad_soft - grad_kv).abs().max().item()
                grad_match = grad_diff < 1e-5
                print(f"    Gradient diff: {grad_diff:.2e}")
                print(f"    Gradients match: {'✅' if grad_match else '❌'}")
            
            if config_passed:
                print(f"\n✅ Configuration '{test_cfg['name']}' PASSED")
            else:
                print(f"\n❌ Configuration '{test_cfg['name']}' FAILED")
                all_tests_passed = False
            
            # Cleanup
            del decoder
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ ERROR in configuration '{test_cfg['name']}': {str(e)}")
            all_tests_passed = False
            continue
    
    return all_tests_passed


def main():
    """Test multiple models."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Models to test
    models = [
        "gpt2",
        "ariG23498/tiny-gemma-2-test",
    ]
    
    all_models_passed = True
    
    for model_name in models:
        try:
            passed = test_model_outputs(model_name, device)
            all_models_passed &= passed
        except Exception as e:
            print(f"\n❌ Failed to test {model_name}: {str(e)}")
            all_models_passed = False
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    if all_models_passed:
        print("✅ ALL TESTS PASSED!")
        print("Both generate_soft and generate_soft_kv_cached produce identical outputs")
        print("across all tested models and configurations.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("There are discrepancies between the two generation methods.")
    
    return 0 if all_models_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)