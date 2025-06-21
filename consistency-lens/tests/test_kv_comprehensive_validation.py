#!/usr/bin/env python3
"""
Comprehensive validation that generate_soft, generate_soft_kv_cached, and generate_soft_kv_cached_nondiff
produce identical outputs across different models and configurations.
Priority on testing multi-layer patching.
"""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig, Generated
import sys
import os

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def compare_three_methods(
    decoder: Decoder,
    activation: torch.Tensor,
    max_length: int,
    gumbel_tau: float,
    seed: int = 42
) -> tuple[bool, dict]:
    """Compare outputs from all three generation methods."""
    
    # Generate with all three methods using same seed
    torch.manual_seed(seed)
    gen_soft = decoder.generate_soft(activation.clone(), max_length, gumbel_tau)
    
    torch.manual_seed(seed)
    gen_kv = decoder.generate_soft_kv_cached(activation.clone(), max_length, gumbel_tau)
    
    torch.manual_seed(seed)
    gen_kv_nondiff = decoder.generate_soft_kv_cached_nondiff(activation.clone(), max_length, gumbel_tau)
    
    # Compare tokens
    tokens_soft = gen_soft.hard_token_ids[0].tolist()
    tokens_kv = gen_kv.hard_token_ids[0].tolist()
    tokens_kv_nondiff = gen_kv_nondiff.hard_token_ids[0].tolist()
    
    tokens_match_kv = tokens_soft == tokens_kv
    tokens_match_nondiff = tokens_soft == tokens_kv_nondiff
    all_tokens_match = tokens_match_kv and tokens_match_nondiff
    
    # Compare logits (if available)
    results = {
        "tokens_match": all_tokens_match,
        "tokens_match_kv": tokens_match_kv,
        "tokens_match_nondiff": tokens_match_nondiff,
        "tokens_soft": tokens_soft,
        "tokens_kv": tokens_kv,
        "tokens_kv_nondiff": tokens_kv_nondiff,
    }
    
    if gen_soft.raw_lm_logits is not None and gen_kv.raw_lm_logits is not None:
        logits_diff_kv = (gen_soft.raw_lm_logits - gen_kv.raw_lm_logits).abs().max().item()
        results["logits_diff_kv"] = logits_diff_kv
        results["logits_match_kv"] = logits_diff_kv < 1e-3
    
    # gen_kv_nondiff might not have logits
    if gen_soft.raw_lm_logits is not None and gen_kv_nondiff.raw_lm_logits is not None:
        logits_diff_nondiff = (gen_soft.raw_lm_logits - gen_kv_nondiff.raw_lm_logits).abs().max().item()
        results["logits_diff_nondiff"] = logits_diff_nondiff
        results["logits_match_nondiff"] = logits_diff_nondiff < 1e-3
    
    # Compare embeddings
    emb_diff_kv = (gen_soft.generated_text_embeddings - gen_kv.generated_text_embeddings).abs().max().item()
    emb_diff_nondiff = (gen_soft.generated_text_embeddings - gen_kv_nondiff.generated_text_embeddings).abs().max().item()
    
    results["emb_diff_kv"] = emb_diff_kv
    results["emb_diff_nondiff"] = emb_diff_nondiff
    results["emb_match_kv"] = emb_diff_kv < 1e-3
    results["emb_match_nondiff"] = emb_diff_nondiff < 1e-3
    
    # Overall match
    results["all_match"] = (
        all_tokens_match and 
        results.get("logits_match_kv", True) and 
        results.get("logits_match_nondiff", True) and
        results["emb_match_kv"] and 
        results["emb_match_nondiff"]
    )
    
    return results["all_match"], results


def test_configuration(
    model_name: str,
    config_name: str,
    config_params: dict,
    device: torch.device,
    tokenizer,
    test_lengths: list = [5, 10, 20],
    test_seeds: list = [42, 123, 456]
) -> bool:
    """Test a specific configuration with multiple parameters."""
    
    print(f"\n{'-'*70}")
    print(f"Configuration: {config_name}")
    print(f"{'-'*70}")
    print(f"Parameters: {config_params}")
    
    try:
        # Create decoder
        config = DecoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
            output_head=False,  # Required for generate_soft
            **config_params
        )
        
        decoder = Decoder(config).to(device).eval()
        decoder.set_prompt("The answer is <embed>:", tokenizer)
        
        d_model = decoder.base.config.hidden_size
        
        all_passed = True
        
        # Test different lengths and seeds
        for max_length in test_lengths:
            for seed_idx, test_seed in enumerate(test_seeds):
                activation = torch.randn(1, d_model, device=device)
                
                match, results = compare_three_methods(
                    decoder, activation, max_length, gumbel_tau=1.0, seed=test_seed
                )
                
                if seed_idx == 0 and max_length == test_lengths[0]:  # Show details for first test
                    text_soft = tokenizer.decode(results["tokens_soft"])
                    text_kv = tokenizer.decode(results["tokens_kv"])
                    text_kv_nondiff = tokenizer.decode(results["tokens_kv_nondiff"])
                    
                    print(f"\nExample (length={max_length}, seed={test_seed}):")
                    print(f"  generate_soft:              '{text_soft}'")
                    print(f"  generate_soft_kv_cached:    '{text_kv}'")
                    print(f"  generate_soft_kv_cached_nondiff: '{text_kv_nondiff}'")
                    print(f"  All match: {'✅' if match else '❌'}")
                    
                    if "logits_diff_kv" in results:
                        print(f"  Logits diff (kv): {results['logits_diff_kv']:.2e}")
                    if "logits_diff_nondiff" in results:
                        print(f"  Logits diff (nondiff): {results['logits_diff_nondiff']:.2e}")
                    print(f"  Emb diff (kv): {results['emb_diff_kv']:.2e}")
                    print(f"  Emb diff (nondiff): {results['emb_diff_nondiff']:.2e}")
                
                if not match:
                    print(f"\n❌ MISMATCH at length={max_length}, seed={test_seed}")
                    if not results["tokens_match_kv"]:
                        print("  Token mismatch: generate_soft vs generate_soft_kv_cached")
                    if not results["tokens_match_nondiff"]:
                        print("  Token mismatch: generate_soft vs generate_soft_kv_cached_nondiff")
                    
                    # Show first few token differences
                    for i in range(min(5, max_length)):
                        t_soft = results["tokens_soft"][i]
                        t_kv = results["tokens_kv"][i]
                        t_nondiff = results["tokens_kv_nondiff"][i]
                        if t_soft != t_kv or t_soft != t_nondiff:
                            print(f"    Position {i}: soft={t_soft}, kv={t_kv}, nondiff={t_nondiff}")
                
                all_passed &= match
        
        if all_passed:
            print(f"\n✅ All tests passed for this configuration")
        else:
            print(f"\n❌ Some tests failed for this configuration")
        
        # Test gradients if applicable
        if config_params.get("end_to_end", True) and not config_params.get("detach_after_each_sample", False):
            print(f"\nGradient test (expecting match):")
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
            print(f"  Gradient diff: {grad_diff:.2e}")
            print(f"  Gradients match: {'✅' if grad_match else '❌'}")
        
        # Cleanup
        del decoder
        torch.cuda.empty_cache()
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive validation tests."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (GPU 3)")
    
    # Models to test
    models = [
        "gpt2",
        # "ariG23498/tiny-gemma-2-test",  # Skip for now due to tiktoken dependency
    ]
    
    all_tests_passed = True
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Testing Model: {model_name}")
        print(f"{'='*80}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test configurations - Only end_to_end=True and detach_after_each_sample=False
        # Priority on multi-layer patching
        configs = [
            # Multi-layer patching test (PRIORITY)
            {
                "name": "Multi-layer patching",
                "params": {
                    "patch_all_layers": True,
                    "per_layer_projections": True,
                    "end_to_end": True,
                    "detach_after_each_sample": False,
                }
            },
            # Single-layer test for comparison
            {
                "name": "Single-layer",
                "params": {
                    "patch_all_layers": False,
                    "per_layer_projections": False,
                    "end_to_end": True,
                    "detach_after_each_sample": False,
                }
            },
        ]
        
        model_passed = True
        for config in configs:
            passed = test_configuration(
                model_name,
                config["name"],
                config["params"],
                device,
                tokenizer,
                test_lengths=[5, 10, 20, 50],
                test_seeds=[42, 123, 456, 789, 999]
            )
            model_passed &= passed
        
        if model_passed:
            print(f"\n✅ All configurations passed for {model_name}")
        else:
            print(f"\n❌ Some configurations failed for {model_name}")
        
        all_tests_passed &= model_passed
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("All three generation methods produce identical outputs across all models and configurations.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("There are discrepancies between the generation methods.")
    
    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)