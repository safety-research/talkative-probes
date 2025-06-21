#!/usr/bin/env python3
"""
Test that fwd_tokens aligns with the patched forward pass and autoregressive KV cached implementations.
This ensures we can use KV cached generation for RL and do policy updates using fwd_tokens.
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


def test_fwd_tokens_alignment():
    """Test that fwd_tokens produces consistent results with generation methods."""
    
    device = torch.device("cuda")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("="*80)
    print("FWD_TOKENS ALIGNMENT TEST")
    print("="*80)
    
    # Test configurations
    configs = [
        {
            "name": "Single-layer patching",
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
        print(f"Testing: {config['name']}")
        print(f"{'-'*70}")
        
        # Create decoder
        decoder_config = DecoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
            output_head=False,
            end_to_end=True,
            detach_after_each_sample=False,
            **config['params']
        )
        
        decoder = Decoder(decoder_config).to(device).eval()
        decoder.set_prompt("The answer is <embed>:", tokenizer)
        
        d_model = decoder.base.config.hidden_size
        
        # Test multiple scenarios
        test_cases = [
            (10, 42),
            (20, 123),
            (50, 456),
        ]
        
        all_passed = True
        
        for max_length, seed in test_cases:
            print(f"\n  Testing length={max_length}, seed={seed}")
            
            # Create activation
            set_all_seeds(seed + 1000)
            activation = torch.randn(1, d_model, device=device)
            
            # Generate sequence using KV cached method
            set_all_seeds(seed)
            gen_kv = decoder.generate_soft_kv_cached(activation.clone(), max_length, gumbel_tau=1.0)
            generated_tokens = gen_kv.hard_token_ids[0]  # Shape: (max_length,)
            
            # Now use fwd_tokens to evaluate the same sequence
            # fwd_tokens expects the tokens to predict (not including the initial activation)
            probs_fwd, entropies_fwd = decoder.fwd_tokens(
                activation_input=activation.clone(),
                use_projection=True,
                input_tokens=generated_tokens
            )
            
            # Also generate using regular generate_soft for comparison
            set_all_seeds(seed)
            gen_soft = decoder.generate_soft(activation.clone(), max_length, gumbel_tau=1.0)
            
            # Extract logits from generated sequences
            logits_kv = gen_kv.raw_lm_logits[0]  # Shape: (max_length, vocab_size)
            logits_soft = gen_soft.raw_lm_logits[0]
            
            # Calculate probabilities from generation logits
            probs_kv = torch.softmax(logits_kv, dim=-1)
            probs_soft = torch.softmax(logits_soft, dim=-1)
            
            # Extract the probabilities of the actually generated tokens
            probs_kv_selected = probs_kv.gather(1, generated_tokens.unsqueeze(-1)).squeeze(-1)
            probs_soft_selected = probs_soft.gather(1, generated_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Compare probabilities
            # fwd_tokens should give the same probabilities as the generation methods
            prob_diff_kv = (probs_fwd[0] - probs_kv_selected).abs().max().item()
            prob_diff_soft = (probs_fwd[0] - probs_soft_selected).abs().max().item()
            
            print(f"    Max prob diff (fwd_tokens vs KV cached): {prob_diff_kv:.2e}")
            print(f"    Max prob diff (fwd_tokens vs generate_soft): {prob_diff_soft:.2e}")
            
            # Check if differences are negligible
            tolerance = 1e-5
            if prob_diff_kv < tolerance and prob_diff_soft < tolerance:
                print(f"    ✅ Probabilities match!")
            else:
                print(f"    ❌ Probabilities differ!")
                all_passed = False
                
                # Debug: show first few probability differences
                print(f"    First 5 token probabilities:")
                for i in range(min(5, max_length)):
                    token_id = generated_tokens[i].item()
                    token_str = tokenizer.decode([token_id])
                    print(f"      Token {i} ('{token_str}', id={token_id}):")
                    print(f"        fwd_tokens: {probs_fwd[0, i].item():.6f}")
                    print(f"        KV cached:  {probs_kv_selected[i].item():.6f}")
                    print(f"        Soft:       {probs_soft_selected[i].item():.6f}")
            
            # Also verify that fwd_tokens works with teacher forcing
            # Generate a different sequence to use as "ground truth"
            set_all_seeds(seed + 2000)
            gen_teacher = decoder.generate_soft(activation.clone(), max_length, gumbel_tau=1.0)
            teacher_tokens = gen_teacher.hard_token_ids[0]
            
            # Evaluate teacher tokens with fwd_tokens
            probs_teacher, _ = decoder.fwd_tokens(
                activation_input=activation.clone(),
                use_projection=True,
                input_tokens=teacher_tokens
            )
            
            print(f"    Teacher forcing test: fwd_tokens evaluated {teacher_tokens.shape[0]} tokens")
            print(f"    Average probability of teacher tokens: {probs_teacher[0].mean().item():.4f}")
        
        if all_passed:
            print(f"\n✅ All tests passed for {config['name']}!")
        else:
            print(f"\n❌ Some tests failed for {config['name']}!")
    
    # Special test: Verify multi-layer patching behavior in fwd_tokens
    print("\n" + "="*80)
    print("MULTI-LAYER PATCHING VERIFICATION IN FWD_TOKENS")
    print("="*80)
    
    # Create decoder with multi-layer patching
    decoder_multi = Decoder(DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,
        patch_all_layers=True,
        per_layer_projections=True,
    )).to(device).eval()
    decoder_multi.set_prompt("The answer is <embed>:", tokenizer)
    
    # Create decoder without multi-layer patching
    decoder_single = Decoder(DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,
        patch_all_layers=False,
        per_layer_projections=False,
    )).to(device).eval()
    decoder_single.set_prompt("The answer is <embed>:", tokenizer)
    
    # Test with same activation and tokens
    set_all_seeds(42)
    activation = torch.randn(1, d_model, device=device)
    test_tokens = torch.randint(0, tokenizer.vocab_size, (1, 10), device=device)
    
    probs_multi, _ = decoder_multi.fwd_tokens(activation, input_tokens=test_tokens[0])
    probs_single, _ = decoder_single.fwd_tokens(activation, input_tokens=test_tokens[0])
    
    if not torch.allclose(probs_multi, probs_single):
        print("✅ Multi-layer and single-layer fwd_tokens produce different results (expected)")
        print(f"   Max probability difference: {(probs_multi - probs_single).abs().max().item():.2e}")
    else:
        print("❌ Multi-layer and single-layer fwd_tokens produce identical results (unexpected)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("The fwd_tokens method can be used for policy gradient updates in RL:")
    print("1. Generate trajectories using generate_soft_kv_cached (fast, O(n) complexity)")
    print("2. Compute policy probabilities using fwd_tokens for gradient updates")
    print("3. Both methods use the same patching mechanism (single or multi-layer)")


if __name__ == "__main__":
    test_fwd_tokens_alignment()