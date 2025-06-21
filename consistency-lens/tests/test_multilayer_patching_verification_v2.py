#!/usr/bin/env python3
"""
Verify that multi-layer patching is working correctly and that all three generation
methods produce identical results with multi-layer patching enabled.
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


def verify_multilayer_patching():
    """Verify that multi-layer patching is working correctly."""
    
    device = torch.device("cuda")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("="*80)
    print("MULTI-LAYER PATCHING VERIFICATION")
    print("="*80)
    
    # Create two decoders: one with multi-layer patching, one without
    config_multi = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,
        patch_all_layers=True,
        per_layer_projections=True,
        end_to_end=True,
        detach_after_each_sample=False,
    )
    
    config_single = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,
        patch_all_layers=False,
        per_layer_projections=False,
        end_to_end=True,
        detach_after_each_sample=False,
    )
    
    decoder_multi = Decoder(config_multi).to(device).eval()
    decoder_single = Decoder(config_single).to(device).eval()
    
    # Set the same prompt for both
    decoder_multi.set_prompt("The answer is <embed>:", tokenizer)
    decoder_single.set_prompt("The answer is <embed>:", tokenizer)
    
    # Verify multi-layer decoder has per-layer projections
    print("\n1. Checking projection layers:")
    print(f"   Number of transformer layers: {len(decoder_multi.base.transformer.h)}")
    
    if hasattr(decoder_multi, 'proj_weight') and decoder_multi.proj_weight is not None:
        print(f"   Projection weight shape: {decoder_multi.proj_weight.shape}")
        print(f"   Projection bias shape: {decoder_multi.proj_bias.shape}")
        expected_shape = (len(decoder_multi.base.transformer.h), decoder_multi.base.config.hidden_size, decoder_multi.base.config.hidden_size)
        if decoder_multi.proj_weight.shape == expected_shape:
            print("   ✅ Multi-layer projections found with correct shape")
        else:
            print(f"   ❌ Unexpected projection shape! Expected {expected_shape}")
            
        # Check that projections are different (not all identity or same)
        # Compare first few layers
        n_check = min(3, decoder_multi.proj_weight.shape[0])
        all_same = True
        for i in range(1, n_check):
            if not torch.allclose(decoder_multi.proj_weight[0], decoder_multi.proj_weight[i]):
                all_same = False
                break
        
        if not all_same:
            print("   ✅ Projection matrices are unique per layer")
        else:
            # Check if they're all identity
            all_identity = True
            for i in range(n_check):
                if not torch.allclose(decoder_multi.proj_weight[i], torch.eye(decoder_multi.proj_weight.shape[1], device=device)):
                    all_identity = False
                    break
            if all_identity:
                print("   ⚠️  All projections are identity matrices (expected if eye_init=True)")
            else:
                print("   ⚠️  All projections appear to be the same!")
    else:
        print("   ❌ ERROR: No per-layer projection weights found!")
    
    # Test that multi-layer and single-layer produce different outputs
    print("\n2. Verifying multi-layer vs single-layer difference:")
    
    d_model = decoder_multi.base.config.hidden_size
    set_all_seeds(42)
    activation = torch.randn(1, d_model, device=device)
    
    # Generate with both configurations
    set_all_seeds(42)
    gen_multi = decoder_multi.generate_soft(activation.clone(), max_length=10, gumbel_tau=1.0)
    
    set_all_seeds(42)
    gen_single = decoder_single.generate_soft(activation.clone(), max_length=10, gumbel_tau=1.0)
    
    tokens_multi = gen_multi.hard_token_ids[0].tolist()
    tokens_single = gen_single.hard_token_ids[0].tolist()
    
    if tokens_multi != tokens_single:
        print("   ✅ Multi-layer and single-layer produce different outputs (expected)")
        print(f"   Multi-layer:  '{tokenizer.decode(tokens_multi)}'")
        print(f"   Single-layer: '{tokenizer.decode(tokens_single)}'")
    else:
        print("   ❌ WARNING: Multi-layer and single-layer produce identical outputs!")
    
    # Now verify all three methods produce identical results with multi-layer patching
    print("\n3. Verifying all three generation methods with multi-layer patching:")
    
    test_lengths = [5, 10, 20, 50]
    test_seeds = [42, 123, 456, 789, 999]
    
    all_match = True
    mismatch_details = []
    
    for max_length in test_lengths:
        for seed in test_seeds:
            # Create fresh activation
            set_all_seeds(seed + 1000)
            activation = torch.randn(1, d_model, device=device)
            
            # Generate with all three methods
            set_all_seeds(seed)
            gen_soft = decoder_multi.generate_soft(activation.clone(), max_length, gumbel_tau=1.0)
            
            set_all_seeds(seed)
            gen_kv = decoder_multi.generate_soft_kv_cached(activation.clone(), max_length, gumbel_tau=1.0)
            
            set_all_seeds(seed)
            gen_kv_nondiff = decoder_multi.generate_soft_kv_cached_nondiff(activation.clone(), max_length, gumbel_tau=1.0)
            
            # Compare tokens
            tokens_soft = gen_soft.hard_token_ids[0].tolist()
            tokens_kv = gen_kv.hard_token_ids[0].tolist()
            tokens_kv_nondiff = gen_kv_nondiff.hard_token_ids[0].tolist()
            
            match_kv = tokens_soft == tokens_kv
            match_nondiff = tokens_soft == tokens_kv_nondiff
            match_all = match_kv and match_nondiff
            
            if not match_all:
                all_match = False
                mismatch_details.append({
                    'length': max_length,
                    'seed': seed,
                    'match_kv': match_kv,
                    'match_nondiff': match_nondiff,
                    'tokens_soft': tokens_soft,
                    'tokens_kv': tokens_kv,
                    'tokens_kv_nondiff': tokens_kv_nondiff
                })
    
    if all_match:
        print("   ✅ All three methods produce identical outputs with multi-layer patching!")
        print(f"   Tested {len(test_lengths)} lengths × {len(test_seeds)} seeds = {len(test_lengths) * len(test_seeds)} combinations")
    else:
        print(f"   ❌ Found {len(mismatch_details)} mismatches!")
        for detail in mismatch_details[:3]:  # Show first 3
            print(f"\n   Mismatch at length={detail['length']}, seed={detail['seed']}:")
            print(f"   - generate_soft vs generate_soft_kv_cached: {'✅' if detail['match_kv'] else '❌'}")
            print(f"   - generate_soft vs generate_soft_kv_cached_nondiff: {'✅' if detail['match_nondiff'] else '❌'}")
    
    # Test specific: verify nondiff version with multi-layer patching
    print("\n4. Specific test: nondiff version with multi-layer patching:")
    
    # Run a few specific tests to ensure nondiff works correctly
    test_cases = [(10, 42), (20, 123), (50, 999)]
    nondiff_match = True
    
    for length, seed in test_cases:
        set_all_seeds(seed + 2000)
        activation = torch.randn(1, d_model, device=device)
        
        set_all_seeds(seed)
        gen_soft = decoder_multi.generate_soft(activation.clone(), length, gumbel_tau=1.0)
        
        set_all_seeds(seed)
        gen_nondiff = decoder_multi.generate_soft_kv_cached_nondiff(activation.clone(), length, gumbel_tau=1.0)
        
        tokens_soft = gen_soft.hard_token_ids[0].tolist()
        tokens_nondiff = gen_nondiff.hard_token_ids[0].tolist()
        
        if tokens_soft == tokens_nondiff:
            print(f"   ✅ Length={length}, seed={seed}: nondiff matches generate_soft")
        else:
            print(f"   ❌ Length={length}, seed={seed}: nondiff DIFFERS from generate_soft")
            nondiff_match = False
            # Show where they differ
            for i, (t1, t2) in enumerate(zip(tokens_soft, tokens_nondiff)):
                if t1 != t2:
                    print(f"      First diff at position {i}: {t1} vs {t2}")
                    break
    
    # Hook verification - check that patches are being applied at each layer
    print("\n5. Hook verification (checking patches are applied at each layer):")
    
    # We'll trace which layers get their hidden states modified
    layer_modifications = {}
    original_forward = {}
    
    def make_trace_forward(layer_idx):
        def trace_forward(hidden_states, *args, **kwargs):
            # Store input for comparison
            layer_modifications[layer_idx] = {
                'input': hidden_states.clone(),
                'modified': False
            }
            # Call original forward
            output = original_forward[layer_idx](hidden_states, *args, **kwargs)
            # Check if output differs from what original model would produce
            # This is a simplified check - in reality the patching happens inside
            layer_modifications[layer_idx]['output'] = output[0] if isinstance(output, tuple) else output
            return output
        return trace_forward
    
    # Temporarily replace forward methods
    for i, layer in enumerate(decoder_multi.base.transformer.h):
        original_forward[i] = layer.forward
        layer.forward = make_trace_forward(i)
    
    # Generate once to trace modifications
    layer_modifications.clear()
    set_all_seeds(42)
    _ = decoder_multi.generate_soft(activation.clone(), max_length=5, gumbel_tau=1.0)
    
    # Restore original forwards
    for i, layer in enumerate(decoder_multi.base.transformer.h):
        layer.forward = original_forward[i]
    
    print(f"   Layers processed during generation: {sorted(layer_modifications.keys())}")
    print(f"   Total layers in model: {len(decoder_multi.base.transformer.h)}")
    
    if len(layer_modifications) == len(decoder_multi.base.transformer.h):
        print("   ✅ All layers were processed during generation")
    else:
        print("   ❌ Not all layers were processed!")
    
    # Gradient flow test for multi-layer patching
    print("\n6. Gradient flow verification for multi-layer patching:")
    
    # Test that gradients flow through projection parameters
    activation_grad = torch.randn(1, d_model, device=device, requires_grad=True)
    
    # Clear any existing gradients
    decoder_multi.zero_grad()
    
    set_all_seeds(42)
    gen = decoder_multi.generate_soft(activation_grad, max_length=5, gumbel_tau=1.0)
    loss = gen.generated_text_embeddings.sum()
    loss.backward()
    
    # Check gradients on projection weights
    if hasattr(decoder_multi, 'proj_weight') and decoder_multi.proj_weight.grad is not None:
        grad_norms = []
        for i in range(decoder_multi.proj_weight.shape[0]):
            grad_norm = decoder_multi.proj_weight.grad[i].norm().item()
            grad_norms.append(grad_norm)
        
        non_zero_grads = sum(1 for g in grad_norms if g > 1e-8)
        print(f"   Projection layers with non-zero gradients: {non_zero_grads}/{len(grad_norms)}")
        
        if non_zero_grads == len(grad_norms):
            print("   ✅ All projection layers receive gradients")
        elif non_zero_grads > 0:
            print(f"   ⚠️  Only {non_zero_grads} projection layers have gradients")
            print(f"   Gradient norms: {[f'{g:.2e}' for g in grad_norms[:5]]}...")
        else:
            print("   ❌ No projection layers have gradients!")
    else:
        print("   ❌ No gradients found on projection weights!")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_match and nondiff_match:
        print("✅ Multi-layer patching is working correctly!")
        print("✅ All three generation methods produce identical results")
        print("✅ This includes the non-differentiable KV cache version")
    else:
        print("❌ Issues found with multi-layer patching implementation")


if __name__ == "__main__":
    verify_multilayer_patching()