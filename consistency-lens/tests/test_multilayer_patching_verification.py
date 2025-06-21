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
    
    # Verify multi-layer decoder has projections for all layers
    print("\n1. Checking projection layers:")
    print(f"   Number of transformer layers: {len(decoder_multi.base.transformer.h)}")
    
    if hasattr(decoder_multi, 'projections'):
        print(f"   Number of projection layers: {len(decoder_multi.projections)}")
        print("   ✅ Multi-layer projections found")
        
        # Check that projections are different (not shared)
        if len(decoder_multi.projections) > 1:
            # Compare parameters of first two projections
            proj0_params = list(decoder_multi.projections[0].parameters())
            proj1_params = list(decoder_multi.projections[1].parameters())
            
            same_params = all(
                torch.equal(p0, p1) 
                for p0, p1 in zip(proj0_params, proj1_params)
            )
            
            if not same_params:
                print("   ✅ Projections are unique per layer (not shared)")
            else:
                print("   ❌ WARNING: Projections appear to be shared!")
    else:
        print("   ❌ ERROR: No projections attribute found!")
    
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
    
    # Test hook verification - check that patches are being applied
    print("\n4. Hook verification (checking patches are applied):")
    
    # We'll trace which layers get patched during generation
    patched_layers = set()
    
    def trace_patch(module, input, output, layer_idx):
        patched_layers.add(layer_idx)
        return output
    
    # Register hooks on all transformer layers
    hooks = []
    for i, layer in enumerate(decoder_multi.base.transformer.h):
        hook = layer.register_forward_hook(
            lambda m, inp, out, idx=i: trace_patch(m, inp, out, idx)
        )
        hooks.append(hook)
    
    # Generate once to trace patches
    patched_layers.clear()
    set_all_seeds(42)
    _ = decoder_multi.generate_soft(activation.clone(), max_length=5, gumbel_tau=1.0)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print(f"   Layers that were accessed during generation: {sorted(patched_layers)}")
    print(f"   Total layers in model: {len(decoder_multi.base.transformer.h)}")
    
    if len(patched_layers) == len(decoder_multi.base.transformer.h):
        print("   ✅ All layers were accessed during generation")
    else:
        print("   ❌ Not all layers were accessed!")
    
    # Gradient flow test for multi-layer patching
    print("\n5. Gradient flow verification for multi-layer patching:")
    
    # Test that gradients flow through all projection layers
    activation_grad = torch.randn(1, d_model, device=device, requires_grad=True)
    
    set_all_seeds(42)
    gen = decoder_multi.generate_soft(activation_grad, max_length=5, gumbel_tau=1.0)
    loss = gen.generated_text_embeddings.sum()
    
    # Check gradients on projection layers
    projection_grads = []
    if hasattr(decoder_multi, 'projections'):
        for i, proj in enumerate(decoder_multi.projections):
            for name, param in proj.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    projection_grads.append((i, name, grad_norm))
    
    # Clear gradients before computing
    decoder_multi.zero_grad()
    loss.backward()
    
    # Check gradients again
    layers_with_grads = set()
    for i, proj in enumerate(decoder_multi.projections):
        has_grad = False
        for param in proj.parameters():
            if param.grad is not None and param.grad.norm() > 0:
                has_grad = True
                break
        if has_grad:
            layers_with_grads.add(i)
    
    print(f"   Projection layers with gradients: {sorted(layers_with_grads)}")
    if len(layers_with_grads) == len(decoder_multi.projections):
        print("   ✅ All projection layers receive gradients")
    else:
        print(f"   ❌ Only {len(layers_with_grads)}/{len(decoder_multi.projections)} projection layers have gradients")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    verify_multilayer_patching()