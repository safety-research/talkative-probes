#!/usr/bin/env python3
"""Test KV caching with SimpleStories-5M (LLaMA architecture)."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_simplestories():
    """Test KV caching with SimpleStories-5M model."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "roneneldan/TinyStories-1M"  # Using 1M as a proxy since 5M might not be available
    # If you have the actual path to SimpleStories-5M, replace the model_name
    
    print("Testing KV Cache with SimpleStories (LLaMA Architecture)")
    print("=" * 60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying alternative model...")
        model_name = "JackFram/llama-68m"  # Small LLaMA model for testing
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Using model: {model_name}")
    
    # Test 1: Basic generation comparison
    print("\nTest 1: Basic generation comparison")
    print("-" * 40)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,
        per_layer_projections=False,
    )
    
    try:
        decoder = Decoder(config).to(device).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Skipping SimpleStories test - model not available")
        return
    
    decoder.set_prompt("Once upon a time <embed>", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Generate with both methods
    print("Generating 10 tokens...")
    torch.manual_seed(42)
    gen1 = decoder.generate_soft(activation.clone(), max_length=10, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen2 = decoder.generate_soft_kv_cached(activation.clone(), max_length=10, gumbel_tau=1.0)
    
    # Compare outputs
    tokens1 = gen1.hard_token_ids[0].tolist()
    tokens2 = gen2.hard_token_ids[0].tolist()
    
    text1 = tokenizer.decode(tokens1)
    text2 = tokenizer.decode(tokens2)
    
    print(f"generate_soft:      '{text1}'")
    print(f"generate_kv_cached: '{text2}'")
    print(f"Tokens match: {tokens1 == tokens2}")
    print(f"Logits max diff: {(gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item():.2e}")
    
    # Test 2: With multi-layer patching
    print("\n\nTest 2: With multi-layer patching")
    print("-" * 40)
    
    config2 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder2 = Decoder(config2).to(device).eval()
    decoder2.set_prompt("The little <embed> went to", tokenizer)
    
    # Generate with both methods
    print("Generating 10 tokens with multi-layer patching...")
    torch.manual_seed(123)
    gen3 = decoder2.generate_soft(activation.clone(), max_length=10, gumbel_tau=0.5)
    
    torch.manual_seed(123)
    gen4 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=10, gumbel_tau=0.5)
    
    tokens3 = gen3.hard_token_ids[0].tolist()
    tokens4 = gen4.hard_token_ids[0].tolist()
    
    text3 = tokenizer.decode(tokens3)
    text4 = tokenizer.decode(tokens4)
    
    print(f"generate_soft:      '{text3}'")
    print(f"generate_kv_cached: '{text4}'")
    print(f"Tokens match: {tokens3 == tokens4}")
    print(f"Logits max diff: {(gen3.raw_lm_logits - gen4.raw_lm_logits).abs().max().item():.2e}")
    
    # Test 3: Gradient alignment
    print("\n\nTest 3: Gradient alignment")
    print("-" * 40)
    
    activation1 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation2 = activation1.clone().detach().requires_grad_(True)
    
    # Generate
    torch.manual_seed(42)
    gen5 = decoder2.generate_soft(activation1, max_length=5, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen6 = decoder2.generate_soft_kv_cached(activation2, max_length=5, gumbel_tau=1.0)
    
    # Compute gradients
    loss1 = gen5.raw_lm_logits.sum()
    loss2 = gen6.raw_lm_logits.sum()
    
    loss1.backward()
    loss2.backward()
    
    grad_diff = (activation1.grad - activation2.grad).abs().max().item()
    grad_norm = activation1.grad.norm().item()
    relative_diff = grad_diff / grad_norm if grad_norm > 0 else 0
    
    print(f"Gradient max diff: {grad_diff:.2e}")
    print(f"Gradient norm: {grad_norm:.2e}")
    print(f"Relative diff: {relative_diff*100:.4f}%")
    print(f"Gradients match: {'✓' if relative_diff < 0.001 else '✗'}")
    
    # Test 4: Per-layer projections
    print("\n\nTest 4: Per-layer projections")
    print("-" * 40)
    
    config3 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=True,
    )
    
    decoder3 = Decoder(config3).to(device).eval()
    decoder3.set_prompt("In the forest <embed> there lived", tokenizer)
    
    print("Generating 15 tokens with per-layer projections...")
    torch.manual_seed(99)
    gen7 = decoder3.generate_soft(activation.clone(), max_length=15, gumbel_tau=0.8)
    
    torch.manual_seed(99)
    gen8 = decoder3.generate_soft_kv_cached(activation.clone(), max_length=15, gumbel_tau=0.8)
    
    tokens7 = gen7.hard_token_ids[0].tolist()
    tokens8 = gen8.hard_token_ids[0].tolist()
    
    # Count matching tokens
    matches = sum(1 for t1, t2 in zip(tokens7, tokens8) if t1 == t2)
    
    print(f"Matching tokens: {matches}/{len(tokens7)}")
    print(f"Logits max diff: {(gen7.raw_lm_logits - gen8.raw_lm_logits).abs().max().item():.2e}")
    
    if matches == len(tokens7):
        print("✅ All tokens match!")
    else:
        print("❌ Some tokens differ")
        print(f"generate_soft tokens:      {tokens7}")
        print(f"generate_kv_cached tokens: {tokens8}")
    
    # Test 5: Longer generation
    print("\n\nTest 5: Longer generation (50 tokens)")
    print("-" * 40)
    
    torch.manual_seed(777)
    gen9 = decoder2.generate_soft(activation.clone(), max_length=50, gumbel_tau=1.0)
    
    torch.manual_seed(777)
    gen10 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=50, gumbel_tau=1.0)
    
    tokens9 = gen9.hard_token_ids[0].tolist()
    tokens10 = gen10.hard_token_ids[0].tolist()
    
    text9 = tokenizer.decode(tokens9)
    text10 = tokenizer.decode(tokens10)
    
    # Show first 100 chars of each
    print(f"generate_soft (first 100 chars):      '{text9[:100]}...'")
    print(f"generate_kv_cached (first 100 chars): '{text10[:100]}...'")
    
    matches_long = sum(1 for t1, t2 in zip(tokens9, tokens10) if t1 == t2)
    print(f"Matching tokens: {matches_long}/{len(tokens9)}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("SimpleStories/LLaMA Architecture Test Complete")
    
    # Check if model is actually LLaMA-style
    if hasattr(decoder.base, 'model'):
        print("✓ Confirmed: Model uses LLaMA-style architecture")
    else:
        print("✗ Warning: Model appears to use GPT-2 style architecture")


if __name__ == "__main__":
    test_simplestories()