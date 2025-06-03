#!/usr/bin/env python3
"""Test KV caching with SimpleStories-5M."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lens.models.decoder import Decoder, DecoderConfig


def test_simplestories_5m():
    """Test KV caching with SimpleStories-5M model."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "SimpleStories/SimpleStories-5M"
    
    print("Testing KV Cache with SimpleStories-5M")
    print("=" * 60)
    
    # First check if model is available and what architecture it uses
    try:
        print(f"Loading model: {model_name}")
        test_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Check architecture
        if hasattr(test_model, 'model'):
            print("✓ Model uses LLaMA-style architecture (has .model attribute)")
            architecture = "llama"
        elif hasattr(test_model, 'transformer'):
            print("✓ Model uses GPT-2 style architecture (has .transformer attribute)")
            architecture = "gpt2"
        else:
            print("? Unknown architecture")
            architecture = "unknown"
            
        # Check model details
        print(f"  Model type: {type(test_model).__name__}")
        print(f"  Hidden size: {test_model.config.hidden_size}")
        print(f"  Num layers: {test_model.config.num_hidden_layers}")
        
        del test_model  # Free memory
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying a fallback LLaMA-style model for testing...")
        # Use a small model that we know has LLaMA architecture
        model_name = "JackFram/llama-68m"
        architecture = "llama"
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Test 1: Basic generation
    print("\n\nTest 1: Basic generation comparison")
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
        print(f"Error creating decoder: {e}")
        return
    
    decoder.set_prompt("Once upon a time <embed> there was", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Test single token first
    print("\nSingle token generation:")
    torch.manual_seed(42)
    gen1 = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen2 = decoder.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    token1 = gen1.hard_token_ids[0, 0].item()
    token2 = gen2.hard_token_ids[0, 0].item()
    print(f"  generate_soft:      {token1} ('{tokenizer.decode([token1])}')")
    print(f"  generate_kv_cached: {token2} ('{tokenizer.decode([token2])}')")
    print(f"  Match: {token1 == token2}")
    
    # Multi-token generation
    print("\nMulti-token generation (10 tokens):")
    torch.manual_seed(42)
    gen3 = decoder.generate_soft(activation.clone(), max_length=10, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen4 = decoder.generate_soft_kv_cached(activation.clone(), max_length=10, gumbel_tau=1.0)
    
    tokens3 = gen3.hard_token_ids[0].tolist()
    tokens4 = gen4.hard_token_ids[0].tolist()
    
    text3 = tokenizer.decode(tokens3)
    text4 = tokenizer.decode(tokens4)
    
    print(f"  generate_soft:      '{text3}'")
    print(f"  generate_kv_cached: '{text4}'")
    print(f"  Tokens match: {tokens3 == tokens4}")
    print(f"  Logits max diff: {(gen3.raw_lm_logits - gen4.raw_lm_logits).abs().max().item():.2e}")
    
    # Test 2: Multi-layer patching
    print("\n\nTest 2: Multi-layer patching")
    print("-" * 40)
    
    config2 = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder2 = Decoder(config2).to(device).eval()
    decoder2.set_prompt("The <embed> was very", tokenizer)
    
    print("\nGenerating 10 tokens with multi-layer patching:")
    torch.manual_seed(123)
    gen5 = decoder2.generate_soft(activation.clone(), max_length=10, gumbel_tau=0.5)
    
    torch.manual_seed(123)
    gen6 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=10, gumbel_tau=0.5)
    
    tokens5 = gen5.hard_token_ids[0].tolist()
    tokens6 = gen6.hard_token_ids[0].tolist()
    
    matches = sum(1 for t1, t2 in zip(tokens5, tokens6) if t1 == t2)
    print(f"  Matching tokens: {matches}/{len(tokens5)}")
    print(f"  Full match: {tokens5 == tokens6}")
    
    if tokens5 != tokens6:
        print(f"  generate_soft:      {tokenizer.decode(tokens5)}")
        print(f"  generate_kv_cached: {tokenizer.decode(tokens6)}")
    
    # Test 3: Gradient alignment
    print("\n\nTest 3: Gradient alignment")
    print("-" * 40)
    
    activation1 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation2 = activation1.clone().detach().requires_grad_(True)
    
    torch.manual_seed(42)
    gen7 = decoder2.generate_soft(activation1, max_length=5, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen8 = decoder2.generate_soft_kv_cached(activation2, max_length=5, gumbel_tau=1.0)
    
    loss1 = gen7.raw_lm_logits.sum()
    loss2 = gen8.raw_lm_logits.sum()
    
    loss1.backward()
    loss2.backward()
    
    grad_diff = (activation1.grad - activation2.grad).abs().max().item()
    grad_norm = activation1.grad.norm().item()
    relative_diff = grad_diff / grad_norm if grad_norm > 0 else 0
    
    print(f"  Gradient max diff: {grad_diff:.2e}")
    print(f"  Relative diff: {relative_diff*100:.4f}%")
    print(f"  Gradients match: {'✓' if relative_diff < 0.001 else '✗'}")
    
    # Test 4: Per-layer projections
    if architecture == "llama" or decoder2.base.config.num_hidden_layers >= 4:
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
        decoder3.set_prompt("In the <embed> forest", tokenizer)
        
        print("Generating 15 tokens with per-layer projections:")
        torch.manual_seed(99)
        gen9 = decoder3.generate_soft(activation.clone(), max_length=15, gumbel_tau=0.8)
        
        torch.manual_seed(99)
        gen10 = decoder3.generate_soft_kv_cached(activation.clone(), max_length=15, gumbel_tau=0.8)
        
        tokens9 = gen9.hard_token_ids[0].tolist()
        tokens10 = gen10.hard_token_ids[0].tolist()
        
        matches = sum(1 for t1, t2 in zip(tokens9, tokens10) if t1 == t2)
        print(f"  Matching tokens: {matches}/{len(tokens9)}")
        print(f"  Full match: {tokens9 == tokens10}")
        
        # Check gradient flow through per-layer projections
        decoder3.zero_grad()
        activation3 = torch.randn(1, d_model, device=device, requires_grad=True)
        gen11 = decoder3.generate_soft_kv_cached(activation3, max_length=3, gumbel_tau=1.0)
        loss3 = gen11.raw_lm_logits.sum()
        loss3.backward()
        
        if hasattr(decoder3, 'proj_weight'):
            proj_grad_norm = decoder3.proj_weight.grad.norm().item()
            print(f"  Per-layer projection gradients flow: {'✓' if proj_grad_norm > 0 else '✗'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SimpleStories-5M KV Cache Test Summary:")
    print(f"  Model architecture: {architecture}")
    print(f"  Basic generation: {'✓' if gen1.hard_token_ids[0, 0] == gen2.hard_token_ids[0, 0] else '✗'}")
    print(f"  Multi-token generation: {'✓' if tokens3 == tokens4 else '✗'}")
    print(f"  Multi-layer patching: {'✓' if tokens5 == tokens6 else '✗'}")
    print(f"  Gradient alignment: {'✓' if relative_diff < 0.001 else '✗'}")


if __name__ == "__main__":
    test_simplestories_5m()