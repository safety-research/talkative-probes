#!/usr/bin/env python3
"""Test that random seeding is handled properly in both methods."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_random_seeding():
    """Test random number generation in both methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Testing Random Seeding")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device, requires_grad=True)
    
    # Test 1: Check if seeding works for single generation
    print("\nTest 1: Single generation with same seed")
    
    torch.manual_seed(42)
    gen1a = decoder.generate_soft(activation.clone(), max_length=5, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen1b = decoder.generate_soft(activation.clone(), max_length=5, gumbel_tau=1.0)
    
    tokens1a = gen1a.hard_token_ids[0].tolist()
    tokens1b = gen1b.hard_token_ids[0].tolist()
    
    print(f"  First run:  {tokens1a}")
    print(f"  Second run: {tokens1b}")
    print(f"  Match: {tokens1a == tokens1b}")
    
    # Test 2: Check KV cached version
    print("\nTest 2: KV cached with same seed")
    
    torch.manual_seed(42)
    gen2a = decoder.generate_soft_kv_cached(activation.clone(), max_length=5, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen2b = decoder.generate_soft_kv_cached(activation.clone(), max_length=5, gumbel_tau=1.0)
    
    tokens2a = gen2a.hard_token_ids[0].tolist()
    tokens2b = gen2b.hard_token_ids[0].tolist()
    
    print(f"  First run:  {tokens2a}")
    print(f"  Second run: {tokens2b}")
    print(f"  Match: {tokens2a == tokens2b}")
    
    # Test 3: Compare between methods with same seed
    print("\nTest 3: Comparing methods with same seed")
    
    torch.manual_seed(42)
    gen3a = decoder.generate_soft(activation.clone(), max_length=5, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    gen3b = decoder.generate_soft_kv_cached(activation.clone(), max_length=5, gumbel_tau=1.0)
    
    tokens3a = gen3a.hard_token_ids[0].tolist()
    tokens3b = gen3b.hard_token_ids[0].tolist()
    
    print(f"  generate_soft:      {tokens3a}")
    print(f"  generate_kv_cached: {tokens3b}")
    print(f"  Match: {tokens3a == tokens3b}")
    
    # Test 4: Check random state consumption
    print("\nTest 4: Random state consumption analysis")
    
    # Count random numbers used
    torch.manual_seed(42)
    state_before = torch.get_rng_state()
    
    gen4a = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    state_after_soft = torch.get_rng_state()
    
    torch.set_rng_state(state_before)
    gen4b = decoder.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    state_after_kv = torch.get_rng_state()
    
    # Check if they consumed the same amount of randomness
    print(f"  Tokens match: {gen4a.hard_token_ids[0].item() == gen4b.hard_token_ids[0].item()}")
    print(f"  RNG states match: {torch.equal(state_after_soft, state_after_kv)}")
    
    # Test 5: Gradient test with proper seeding
    print("\nTest 5: Gradients with proper seeding")
    
    activation1 = torch.randn(1, d_model, device=device, requires_grad=True)
    activation2 = activation1.clone().detach().requires_grad_(True)
    
    # Set seed before each generation
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # Also set CUDA seed
    gen5a = decoder.generate_soft(activation1, max_length=5, gumbel_tau=1.0)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # Also set CUDA seed
    gen5b = decoder.generate_soft_kv_cached(activation2, max_length=5, gumbel_tau=1.0)
    
    # Check outputs
    print(f"  Outputs match: {torch.allclose(gen5a.raw_lm_logits, gen5b.raw_lm_logits, atol=1e-4)}")
    print(f"  Tokens match: {torch.all(gen5a.hard_token_ids == gen5b.hard_token_ids).item()}")
    
    # Compute gradients
    loss1 = gen5a.raw_lm_logits.sum()
    loss2 = gen5b.raw_lm_logits.sum()
    
    loss1.backward()
    loss2.backward()
    
    grad_diff = (activation1.grad - activation2.grad).abs().max().item()
    grad_norm = activation1.grad.norm().item()
    relative_diff = grad_diff / grad_norm if grad_norm > 0 else 0
    
    print(f"  Gradient diff: {grad_diff:.2e}")
    print(f"  Relative diff: {relative_diff*100:.6f}%")
    
    # Detailed token-by-token analysis
    print("\nDetailed token generation:")
    for i in range(5):
        print(f"  Token {i}: {gen5a.hard_token_ids[0, i].item()} vs {gen5b.hard_token_ids[0, i].item()}")


if __name__ == "__main__":
    test_random_seeding()