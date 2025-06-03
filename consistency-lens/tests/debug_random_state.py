#!/usr/bin/env python3
"""Debug random state handling in checkpointing."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_random_state_handling():
    """Test if random state is preserved correctly."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Testing Random State Handling")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    # Create identical decoders
    torch.manual_seed(42)
    decoder1 = Decoder(config).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    torch.manual_seed(42)
    decoder2 = Decoder(config).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder1.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Test 1: Check if setting same seed gives same results
    print("\nTest 1: Same seed, same results?")
    
    results = []
    for i in range(3):
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        gen = decoder1.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
        token = gen.hard_token_ids[0, 0].item()
        results.append(token)
        print(f"  Run {i+1}: token = {token}")
    
    if len(set(results)) == 1:
        print("  ✓ Same seed gives same results")
    else:
        print("  ✗ Same seed gives different results!")
    
    # Test 2: Compare methods with same seed
    print("\nTest 2: Do both methods use randomness the same way?")
    
    torch.manual_seed(456)
    torch.cuda.manual_seed_all(456)
    # Get the initial RNG state
    cpu_state_before = torch.get_rng_state()
    cuda_state_before = torch.cuda.get_rng_state()
    
    gen1 = decoder1.generate_soft(activation.clone(), max_length=2, gumbel_tau=1.0)
    
    # Check how much randomness was consumed
    cpu_state_after1 = torch.get_rng_state()
    cuda_state_after1 = torch.cuda.get_rng_state()
    
    # Reset to same initial state
    torch.manual_seed(456)
    torch.cuda.manual_seed_all(456)
    torch.set_rng_state(cpu_state_before)
    torch.cuda.set_rng_state(cuda_state_before)
    
    gen2 = decoder2.generate_soft_chkpt(activation.clone(), max_length=2, gumbel_tau=1.0)
    
    cpu_state_after2 = torch.get_rng_state()
    cuda_state_after2 = torch.cuda.get_rng_state()
    
    # Compare
    tokens1 = gen1.hard_token_ids[0].tolist()
    tokens2 = gen2.hard_token_ids[0].tolist()
    
    print(f"  Tokens method 1: {tokens1}")
    print(f"  Tokens method 2: {tokens2}")
    print(f"  Tokens match: {tokens1 == tokens2}")
    
    # Test 3: Trace random number usage
    print("\nTest 3: Trace random number usage")
    
    # Count random numbers used
    torch.manual_seed(789)
    
    # Mock gumbel_softmax to count calls
    original_gumbel = torch.nn.functional.gumbel_softmax
    call_count = [0]
    
    def counting_gumbel(*args, **kwargs):
        call_count[0] += 1
        return original_gumbel(*args, **kwargs)
    
    torch.nn.functional.gumbel_softmax = counting_gumbel
    
    gen1 = decoder1.generate_soft(activation.clone(), max_length=2, gumbel_tau=1.0)
    count1 = call_count[0]
    
    call_count[0] = 0
    gen2 = decoder2.generate_soft_chkpt(activation.clone(), max_length=2, gumbel_tau=1.0)
    count2 = call_count[0]
    
    torch.nn.functional.gumbel_softmax = original_gumbel
    
    print(f"  Gumbel calls method 1: {count1}")
    print(f"  Gumbel calls method 2: {count2}")
    
    if count1 == count2:
        print("  ✓ Same number of random calls")
    else:
        print("  ✗ Different number of random calls!")


if __name__ == "__main__":
    test_random_state_handling()