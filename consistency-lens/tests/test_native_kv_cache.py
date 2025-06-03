#!/usr/bin/env python3
"""Test using GPT2's native KV caching."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_native_caching():
    """Test GPT2's built-in KV caching."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("Testing Native KV Caching")
    print("=" * 60)
    
    # Create input
    text = "The quick brown"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    print(f"Input: '{text}'")
    print(f"Input IDs: {input_ids}")
    
    # Method 1: Without caching
    print("\nMethod 1: Without caching")
    with torch.no_grad():
        outputs1 = model(input_ids=input_ids, use_cache=False)
        logits1 = outputs1.logits
    
    print(f"  Logits shape: {logits1.shape}")
    
    # Method 2: With caching
    print("\nMethod 2: With caching")
    with torch.no_grad():
        outputs2 = model(input_ids=input_ids, use_cache=True)
        logits2 = outputs2.logits
        past_key_values = outputs2.past_key_values
    
    print(f"  Logits shape: {logits2.shape}")
    print(f"  Past KV: {len(past_key_values)} layers")
    print(f"  Each layer KV shape: {past_key_values[0][0].shape} (key), {past_key_values[0][1].shape} (value)")
    
    # Check they're the same
    print(f"\nLogits difference: {(logits1 - logits2).abs().max().item():.2e}")
    
    # Method 3: Incremental generation with cache
    print("\nMethod 3: Incremental generation")
    # Generate one more token
    next_token_logits = logits2[0, -1]
    next_token_id = next_token_logits.argmax().unsqueeze(0).unsqueeze(0)
    print(f"  Next token ID: {next_token_id.item()} ('{tokenizer.decode([next_token_id.item()])}')")
    
    # Process just the new token with cached KV
    with torch.no_grad():
        outputs3 = model(
            input_ids=next_token_id,
            past_key_values=past_key_values,
            use_cache=True
        )
        logits3 = outputs3.logits
        new_past = outputs3.past_key_values
    
    print(f"  New logits shape: {logits3.shape}")
    print(f"  New past KV shape: {new_past[0][0].shape}")
    
    # Compare with full forward pass
    full_input = torch.cat([input_ids, next_token_id], dim=1)
    with torch.no_grad():
        outputs_full = model(input_ids=full_input, use_cache=False)
        logits_full = outputs_full.logits
    
    print(f"\nIncremental vs full forward:")
    print(f"  Last logits diff: {(logits3[0, -1] - logits_full[0, -1]).abs().max().item():.2e}")


if __name__ == "__main__":
    test_native_caching()