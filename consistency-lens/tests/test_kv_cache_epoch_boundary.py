"""Test KV cache behavior at epoch boundaries and edge cases."""

import torch
import torch.nn as nn
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.kv_cache import KVCache, compute_with_kv_cache
from transformers import AutoTokenizer, GPT2Model
import gc


def test_cache_persistence():
    """Test that KV cache persists correctly across multiple generation calls."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing KV cache persistence across calls")
    print("=" * 70)
    
    # Create a simple GPT2 model
    model = GPT2Model.from_pretrained("gpt2").to(device)
    d_model = model.config.hidden_size
    
    # Test sequence
    batch_size = 2
    initial_seq_len = 10
    
    # Initial sequence
    initial_embeds = torch.randn(batch_size, initial_seq_len, d_model, device=device)
    
    # Process initial sequence
    print("\n1. Processing initial sequence (length 10)...")
    kv_cache = KVCache()
    hidden_states, kv_cache = compute_with_kv_cache(
        model, initial_embeds, kv_cache, position_offset=0
    )
    
    print(f"   Cache initialized: {len(kv_cache)} layers")
    print(f"   Cached sequence length: {kv_cache.get_seq_length()}")
    
    # Generate tokens one by one
    print("\n2. Generating tokens incrementally...")
    position = initial_seq_len
    
    for i in range(5):
        # Single new token
        new_token_embed = torch.randn(batch_size, 1, d_model, device=device)
        
        # Process with cache
        hidden_states, kv_cache = compute_with_kv_cache(
            model, new_token_embed, kv_cache, position_offset=position
        )
        
        position += 1
        print(f"   Step {i+1}: Cached sequence length = {kv_cache.get_seq_length()}")
    
    # Verify cache integrity
    print("\n3. Verifying cache integrity...")
    expected_len = initial_seq_len + 5
    actual_len = kv_cache.get_seq_length()
    print(f"   Expected length: {expected_len}")
    print(f"   Actual length: {actual_len}")
    print(f"   ✓ Cache integrity maintained" if expected_len == actual_len else "   ✗ Cache corrupted!")
    
    return kv_cache


def test_epoch_boundary_scenario():
    """Test what might happen at epoch boundaries."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n\nTesting epoch boundary scenario")
    print("=" * 70)
    
    # Create decoder
    config = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        eye_init=True,
        use_kv_cache=True
    )
    
    decoder = Decoder(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    
    # Simulate multiple batches
    print("\n1. Simulating training batches...")
    
    for batch_idx in range(3):
        print(f"\n   Batch {batch_idx + 1}:")
        
        # Different batch sizes to test robustness
        batch_size = 2 if batch_idx < 2 else 4  # Change at "epoch boundary"
        
        # Generate
        test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        gen = decoder.generate_soft_kv_cached(test_activation, max_length=8, gumbel_tau=1.0)
        
        # Backward
        loss = gen.generated_text_embeddings.sum()
        loss.backward()
        
        print(f"     Batch size: {batch_size}")
        print(f"     Generated shape: {gen.generated_text_embeddings.shape}")
        print(f"     Gradient norm: {test_activation.grad.norm().item():.4f}")
        
        # Clear gradients (but not the model state)
        decoder.zero_grad()
    
    print("\n2. Testing cache reuse across batches...")
    
    # The issue might be that the KV cache is not properly cleared between batches
    # Let's test if the cache state persists incorrectly
    
    # First generation
    test_act1 = torch.randn(2, d_model, device=device, requires_grad=True)
    gen1 = decoder.generate_soft_kv_cached(test_act1, max_length=4, gumbel_tau=1.0)
    
    # Second generation with different batch size
    test_act2 = torch.randn(4, d_model, device=device, requires_grad=True)
    try:
        gen2 = decoder.generate_soft_kv_cached(test_act2, max_length=4, gumbel_tau=1.0)
        print("   ✓ Different batch sizes handled correctly")
    except Exception as e:
        print(f"   ✗ Error with different batch size: {str(e)}")


def test_cache_state_management():
    """Test cache state management and potential issues."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n\nTesting cache state management")
    print("=" * 70)
    
    # Create model
    model = GPT2Model.from_pretrained("gpt2").to(device)
    d_model = model.config.hidden_size
    
    # Test 1: Cache with wrong batch size
    print("\n1. Testing cache with mismatched batch sizes...")
    
    # Initialize with batch size 2
    kv_cache = KVCache()
    embeds1 = torch.randn(2, 5, d_model, device=device)
    _, kv_cache = compute_with_kv_cache(model, embeds1, kv_cache)
    
    # Try to use with batch size 4
    embeds2 = torch.randn(4, 1, d_model, device=device)
    try:
        _, kv_cache = compute_with_kv_cache(model, embeds2, kv_cache, position_offset=5)
        print("   ✗ Should have failed with mismatched batch size!")
    except Exception as e:
        print(f"   ✓ Correctly caught batch size mismatch: {type(e).__name__}")
    
    # Test 2: Cache clearing
    print("\n2. Testing cache clearing...")
    kv_cache.clear()
    print(f"   Cache length after clear: {len(kv_cache)}")
    print(f"   ✓ Cache cleared successfully" if len(kv_cache) == 0 else "   ✗ Cache not cleared!")
    
    # Test 3: Memory accumulation
    print("\n3. Testing memory accumulation...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        initial_mem = torch.cuda.memory_allocated()
        
        # Generate many sequences without clearing cache
        for i in range(10):
            kv_cache = KVCache()  # New cache each time
            embeds = torch.randn(2, 10, d_model, device=device)
            _, kv_cache = compute_with_kv_cache(model, embeds, kv_cache)
            
            # Generate a few tokens
            for j in range(5):
                new_token = torch.randn(2, 1, d_model, device=device)
                _, kv_cache = compute_with_kv_cache(
                    model, new_token, kv_cache, position_offset=10+j
                )
        
        final_mem = torch.cuda.memory_allocated()
        mem_increase = (final_mem - initial_mem) / 1024 / 1024
        print(f"   Memory increase: {mem_increase:.1f} MB")
        
        # Clear everything
        del kv_cache
        torch.cuda.empty_cache()
        gc.collect()


def test_edge_cases():
    """Test various edge cases that might occur at boundaries."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n\nTesting edge cases")
    print("=" * 70)
    
    config = DecoderConfig(
        model_name="gpt2",
        use_kv_cache=True
    )
    decoder = Decoder(config).to(device)
    d_model = decoder.base.config.hidden_size
    
    # Test 1: Zero-length generation
    print("\n1. Testing zero-length generation...")
    try:
        test_act = torch.randn(2, d_model, device=device)
        gen = decoder.generate_soft_kv_cached(test_act, max_length=0, gumbel_tau=1.0)
        print(f"   ✓ Zero-length generation handled: shape={gen.generated_text_embeddings.shape}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Test 2: Very long generation
    print("\n2. Testing long generation...")
    try:
        test_act = torch.randn(1, d_model, device=device)
        gen = decoder.generate_soft_kv_cached(test_act, max_length=100, gumbel_tau=1.0)
        print(f"   ✓ Long generation successful: shape={gen.generated_text_embeddings.shape}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Test 3: Gradient flow through very long sequence
    print("\n3. Testing gradient flow through long sequence...")
    test_act = torch.randn(2, d_model, device=device, requires_grad=True)
    gen = decoder.generate_soft_kv_cached(test_act, max_length=50, gumbel_tau=1.0)
    loss = gen.generated_text_embeddings.mean()
    loss.backward()
    
    grad_norm = test_act.grad.norm().item()
    print(f"   Gradient norm: {grad_norm:.6f}")
    print(f"   ✓ Gradients flow correctly" if grad_norm > 0 else "   ✗ No gradients!")


if __name__ == "__main__":
    # Run all tests
    test_cache_persistence()
    test_epoch_boundary_scenario()
    test_cache_state_management()
    test_edge_cases()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)