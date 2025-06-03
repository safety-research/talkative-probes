"""Test that KV-cached generation produces identical results to original."""

import torch
import torch.nn as nn
from lens.models.decoder import Decoder, DecoderConfig
from transformers import AutoTokenizer
import numpy as np


def test_forward_backward_equivalence():
    """Test that KV-cached and original methods produce identical forward and backward passes."""
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create decoder
    config = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        embedding_head=False,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=False
    )
    
    decoder = Decoder(config).to(device)
    
    # Setup tokenizer and prompt
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    # Create test input
    batch_size = 2
    d_model = decoder.base.config.hidden_size
    test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    # Test parameters
    max_length = 8
    gumbel_tau = 1.0
    
    print("\n1. Testing forward pass equivalence...")
    
    # Generate with original method
    torch.manual_seed(123)
    test_act_orig = test_activation.clone().detach().requires_grad_(True)
    gen_original = decoder.generate_soft(
        test_act_orig,
        max_length=max_length,
        gumbel_tau=gumbel_tau
    )
    
    # Generate with KV-cached method
    torch.manual_seed(123)
    test_act_cached = test_activation.clone().detach().requires_grad_(True)
    gen_cached = decoder.generate_soft_kv_cached(
        test_act_cached,
        max_length=max_length,
        gumbel_tau=gumbel_tau
    )
    
    # Check forward pass equivalence
    embeddings_match = torch.allclose(gen_original.generated_text_embeddings, 
                                     gen_cached.generated_text_embeddings, 
                                     rtol=1e-4, atol=1e-5)
    logits_match = torch.allclose(gen_original.raw_lm_logits, 
                                 gen_cached.raw_lm_logits, 
                                 rtol=1e-4, atol=1e-5)
    ids_match = torch.equal(gen_original.hard_token_ids, gen_cached.hard_token_ids)
    
    print(f"Generated embeddings match: {embeddings_match}")
    print(f"Raw logits match: {logits_match}")
    print(f"Hard token IDs match: {ids_match}")
    
    if not embeddings_match:
        diff = torch.abs(gen_original.generated_text_embeddings - gen_cached.generated_text_embeddings)
        print(f"Max embedding difference: {diff.max().item()}")
        print(f"Mean embedding difference: {diff.mean().item()}")
    
    # Test backward pass
    print("\n2. Testing backward pass equivalence...")
    
    # Create identical loss targets
    target = torch.randn_like(gen_original.generated_text_embeddings)
    
    # Backward pass - original
    loss_original = nn.functional.mse_loss(gen_original.generated_text_embeddings, target)
    loss_original.backward()
    grad_original = test_act_orig.grad.clone()
    
    # Backward pass - cached
    loss_cached = nn.functional.mse_loss(gen_cached.generated_text_embeddings, target)
    loss_cached.backward()
    grad_cached = test_act_cached.grad.clone()
    
    losses_match = torch.allclose(loss_original, loss_cached, rtol=1e-4, atol=1e-5)
    gradients_match = torch.allclose(grad_original, grad_cached, rtol=1e-4, atol=1e-5)
    
    print(f"Losses match: {losses_match}")
    print(f"Gradients match: {gradients_match}")
    
    if not gradients_match:
        grad_diff = torch.abs(grad_original - grad_cached)
        print(f"Max gradient difference: {grad_diff.max().item()}")
        print(f"Mean gradient difference: {grad_diff.mean().item()}")
    
    # Memory usage comparison
    if torch.cuda.is_available():
        print("\n3. Memory usage comparison...")
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Measure original method
        start_mem = torch.cuda.memory_allocated()
        _ = decoder.generate_soft(test_activation.clone(), max_length=16, gumbel_tau=1.0)
        torch.cuda.synchronize()
        original_mem = torch.cuda.memory_allocated() - start_mem
        
        # Clear for fair comparison
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Measure cached method
        start_mem = torch.cuda.memory_allocated()
        _ = decoder.generate_soft_kv_cached(test_activation.clone(), max_length=16, gumbel_tau=1.0)
        torch.cuda.synchronize()
        cached_mem = torch.cuda.memory_allocated() - start_mem
        
        print(f"Original method: {original_mem / 1024 / 1024:.1f} MB")
        print(f"Cached method: {cached_mem / 1024 / 1024:.1f} MB")
        
        if cached_mem < original_mem:
            print(f"Memory saved: {(original_mem - cached_mem) / 1024 / 1024:.1f} MB ({(original_mem - cached_mem) / original_mem * 100:.1f}%)")
        else:
            print(f"Note: Cached method uses more memory (likely due to storing K,V tensors)")
    
    # Test with different sequence lengths
    print("\n4. Testing scalability...")
    for seq_len in [4, 8, 16, 32]:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Set same seed for both
        torch.manual_seed(456)
        gen1 = decoder.generate_soft(test_activation.clone(), max_length=seq_len, gumbel_tau=1.0)
        
        torch.manual_seed(456)
        gen2 = decoder.generate_soft_kv_cached(test_activation.clone(), max_length=seq_len, gumbel_tau=1.0)
        
        match = torch.allclose(gen1.generated_text_embeddings, gen2.generated_text_embeddings, rtol=1e-4, atol=1e-5)
        print(f"Sequence length {seq_len}: {'✓' if match else '✗'}")
    
    overall_pass = embeddings_match and logits_match and ids_match and losses_match and gradients_match
    print(f"\n{'✓ All tests passed!' if overall_pass else '✗ Some tests failed!'}")
    
    return overall_pass


def test_computational_complexity():
    """Test that KV caching actually reduces computation."""
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        embedding_head=False,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=False
    )
    
    decoder = Decoder(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    batch_size = 8
    d_model = decoder.base.config.hidden_size
    test_activation = torch.randn(batch_size, d_model, device=device)
    
    print("\n5. Timing comparison (forward pass only)...")
    
    # Warmup
    for _ in range(3):
        _ = decoder.generate_soft(test_activation, max_length=8, gumbel_tau=1.0)
        _ = decoder.generate_soft_kv_cached(test_activation, max_length=8, gumbel_tau=1.0)
    
    # Time different sequence lengths
    for seq_len in [8, 16, 32]:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Original method
        start = time.time()
        for _ in range(5):
            _ = decoder.generate_soft(test_activation, max_length=seq_len, gumbel_tau=1.0)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        original_time = (time.time() - start) / 5
        
        # Cached method
        start = time.time()
        for _ in range(5):
            _ = decoder.generate_soft_kv_cached(test_activation, max_length=seq_len, gumbel_tau=1.0)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cached_time = (time.time() - start) / 5
        
        speedup = original_time / cached_time
        print(f"Seq length {seq_len}: Original {original_time*1000:.1f}ms, Cached {cached_time*1000:.1f}ms, Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing KV-Cached Generation Implementation")
    print("=" * 70)
    
    # Run main equivalence test
    test_forward_backward_equivalence()
    
    # Run timing test
    test_computational_complexity()
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)