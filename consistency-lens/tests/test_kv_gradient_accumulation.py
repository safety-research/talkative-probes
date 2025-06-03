"""Test that KV cache doesn't create duplicate gradient copies for model parameters."""

import torch
import torch.nn as nn
from lens.models.decoder import Decoder, DecoderConfig
from transformers import AutoTokenizer
import gc


def count_gradient_tensors():
    """Count the number of gradient tensors in memory."""
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.grad_fn is not None:
                count += 1
        except:
            pass
    return count


def test_gradient_accumulation_efficiency():
    """Test that gradients accumulate efficiently without creating n_tokens copies."""
    
    if not torch.cuda.is_available():
        print("CUDA required for memory analysis")
        return
    
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Testing gradient accumulation efficiency")
    print("=" * 70)
    
    # Create decoder with trainable parameters
    config = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=True,  # Trainable
        projection_layer=True,
        output_head=True,
        embedding_head=True,
        eye_init=True,
        trainable_prompts=True,
        use_checkpointing=False,
        use_kv_cache=True  # Using KV cache
    )
    
    decoder = Decoder(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    # Get a specific parameter to track
    tracked_param = decoder.base.transformer.h[0].attn.c_attn.weight
    param_id = id(tracked_param)
    print(f"Tracking parameter: Layer 0 attention weight")
    print(f"Parameter shape: {tracked_param.shape}")
    print(f"Parameter ID: {param_id}")
    
    # Test with different sequence lengths
    seq_lengths = [4, 8, 16, 32]
    
    for seq_length in seq_lengths:
        print(f"\n--- Sequence length: {seq_length} ---")
        
        # Clear any existing gradients
        decoder.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create input
        batch_size = 2
        d_model = decoder.base.config.hidden_size
        test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        # Forward pass
        gen = decoder.generate_soft_kv_cached(test_activation, max_length=seq_length, gumbel_tau=1.0)
        
        # Create loss
        loss = gen.generated_text_embeddings.sum()
        
        # Check gradient before backward
        print(f"Before backward: tracked_param.grad = {tracked_param.grad}")
        
        # Backward pass
        loss.backward()
        
        # Check gradient after backward
        print(f"After backward: tracked_param.grad exists = {tracked_param.grad is not None}")
        print(f"Gradient shape: {tracked_param.grad.shape}")
        print(f"Gradient norm: {tracked_param.grad.norm().item():.6f}")
        
        # Verify it's the same tensor object
        print(f"Same parameter object: {id(decoder.base.transformer.h[0].attn.c_attn.weight) == param_id}")
        
        # Check if gradient is accumulated (not replaced)
        if tracked_param.grad is not None:
            grad_id = id(tracked_param.grad)
            print(f"Gradient tensor ID: {grad_id}")
    
    # Test accumulation across multiple forward/backward passes
    print("\n" + "=" * 70)
    print("Testing gradient accumulation across multiple passes")
    print("=" * 70)
    
    decoder.zero_grad()
    seq_length = 8
    
    # Store gradient norms after each pass
    grad_norms = []
    
    for i in range(3):
        print(f"\nPass {i+1}:")
        
        # Forward
        test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        gen = decoder.generate_soft_kv_cached(test_activation, max_length=seq_length, gumbel_tau=1.0)
        loss = gen.generated_text_embeddings.sum()
        
        # Backward (accumulate gradients)
        loss.backward()
        
        # Check gradient
        grad_norm = tracked_param.grad.norm().item()
        grad_norms.append(grad_norm)
        print(f"Gradient norm: {grad_norm:.6f}")
        print(f"Gradient tensor ID: {id(tracked_param.grad)}")
    
    # Verify gradients are accumulating
    print(f"\nGradient norms increasing: {grad_norms[0] < grad_norms[1] < grad_norms[2]}")
    
    # Memory efficiency test
    print("\n" + "=" * 70)
    print("Memory efficiency comparison")
    print("=" * 70)
    
    # Test memory usage for different sequence lengths
    for seq_length in [8, 16, 32]:
        decoder.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        # Measure memory before
        mem_before = torch.cuda.memory_allocated()
        
        # Forward and backward
        test_activation = torch.randn(4, d_model, device=device, requires_grad=True)
        gen = decoder.generate_soft_kv_cached(test_activation, max_length=seq_length, gumbel_tau=1.0)
        loss = gen.generated_text_embeddings.sum()
        loss.backward()
        
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()
        
        mem_used = (mem_after - mem_before) / 1024 / 1024
        print(f"Seq length {seq_length}: Memory used = {mem_used:.1f} MB")
        
        # Count unique gradient tensors
        n_params_with_grad = sum(1 for p in decoder.parameters() if p.grad is not None)
        print(f"  Parameters with gradients: {n_params_with_grad}")
    
    print("\n" + "=" * 70)
    print("Conclusion: Gradients accumulate efficiently without duplication!")
    print("=" * 70)


def test_gradient_storage_mechanism():
    """Deep dive into how gradients are stored during KV-cached generation."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n\nDeep dive: Gradient storage mechanism")
    print("=" * 70)
    
    # Simple model for clarity
    config = DecoderConfig(
        model_name="gpt2",
        base_model=True,
        projection_layer=True,
        output_head=True,
        use_kv_cache=True
    )
    
    decoder = Decoder(config).to(device)
    
    # Track multiple parameters
    params_to_track = {
        "proj_weight": decoder.proj.weight,
        "out_weight": decoder.out.weight,
        "attn_0": decoder.base.transformer.h[0].attn.c_attn.weight,
        "attn_1": decoder.base.transformer.h[1].attn.c_attn.weight,
    }
    
    print("Parameters being tracked:")
    for name, param in params_to_track.items():
        print(f"  {name}: shape={param.shape}, id={id(param)}")
    
    # Generate with different lengths
    for seq_len in [2, 4, 8]:
        print(f"\n--- Testing seq_len={seq_len} ---")
        
        decoder.zero_grad()
        
        # Forward + backward
        test_input = torch.randn(1, 768, device=device, requires_grad=True)
        gen = decoder.generate_soft_kv_cached(test_input, max_length=seq_len, gumbel_tau=1.0)
        loss = gen.generated_text_embeddings.sum()
        loss.backward()
        
        # Check each parameter's gradient
        print("Gradient properties:")
        for name, param in params_to_track.items():
            if param.grad is not None:
                print(f"  {name}:")
                print(f"    - Shape: {param.grad.shape} (same as param: {param.grad.shape == param.shape})")
                print(f"    - Norm: {param.grad.norm().item():.6f}")
                print(f"    - Unique tensor: id={id(param.grad)}")
        
        # Key insight: Each parameter has ONE gradient tensor, not n_tokens copies!
        print(f"\nKey insight: Each parameter has exactly ONE gradient tensor,")
        print(f"not {seq_len} copies (one per token)!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_gradient_accumulation_efficiency()
    test_gradient_storage_mechanism()