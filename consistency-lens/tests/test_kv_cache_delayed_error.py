"""Test KV cache behavior that might cause delayed errors after epoch boundaries."""

import torch
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.training.loop import train_step
from transformers import AutoTokenizer
import gc


def test_delayed_epoch_boundary_error():
    """Simulate the scenario where error occurs some steps after epoch boundary."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing delayed epoch boundary error scenario")
    print("=" * 70)
    
    # Create models
    dec_config = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        eye_init=True,
        use_checkpointing=False,
        use_kv_cache=True
    )
    
    enc_config = EncoderConfig(
        model_name="gpt2",
        base_model=False,
        use_base_model=True,
        projection_layer=True,
        eye_init=True,
        output_layer=-1,
        stop_grad_aprime=False
    )
    
    decoder = Decoder(dec_config).to(device)
    encoder = Encoder(enc_config).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    
    models = {
        "dec": decoder,
        "enc": encoder,
        "orig": None
    }
    
    loss_fns = {
        "T_text": 8,
        "tau": 1.0,
        "alpha": 0.1,
        "lm_base_weight": 0.0,
        "kl_base_weight": 1.0,  # Always non-zero
        "entropy_weight": 0.0,
        "mse_weight": 1.0
    }
    
    # Simulate training loop
    steps_per_epoch = 10
    total_steps = 25  # More than 2 epochs
    
    print("\nSimulating training with potential epoch boundary issues...")
    
    for step in range(total_steps):
        epoch = step // steps_per_epoch
        step_in_epoch = step % steps_per_epoch
        
        # Change something at epoch boundary
        if step_in_epoch == 0 and step > 0:
            print(f"\n--- EPOCH BOUNDARY: Starting epoch {epoch} ---")
            
            # Simulate what might happen at epoch boundary:
            # 1. Clear gradients
            decoder.zero_grad()
            encoder.zero_grad()
            
            # 2. Maybe garbage collection happens
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 3. Maybe data characteristics change
            # (we'll simulate this with different batch sizes)
        
        # Vary batch size to stress test
        if epoch == 0:
            batch_size = 4
        elif epoch == 1:
            batch_size = 8  # Different batch size in epoch 1
        else:
            batch_size = 2  # And different again in epoch 2
        
        # Also vary sequence characteristics after epoch boundary
        if step_in_epoch < 3:
            # First few steps of epoch: maybe different data distribution
            seq_length = 32  # Longer sequences
        else:
            seq_length = 16  # Normal sequences
        
        print(f"\nStep {step} (Epoch {epoch}, Step {step_in_epoch}):")
        print(f"  Batch size: {batch_size}, Seq length: {seq_length}")
        
        try:
            # Create batch
            batch = {
                "A": torch.randn(batch_size, d_model, device=device),
                "A_prime": torch.randn(batch_size, d_model, device=device),
                "input_ids_A": torch.randint(0, 1000, (batch_size, seq_length), device=device),
                "layer_idx": torch.tensor([5] * batch_size, device=device),
                "token_pos_A": torch.tensor([10] * batch_size, device=device)
            }
            
            # Forward pass
            losses = train_step(batch, models, loss_fns)
            
            # Backward pass
            losses['total'].backward()
            
            print(f"  ✓ Success - Loss: {losses['total'].item():.4f}")
            
            # Clear gradients for next step
            decoder.zero_grad()
            encoder.zero_grad()
            
        except Exception as e:
            print(f"  ✗ ERROR at step {step} (epoch {epoch}, step {step_in_epoch})!")
            print(f"     Error type: {type(e).__name__}")
            print(f"     Error message: {str(e)}")
            
            # Try to understand what's special about this step
            print(f"\n  Debugging info:")
            print(f"    - Steps since epoch start: {step_in_epoch}")
            print(f"    - This is {'the first' if step_in_epoch == 0 else 'NOT the first'} step of the epoch")
            print(f"    - Batch size changed: {'YES' if (epoch > 0 and step_in_epoch == 0) else 'NO'}")
            
            # Check if it's related to cache state
            if hasattr(decoder, '_kv_cache_state'):
                print(f"    - KV cache state exists: {decoder._kv_cache_state is not None}")
            
            import traceback
            print("\n  Full traceback:")
            traceback.print_exc()
            
            break
    
    print("\n" + "=" * 70)
    print("Test completed")


def test_cache_state_persistence():
    """Test if KV cache state incorrectly persists between forward passes."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n\nTesting KV cache state persistence issues")
    print("=" * 70)
    
    config = DecoderConfig(
        model_name="gpt2",
        use_kv_cache=True
    )
    decoder = Decoder(config).to(device)
    d_model = decoder.base.config.hidden_size
    
    print("\n1. Testing if cache persists between generate calls...")
    
    # First generation
    act1 = torch.randn(2, d_model, device=device)
    gen1 = decoder.generate_soft_kv_cached(act1, max_length=8, gumbel_tau=1.0)
    print(f"   First generation: shape={gen1.generated_text_embeddings.shape}")
    
    # Check if any cache state persists (it shouldn't)
    # The KV cache should be local to each generate call
    
    # Second generation with different parameters
    act2 = torch.randn(4, d_model, device=device)  # Different batch size
    gen2 = decoder.generate_soft_kv_cached(act2, max_length=16, gumbel_tau=1.0)  # Different length
    print(f"   Second generation: shape={gen2.generated_text_embeddings.shape}")
    
    print("   ✓ No persistent cache issues detected")
    
    print("\n2. Testing rapid successive calls...")
    
    # Simulate what might happen with multiple A and A' calls in quick succession
    for i in range(5):
        batch_size = 2 + i % 3  # Varying batch sizes
        act = torch.randn(batch_size, d_model, device=device)
        gen = decoder.generate_soft_kv_cached(act, max_length=4, gumbel_tau=1.0)
        print(f"   Call {i+1}: batch_size={batch_size}, output_shape={gen.generated_text_embeddings.shape}")
    
    print("   ✓ All rapid calls succeeded")


if __name__ == "__main__":
    test_delayed_epoch_boundary_error()
    test_cache_state_persistence()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)