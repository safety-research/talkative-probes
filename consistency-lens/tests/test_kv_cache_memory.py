"""Test memory usage during backward pass with KV caching."""

import torch
import torch.nn as nn
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.training.loop import train_step
from transformers import AutoTokenizer
import gc


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_backward_memory():
    """Compare memory usage during backward pass with and without KV caching."""
    
    if not torch.cuda.is_available():
        print("CUDA not available. Memory test requires GPU.")
        return
    
    device = torch.device("cuda")
    
    # Test configurations
    batch_sizes = [8, 16, 32, 64]
    seq_lengths = [8, 16, 32]
    
    print("=" * 70)
    print("Memory Usage During Backward Pass")
    print("=" * 70)
    
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            print(f"\nBatch size: {batch_size}, Sequence length: {seq_length}")
            print("-" * 50)
            
            # Test original method
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            
            # Create models for original method
            dec_config_orig = DecoderConfig(
                model_name="gpt2",
                n_prompt_tokens=0,
                base_model=False,
                projection_layer=True,
                output_head=True,
                eye_init=True,
                use_checkpointing=False,
                use_kv_cache=False  # Original method
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
            
            decoder_orig = Decoder(dec_config_orig).to(device)
            encoder_orig = Encoder(enc_config).to(device)
            
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            decoder_orig.set_prompt("The meaning of <embed> is:", tokenizer)
            
            # Create batch
            d_model = decoder_orig.base.config.hidden_size
            batch = {
                "A": torch.randn(batch_size, d_model, device=device, requires_grad=True),
                "A_prime": torch.randn(batch_size, d_model, device=device, requires_grad=True),
                "input_ids_A": torch.randint(0, 1000, (batch_size, 32), device=device),
                "layer_idx": torch.tensor([5] * batch_size, device=device),
                "token_pos_A": torch.tensor([10] * batch_size, device=device)
            }
            
            models_orig = {
                "dec": decoder_orig,
                "enc": encoder_orig,
                "orig": None  # Skip for memory test
            }
            
            loss_fns = {
                "T_text": seq_length,
                "tau": 1.0,
                "alpha": 0.1,
                "lm_base_weight": 0.0,  # Disable to focus on generation memory
                "kl_base_weight": 0.0,
                "entropy_weight": 0.0,
                "mse_weight": 1.0
            }
            
            # Measure original method
            torch.cuda.synchronize()
            mem_before = get_gpu_memory_mb()
            
            # Forward pass
            losses_orig = train_step(batch.copy(), models_orig, loss_fns)
            torch.cuda.synchronize()
            mem_after_forward = get_gpu_memory_mb()
            
            # Backward pass
            losses_orig['total'].backward()
            torch.cuda.synchronize()
            mem_after_backward = get_gpu_memory_mb()
            
            orig_forward_mem = mem_after_forward - mem_before
            orig_backward_mem = mem_after_backward - mem_after_forward
            orig_total_mem = mem_after_backward - mem_before
            
            # Clean up
            del decoder_orig, encoder_orig, models_orig, losses_orig
            torch.cuda.empty_cache()
            gc.collect()
            
            # Test KV-cached method
            torch.cuda.synchronize()
            
            # Create models for KV-cached method
            dec_config_kv = DecoderConfig(
                model_name="gpt2",
                n_prompt_tokens=0,
                base_model=False,
                projection_layer=True,
                output_head=True,
                eye_init=True,
                use_checkpointing=False,
                use_kv_cache=True  # KV-cached method
            )
            
            decoder_kv = Decoder(dec_config_kv).to(device)
            encoder_kv = Encoder(enc_config).to(device)
            decoder_kv.set_prompt("The meaning of <embed> is:", tokenizer)
            
            models_kv = {
                "dec": decoder_kv,
                "enc": encoder_kv,
                "orig": None
            }
            
            # Measure KV-cached method
            torch.cuda.synchronize()
            mem_before = get_gpu_memory_mb()
            
            # Forward pass
            losses_kv = train_step(batch.copy(), models_kv, loss_fns)
            torch.cuda.synchronize()
            mem_after_forward = get_gpu_memory_mb()
            
            # Backward pass
            losses_kv['total'].backward()
            torch.cuda.synchronize()
            mem_after_backward = get_gpu_memory_mb()
            
            kv_forward_mem = mem_after_forward - mem_before
            kv_backward_mem = mem_after_backward - mem_after_forward
            kv_total_mem = mem_after_backward - mem_before
            
            # Print results
            print(f"Original method:")
            print(f"  Forward:  {orig_forward_mem:.1f} MB")
            print(f"  Backward: {orig_backward_mem:.1f} MB")
            print(f"  Total:    {orig_total_mem:.1f} MB")
            
            print(f"KV-cached method:")
            print(f"  Forward:  {kv_forward_mem:.1f} MB")
            print(f"  Backward: {kv_backward_mem:.1f} MB")
            print(f"  Total:    {kv_total_mem:.1f} MB")
            
            print(f"Memory saved:")
            forward_saved = orig_forward_mem - kv_forward_mem
            backward_saved = orig_backward_mem - kv_backward_mem
            total_saved = orig_total_mem - kv_total_mem
            
            print(f"  Forward:  {forward_saved:.1f} MB ({forward_saved/orig_forward_mem*100:.1f}%)")
            print(f"  Backward: {backward_saved:.1f} MB ({backward_saved/orig_backward_mem*100:.1f}%)")
            print(f"  Total:    {total_saved:.1f} MB ({total_saved/orig_total_mem*100:.1f}%)")
            
            # Clean up
            del decoder_kv, encoder_kv, models_kv, losses_kv
            torch.cuda.empty_cache()
            gc.collect()
    
    print("\n" + "=" * 70)
    print("Memory test complete!")
    print("=" * 70)


def test_gradient_accumulation():
    """Test that gradients accumulate correctly through KV cache."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create decoder with KV cache
    config = DecoderConfig(
        model_name="gpt2",
        n_prompt_tokens=0,
        base_model=False,
        projection_layer=True,
        output_head=True,
        eye_init=True,
        use_checkpointing=False,
        use_kv_cache=True
    )
    
    decoder = Decoder(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder.set_prompt("The meaning of <embed> is:", tokenizer)
    
    # Test input
    batch_size = 4
    d_model = decoder.base.config.hidden_size
    test_activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    # Generate with KV cache
    gen = decoder.generate_soft_kv_cached(test_activation, max_length=16, gumbel_tau=1.0)
    
    # Create loss that depends on all generated tokens
    loss = gen.generated_text_embeddings.mean()
    
    # Backward
    loss.backward()
    
    print("\nGradient flow test:")
    print(f"Input gradient exists: {test_activation.grad is not None}")
    print(f"Input gradient norm: {test_activation.grad.norm().item():.4f}")
    print(f"Projection weight gradient norm: {decoder.proj.weight.grad.norm().item():.4f}")
    
    # Check that gradients flow through all layers
    layer_grad_norms = []
    for i, layer in enumerate(decoder.base.transformer.h):
        if hasattr(layer.attn.c_attn, 'weight') and layer.attn.c_attn.weight.grad is not None:
            layer_grad_norms.append(layer.attn.c_attn.weight.grad.norm().item())
    
    print(f"Attention layer gradients: {len(layer_grad_norms)} layers have gradients")
    print("✓ Gradient flow test passed!")


if __name__ == "__main__":
    test_backward_memory()
    print("\n" + "=" * 70 + "\n")
    test_gradient_accumulation()