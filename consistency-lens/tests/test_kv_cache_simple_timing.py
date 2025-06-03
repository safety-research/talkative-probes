#!/usr/bin/env python3
"""Simple timing test for KV cache during training."""

import torch
import time
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig


def time_generation_methods():
    """Compare generation time for different methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    decoder_config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        trainable_prompts=True,
    )
    decoder = Decoder(decoder_config).to(device)
    
    encoder_config = EncoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
    )
    encoder = Encoder(encoder_config).to(device)
    
    # Set prompt
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    print("KV Cache Training Timing Analysis")
    print("="*60)
    
    # Test parameters
    batch_sizes = [1, 2, 4]
    lengths = [8, 16, 32, 64]
    d_model = decoder.base.config.hidden_size
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-"*40)
        
        for length in lengths:
            # Create activation
            activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
            
            times = {}
            
            # Test each method
            for method in ["naive", "checkpoint", "kv_cache"]:
                # Configure decoder
                if method == "naive":
                    decoder.config.use_kv_cache = False
                    decoder.config.use_checkpointing = False
                elif method == "checkpoint":
                    decoder.config.use_kv_cache = False
                    decoder.config.use_checkpointing = True
                else:  # kv_cache
                    decoder.config.use_kv_cache = True
                    decoder.config.use_checkpointing = False
                
                # Warm up
                with torch.no_grad():
                    if method == "naive":
                        _ = decoder.generate_soft(activation, max_length=4, gumbel_tau=1.0)
                    elif method == "checkpoint":
                        _ = decoder.generate_soft_chkpt(activation, max_length=4, gumbel_tau=1.0, checkpoint_every_n_tokens=1)
                    else:
                        _ = decoder.generate_soft_kv_cached(activation, max_length=4, gumbel_tau=1.0)
                
                # Time forward pass with gradients
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()
                
                if method == "naive":
                    gen = decoder.generate_soft(activation, max_length=length, gumbel_tau=1.0)
                elif method == "checkpoint":
                    gen = decoder.generate_soft_chkpt(activation, max_length=length, gumbel_tau=1.0, checkpoint_every_n_tokens=1)
                else:
                    gen = decoder.generate_soft_kv_cached(activation, max_length=length, gumbel_tau=1.0)
                
                # Encode and compute loss to ensure full computation
                reconstructed = encoder(gen.generated_text_embeddings)
                loss = torch.nn.functional.mse_loss(reconstructed, activation.detach())
                
                # Backward pass
                loss.backward()
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end = time.time()
                
                times[method] = end - start
                
                # Clear gradients
                decoder.zero_grad()
                encoder.zero_grad()
            
            # Print results
            print(f"\n  Length {length}:")
            print(f"    Naive:      {times['naive']:.4f}s")
            print(f"    Checkpoint: {times['checkpoint']:.4f}s")
            print(f"    KV Cache:   {times['kv_cache']:.4f}s")
            
            # Calculate speedups
            naive_vs_kv = times['naive'] / times['kv_cache']
            checkpoint_vs_kv = times['checkpoint'] / times['kv_cache']
            
            print(f"    Speedup (Naive vs KV):      {naive_vs_kv:.2f}x")
            print(f"    Speedup (Checkpoint vs KV): {checkpoint_vs_kv:.2f}x")
    
    # Scaling analysis
    print("\n" + "="*60)
    print("Computational Complexity Analysis")
    print("="*60)
    
    print("\nExpected scaling:")
    print("- Naive: O(n²) attention computation at each position")
    print("- KV Cache: O(n) attention computation (reuse previous K,V)")
    print("- Speedup should increase with sequence length")
    
    print("\nMemory-compute tradeoff:")
    print("- KV Cache stores K,V tensors: 2 * layers * batch * heads * seq * head_dim")
    print("- This memory allows O(n) instead of O(n²) computation")
    print("- For GPT-2: ~1.5MB per token in KV cache (batch=1)")


if __name__ == "__main__":
    time_generation_methods()