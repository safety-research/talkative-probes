#!/usr/bin/env python3
"""Test KV cache performance during training (forward + backward)."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import time
import gc
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step


def profile_training_step(models, batch, loss_fns, method="naive", tokenizer=None, cached_prefix_ids=None):
    """Profile a single training step with timing and memory."""
    
    # Configure decoder for the specific method
    decoder = models["dec"]
    if method == "naive":
        decoder.config.use_kv_cache = False
        decoder.config.use_checkpointing = False
    elif method == "checkpoint":
        decoder.config.use_kv_cache = False
        decoder.config.use_checkpointing = True
        decoder.config.checkpoint_every_n_tokens = 1
    elif method == "kv_cache":
        decoder.config.use_kv_cache = True
        decoder.config.use_checkpointing = False
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024**3
    
    # Time the training step
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    # Forward + backward
    losses = train_step(batch, models, loss_fns, tokenizer=tokenizer, cached_prefix_ids=cached_prefix_ids)
    
    # Compute total loss and backward
    total_loss = losses["total"]
    total_loss.backward()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    # Get memory stats
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        mem_delta = peak_mem - mem_before
    else:
        peak_mem = 0
        mem_delta = 0
    
    return {
        "time": end_time - start_time,
        "peak_memory_gb": peak_mem,
        "memory_delta_gb": mem_delta,
        "loss": total_loss.item(),
        "losses": {k: v.item() if hasattr(v, 'item') else v for k, v in losses.items() if k != 'total'}
    }


def test_training_performance():
    """Compare training performance of different generation methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("KV Cache Training Performance Analysis")
    print("="*80)
    
    # Initialize models
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create models for each test
    decoder_config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        output_head=False,
        trainable_prompts=True,
        eye_init=True,
    )
    
    encoder_config = EncoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        eye_init=True,
    )
    
    # Test configurations
    batch_sizes = [2, 4]
    t_text_values = [8, 16]  # Token generation lengths
    
    results = {}
    
    for batch_size in batch_sizes:
        for t_text in t_text_values:
            print(f"\nTesting batch_size={batch_size}, t_text={t_text}")
            print("-"*60)
            
            # Create fresh models for each test
            decoder = Decoder(decoder_config).to(device)
            encoder = Encoder(encoder_config).to(device)
            orig_model = OrigWrapper(model_name).to(device)
            
            # Set decoder prompt
            decoder.set_prompt("explain <embed>:", tokenizer)
            
            models = {
                "dec": decoder,
                "enc": encoder,
                "orig": orig_model
            }
            
            # Create batch
            d_model = decoder.base.config.hidden_size
            layer = 6
            batch = {
                "A": torch.randn(batch_size, d_model, device=device),
                "A_prime": torch.randn(batch_size, d_model, device=device),
                "input_ids_A": torch.randint(0, 50257, (batch_size, 64), device=device),
                "layer_idx": torch.tensor([layer] * batch_size, device=device).unsqueeze(1),
                "token_pos_A": torch.tensor([32] * batch_size, device=device).unsqueeze(1),  # Middle of sequence
            }
            
            # Cache natural language prefix for LM loss
            lm_loss_natural_prefix = "explain something:"
            cached_prefix_ids = tokenizer(lm_loss_natural_prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            
            # Loss function parameters
            loss_fns = {
                "t_text": t_text,
                "tau": 1.0,
                "alpha": 0.1,
                "kl_base_weight": 1.0,
                "entropy_weight": 0.01,
                "mse_weight": 0.0,
                "lm_weight": 0.1,
            }
            
            # Test each method
            test_results = {}
            
            for method in ["naive", "checkpoint", "kv_cache"]:
                print(f"\n  {method.upper()} method:")
                
                # Warm up
                _ = profile_training_step(models, batch, loss_fns, method, tokenizer, cached_prefix_ids)
                
                # Clear gradients
                decoder.zero_grad()
                encoder.zero_grad()
                orig_model.zero_grad()
                
                # Run multiple iterations for timing
                num_iterations = 5
                times = []
                memories = []
                
                for i in range(num_iterations):
                    # Clear gradients
                    decoder.zero_grad()
                    encoder.zero_grad()
                    orig_model.zero_grad()
                    
                    result = profile_training_step(models, batch, loss_fns, method, tokenizer, cached_prefix_ids)
                    times.append(result["time"])
                    memories.append(result["memory_delta_gb"])
                    
                    if i == 0:
                        print(f"    Loss: {result['loss']:.6f}")
                        print(f"    Losses: KL={result['losses']['kl']:.4f}, LM={result['losses']['lm']:.4f}, MSE={result['losses']['mse']:.4f}")
                
                avg_time = sum(times) / len(times)
                avg_memory = sum(memories) / len(memories)
                
                print(f"    Avg time: {avg_time:.4f}s")
                print(f"    Avg memory delta: {avg_memory:.3f} GB")
                
                test_results[method] = {
                    "avg_time": avg_time,
                    "avg_memory": avg_memory,
                    "times": times,
                    "memories": memories
                }
            
            # Calculate speedups and savings
            if "naive" in test_results and "kv_cache" in test_results:
                speedup = test_results["naive"]["avg_time"] / test_results["kv_cache"]["avg_time"]
                memory_ratio = test_results["kv_cache"]["avg_memory"] / test_results["naive"]["avg_memory"]
                
                print(f"\n  KV Cache vs Naive:")
                print(f"    Speedup: {speedup:.2f}x")
                print(f"    Memory ratio: {memory_ratio:.2f}x")
                
                results[f"B{batch_size}_T{t_text}"] = {
                    "speedup": speedup,
                    "memory_ratio": memory_ratio,
                    "methods": test_results
                }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - KV Cache Training Performance")
    print("="*80)
    
    for config, data in results.items():
        print(f"\n{config}:")
        print(f"  Training speedup: {data['speedup']:.2f}x")
        print(f"  Memory ratio: {data['memory_ratio']:.2f}x")
    
    # Analyze scaling
    print("\n" + "="*80)
    print("Scaling Analysis")
    print("="*80)
    
    # Extract speedups for different T values
    for batch_size in batch_sizes:
        print(f"\nBatch size {batch_size}:")
        t_speedups = []
        for t in t_text_values:
            key = f"B{batch_size}_T{t}"
            if key in results:
                t_speedups.append((t, results[key]["speedup"]))
        
        if len(t_speedups) >= 2:
            # Check if speedup increases with sequence length
            print("  t_text -> Speedup:")
            for t, speedup in t_speedups:
                print(f"    {t} -> {speedup:.2f}x")
            
            # Calculate speedup growth
            if len(t_speedups) >= 3:
                speedup_growth = (t_speedups[-1][1] - t_speedups[0][1]) / (t_speedups[-1][0] - t_speedups[0][0])
                print(f"  Speedup growth rate: {speedup_growth:.3f} per token")


if __name__ == "__main__":
    test_training_performance()