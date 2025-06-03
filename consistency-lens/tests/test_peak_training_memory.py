#!/usr/bin/env python3
"""Test peak memory usage during training for all generation methods."""

import torch
import gc
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from lens.models.flash_kv_cache import FLASH_AVAILABLE


def get_memory_stats():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return allocated, peak
    return 0, 0


def reset_memory():
    """Reset GPU memory stats."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def measure_training_step(method, batch_size, seq_length, device):
    """Measure peak memory for a single training step."""
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    d_model = 768
    
    # Create models
    if method == "naive":
        decoder_config = DecoderConfig(
            model_name=model_name,
            use_kv_cache=False,
            use_flash_attention=False,
            base_model=False,
            projection_layer=True,
        )
    elif method == "kv_cache":
        decoder_config = DecoderConfig(
            model_name=model_name,
            use_kv_cache=True,
            use_flash_attention=False,
            base_model=False,
            projection_layer=True,
        )
    else:  # flash
        decoder_config = DecoderConfig(
            model_name=model_name,
            use_kv_cache=False,
            use_flash_attention=True,
            base_model=False,
            projection_layer=True,
        )
    
    decoder = Decoder(decoder_config).to(device)
    encoder = Encoder(EncoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
    )).to(device)
    orig_model = OrigWrapper(model_name).to(device)
    
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    models = {
        "dec": decoder,
        "enc": encoder,
        "orig": orig_model
    }
    
    # Create batch
    batch = {
        "A": torch.randn(batch_size, d_model, device=device),
        "A_prime": torch.randn(batch_size, d_model, device=device),
        "input_ids_A": torch.randint(0, 50257, (batch_size, 128), device=device),
        "layer_idx": torch.tensor([6] * batch_size, device=device).unsqueeze(1),
        "token_pos_A": torch.tensor([64] * batch_size, device=device).unsqueeze(1),
    }
    
    # Loss parameters
    loss_fns = {
        "T_text": seq_length,
        "tau": 1.0,
        "alpha": 0.1,
        "kl_base_weight": 1.0,
        "entropy_weight": 0.01,
        "mse_weight": 0.0,
        "lm_weight": 0.1,
    }
    
    # Natural language prefix
    cached_prefix_ids = tokenizer("explain something:", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    
    # Reset memory
    reset_memory()
    
    # Run training step
    losses = train_step(batch, models, loss_fns, tokenizer, cached_prefix_ids)
    total_loss = losses["total"]
    
    # Backward pass
    total_loss.backward()
    
    # Get peak memory
    _, peak_memory = get_memory_stats()
    
    # Cleanup
    decoder.zero_grad()
    encoder.zero_grad()
    orig_model.zero_grad()
    del models, batch, losses, total_loss
    
    return peak_memory


def main():
    """Test peak memory usage during training."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, cannot measure GPU memory")
        return
    
    print("Peak Memory Usage During Training (Forward + Backward)")
    print("=" * 80)
    print("Measuring the actual memory constraint for training")
    print()
    
    # Test configurations
    test_configs = [
        (2, 16),   # Small
        (4, 32),   # Medium 
        (4, 64),   # Large
        (2, 128),  # Very large
        (1, 256),  # Extra long
    ]
    
    # Methods to test
    methods = ["naive", "kv_cache"]
    if FLASH_AVAILABLE:
        methods.append("flash")
    
    results = {}
    
    for batch_size, seq_length in test_configs:
        print(f"\nBatch={batch_size}, Length={seq_length}")
        print("-" * 50)
        
        config_results = {}
        
        for method in methods:
            try:
                peak_memory = measure_training_step(method, batch_size, seq_length, device)
                config_results[method] = peak_memory
                print(f"  {method:12}: {peak_memory:.3f} GB")
            except torch.cuda.OutOfMemoryError:
                print(f"  {method:12}: OOM")
                config_results[method] = float('inf')
            except Exception as e:
                print(f"  {method:12}: Error - {str(e)}")
                config_results[method] = None
        
        results[(batch_size, seq_length)] = config_results
    
    # Summary
    print("\n" + "=" * 80)
    print("PEAK MEMORY SUMMARY")
    print("=" * 80)
    
    print("\nPeak Memory Comparison:")
    print(f"{'Config':<15} {'Naive':<12} {'KV Cache':<12} {'Flash':<12} {'Savings':<20}")
    print("-" * 70)
    
    for (batch, length), methods_data in results.items():
        config_str = f"B={batch}, L={length}"
        row = [config_str]
        
        naive_mem = methods_data.get('naive', float('inf'))
        
        # Format memory values
        for method in ['naive', 'kv_cache', 'flash']:
            if method in methods_data:
                mem = methods_data[method]
                if mem == float('inf'):
                    row.append("OOM")
                elif mem is None:
                    row.append("Error")
                else:
                    row.append(f"{mem:.2f} GB")
            else:
                row.append("N/A")
        
        # Calculate savings
        if 'flash' in methods_data and methods_data['flash'] is not None and naive_mem != float('inf'):
            flash_mem = methods_data['flash']
            if flash_mem != float('inf'):
                savings = (1 - flash_mem / naive_mem) * 100
                row.append(f"{savings:.0f}% with Flash")
            else:
                row.append("N/A")
        else:
            row.append("N/A")
        
        print(f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<20}")
    
    # Practical implications
    print("\nPractical Implications:")
    print("-" * 40)
    
    # Find max batch sizes
    if results:
        print("\nMaximum trainable configurations (assuming 24GB GPU):")
        for method in methods:
            max_configs = []
            for (batch, length), methods_data in results.items():
                if method in methods_data and methods_data[method] is not None:
                    if methods_data[method] < 20:  # Leave some headroom
                        max_configs.append((batch * length, batch, length, methods_data[method]))
            
            if max_configs:
                max_configs.sort(reverse=True)
                batch, length, mem = max_configs[0][1:4]
                print(f"  {method:12}: Batch={batch}, Length={length} ({mem:.1f} GB)")
    
    # Memory efficiency
    print("\nMemory Efficiency Gains:")
    for (batch, length), methods_data in results.items():
        if all(m in methods_data and methods_data[m] is not None and methods_data[m] != float('inf') 
               for m in ['naive', 'flash']):
            ratio = methods_data['naive'] / methods_data['flash']
            print(f"  B={batch}, L={length}: Flash uses {ratio:.1f}x less memory than naive")


if __name__ == "__main__":
    main()