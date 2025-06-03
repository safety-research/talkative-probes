#!/usr/bin/env python3
"""Test memory usage during backpropagation for all generation methods."""

import torch
import gc
import time
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
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return allocated, reserved, peak
    return 0, 0, 0


def reset_memory():
    """Reset GPU memory stats."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def test_backprop_memory():
    """Test memory usage during full forward + backward pass."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, cannot measure GPU memory")
        return
    
    print("Memory Usage During Backpropagation Test")
    print("=" * 80)
    print("Testing full forward + backward pass with gradient accumulation")
    print()
    
    # Initialize models
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    d_model = 768
    
    # Test configurations
    test_configs = [
        (2, 16),   # Small
        (4, 32),   # Medium
        (4, 64),   # Large
        (2, 128),  # Very large
    ]
    
    # Methods to test
    methods = ["naive", "kv_cache"]
    if FLASH_AVAILABLE:
        methods.append("flash")
    
    results = {}
    
    for batch_size, seq_length in test_configs:
        print(f"\nBatch size: {batch_size}, Sequence length: {seq_length}")
        print("-" * 70)
        
        config_results = {}
        
        for method in methods:
            # Create models
            if method == "naive":
                decoder_config = DecoderConfig(
                    model_name=model_name,
                    use_kv_cache=False,
                    use_flash_attention=False,
                    base_model=False,
                    projection_layer=True,
                    eye_init=True,
                )
            elif method == "kv_cache":
                decoder_config = DecoderConfig(
                    model_name=model_name,
                    use_kv_cache=True,
                    use_flash_attention=False,
                    base_model=False,
                    projection_layer=True,
                    eye_init=True,
                )
            else:  # flash
                decoder_config = DecoderConfig(
                    model_name=model_name,
                    use_kv_cache=False,
                    use_flash_attention=True,
                    base_model=False,
                    projection_layer=True,
                    eye_init=True,
                )
            
            decoder = Decoder(decoder_config).to(device)
            encoder = Encoder(EncoderConfig(
                model_name=model_name,
                base_model=False,
                projection_layer=True,
                eye_init=True,
            )).to(device)
            
            decoder.set_prompt("explain <embed>:", tokenizer)
            
            # Reset memory
            reset_memory()
            
            # Measure memory before
            alloc_before, _, _ = get_memory_stats()
            
            # Create batch
            activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
            
            # Time the operation
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Forward pass
            if method == "naive":
                gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
            elif method == "kv_cache":
                gen = decoder.generate_soft_kv_cached(activation, max_length=seq_length, gumbel_tau=1.0)
            else:  # flash
                gen = decoder.generate_soft_kv_flash(activation, max_length=seq_length, gumbel_tau=1.0)
            
            # Get memory after forward
            alloc_after_forward, _, _ = get_memory_stats()
            forward_memory = alloc_after_forward - alloc_before
            
            # Full reconstruction and loss
            reconstructed = encoder(gen.generated_text_embeddings)
            loss = torch.nn.functional.mse_loss(reconstructed, activation.detach())
            
            # Backward pass
            loss.backward()
            
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            
            # Get final memory stats
            alloc_after_backward, _, peak_memory = get_memory_stats()
            backward_memory = alloc_after_backward - alloc_after_forward
            total_memory = alloc_after_backward - alloc_before
            
            # Get gradient memory estimate
            grad_memory = sum(p.grad.numel() * p.grad.element_size() if p.grad is not None else 0 
                            for p in decoder.parameters()) / 1024**3
            grad_memory += sum(p.grad.numel() * p.grad.element_size() if p.grad is not None else 0 
                             for p in encoder.parameters()) / 1024**3
            
            config_results[method] = {
                'forward_memory': forward_memory,
                'backward_memory': backward_memory,
                'total_memory': total_memory,
                'peak_memory': peak_memory,
                'grad_memory': grad_memory,
                'time': elapsed_time
            }
            
            print(f"\n  {method.upper()}:")
            print(f"    Forward memory:   {forward_memory:.3f} GB")
            print(f"    Backward memory:  {backward_memory:.3f} GB (additional)")
            print(f"    Total memory:     {total_memory:.3f} GB")
            print(f"    Peak memory:      {peak_memory:.3f} GB")
            print(f"    Gradient storage: {grad_memory:.3f} GB")
            print(f"    Time:            {elapsed_time:.3f}s")
            
            # Cleanup
            decoder.zero_grad()
            encoder.zero_grad()
            del decoder, encoder, gen, activation, loss
        
        results[(batch_size, seq_length)] = config_results
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("BACKPROPAGATION MEMORY PENALTY SUMMARY")
    print("=" * 80)
    
    print("\nTotal Memory During Training (Forward + Backward):")
    print(f"{'Config':<15} {'Naive':<15} {'KV Cache':<15} {'Flash':<15} {'Flash vs Naive':<20}")
    print("-" * 80)
    
    for (batch, length), methods_data in results.items():
        config_str = f"B={batch}, L={length}"
        row = [config_str]
        
        naive_total = methods_data.get('naive', {}).get('total_memory', 0)
        
        for method in ['naive', 'kv_cache', 'flash']:
            if method in methods_data:
                total = methods_data[method]['total_memory']
                row.append(f"{total:.3f} GB")
            else:
                row.append("N/A")
        
        # Add savings comparison
        if 'flash' in methods_data and naive_total > 0:
            flash_total = methods_data['flash']['total_memory']
            savings = (1 - flash_total / naive_total) * 100
            row.append(f"{savings:.1f}% savings")
        else:
            row.append("N/A")
        
        print(f"{row[0]:<15} {row[1]:<15} {row[2]:<15} {row[3]:<15} {row[4]:<20}")
    
    print("\nBackward Pass Memory Overhead (Additional memory for gradients):")
    print(f"{'Config':<15} {'Naive':<15} {'KV Cache':<15} {'Flash':<15}")
    print("-" * 60)
    
    for (batch, length), methods_data in results.items():
        config_str = f"B={batch}, L={length}"
        row = [config_str]
        
        for method in ['naive', 'kv_cache', 'flash']:
            if method in methods_data:
                backward = methods_data[method]['backward_memory']
                row.append(f"{backward:.3f} GB")
            else:
                row.append("N/A")
        
        print(f"{row[0]:<15} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
    
    # Analyze scaling
    print("\nMemory Scaling Analysis:")
    print("-" * 40)
    
    # Check if backward memory scales with sequence length
    for method in methods:
        print(f"\n{method.upper()} scaling:")
        method_results = [(k[1], v[method]['backward_memory']) 
                         for k, v in results.items() if method in v]
        method_results.sort()
        
        if len(method_results) >= 2:
            for i in range(1, len(method_results)):
                length1, mem1 = method_results[i-1]
                length2, mem2 = method_results[i]
                ratio = length2 / length1
                mem_ratio = mem2 / mem1 if mem1 > 0 else 0
                print(f"  {length1} → {length2} tokens: {ratio:.1f}x length, {mem_ratio:.1f}x backward memory")


def test_realistic_training_step():
    """Test memory with a realistic training step including all losses."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("\nCUDA not available for realistic training test")
        return
    
    print("\n" + "=" * 80)
    print("Realistic Training Step Memory Test")
    print("=" * 80)
    print("Including KL divergence, language modeling, and entropy losses")
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test medium-sized batch
    batch_size = 4
    seq_length = 64
    d_model = 768
    layer = 6
    
    methods = ["naive", "kv_cache"]
    if FLASH_AVAILABLE:
        methods.append("flash")
    
    for method in methods:
        print(f"\n{method.upper()} method:")
        print("-" * 40)
        
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
        
        # Create realistic batch
        batch = {
            "A": torch.randn(batch_size, d_model, device=device),
            "A_prime": torch.randn(batch_size, d_model, device=device),
            "input_ids_A": torch.randint(0, 50257, (batch_size, 128), device=device),
            "layer_idx": torch.tensor([layer] * batch_size, device=device).unsqueeze(1),
            "token_pos_A": torch.tensor([64] * batch_size, device=device).unsqueeze(1),
        }
        
        # Loss function parameters
        loss_fns = {
            "T_text": seq_length,
            "tau": 1.0,
            "alpha": 0.1,
            "kl_base_weight": 1.0,
            "entropy_weight": 0.01,
            "mse_weight": 0.0,
            "lm_weight": 0.1,
        }
        
        # Reset memory
        reset_memory()
        alloc_before, _, _ = get_memory_stats()
        
        # Cache natural language prefix for LM loss
        lm_loss_natural_prefix = "explain something:"
        cached_prefix_ids = tokenizer(lm_loss_natural_prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        # Run training step
        torch.cuda.synchronize()
        start_time = time.time()
        
        losses = train_step(batch, models, loss_fns, tokenizer, cached_prefix_ids)
        total_loss = losses["total"]
        
        # Get memory after forward
        alloc_after_forward, _, _ = get_memory_stats()
        
        # Backward pass
        total_loss.backward()
        
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        # Get final memory stats
        alloc_after_backward, _, peak_memory = get_memory_stats()
        
        forward_memory = alloc_after_forward - alloc_before
        backward_memory = alloc_after_backward - alloc_after_forward
        total_memory = alloc_after_backward - alloc_before
        
        print(f"  Forward memory:  {forward_memory:.3f} GB")
        print(f"  Backward memory: {backward_memory:.3f} GB")
        print(f"  Total memory:    {total_memory:.3f} GB")
        print(f"  Peak memory:     {peak_memory:.3f} GB")
        print(f"  Time:           {elapsed_time:.3f}s")
        print(f"  Losses: KL={losses['kl']:.4f}, LM={losses['lm']:.4f}, Total={total_loss.item():.4f}")
        
        # Cleanup
        decoder.zero_grad()
        encoder.zero_grad()
        orig_model.zero_grad()
        del models, batch, losses, total_loss


if __name__ == "__main__":
    test_backprop_memory()
    test_realistic_training_step()