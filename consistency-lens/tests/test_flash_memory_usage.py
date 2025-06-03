#!/usr/bin/env python3
"""Test memory usage comparison between Flash Attention and standard KV cache."""

import torch
import gc
import time
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.flash_kv_cache import FLASH_AVAILABLE


def get_memory_stats():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return allocated, reserved
    return 0, 0


def reset_memory():
    """Reset GPU memory stats."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def test_memory_usage():
    """Compare memory usage between Flash Attention and standard KV cache."""
    
    if not FLASH_AVAILABLE:
        print("Flash Attention not available. Install with: make flash-attention")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, cannot measure GPU memory")
        return
    
    print("Memory Usage Comparison: Flash Attention vs Standard KV Cache")
    print("=" * 80)
    
    # Test configurations
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    d_model = 768  # GPT-2 hidden size
    
    test_configs = [
        # (batch_size, sequence_length)
        (1, 32),
        (2, 32),
        (4, 32),
        (1, 64),
        (2, 64),
        (4, 64),
        (1, 128),
        (2, 128),
    ]
    
    results = []
    
    for batch_size, seq_length in test_configs:
        print(f"\nBatch size: {batch_size}, Sequence length: {seq_length}")
        print("-" * 60)
        
        # Test Standard KV Cache
        reset_memory()
        
        decoder_config = DecoderConfig(
            model_name=model_name,
            use_kv_cache=True,
            use_flash_attention=False,
            base_model=False,
            projection_layer=True,
        )
        decoder = Decoder(decoder_config).to(device)
        decoder.set_prompt("explain <embed>:", tokenizer)
        
        # Create activation
        activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        # Measure memory before generation
        alloc_before, _ = get_memory_stats()
        
        # Generate with standard KV cache
        gen = decoder.generate_soft_kv_cached(activation, max_length=seq_length, gumbel_tau=1.0)
        
        # Force computation
        _ = gen.generated_text_embeddings.sum()
        
        torch.cuda.synchronize()
        alloc_after, _ = get_memory_stats()
        peak_memory_kv = torch.cuda.max_memory_allocated() / 1024**3
        
        kv_memory_delta = alloc_after - alloc_before
        
        # Cleanup
        del decoder, gen, activation
        
        # Test Flash Attention
        reset_memory()
        
        decoder_config = DecoderConfig(
            model_name=model_name,
            use_flash_attention=True,
            use_kv_cache=False,
            base_model=False,
            projection_layer=True,
        )
        decoder = Decoder(decoder_config).to(device)
        decoder.set_prompt("explain <embed>:", tokenizer)
        
        # Create activation
        activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        # Measure memory before generation
        alloc_before, _ = get_memory_stats()
        
        # Generate with Flash Attention
        gen = decoder.generate_soft_kv_flash(activation, max_length=seq_length, gumbel_tau=1.0)
        
        # Force computation
        _ = gen.generated_text_embeddings.sum()
        
        torch.cuda.synchronize()
        alloc_after, _ = get_memory_stats()
        peak_memory_flash = torch.cuda.max_memory_allocated() / 1024**3
        
        flash_memory_delta = alloc_after - alloc_before
        
        # Cleanup
        del decoder, gen, activation
        
        # Calculate savings
        memory_saved = kv_memory_delta - flash_memory_delta
        memory_ratio = flash_memory_delta / kv_memory_delta if kv_memory_delta > 0 else 1.0
        
        print(f"  Standard KV Cache:")
        print(f"    Memory increase: {kv_memory_delta:.3f} GB")
        print(f"    Peak memory: {peak_memory_kv:.3f} GB")
        
        print(f"  Flash Attention:")
        print(f"    Memory increase: {flash_memory_delta:.3f} GB")
        print(f"    Peak memory: {peak_memory_flash:.3f} GB")
        
        print(f"  Memory saved: {memory_saved:.3f} GB ({(1-memory_ratio)*100:.1f}% reduction)")
        
        results.append({
            'batch_size': batch_size,
            'seq_length': seq_length,
            'kv_memory': kv_memory_delta,
            'flash_memory': flash_memory_delta,
            'memory_saved': memory_saved,
            'memory_ratio': memory_ratio
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nMemory Usage (GB):")
    print(f"{'Config':<20} {'Standard KV':<15} {'Flash Attn':<15} {'Saved':<15} {'Reduction':<10}")
    print("-" * 80)
    
    for r in results:
        config = f"B={r['batch_size']}, L={r['seq_length']}"
        print(f"{config:<20} {r['kv_memory']:<15.3f} {r['flash_memory']:<15.3f} "
              f"{r['memory_saved']:<15.3f} {r['memory_ratio']*100:<10.1f}%")


def test_training_memory():
    """Test memory usage during training (forward + backward)."""
    
    if not FLASH_AVAILABLE:
        print("\nFlash Attention not available for training memory test.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("\nCUDA not available, cannot measure training memory")
        return
    
    print("\n" + "=" * 80)
    print("Training Memory Usage Comparison (Forward + Backward)")
    print("=" * 80)
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    d_model = 768
    
    # Test with typical training batch sizes
    test_configs = [
        (4, 32),   # Small batch, short sequence
        (8, 32),   # Medium batch, short sequence
        (4, 64),   # Small batch, medium sequence
        (8, 64),   # Medium batch, medium sequence
    ]
    
    for batch_size, seq_length in test_configs:
        print(f"\nBatch size: {batch_size}, Sequence length: {seq_length}")
        print("-" * 60)
        
        # Test Standard KV Cache
        reset_memory()
        
        decoder_config = DecoderConfig(
            model_name=model_name,
            use_kv_cache=True,
            use_flash_attention=False,
            base_model=False,
            projection_layer=True,
        )
        decoder = Decoder(decoder_config).to(device)
        encoder = Encoder(EncoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
        )).to(device)
        
        decoder.set_prompt("explain <embed>:", tokenizer)
        
        # Measure training step
        activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        alloc_before, _ = get_memory_stats()
        
        # Forward pass
        gen = decoder.generate_soft_kv_cached(activation, max_length=seq_length, gumbel_tau=1.0)
        reconstructed = encoder(gen.generated_text_embeddings)
        loss = torch.nn.functional.mse_loss(reconstructed, activation)
        
        # Backward pass
        loss.backward()
        
        torch.cuda.synchronize()
        alloc_after, _ = get_memory_stats()
        peak_memory_kv = torch.cuda.max_memory_allocated() / 1024**3
        
        kv_train_memory = alloc_after - alloc_before
        
        # Cleanup
        decoder.zero_grad()
        encoder.zero_grad()
        del decoder, encoder, gen, activation, loss
        
        # Test Flash Attention
        reset_memory()
        
        decoder_config = DecoderConfig(
            model_name=model_name,
            use_flash_attention=True,
            use_kv_cache=False,
            base_model=False,
            projection_layer=True,
        )
        decoder = Decoder(decoder_config).to(device)
        encoder = Encoder(EncoderConfig(
            model_name=model_name,
            base_model=False,
            projection_layer=True,
        )).to(device)
        
        decoder.set_prompt("explain <embed>:", tokenizer)
        
        # Measure training step
        activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        alloc_before, _ = get_memory_stats()
        
        # Forward pass
        gen = decoder.generate_soft_kv_flash(activation, max_length=seq_length, gumbel_tau=1.0)
        reconstructed = encoder(gen.generated_text_embeddings)
        loss = torch.nn.functional.mse_loss(reconstructed, activation)
        
        # Backward pass
        loss.backward()
        
        torch.cuda.synchronize()
        alloc_after, _ = get_memory_stats()
        peak_memory_flash = torch.cuda.max_memory_allocated() / 1024**3
        
        flash_train_memory = alloc_after - alloc_before
        
        # Cleanup
        decoder.zero_grad()
        encoder.zero_grad()
        del decoder, encoder, gen, activation, loss
        
        # Results
        memory_saved = kv_train_memory - flash_train_memory
        memory_ratio = flash_train_memory / kv_train_memory if kv_train_memory > 0 else 1.0
        
        print(f"  Standard KV Cache:")
        print(f"    Training memory: {kv_train_memory:.3f} GB")
        print(f"    Peak memory: {peak_memory_kv:.3f} GB")
        
        print(f"  Flash Attention:")
        print(f"    Training memory: {flash_train_memory:.3f} GB")
        print(f"    Peak memory: {peak_memory_flash:.3f} GB")
        
        print(f"  Memory saved: {memory_saved:.3f} GB ({(1-memory_ratio)*100:.1f}% reduction)")
    
    print("\n" + "=" * 80)
    print("Memory scaling analysis:")
    print("- Flash Attention uses O(n) memory vs O(n²) for standard attention")
    print("- Benefits increase with sequence length")
    print("- Training (with gradients) shows larger memory savings than inference")


if __name__ == "__main__":
    test_memory_usage()
    test_training_memory()