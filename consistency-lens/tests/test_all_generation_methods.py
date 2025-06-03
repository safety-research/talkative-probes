#!/usr/bin/env python3
"""Comprehensive test suite for all generation methods: Naive, KV Cache, and Flash Attention."""

import torch
import time
import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.flash_kv_cache import FLASH_AVAILABLE


@dataclass
class TestResult:
    """Results from a single test."""
    method: str
    batch_size: int
    seq_length: int
    time: float
    memory_used: float
    peak_memory: float
    passed: bool
    error: str = ""


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


def test_generation_method(
    method: str,
    decoder: Decoder,
    activation: torch.Tensor,
    seq_length: int,
    gumbel_tau: float = 1.0
) -> Tuple[torch.Tensor, float, float, float]:
    """Test a single generation method and return results."""
    
    # Reset memory tracking
    reset_memory()
    alloc_before, _, _ = get_memory_stats()
    
    # Time the generation
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    # Generate based on method
    if method == "naive":
        gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=gumbel_tau)
    elif method == "kv_cache":
        gen = decoder.generate_soft_kv_cached(activation, max_length=seq_length, gumbel_tau=gumbel_tau)
    elif method == "flash":
        gen = decoder.generate_soft_kv_flash(activation, max_length=seq_length, gumbel_tau=gumbel_tau)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Force computation
    _ = gen.generated_text_embeddings.sum()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_time = time.time() - start_time
    
    # Get memory stats
    alloc_after, _, peak_memory = get_memory_stats()
    memory_used = alloc_after - alloc_before
    
    return gen, elapsed_time, memory_used, peak_memory


def run_comprehensive_tests() -> List[TestResult]:
    """Run tests on all generation methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    print("Comprehensive Generation Methods Test")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Flash Attention available: {FLASH_AVAILABLE}")
    print()
    
    # Initialize models
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    d_model = 768  # GPT-2 hidden size
    
    # Test configurations
    test_configs = [
        (1, 16),   # Very short
        (2, 32),   # Short
        (4, 64),   # Medium
        (2, 128),  # Long
    ]
    
    # Methods to test
    methods = ["naive", "kv_cache"]
    if FLASH_AVAILABLE:
        methods.append("flash")
    
    for batch_size, seq_length in test_configs:
        print(f"\nTesting Batch={batch_size}, Length={seq_length}")
        print("-" * 60)
        
        for method in methods:
            try:
                # Create decoder with appropriate config
                if method == "naive":
                    config = DecoderConfig(
                        model_name=model_name,
                        use_kv_cache=False,
                        use_flash_attention=False,
                        base_model=False,
                        projection_layer=True,
                    )
                elif method == "kv_cache":
                    config = DecoderConfig(
                        model_name=model_name,
                        use_kv_cache=True,
                        use_flash_attention=False,
                        base_model=False,
                        projection_layer=True,
                    )
                else:  # flash
                    config = DecoderConfig(
                        model_name=model_name,
                        use_kv_cache=False,
                        use_flash_attention=True,
                        base_model=False,
                        projection_layer=True,
                    )
                
                decoder = Decoder(config).to(device)
                decoder.set_prompt("explain <embed>:", tokenizer)
                
                # Create activation
                activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
                
                # Run test
                gen, elapsed_time, memory_used, peak_memory = test_generation_method(
                    method, decoder, activation, seq_length
                )
                
                # Verify output shapes
                assert gen.generated_text_embeddings.shape == (batch_size, seq_length, d_model)
                assert gen.raw_lm_logits.shape == (batch_size, seq_length, 50257)  # GPT-2 vocab size
                assert gen.hard_token_ids.shape == (batch_size, seq_length)
                
                result = TestResult(
                    method=method,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    time=elapsed_time,
                    memory_used=memory_used,
                    peak_memory=peak_memory,
                    passed=True
                )
                results.append(result)
                
                print(f"  {method:12} - Time: {elapsed_time:.3f}s, Memory: {memory_used:.3f}GB, Peak: {peak_memory:.3f}GB ✓")
                
                # Cleanup
                del decoder, gen, activation
                
            except Exception as e:
                result = TestResult(
                    method=method,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    time=0,
                    memory_used=0,
                    peak_memory=0,
                    passed=False,
                    error=str(e)
                )
                results.append(result)
                print(f"  {method:12} - FAILED: {str(e)}")
    
    return results


def test_gradient_flow():
    """Test that gradients flow through all methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 80)
    print("Gradient Flow Test")
    print("=" * 80)
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    d_model = 768
    batch_size = 2
    seq_length = 16
    
    methods = ["naive", "kv_cache"]
    if FLASH_AVAILABLE:
        methods.append("flash")
    
    # Also need encoder for full gradient test
    encoder = Encoder(EncoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    for method in methods:
        print(f"\nTesting {method}...")
        
        # Create decoder
        if method == "naive":
            config = DecoderConfig(
                model_name=model_name,
                use_kv_cache=False,
                use_flash_attention=False,
                base_model=False,
                projection_layer=True,
            )
        elif method == "kv_cache":
            config = DecoderConfig(
                model_name=model_name,
                use_kv_cache=True,
                use_flash_attention=False,
                base_model=False,
                projection_layer=True,
            )
        else:  # flash
            config = DecoderConfig(
                model_name=model_name,
                use_kv_cache=False,
                use_flash_attention=True,
                base_model=False,
                projection_layer=True,
            )
        
        decoder = Decoder(config).to(device)
        decoder.set_prompt("explain <embed>:", tokenizer)
        
        # Create activation that requires grad
        activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        # Forward pass
        if method == "naive":
            gen = decoder.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
        elif method == "kv_cache":
            gen = decoder.generate_soft_kv_cached(activation, max_length=seq_length, gumbel_tau=1.0)
        else:  # flash
            gen = decoder.generate_soft_kv_flash(activation, max_length=seq_length, gumbel_tau=1.0)
        
        # Encode back
        reconstructed = encoder(gen.generated_text_embeddings)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(reconstructed, activation.detach())
        
        # Backward
        loss.backward()
        
        # Check gradients
        grad_norm = activation.grad.norm().item()
        decoder_grads = sum(p.grad.norm().item() for p in decoder.parameters() if p.grad is not None)
        encoder_grads = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None)
        
        print(f"  Activation grad norm: {grad_norm:.4f}")
        print(f"  Decoder grad norm: {decoder_grads:.4f}")
        print(f"  Encoder grad norm: {encoder_grads:.4f}")
        
        if grad_norm > 0 and decoder_grads > 0 and encoder_grads > 0:
            print(f"  ✓ Gradients flow correctly")
        else:
            print(f"  ✗ Gradient flow issue detected")
        
        # Cleanup
        decoder.zero_grad()
        encoder.zero_grad()
        if activation.grad is not None:
            activation.grad.zero_()
        del decoder, gen


def analyze_results(results: List[TestResult]):
    """Analyze and summarize test results."""
    
    print("\n" + "=" * 80)
    print("Summary Analysis")
    print("=" * 80)
    
    # Group by configuration
    configs = {}
    for r in results:
        if r.passed:
            key = (r.batch_size, r.seq_length)
            if key not in configs:
                configs[key] = {}
            configs[key][r.method] = r
    
    # Performance comparison table
    print("\nPerformance Comparison:")
    print(f"{'Config':<15} {'Naive':<20} {'KV Cache':<20} {'Flash Attention':<20}")
    print("-" * 75)
    
    for (batch, length), methods in sorted(configs.items()):
        config_str = f"B={batch}, L={length}"
        
        # Get baseline (naive) time
        naive_time = methods.get('naive', TestResult('', 0, 0, 1.0, 0, 0, False)).time
        
        row = [config_str]
        for method in ['naive', 'kv_cache', 'flash']:
            if method in methods:
                r = methods[method]
                speedup = naive_time / r.time if r.time > 0 else 0
                row.append(f"{r.time:.3f}s ({speedup:.2f}x)")
            else:
                row.append("N/A")
        
        print(f"{row[0]:<15} {row[1]:<20} {row[2]:<20} {row[3]:<20}")
    
    # Memory comparison table
    print("\nMemory Usage Comparison:")
    print(f"{'Config':<15} {'Naive':<20} {'KV Cache':<20} {'Flash Attention':<20}")
    print("-" * 75)
    
    for (batch, length), methods in sorted(configs.items()):
        config_str = f"B={batch}, L={length}"
        
        # Get baseline (naive) memory
        naive_mem = methods.get('naive', TestResult('', 0, 0, 0, 1.0, 0, False)).memory_used
        
        row = [config_str]
        for method in ['naive', 'kv_cache', 'flash']:
            if method in methods:
                r = methods[method]
                reduction = (1 - r.memory_used / naive_mem) * 100 if naive_mem > 0 else 0
                row.append(f"{r.memory_used:.3f}GB ({reduction:+.0f}%)")
            else:
                row.append("N/A")
        
        print(f"{row[0]:<15} {row[1]:<20} {row[2]:<20} {row[3]:<20}")
    
    # Recommendations
    print("\nRecommendations based on results:")
    print("-" * 40)
    
    # Find crossover points
    for (batch, length), methods in sorted(configs.items()):
        if 'naive' in methods and 'kv_cache' in methods:
            if methods['kv_cache'].time < methods['naive'].time:
                print(f"- Use KV Cache for sequences ≥ {length} tokens (observed {(methods['naive'].time/methods['kv_cache'].time):.1f}x speedup)")
                break
    
    if FLASH_AVAILABLE:
        # Find where Flash becomes beneficial
        flash_benefits = []
        for (batch, length), methods in sorted(configs.items()):
            if 'kv_cache' in methods and 'flash' in methods:
                mem_savings = (1 - methods['flash'].memory_used / methods['kv_cache'].memory_used) * 100
                if mem_savings > 20:  # Significant memory savings
                    flash_benefits.append((length, mem_savings))
        
        if flash_benefits:
            min_length = min(fb[0] for fb in flash_benefits)
            avg_savings = sum(fb[1] for fb in flash_benefits) / len(flash_benefits)
            print(f"- Use Flash Attention for sequences ≥ {min_length} tokens (average {avg_savings:.0f}% memory savings)")
    
    # Check for failures
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n⚠️  {len(failures)} tests failed:")
        for f in failures:
            print(f"  - {f.method} (B={f.batch_size}, L={f.seq_length}): {f.error}")


if __name__ == "__main__":
    # Run all tests
    results = run_comprehensive_tests()
    test_gradient_flow()
    analyze_results(results)