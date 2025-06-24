#!/usr/bin/env python3
"""Test Flash Attention KV cache implementation."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.flash_kv_cache import FLASH_AVAILABLE


def test_flash_attention_generation():
    """Test that Flash Attention generation produces valid outputs."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Flash Attention Test")
    print("=" * 60)
    print(f"Flash Attention available: {FLASH_AVAILABLE}")
    
    # Initialize models
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create decoder with Flash Attention enabled
    decoder_config = DecoderConfig(
        model_name=model_name,
        use_flash_attention=True,
        base_model=False,
        projection_layer=True,
        output_head=False,
        trainable_prompts=True,
    )
    decoder = Decoder(decoder_config).to(device)
    
    # Set prompt
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    if not FLASH_AVAILABLE:
        print("\nFlash Attention not installed.")
        print("To install: make flash-attention")
        print("\nTesting that Flash Attention request fails properly...")
        
        # Test parameters
        batch_size = 2
        d_model = decoder.base.config.hidden_size
        t_text = 16
        
        # Create test activation
        activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
        
        try:
            gen = decoder.generate_soft_kv_flash(activation, max_length=t_text, gumbel_tau=1.0)
            print("✗ ERROR: Flash Attention fallback should not happen!")
        except RuntimeError as e:
            print(f"✓ Correctly raised error: {e}")
        return
    
    print("\nTesting Flash Attention generation...")
    
    # Test parameters
    batch_size = 2
    d_model = decoder.base.config.hidden_size
    t_text = 16
    
    # Create test activation
    activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    
    try:
        # Generate with Flash Attention
        gen = decoder.generate_soft_kv_flash(activation, max_length=t_text, gumbel_tau=1.0)
        
        print(f"✓ Generation successful!")
        print(f"  Output shape: {gen.generated_text_embeddings.shape}")
        print(f"  Logits shape: {gen.raw_lm_logits.shape}")
        print(f"  Token IDs shape: {gen.hard_token_ids.shape}")
        
        # Test gradient flow
        print("\nTesting gradient flow...")
        loss = gen.generated_text_embeddings.sum()
        loss.backward()
        
        if activation.grad is not None and activation.grad.abs().max() > 0:
            print(f"✓ Gradients flow correctly!")
            print(f"  Activation grad norm: {activation.grad.norm():.6f}")
        else:
            print("✗ No gradients detected!")
            
    except Exception as e:
        print(f"✗ Flash Attention generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_flash_vs_standard_equivalence():
    """Test that Flash Attention produces similar results to standard KV cache."""
    
    if not FLASH_AVAILABLE:
        print("\nFlash Attention not available, skipping equivalence test.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 60)
    print("Flash Attention vs Standard KV Cache Equivalence Test")
    print("=" * 60)
    
    # Initialize models
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create two decoders
    decoder_flash = Decoder(DecoderConfig(
        model_name=model_name,
        use_flash_attention=True,
        base_model=False,
        projection_layer=True,
        trainable_prompts=True,
    )).to(device)
    
    decoder_kv = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        base_model=False,
        projection_layer=True,
        trainable_prompts=True,
    )).to(device)
    
    # Ensure they have the same weights
    decoder_flash.load_state_dict(decoder_kv.state_dict())
    
    # Set same prompt
    decoder_flash.set_prompt("explain <embed>:", tokenizer)
    decoder_kv.set_prompt("explain <embed>:", tokenizer)
    
    # Test parameters
    batch_size = 1
    d_model = decoder_flash.base.config.hidden_size
    t_text = 8
    
    # Create test activation
    torch.manual_seed(42)
    activation = torch.randn(batch_size, d_model, device=device)
    
    # Set deterministic mode for comparison
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    try:
        # Generate with both methods
        print("Generating with Flash Attention...")
        gen_flash = decoder_flash.generate_soft_kv_flash(
            activation.clone(), max_length=t_text, gumbel_tau=0.0  # tau=0 for deterministic
        )
        
        print("Generating with standard KV cache...")
        gen_kv = decoder_kv.generate_soft_kv_cached(
            activation.clone(), max_length=t_text, gumbel_tau=0.0
        )
        
        # Compare outputs
        print("\nComparing outputs...")
        
        # Token IDs should match exactly
        tokens_match = torch.allclose(gen_flash.hard_token_ids, gen_kv.hard_token_ids)
        print(f"  Token IDs match: {tokens_match}")
        
        # Embeddings should be very close
        emb_diff = (gen_flash.generated_text_embeddings - gen_kv.generated_text_embeddings).abs().max()
        print(f"  Max embedding difference: {emb_diff:.6f}")
        
        # Logits should be very close
        logit_diff = (gen_flash.raw_lm_logits - gen_kv.raw_lm_logits).abs().max()
        print(f"  Max logit difference: {logit_diff:.6f}")
        
        if tokens_match and emb_diff < 1e-4 and logit_diff < 1e-4:
            print("\n✓ Flash Attention produces equivalent results!")
        else:
            print("\n✗ Results differ significantly")
            
    except Exception as e:
        print(f"\n✗ Equivalence test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        torch.use_deterministic_algorithms(False)


def test_flash_performance():
    """Benchmark Flash Attention performance vs standard KV cache."""
    
    if not FLASH_AVAILABLE:
        print("\nFlash Attention not available, skipping performance test.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 60)
    print("Flash Attention Performance Benchmark")
    print("=" * 60)
    
    import time
    
    # Initialize models
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    decoder_config_flash = DecoderConfig(
        model_name=model_name,
        use_flash_attention=True,
        base_model=False,
        projection_layer=True,
    )
    decoder_flash = Decoder(decoder_config_flash).to(device)
    
    decoder_config_kv = DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        base_model=False,
        projection_layer=True,
    )
    decoder_kv = Decoder(decoder_config_kv).to(device)
    
    # Set prompts
    decoder_flash.set_prompt("explain <embed>:", tokenizer)
    decoder_kv.set_prompt("explain <embed>:", tokenizer)
    
    # Test configurations
    test_configs = [
        (2, 16),   # batch_size, t_text
        (4, 32),
        (8, 64),
    ]
    
    d_model = decoder_flash.base.config.hidden_size
    
    for batch_size, t_text in test_configs:
        print(f"\nBatch size: {batch_size}, Sequence length: {t_text}")
        
        # Create activation
        activation = torch.randn(batch_size, d_model, device=device)
        
        # Warm up
        _ = decoder_flash.generate_soft_kv_flash(activation, max_length=4, gumbel_tau=1.0)
        _ = decoder_kv.generate_soft_kv_cached(activation, max_length=4, gumbel_tau=1.0)
        
        # Time Flash Attention
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(5):
            _ = decoder_flash.generate_soft_kv_flash(activation, max_length=t_text, gumbel_tau=1.0)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        flash_time = (time.time() - start) / 5
        
        # Time standard KV cache
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(5):
            _ = decoder_kv.generate_soft_kv_cached(activation, max_length=t_text, gumbel_tau=1.0)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        kv_time = (time.time() - start) / 5
        
        speedup = kv_time / flash_time
        print(f"  Flash Attention: {flash_time:.4f}s")
        print(f"  Standard KV: {kv_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    test_flash_attention_generation()
    test_flash_vs_standard_equivalence()
    test_flash_performance()