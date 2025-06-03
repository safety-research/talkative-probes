#!/usr/bin/env python3
"""Test autoregressive generation step by step."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.flash_kv_cache_v2 import FLASH_AVAILABLE


def test_autoregressive():
    """Test autoregressive generation step by step."""
    
    if not FLASH_AVAILABLE:
        print("Flash Attention not available")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available")
        return
    
    print("Autoregressive Generation Test")
    print("=" * 60)
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create two decoders
    decoder_kv = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        use_flash_attention=False,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    decoder_flash = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=False,
        use_flash_attention=True,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    # Ensure same weights
    decoder_flash.load_state_dict(decoder_kv.state_dict(), strict=False)
    
    # Set same prompt
    decoder_kv.set_prompt("explain <embed>:", tokenizer)
    decoder_flash.set_prompt("explain <embed>:", tokenizer)
    
    # Test activation
    torch.manual_seed(42)
    activation = torch.randn(1, 768, device=device)
    
    # Generate just 3 tokens to debug
    max_length = 3
    
    print(f"\nGenerating {max_length} tokens...")
    
    with torch.no_grad():
        gen_kv = decoder_kv.generate_soft_kv_cached(
            activation.clone(), max_length=max_length, gumbel_tau=0.0
        )
        
        gen_flash = decoder_flash.generate_soft_kv_flash(
            activation.clone(), max_length=max_length, gumbel_tau=0.0
        )
    
    print(f"\nStep-by-step comparison:")
    for step in range(max_length):
        print(f"\nStep {step}:")
        
        # Compare logits
        logits_kv = gen_kv.raw_lm_logits[0, step]
        logits_flash = gen_flash.raw_lm_logits[0, step]
        
        # Top predictions
        kv_top5 = logits_kv.topk(5)
        flash_top5 = logits_flash.topk(5)
        
        print(f"  KV top-5 values: {[f'{v:.3f}' for v in kv_top5.values.tolist()]}")
        print(f"  Flash top-5 values: {[f'{v:.3f}' for v in flash_top5.values.tolist()]}")
        
        # Check if tokens match
        kv_token = gen_kv.hard_token_ids[0, step].item()
        flash_token = gen_flash.hard_token_ids[0, step].item()
        
        print(f"  KV selected token: {kv_token} ({tokenizer.decode(kv_token)})")
        print(f"  Flash selected token: {flash_token} ({tokenizer.decode(flash_token)})")
        
        # Logit difference
        diff = (logits_flash - logits_kv).abs()
        print(f"  Max logit diff: {diff.max().item():.3f}")
        print(f"  Mean logit diff: {diff.mean().item():.3f}")
    
    # Check embeddings
    print(f"\n\nEmbedding comparison:")
    for step in range(max_length):
        emb_kv = gen_kv.generated_text_embeddings[0, step]
        emb_flash = gen_flash.generated_text_embeddings[0, step]
        
        emb_diff = (emb_flash - emb_kv).abs()
        print(f"  Step {step}: max diff = {emb_diff.max().item():.6f}, mean = {emb_diff.mean().item():.6f}")
    
    # Decode the generated sequences
    print(f"\n\nGenerated sequences:")
    kv_tokens = gen_kv.hard_token_ids[0].tolist()
    flash_tokens = gen_flash.hard_token_ids[0].tolist()
    
    print(f"  KV: {tokenizer.decode(kv_tokens)}")
    print(f"  Flash: {tokenizer.decode(flash_tokens)}")
    
    # Check if the issue is cumulative
    print(f"\n\nChecking if errors accumulate:")
    total_diff = 0
    for step in range(max_length):
        step_diff = (gen_flash.raw_lm_logits[0, step] - gen_kv.raw_lm_logits[0, step]).abs().mean().item()
        total_diff += step_diff
        print(f"  Step {step}: mean diff = {step_diff:.3f}, cumulative = {total_diff:.3f}")


if __name__ == "__main__":
    test_autoregressive()