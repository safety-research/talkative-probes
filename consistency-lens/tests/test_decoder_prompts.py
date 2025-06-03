#!/usr/bin/env python3
"""Test decoder with prompts to isolate the issue."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.flash_kv_cache_v2 import FLASH_AVAILABLE


def test_decoder_prompts():
    """Test decoder prompt handling."""
    
    if not FLASH_AVAILABLE:
        print("Flash Attention not available")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available")
        return
    
    print("Decoder Prompt Test")
    print("=" * 60)
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create decoder
    decoder = Decoder(DecoderConfig(
        model_name=model_name,
        use_kv_cache=True,
        use_flash_attention=False,
        base_model=False,
        projection_layer=True,
    )).to(device)
    
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    # Test activation
    torch.manual_seed(42)
    activation = torch.randn(1, 768, device=device)
    
    print("Testing prompt processing...")
    
    # Get the projected activation
    emb_a = decoder.proj(activation)
    print(f"Projected activation norm: {emb_a.norm():.3f}")
    
    # Build prompt manually
    B = 1
    prompt_left_emb = decoder.prompt_left_emb
    prompt_right_emb = decoder.prompt_right_emb
    
    left_prompt_embs = prompt_left_emb.unsqueeze(0).expand(B, -1, -1)
    right_prompt_embs = prompt_right_emb.unsqueeze(0).expand(B, -1, -1)
    prompt_embs = torch.cat([left_prompt_embs, emb_a.unsqueeze(1), right_prompt_embs], dim=1)
    
    print(f"Prompt shape: {prompt_embs.shape}")
    print(f"Prompt norm: {prompt_embs.norm():.3f}")
    
    # Process through base model
    transformer = decoder.base.transformer if hasattr(decoder.base, 'transformer') else decoder.base
    
    # Test both methods
    from lens.models.kv_cache import compute_with_kv_cache
    from lens.models.flash_kv_cache_v2 import compute_with_flash_kv_cache
    
    # KV cache
    with torch.no_grad():
        hidden_kv, _ = compute_with_kv_cache(transformer, prompt_embs, use_cache=False)
        logits_kv = decoder.out(hidden_kv[:, -1, :])
    
    print(f"\nKV cache:")
    print(f"  Hidden norm: {hidden_kv.norm():.3f}")
    print(f"  Logits norm: {logits_kv.norm():.3f}")
    print(f"  Top 5 logits: {[f'{v:.3f}' for v in logits_kv[0].topk(5).values.tolist()]}")
    
    # Flash
    with torch.no_grad():
        hidden_flash, _ = compute_with_flash_kv_cache(transformer, prompt_embs, use_cache=False)
        logits_flash = decoder.out(hidden_flash[:, -1, :])
    
    print(f"\nFlash:")
    print(f"  Hidden norm: {hidden_flash.norm():.3f}")
    print(f"  Logits norm: {logits_flash.norm():.3f}")
    print(f"  Top 5 logits: {[f'{v:.3f}' for v in logits_flash[0].topk(5).values.tolist()]}")
    
    print(f"\nDifferences:")
    print(f"  Hidden diff: {(hidden_flash - hidden_kv).abs().max():.6f}")
    print(f"  Logits diff: {(logits_flash - logits_kv).abs().max():.6f}")
    
    # Now test with actual generation methods
    print(f"\n\nTesting generation methods:")
    
    # KV cached generation
    with torch.no_grad():
        gen_kv = decoder.generate_soft_kv_cached(
            activation.clone(), max_length=4, gumbel_tau=0.0
        )
    
    print(f"\nKV cached generation:")
    print(f"  Logits shape: {gen_kv.raw_lm_logits.shape}")
    print(f"  Logits norm: {gen_kv.raw_lm_logits.norm():.3f}")
    print(f"  First position logits: {[f'{v:.3f}' for v in gen_kv.raw_lm_logits[0, 0].topk(5).values.tolist()]}")
    
    # Use the same decoder but call the flash method
    # (The decoder has both methods available)
    with torch.no_grad():
        gen_flash = decoder.generate_soft_kv_flash(
            activation.clone(), max_length=4, gumbel_tau=0.0
        )
    
    print(f"\nFlash generation:")
    print(f"  Logits shape: {gen_flash.raw_lm_logits.shape}")
    print(f"  Logits norm: {gen_flash.raw_lm_logits.norm():.3f}")
    print(f"  First position logits: {[f'{v:.3f}' for v in gen_flash.raw_lm_logits[0, 0].topk(5).values.tolist()]}")
    
    print(f"\nGeneration differences:")
    print(f"  Max logit diff: {(gen_flash.raw_lm_logits - gen_kv.raw_lm_logits).abs().max():.6f}")
    print(f"  Mean logit diff: {(gen_flash.raw_lm_logits - gen_kv.raw_lm_logits).abs().mean():.6f}")


if __name__ == "__main__":
    test_decoder_prompts()