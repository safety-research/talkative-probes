#!/usr/bin/env python3
"""Debug generation step by step."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_generation():
    """Debug generation step by step."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    print("Step-by-step KV generation debug")
    print("=" * 60)
    
    # Manually do what generate_soft_kv_cached does
    B, d_model = activation.shape
    emb_a = decoder.proj(activation)
    
    # Build prompt
    prompt_left_emb = decoder.prompt_left_emb
    prompt_right_emb = decoder.prompt_right_emb
    
    left_prompt_embs = prompt_left_emb.unsqueeze(0).expand(B, -1, -1)
    right_prompt_embs = prompt_right_emb.unsqueeze(0).expand(B, -1, -1)
    prompt_embs = torch.cat([left_prompt_embs, emb_a.unsqueeze(1), right_prompt_embs], dim=1)
    
    print(f"Prompt shape: {prompt_embs.shape}")
    
    # Get transformer and output layer
    transformer = decoder.base.transformer if hasattr(decoder.base, 'transformer') else decoder.base
    main_out = decoder.out
    
    # Process prompt
    from lens.models.kv_cache import compute_with_kv_cache
    
    with torch.no_grad():
        # Process initial prompt
        hidden_states, kv_cache = compute_with_kv_cache(
            transformer, 
            prompt_embs,
            use_cache=True
        )
        
        print(f"\nAfter initial prompt:")
        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"  Hidden states norm: {hidden_states.norm():.3f}")
        
        # Get logits - try different ways
        # Method 1: Last position of all hidden states
        logits_method1 = main_out(hidden_states[:, -1])
        print(f"\nMethod 1 (hidden[:, -1]):")
        print(f"  Logits shape: {logits_method1.shape}")
        print(f"  Logits norm: {logits_method1.norm():.3f}")
        print(f"  Top 5: {[f'{v:.3f}' for v in logits_method1[0].topk(5).values.tolist()]}")
        
        # Method 2: Last position with extra dim
        logits_method2 = main_out(hidden_states[:, -1:])
        print(f"\nMethod 2 (hidden[:, -1:]):")
        print(f"  Logits shape: {logits_method2.shape}")
        print(f"  Logits norm: {logits_method2.norm():.3f}")
        if logits_method2.dim() == 3:
            print(f"  Top 5: {[f'{v:.3f}' for v in logits_method2[0, 0].topk(5).values.tolist()]}")
        else:
            print(f"  Top 5: {[f'{v:.3f}' for v in logits_method2[0].topk(5).values.tolist()]}")
        
        # Check what the actual KV generation does
        print(f"\n\nRunning actual KV generation:")
        gen_kv = decoder.generate_soft_kv_cached(
            activation.clone(), max_length=1, gumbel_tau=0.0
        )
        print(f"  First logits: {[f'{v:.3f}' for v in gen_kv.raw_lm_logits[0, 0].topk(5).values.tolist()]}")
        
        # Also test Flash
        print(f"\n\nRunning Flash generation:")
        gen_flash = decoder.generate_soft_kv_flash(
            activation.clone(), max_length=1, gumbel_tau=0.0
        )
        print(f"  First logits: {[f'{v:.3f}' for v in gen_flash.raw_lm_logits[0, 0].topk(5).values.tolist()]}")


if __name__ == "__main__":
    debug_generation()