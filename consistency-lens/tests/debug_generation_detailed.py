#!/usr/bin/env python3
"""Detailed debug of generation issue."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.kv_cache import compute_with_kv_cache, KVCache


def debug_generation_detailed():
    """Debug generation in detail."""
    
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
    
    # Set prompt
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    # Test activation
    torch.manual_seed(42)
    activation = torch.randn(1, 768, device=device)
    
    print("Detailed generation debugging")
    print("=" * 60)
    
    # Manually trace through generate_soft_kv_cached
    B = 1
    
    # Get components (mimicking generate_soft_kv_cached)
    main_base = decoder.base
    main_out = decoder.out
    
    # Get embeddings
    emb_a = decoder.proj(activation)
    left_prompt_embs = decoder.prompt_left_emb.unsqueeze(0).expand(B, -1, -1)
    right_prompt_embs = decoder.prompt_right_emb.unsqueeze(0).expand(B, -1, -1)
    
    # Build sequence
    parts = []
    if left_prompt_embs is not None:
        parts.append(left_prompt_embs)
    parts.append(emb_a.unsqueeze(1))
    if right_prompt_embs is not None:
        parts.append(right_prompt_embs)
    seq_embs = torch.cat(parts, dim=1)
    
    print(f"Sequence shape: {seq_embs.shape}")
    print(f"Parts: {[p.shape for p in parts]}")
    
    # Get transformer
    transformer = main_base.transformer if hasattr(main_base, 'transformer') else main_base.model
    
    # Process with KV cache
    kv_cache = KVCache()
    hidden_states, kv_cache = compute_with_kv_cache(
        transformer, seq_embs, kv_cache, position_offset=0
    )
    
    print(f"\nAfter compute_with_kv_cache:")
    print(f"  Hidden states shape: {hidden_states.shape}")
    print(f"  Hidden states last pos norm: {hidden_states[:, -1].norm().item():.3f}")
    
    # Get logits (this is line 630 in generate_soft_kv_cached)
    logits_t = main_out(hidden_states[:, -1])
    print(f"\nLogits computation:")
    print(f"  Logits shape: {logits_t.shape}")
    print(f"  Logits top 5: {[f'{v:.3f}' for v in logits_t[0].topk(5).values.tolist()]}")
    
    # Compare with manual forward
    print("\n\nComparing with manual forward:")
    outputs = transformer(inputs_embeds=seq_embs)
    hidden_manual = outputs.last_hidden_state
    logits_manual = main_out(hidden_manual[:, -1])
    print(f"  Manual hidden last pos norm: {hidden_manual[:, -1].norm().item():.3f}")
    print(f"  Manual logits top 5: {[f'{v:.3f}' for v in logits_manual[0].topk(5).values.tolist()]}")
    
    # Check if hidden states are different
    print(f"\n  Hidden states match: {torch.allclose(hidden_states, hidden_manual)}")
    if not torch.allclose(hidden_states, hidden_manual):
        diff = (hidden_states - hidden_manual).abs()
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")
    
    # Now let's call the actual method and see what happens
    print("\n\nCalling generate_soft_kv_cached:")
    with torch.no_grad():
        gen_result = decoder.generate_soft_kv_cached(
            activation.clone(),
            max_length=1,
            gumbel_tau=0.0,
            print_prompt=False
        )
    
    print(f"  Result logits shape: {gen_result.raw_lm_logits.shape}")
    print(f"  Result logits top 5: {[f'{v:.3f}' for v in gen_result.raw_lm_logits[0, 0].topk(5).values.tolist()]}")
    
    # Check if there's a difference in the sequence construction
    print("\n\nChecking sequence construction in generate_soft_kv_cached:")
    # Look at the exact order of parts in the method
    print("  Expected order: [left_prompt, activation, right_prompt]")
    print(f"  Our order has {len(parts)} parts")


if __name__ == "__main__":
    debug_generation_detailed()