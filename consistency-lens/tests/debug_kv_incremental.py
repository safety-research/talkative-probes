#!/usr/bin/env python3
"""Debug incremental generation in KV cache."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def trace_generation_step_by_step():
    """Trace generation step by step to find where they diverge."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Step-by-Step Generation Trace")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    # Create two decoders
    decoder1 = Decoder(config).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    decoder2 = Decoder(config).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    decoder2.load_state_dict(decoder1.state_dict())
    
    d_model = decoder1.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Generate just 2 tokens to see where they diverge
    print("\nGenerating 2 tokens...")
    torch.manual_seed(123)
    gen1 = decoder1.generate_soft(activation.clone(), max_length=2, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen2 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=2, gumbel_tau=1.0)
    
    print("\nGenerated tokens:")
    for i in range(2):
        t1 = gen1.hard_token_ids[0, i].item()
        t2 = gen2.hard_token_ids[0, i].item()
        print(f"  Token {i}: {t1} vs {t2} ('{tokenizer.decode([t1])}' vs '{tokenizer.decode([t2])}')")
    
    # Check logits
    print("\nLogits comparison:")
    for i in range(2):
        diff = (gen1.raw_lm_logits[0, i] - gen2.raw_lm_logits[0, i]).abs().max().item()
        print(f"  Time {i} logits diff: {diff:.2e}")
        
        if diff > 1e-3:
            # Show top 5 for each
            logits1 = gen1.raw_lm_logits[0, i]
            logits2 = gen2.raw_lm_logits[0, i]
            
            top1 = logits1.topk(5)
            top2 = logits2.topk(5)
            
            print(f"    Top 5 from generate_soft:")
            for val, idx in zip(top1.values, top1.indices):
                print(f"      {idx.item()}: {val.item():.2f}")
                
            print(f"    Top 5 from generate_soft_kv_cached:")  
            for val, idx in zip(top2.values, top2.indices):
                print(f"      {idx.item()}: {val.item():.2f}")


def check_kv_cache_state():
    """Check the state of KV cache after initial pass."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nKV Cache State Analysis")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Manually run the initial KV cache setup
    from lens.models.kv_cache import KVCache, GPT2AttentionWithCache
    
    # Build initial sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
    a_proj = decoder._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
    seq_embs = torch.cat(parts, dim=1)
    
    print(f"Initial sequence shape: {seq_embs.shape}")
    
    # Process through first layer manually
    kv_cache = KVCache()
    transformer = decoder.base.transformer
    
    # Add position embeddings
    seq_length = seq_embs.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    position_embeds = transformer.wpe(position_ids[0])
    hidden_states = transformer.drop(seq_embs + position_embeds)
    
    print(f"Position IDs: {position_ids}")
    
    # Process first layer
    layer_0 = transformer.h[0]
    wrapper = GPT2AttentionWithCache(layer_0)
    hidden_states_out, kv_cache = wrapper(hidden_states, kv_cache, 0, use_cache=True)
    
    print(f"\nAfter first layer:")
    print(f"  KV cache has layer 0: {hasattr(kv_cache, 'keys') and len(kv_cache.keys) > 0}")
    if hasattr(kv_cache, 'keys') and len(kv_cache.keys) > 0:
        print(f"  Key shape: {kv_cache.keys[0].shape}")
        print(f"  Value shape: {kv_cache.values[0].shape}")
    
    # Check what happens when we process a new token
    print("\nProcessing new token...")
    new_token_emb = torch.randn(1, 1, d_model, device=device)
    new_position_ids = torch.tensor([[seq_length]], device=device)
    new_position_embeds = transformer.wpe(new_position_ids[0])
    new_hidden = transformer.drop(new_token_emb + new_position_embeds)
    
    print(f"New position IDs: {new_position_ids}")
    
    # Process through first layer with cache
    new_hidden_out, kv_cache = wrapper(new_hidden, kv_cache, 0, use_cache=True)
    
    print(f"After processing new token:")
    print(f"  Key shape: {kv_cache.keys[0].shape}")
    print(f"  Value shape: {kv_cache.values[0].shape}")


if __name__ == "__main__":
    trace_generation_step_by_step()
    check_kv_cache_state()