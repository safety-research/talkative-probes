#!/usr/bin/env python3
"""Trace KV cache with multi-layer patching to find the issue."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.kv_cache import KVCache, GPT2AttentionWithCache


def trace_initial_pass():
    """Trace the initial pass through layers with patching."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Tracing Initial Pass with Multi-Layer Patching")
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
    torch.manual_seed(42)
    activation = torch.randn(1, d_model, device=device)
    
    # Get components
    transformer = decoder.base.transformer
    layers = transformer.h
    final_norm = transformer.ln_f
    
    # Build initial sequence (no activation token with patch_all_layers)
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
    
    seq_embs = torch.cat(parts, dim=1)
    seq_len = seq_embs.size(1)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    
    # Project activation
    a_proj = decoder._apply_projection(activation)
    embed_pos = decoder.prompt_left_emb.size(0) if decoder.prompt_left_emb is not None else 0
    
    print(f"Initial setup:")
    print(f"  Sequence shape: {seq_embs.shape}")
    print(f"  Embed position: {embed_pos}")
    print(f"  Projected activation norm: {a_proj.norm().item():.4f}")
    
    # Process with standard method first
    print("\n1. Standard generate_soft:")
    hidden_states = seq_embs.clone()
    
    for i in range(3):  # First 3 layers
        layer = layers[i]
        
        # Before layer
        pre_hidden = hidden_states[:, embed_pos].clone()
        
        # Apply layer
        layer_out = layer(hidden_states, position_ids=position_ids)
        hidden_states = layer_out[0]
        
        # Patch activation
        hidden_states[:, embed_pos] = a_proj
        
        post_hidden = hidden_states[:, embed_pos]
        
        print(f"  Layer {i}: pre_norm={pre_hidden.norm().item():.4f}, "
              f"post_norm={post_hidden.norm().item():.4f}, "
              f"matches_proj={torch.allclose(post_hidden, a_proj)}")
    
    # Process with KV cache method
    print("\n2. KV cache method:")
    hidden_states = seq_embs.clone()
    kv_cache = KVCache()
    
    for i in range(3):  # First 3 layers
        layer = layers[i]
        
        # Before layer
        pre_hidden = hidden_states[:, embed_pos].clone()
        
        # Apply layer with caching
        wrapper = GPT2AttentionWithCache(layer)
        hidden_states, kv_cache = wrapper(
            hidden_states, kv_cache, i, use_cache=True
        )
        
        # Patch activation
        hidden_states[:, embed_pos] = a_proj
        
        post_hidden = hidden_states[:, embed_pos]
        
        print(f"  Layer {i}: pre_norm={pre_hidden.norm().item():.4f}, "
              f"post_norm={post_hidden.norm().item():.4f}, "
              f"matches_proj={torch.allclose(post_hidden, a_proj)}")
    
    print(f"\nKV cache initialized with {len(kv_cache)} layers")
    print(f"First key shape: {kv_cache.keys[0].shape}")
    
    # Check if the cached values contain the patched activation effect
    print("\n3. Checking if patching affected cached values:")
    
    # Get final hidden states for both methods
    hidden1 = seq_embs.clone()
    hidden2 = seq_embs.clone()
    
    # Process all layers with standard method
    for layer in layers:
        layer_out = layer(hidden1, position_ids=position_ids)
        hidden1 = layer_out[0]
        hidden1[:, embed_pos] = a_proj
    
    # Process all layers with KV cache
    kv_cache2 = KVCache()
    for i, layer in enumerate(layers):
        wrapper = GPT2AttentionWithCache(layer)
        hidden2, kv_cache2 = wrapper(hidden2, kv_cache2, i, use_cache=True)
        hidden2[:, embed_pos] = a_proj
    
    # Compare
    diff = (hidden1 - hidden2).abs().max().item()
    print(f"  Hidden states max diff: {diff:.2e}")
    
    # Apply final norm and get logits
    hidden1 = final_norm(hidden1)
    hidden2 = final_norm(hidden2)
    
    logits1 = decoder.out(hidden1[:, -1])
    logits2 = decoder.out(hidden2[:, -1])
    
    logits_diff = (logits1 - logits2).abs().max().item()
    print(f"  Logits max diff: {logits_diff:.2e}")
    
    return kv_cache2, a_proj, embed_pos


def test_incremental_generation():
    """Test incremental generation with cached KV."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nTesting Incremental Generation")
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
    torch.manual_seed(42)
    activation = torch.randn(1, d_model, device=device)
    
    # Get cached state from initial pass
    kv_cache, a_proj, embed_pos = trace_initial_pass()
    
    print("\n4. Simulating incremental generation:")
    
    # Get a new token embedding (simulate Gumbel-Softmax output)
    torch.manual_seed(123)
    new_token_emb = torch.randn(1, 1, d_model, device=device)
    
    transformer = decoder.base.transformer
    layers = transformer.h
    
    # Process new token through layers
    hidden_states = new_token_emb
    current_position = 4  # After initial sequence
    
    print(f"  Processing new token at position {current_position}")
    print(f"  New token embedding norm: {new_token_emb.norm().item():.4f}")
    
    for i in range(3):  # First 3 layers
        layer = layers[i]
        
        # Apply layer with cached KV
        wrapper = GPT2AttentionWithCache(layer)
        hidden_states, kv_cache = wrapper(
            hidden_states, kv_cache, i, use_cache=True
        )
        
        print(f"  Layer {i}: output norm = {hidden_states.norm().item():.4f}")
        
        # Check KV cache update
        print(f"    Key shape now: {kv_cache.keys[i].shape}")
    
    print("\nThe issue might be that the patched activation is not properly")
    print("influencing the attention computation for new tokens!")


if __name__ == "__main__":
    test_incremental_generation()