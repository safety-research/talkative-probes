#!/usr/bin/env python3
"""Detailed debugging of KV cache with multi-layer patching."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def compare_first_logits():
    """Compare just the first logits between methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Comparing First Logits with Multi-Layer Patching")
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
    
    # Build initial sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
    
    seq_embs = torch.cat(parts, dim=1)
    B = 1
    seq_length = seq_embs.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    
    # Project activation
    a_proj = decoder._apply_projection(activation)
    embed_pos = decoder.prompt_left_emb.size(0) if decoder.prompt_left_emb is not None else 0
    
    print(f"Setup:")
    print(f"  Sequence shape: {seq_embs.shape}")
    print(f"  Embed position: {embed_pos}")
    print(f"  Activation norm: {a_proj.norm().item():.4f}")
    
    # Method 1: Standard forward (no KV cache)
    print("\n1. Standard forward (generate_soft):")
    hidden1 = seq_embs.clone()
    
    # Add position embeddings
    position_embeds = transformer.wpe(position_ids[0])
    hidden1 = transformer.drop(hidden1 + position_embeds)
    
    # Process through layers
    for i, layer in enumerate(layers):
        layer_out = layer(hidden1, position_ids=position_ids)
        hidden1 = layer_out[0]
        # Patch at embed position
        hidden1[:, embed_pos] = a_proj
        if i < 3:
            print(f"  Layer {i}: hidden[:, {embed_pos}] norm = {hidden1[:, embed_pos].norm().item():.4f}")
    
    # Final norm and logits
    hidden1 = final_norm(hidden1)
    logits1 = decoder.out(hidden1[:, -1])
    print(f"  Final logits sum: {logits1.sum().item():.4f}")
    print(f"  Final logits max: {logits1.max().item():.4f}")
    
    # Method 2: With KV cache
    print("\n2. With KV cache (generate_soft_kv_cached):")
    from lens.models.kv_cache import KVCache, GPT2AttentionWithCache
    
    hidden2 = seq_embs.clone()
    kv_cache = KVCache()
    
    # Add position embeddings
    position_embeds = transformer.wpe(position_ids[0])
    hidden2 = transformer.drop(hidden2 + position_embeds)
    
    # Process through layers with caching
    for i, layer in enumerate(layers):
        wrapper = GPT2AttentionWithCache(layer)
        hidden2, kv_cache = wrapper(hidden2, kv_cache, i, use_cache=True)
        # Patch at embed position
        hidden2[:, embed_pos] = a_proj
        if i < 3:
            print(f"  Layer {i}: hidden[:, {embed_pos}] norm = {hidden2[:, embed_pos].norm().item():.4f}")
    
    # Final norm and logits
    hidden2 = final_norm(hidden2)
    logits2 = decoder.out(hidden2[:, -1])
    print(f"  Final logits sum: {logits2.sum().item():.4f}")
    print(f"  Final logits max: {logits2.max().item():.4f}")
    
    # Compare
    diff = (logits1 - logits2).abs().max().item()
    print(f"\n  Logits max diff: {diff:.2e}")
    
    if diff > 1e-4:
        print("\n  Something is wrong with the initial pass!")
        # Check hidden states before final norm
        h_diff = (hidden1 - hidden2).abs().max().item()
        print(f"  Hidden states before final norm diff: {h_diff:.2e}")
    else:
        print("\n  ✓ Initial pass matches!")
    
    # Now test first generated token
    print("\n3. Testing first generated token:")
    
    # Get first token via Gumbel-Softmax (use same seed)
    torch.manual_seed(123)
    with torch.amp.autocast('cuda', enabled=False):
        logits_f32 = logits1.float()
        logits_stable = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
        ste_token_dist = torch.nn.functional.gumbel_softmax(
            logits_stable, tau=1.0, hard=True
        ).to(logits1.dtype)
    
    input_emb_table = decoder.base.get_input_embeddings().weight
    emb_t = ste_token_dist @ input_emb_table
    print(f"  Generated token embedding norm: {emb_t.norm().item():.4f}")
    
    # Process this token through both methods
    current_position = seq_length
    new_pos_ids = torch.arange(current_position, current_position + 1, device=device).unsqueeze(0)
    
    # Method 1: Full recomputation (what generate_soft would do)
    print("\n  Method 1 (full recompute):")
    new_seq = torch.cat([seq_embs, emb_t.unsqueeze(1)], dim=1)
    new_seq_len = new_seq.size(1)
    new_position_ids = torch.arange(new_seq_len, device=device).unsqueeze(0)
    
    hidden1_new = new_seq
    position_embeds_new = transformer.wpe(new_position_ids[0])
    hidden1_new = transformer.drop(hidden1_new + position_embeds_new)
    
    # Process all positions
    for i, layer in enumerate(layers[:3]):
        layer_out = layer(hidden1_new, position_ids=new_position_ids)
        hidden1_new = layer_out[0]
        # Patch at embed position
        hidden1_new[:, embed_pos] = a_proj
        print(f"    Layer {i}: new token hidden norm = {hidden1_new[:, -1].norm().item():.4f}")
    
    # Method 2: Incremental with KV cache
    print("\n  Method 2 (incremental KV):")
    hidden2_new = emb_t.unsqueeze(1)
    position_embeds_inc = transformer.wpe(new_pos_ids[0])
    hidden2_new = transformer.drop(hidden2_new + position_embeds_inc)
    
    for i, layer in enumerate(layers[:3]):
        wrapper = GPT2AttentionWithCache(layer)
        hidden2_new, kv_cache = wrapper(hidden2_new, kv_cache, i, use_cache=True)
        print(f"    Layer {i}: new token hidden norm = {hidden2_new[:, -1].norm().item():.4f}")
    
    # The key insight: with multi-layer patching, the activation at embed_pos
    # affects attention computation for all subsequent tokens!
    print("\n  Key insight: The patched activation at position {} affects".format(embed_pos))
    print("  attention for all subsequent tokens, but KV cache doesn't know this!")


if __name__ == "__main__":
    compare_first_logits()