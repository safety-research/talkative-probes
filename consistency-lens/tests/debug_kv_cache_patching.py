#!/usr/bin/env python3
"""Debug KV cache with multi-layer patching."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def compare_first_token_generation():
    """Compare just the first generated token between methods."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Comparing First Token Generation")
    print("=" * 60)
    
    # Config with multi-layer patching
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    # Create two identical decoders
    decoder1 = Decoder(config).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    decoder2 = Decoder(config).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    decoder2.load_state_dict(decoder1.state_dict())
    
    # Create activation
    d_model = decoder1.base.config.hidden_size
    torch.manual_seed(42)
    activation = torch.randn(1, d_model, device=device)
    
    # Generate just 1 token with each method
    print("\nGenerating 1 token...")
    gen1 = decoder1.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    gen2 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    # Compare
    logits1 = gen1.raw_lm_logits[0, 0]  # First token logits
    logits2 = gen2.raw_lm_logits[0, 0]
    
    diff = (logits1 - logits2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nLogits comparison:")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    
    # Find top differing positions
    if max_diff > 1e-4:
        top_diffs = diff.topk(10)
        print(f"\n  Top 10 differing vocab items:")
        for i, (d, idx) in enumerate(zip(top_diffs.values, top_diffs.indices)):
            token = tokenizer.decode([idx.item()])
            print(f"    {idx.item():5d} ('{token}'): diff = {d.item():.2e}, "
                  f"logit1 = {logits1[idx].item():.2f}, logit2 = {logits2[idx].item():.2f}")
    
    # Check hard token selection
    hard1 = gen1.hard_token_ids[0, 0].item()
    hard2 = gen2.hard_token_ids[0, 0].item()
    print(f"\n  Hard token 1: {hard1} ('{tokenizer.decode([hard1])}')")
    print(f"  Hard token 2: {hard2} ('{tokenizer.decode([hard2])}')")
    
    return max_diff < 1e-4


def trace_multilayer_patching():
    """Trace through multi-layer patching step by step."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nTracing Multi-Layer Patching")
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
    
    # Get components
    transformer = decoder.base.transformer
    layers = transformer.h
    n_layers = len(layers)
    
    # Project activation
    a_proj = decoder._apply_projection(activation)
    print(f"Projected activation norm: {a_proj.norm().item():.4f}")
    
    # Build initial sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
    
    seq_embs = torch.cat(parts, dim=1)
    seq_len = seq_embs.size(1)
    embed_pos = decoder.prompt_left_emb.size(0) if decoder.prompt_left_emb is not None else 0
    
    print(f"\nInitial sequence:")
    print(f"  Shape: {seq_embs.shape}")
    print(f"  Embed position: {embed_pos}")
    print(f"  Sequence length: {seq_len}")
    
    # Process through layers manually
    hidden_states = seq_embs.clone()
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\nProcessing through {n_layers} layers:")
    for i in range(min(3, n_layers)):  # Just first 3 layers
        layer = layers[i]
        
        # Before layer
        pre_norm = hidden_states[:, embed_pos].norm().item()
        
        # Apply layer
        layer_out = layer(hidden_states, position_ids=position_ids)
        hidden_states = layer_out[0]
        
        # Patch activation
        hidden_states[:, embed_pos] = a_proj
        
        # After patching
        post_norm = hidden_states[:, embed_pos].norm().item()
        
        print(f"  Layer {i}: pre_patch_norm = {pre_norm:.4f}, post_patch_norm = {post_norm:.4f}")
    
    return True


def test_without_patching():
    """Test KV cache without multi-layer patching as sanity check."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nTesting KV Cache WITHOUT Multi-Layer Patching (Sanity Check)")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,  # No multi-layer patching
    )
    
    decoder1 = Decoder(config).to(device)
    decoder1.set_prompt("explain <embed>:", tokenizer)
    
    decoder2 = Decoder(config).to(device)
    decoder2.set_prompt("explain <embed>:", tokenizer)
    decoder2.load_state_dict(decoder1.state_dict())
    
    d_model = decoder1.base.config.hidden_size
    torch.manual_seed(42)
    activation = torch.randn(2, d_model, device=device)
    
    # Generate with both methods
    print("\nGenerating 8 tokens...")
    gen1 = decoder1.generate_soft(activation.clone(), max_length=8, gumbel_tau=1.0)
    gen2 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=8, gumbel_tau=1.0)
    
    # Compare
    emb_diff = (gen1.generated_text_embeddings - gen2.generated_text_embeddings).abs().max().item()
    logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
    ids_same = torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)
    
    print(f"\nComparison:")
    print(f"  Embeddings max diff: {emb_diff:.2e}")
    print(f"  Logits max diff: {logits_diff:.2e}")
    print(f"  Hard IDs identical: {ids_same}")
    
    if emb_diff < 1e-4 and logits_diff < 1e-4:
        print(f"  ✓ KV cache works correctly WITHOUT multi-layer patching")
    else:
        print(f"  ✗ KV cache has issues even without multi-layer patching")
    
    return emb_diff < 1e-4


if __name__ == "__main__":
    # Test basic functionality first
    test_without_patching()
    
    # Then test with multi-layer patching
    compare_first_token_generation()
    trace_multilayer_patching()