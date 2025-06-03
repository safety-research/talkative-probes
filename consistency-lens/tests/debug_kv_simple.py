#!/usr/bin/env python3
"""Debug simple KV cache issue."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_position_offset_issue():
    """Test if position offset is the issue."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Testing Position Offset in KV Cache")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,  # Simple case first
    )
    
    decoder = Decoder(config).to(device)
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    torch.manual_seed(42)
    activation = torch.randn(1, d_model, device=device)
    
    # Manually trace through both methods
    print("\n1. generate_soft (reference):")
    
    # Build initial sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
    
    a_proj = decoder._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
    
    seq_embs = torch.cat(parts, dim=1)
    print(f"  Initial seq shape: {seq_embs.shape}")
    
    # First forward pass
    out1 = decoder.base(inputs_embeds=seq_embs, output_hidden_states=True)
    h_last = out1.last_hidden_state if hasattr(out1, "last_hidden_state") else out1.hidden_states[-1]
    logits1 = decoder.out(h_last[:, -1])
    print(f"  First logits sum: {logits1.sum().item():.4f}")
    print(f"  First logits max: {logits1.max().item():.4f}")
    
    print("\n2. KV cache method:")
    
    # Process initial sequence with KV cache
    from lens.models.kv_cache import compute_with_kv_cache, KVCache
    
    transformer = decoder.base.transformer
    kv_cache = KVCache()
    
    # Process initial sequence
    hidden_states, kv_cache = compute_with_kv_cache(
        transformer, seq_embs, kv_cache, position_offset=0
    )
    
    logits2 = decoder.out(hidden_states[:, -1])
    print(f"  First logits sum: {logits2.sum().item():.4f}")
    print(f"  First logits max: {logits2.max().item():.4f}")
    
    # Compare
    diff = (logits1 - logits2).abs().max().item()
    print(f"\n  Logits max diff: {diff:.2e}")
    
    if diff < 1e-4:
        print("  ✓ Initial processing matches!")
    else:
        print("  ✗ Initial processing differs!")
        
        # Debug more
        print("\n  Checking hidden states:")
        h1 = out1.last_hidden_state
        h2 = hidden_states
        h_diff = (h1 - h2).abs().max().item()
        print(f"  Hidden states max diff: {h_diff:.2e}")
        
        # Check shape
        print(f"  h1 shape: {h1.shape}, h2 shape: {h2.shape}")
    
    # Check KV cache contents
    print(f"\n3. KV cache contents:")
    print(f"  Number of layers cached: {len(kv_cache)}")
    if len(kv_cache) > 0:
        print(f"  First layer K shape: {kv_cache.keys[0].shape}")
        print(f"  First layer V shape: {kv_cache.values[0].shape}")


def test_gumbel_temperature():
    """Test if low Gumbel temperature causes divergence."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nTesting Gumbel Temperature Effect")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=False,
    )
    
    for tau in [10.0, 5.0, 2.0, 1.0, 0.5]:
        print(f"\nTau = {tau}:")
        
        decoder1 = Decoder(config).to(device)
        decoder1.set_prompt("explain <embed>:", tokenizer)
        
        decoder2 = Decoder(config).to(device)
        decoder2.set_prompt("explain <embed>:", tokenizer)
        decoder2.load_state_dict(decoder1.state_dict())
        
        d_model = decoder1.base.config.hidden_size
        torch.manual_seed(42)
        activation = torch.randn(1, d_model, device=device)
        
        # Generate 1 token
        gen1 = decoder1.generate_soft(activation.clone(), max_length=1, gumbel_tau=tau)
        gen2 = decoder2.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=tau)
        
        # Compare
        logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
        ids_same = torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)
        
        print(f"  Logits diff: {logits_diff:.2e}, IDs same: {ids_same}")


if __name__ == "__main__":
    test_position_offset_issue()
    test_gumbel_temperature()