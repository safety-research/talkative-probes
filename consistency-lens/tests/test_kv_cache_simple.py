#!/usr/bin/env python3
"""Simple test for KV cache with multi-layer patching."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def test_kv_cache_multilayer_simple():
    """Test KV cache with multi-layer patching - simple version."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Testing KV Cache with Multi-Layer Patching")
    print("=" * 60)
    
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
    
    d_model = decoder1.base.config.hidden_size
    batch_size = 2
    seq_length = 8
    
    # Create activation
    torch.manual_seed(42)
    activation = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    activation2 = activation.detach().clone().requires_grad_(True)
    
    # Generate with both methods
    print("\nGenerating with both methods...")
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    gen1 = decoder1.generate_soft(activation, max_length=seq_length, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    gen2 = decoder2.generate_soft_kv_cached(activation2, max_length=seq_length, gumbel_tau=1.0)
    
    # Compare outputs
    print("\nComparing outputs:")
    emb_diff = (gen1.generated_text_embeddings - gen2.generated_text_embeddings).abs().max().item()
    logits_diff = (gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item()
    ids_same = torch.equal(gen1.hard_token_ids, gen2.hard_token_ids)
    
    print(f"  Embeddings max diff: {emb_diff:.2e}")
    print(f"  Logits max diff: {logits_diff:.2e}")
    print(f"  Hard IDs identical: {ids_same}")
    
    # Test gradients
    print("\nTesting gradients:")
    loss1 = gen1.generated_text_embeddings.sum()
    loss2 = gen2.generated_text_embeddings.sum()
    
    loss1.backward()
    loss2.backward()
    
    grad_diff = (activation.grad - activation2.grad).abs().max().item()
    proj_grad_diff = (decoder1.proj.weight.grad - decoder2.proj.weight.grad).abs().max().item()
    
    print(f"  Activation grad diff: {grad_diff:.2e}")
    print(f"  Projection grad diff: {proj_grad_diff:.2e}")
    
    # Check if results are good
    threshold = 1e-4
    if emb_diff < threshold and logits_diff < threshold and ids_same and grad_diff < threshold:
        print("\n✓ KV cache works correctly with multi-layer patching!")
    else:
        print("\n✗ KV cache has issues with multi-layer patching")
        
        # Debug which tokens differ
        if not ids_same:
            print("\n  Token differences:")
            for b in range(batch_size):
                for t in range(seq_length):
                    id1 = gen1.hard_token_ids[b, t].item()
                    id2 = gen2.hard_token_ids[b, t].item()
                    if id1 != id2:
                        print(f"    Batch {b}, Time {t}: {id1} vs {id2}")


def trace_kv_cache_behavior():
    """Trace what's happening in KV cache."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n\nTracing KV Cache Behavior")
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
    
    # Manually trace first token generation
    print("\nManual trace of first token:")
    
    # Build initial sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
    a_proj = decoder._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
    
    seq_embs = torch.cat(parts, dim=1)
    print(f"  Initial sequence shape: {seq_embs.shape}")
    
    # Get position where activation is patched
    embed_pos = decoder.prompt_left_emb.size(0) if decoder.prompt_left_emb is not None else 0
    print(f"  Embed position: {embed_pos}")
    
    # The key insight: with multi-layer patching, the activation at embed_pos
    # affects the computation at ALL positions through attention!
    # So the cached K,V already include the effect of the patched activation.
    print("\n  Key insight: The cached K,V from initial pass already include")
    print("  the effect of the patched activation at all positions!")


if __name__ == "__main__":
    test_kv_cache_multilayer_simple()
    trace_kv_cache_behavior()