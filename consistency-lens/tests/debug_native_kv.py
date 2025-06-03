#!/usr/bin/env python3
"""Debug native KV cache implementation."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_native_kv():
    """Debug what's happening with native KV caching."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Debugging Native KV Cache")
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
    
    # Test with just 1 token
    print("\nGenerating 1 token with both methods...")
    
    torch.manual_seed(123)
    gen1 = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    torch.manual_seed(123)
    gen2 = decoder.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    
    print(f"\nResults:")
    print(f"  Token from generate_soft: {gen1.hard_token_ids[0, 0].item()}")
    print(f"  Token from KV cached: {gen2.hard_token_ids[0, 0].item()}")
    print(f"  Logits diff: {(gen1.raw_lm_logits - gen2.raw_lm_logits).abs().max().item():.2e}")
    
    # Let's manually trace through the initial pass
    print("\n\nManual trace of initial pass:")
    
    # Build sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
    a_proj = decoder._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
    seq_embs = torch.cat(parts, dim=1)
    
    print(f"  Sequence shape: {seq_embs.shape}")
    
    # Get transformer
    transformer = decoder.base.transformer
    layers = transformer.h
    final_norm = transformer.ln_f
    
    # Initial pass with patching
    hidden_states = seq_embs
    seq_length = hidden_states.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    
    # Add position embeddings
    position_embeds = transformer.wpe(position_ids)
    hidden_states = transformer.drop(hidden_states + position_embeds)
    
    print(f"  After position embeddings: hidden norm = {hidden_states[0, -1].norm().item():.2f}")
    
    # Process first layer
    layer_outputs = layers[0](
        hidden_states,
        position_ids=position_ids,
        use_cache=True
    )
    hidden_states = layer_outputs[0]
    
    print(f"  After layer 0: hidden norm = {hidden_states[0, -1].norm().item():.2f}")
    print(f"  Layer 0 output length: {len(layer_outputs)}")
    if len(layer_outputs) > 1 and layer_outputs[1] is not None:
        print(f"  Past KV shape: {layer_outputs[1][0].shape} (key), {layer_outputs[1][1].shape} (value)")
    
    # Check if position_ids is being used correctly
    print("\n\nChecking position_ids handling:")
    
    # Try with and without position_ids
    torch.manual_seed(42)
    h1 = torch.randn(1, 5, 768, device=device)
    
    out1 = layers[0](h1, use_cache=False)[0]
    out2 = layers[0](h1, position_ids=position_ids, use_cache=False)[0]
    
    print(f"  Difference with/without position_ids: {(out1 - out2).abs().max().item():.2e}")
    
    # The issue might be that position embeddings are added differently
    print("\n\nChecking full model forward:")
    
    # Method 1: Our custom forward
    # Already computed above as hidden_states
    
    # Method 2: Model's native forward
    outputs = decoder.base(
        inputs_embeds=seq_embs,
        use_cache=True,
        output_hidden_states=True
    )
    native_hidden = outputs.hidden_states[-1]
    
    print(f"  Native forward last hidden norm: {native_hidden[0, -1].norm().item():.2f}")
    print(f"  Our forward last hidden norm: {hidden_states[0, -1].norm().item():.2f}")
    
    # The native forward adds position embeddings internally!


if __name__ == "__main__":
    debug_native_kv()