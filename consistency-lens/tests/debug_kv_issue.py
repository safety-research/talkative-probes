#!/usr/bin/env python3
"""Debug KV caching issue with multi-layer patching."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_issue():
    """Debug the KV caching issue."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Debugging KV Caching Issue")
    print("=" * 60)
    
    config = DecoderConfig(
        model_name=model_name,
        base_model=False,
        projection_layer=True,
        patch_all_layers=True,
        per_layer_projections=False,
    )
    
    decoder = Decoder(config).to(device).eval()
    decoder.set_prompt("explain <embed>:", tokenizer)
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Compare first token generation between both methods
    print("\nComparing first token generation:")
    
    # Method 1: generate_soft
    torch.manual_seed(42)
    gen1 = decoder.generate_soft(activation.clone(), max_length=1, gumbel_tau=1.0)
    logits1 = gen1.raw_lm_logits[0, 0]
    token1 = gen1.hard_token_ids[0, 0].item()
    
    # Method 2: generate_soft_kv_cached
    torch.manual_seed(42)
    gen2 = decoder.generate_soft_kv_cached(activation.clone(), max_length=1, gumbel_tau=1.0)
    logits2 = gen2.raw_lm_logits[0, 0]
    token2 = gen2.hard_token_ids[0, 0].item()
    
    print(f"First token from generate_soft: {token1} ('{tokenizer.decode([token1])}')")
    print(f"First token from generate_kv_cached: {token2} ('{tokenizer.decode([token2])}')")
    print(f"Tokens match: {token1 == token2}")
    print(f"Logits diff: {(logits1 - logits2).abs().max().item():.2e}")
    
    # Now let's trace through what happens for the second token
    print("\n\nTracing second token generation:")
    
    # Get transformer components
    transformer = decoder.base.transformer
    layers = transformer.h
    final_norm = transformer.ln_f
    
    # Build initial sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
    a_proj = decoder._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
    seq_embs = torch.cat(parts, dim=1)
    
    print(f"Initial sequence length: {seq_embs.size(1)}")
    
    # Process initial sequence to get past_key_values
    hidden_states = seq_embs
    seq_length = hidden_states.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    
    # Add position embeddings
    position_embeds = transformer.wpe(position_ids)
    hidden_states = transformer.drop(hidden_states + position_embeds)
    
    # Process through layers
    past_key_values = []
    single_proj = decoder._apply_projection(activation)
    embed_pos = 3  # Position after "explain "
    
    for layer_idx, layer_module in enumerate(layers):
        # Apply patching for layer > 0
        if layer_idx > 0:
            hidden_states[:, embed_pos] = single_proj
            
        # Process through layer
        layer_outputs = layer_module(
            hidden_states,
            use_cache=True
        )
        hidden_states = layer_outputs[0]
        
        # Collect past_key_values
        if len(layer_outputs) > 1 and layer_outputs[1] is not None:
            past_key_values.append(layer_outputs[1])
    
    # Final layer norm
    hidden_states = final_norm(hidden_states)
    past_key_values = tuple(past_key_values)
    
    # Get first token logits
    logits_0 = decoder.out(hidden_states[:, -1])
    
    # Generate first token
    torch.manual_seed(42)
    with torch.amp.autocast('cuda', enabled=False):
        logits_f32 = logits_0.float()
        logits_f32 = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
        ste_token_dist = torch.nn.functional.gumbel_softmax(
            logits_f32,
            tau=1.0,
            hard=True
        ).to(logits_0.dtype)
    
    # Get first token embedding
    input_emb_table = decoder.base.get_input_embeddings().weight
    emb_t = ste_token_dist @ input_emb_table
    first_token = ste_token_dist.argmax().item()
    
    print(f"\nGenerated first token: {first_token} ('{tokenizer.decode([first_token])}')")
    
    # Now process this token incrementally
    print("\nProcessing first token incrementally with KV cache:")
    current_position = seq_embs.size(1)
    new_hidden = emb_t.unsqueeze(1)
    new_position_ids = torch.arange(
        current_position, current_position + 1,
        dtype=torch.long, device=device
    ).unsqueeze(0)
    
    print(f"Current position: {current_position}")
    print(f"New position IDs: {new_position_ids[0].item()}")
    
    # Add position embeddings for the new token
    position_embeds = transformer.wpe(new_position_ids)
    new_hidden = transformer.drop(new_hidden + position_embeds)
    
    print(f"New hidden shape after position embeddings: {new_hidden.shape}")
    print(f"New hidden norm: {new_hidden[0, 0].norm().item():.2f}")
    
    # Process through layers with cached KV
    print("\nProcessing through layers:")
    new_past_key_values = []
    for layer_idx, (layer_module, past_kv) in enumerate(zip(layers, past_key_values)):
        print(f"  Layer {layer_idx}:")
        print(f"    Past KV shapes: K={past_kv[0].shape}, V={past_kv[1].shape}")
        print(f"    Input hidden norm: {new_hidden[0, 0].norm().item():.2f}")
        
        # Process through layer
        layer_outputs = layer_module(
            new_hidden,
            past_key_value=past_kv,
            use_cache=True
        )
        new_hidden = layer_outputs[0]
        
        print(f"    Output hidden norm: {new_hidden[0, 0].norm().item():.2f}")
        
        # Break after first few layers to save output
        if layer_idx >= 2:
            print("    ... (remaining layers omitted)")
            break
    
    print("\nThe issue might be that during incremental generation,")
    print("we're not maintaining the activation patching that was done")
    print("in the initial forward pass. The cached KV contains the")
    print("patched activations, but new tokens don't get the benefit")
    print("of the multi-layer patching.")


if __name__ == "__main__":
    debug_issue()