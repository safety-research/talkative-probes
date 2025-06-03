#!/usr/bin/env python3
"""Debug incremental token processing."""

import torch
from transformers import AutoTokenizer
from lens.models.decoder import Decoder, DecoderConfig


def debug_incremental():
    """Debug what happens during incremental generation."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Debugging Incremental Generation")
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
    decoder.eval()
    
    d_model = decoder.base.config.hidden_size
    activation = torch.randn(1, d_model, device=device)
    
    # Manually trace through KV cache generation
    # Build initial sequence
    parts = []
    if decoder.prompt_left_emb is not None:
        parts.append(decoder.prompt_left_emb.expand(1, -1, -1))
    a_proj = decoder._apply_projection(activation).unsqueeze(1)
    parts.append(a_proj)
    if decoder.prompt_right_emb is not None:
        parts.append(decoder.prompt_right_emb.expand(1, -1, -1))
    seq_embs = torch.cat(parts, dim=1)
    
    # Get components
    transformer = decoder.base.transformer
    layers = transformer.h
    final_norm = transformer.ln_f
    
    # Initial forward pass
    hidden_states = seq_embs
    seq_length = hidden_states.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
    
    # Process through layers with caching
    past_key_values = []
    embed_pos = 3
    single_proj = decoder._apply_projection(activation)
    
    for layer_idx, layer_module in enumerate(layers):
        layer_outputs = layer_module(
            hidden_states,
            position_ids=position_ids,
            use_cache=True
        )
        hidden_states = layer_outputs[0]
        
        # Collect past_key_values
        if len(layer_outputs) > 1 and layer_outputs[1] is not None:
            past_key_values.append(layer_outputs[1])
        
        # Replace activation
        if layer_idx > 0:
            hidden_states[:, embed_pos] = single_proj
    
    # Final layer norm
    hidden_states = final_norm(hidden_states)
    past_key_values = tuple(past_key_values)
    
    # Get first token logits
    logits_0 = decoder.out(hidden_states[:, -1])
    print(f"Initial logits range: [{logits_0.min().item():.2f}, {logits_0.max().item():.2f}]")
    print(f"Top token: {logits_0.argmax().item()} ('{tokenizer.decode([logits_0.argmax().item()])}')")
    
    # Generate first token using Gumbel-Softmax
    torch.manual_seed(123)
    with torch.amp.autocast('cuda', enabled=False):
        logits_f32 = logits_0.float()
        logits_f32 = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
        ste_token_dist = torch.nn.functional.gumbel_softmax(
            logits_f32,
            tau=1.0,
            hard=True
        ).to(logits_0.dtype)
    
    # Get embeddings
    input_emb_table = decoder.base.get_input_embeddings().weight
    emb_t = ste_token_dist @ input_emb_table
    
    print(f"\nFirst generated token: {ste_token_dist.argmax().item()} ('{tokenizer.decode([ste_token_dist.argmax().item()])}')")
    print(f"Token embedding shape: {emb_t.shape}")
    print(f"Token embedding norm: {emb_t.norm().item():.2f}")
    
    # Process this token incrementally
    print("\nProcessing token incrementally...")
    
    # Current position is 5 (after initial sequence of length 5)
    current_position = seq_embs.size(1)
    new_hidden = emb_t.unsqueeze(1)  # Shape: (1, 1, 768)
    new_position_ids = torch.arange(
        current_position, current_position + 1,
        dtype=torch.long, device=device
    ).unsqueeze(0)
    
    print(f"New position IDs: {new_position_ids}")
    print(f"New hidden shape: {new_hidden.shape}")
    
    # Process through layers with cached KV
    new_past_key_values = []
    for layer_idx, (layer_module, past_kv) in enumerate(zip(layers, past_key_values)):
        layer_outputs = layer_module(
            new_hidden,
            past_key_value=past_kv,
            position_ids=new_position_ids,
            use_cache=True
        )
        new_hidden = layer_outputs[0]
        
        if layer_idx == 0:
            print(f"After layer 0: hidden norm = {new_hidden[0, 0].norm().item():.2f}")
        
        # Update past_key_values
        if len(layer_outputs) > 1 and layer_outputs[1] is not None:
            new_past_key_values.append(layer_outputs[1])
    
    # Final norm
    new_hidden = final_norm(new_hidden)
    
    # Get next logits
    logits_1 = decoder.out(new_hidden[:, -1])
    print(f"\nNext logits range: [{logits_1.min().item():.2f}, {logits_1.max().item():.2f}]")
    print(f"Next top token: {logits_1.argmax().item()} ('{tokenizer.decode([logits_1.argmax().item()])}')")
    
    # Compare with what generate_soft would produce for the second token
    print("\n\nComparing with generate_soft...")
    torch.manual_seed(123)
    gen_soft = decoder.generate_soft(activation.clone(), max_length=2, gumbel_tau=1.0)
    
    print(f"generate_soft tokens: {gen_soft.hard_token_ids[0].tolist()}")
    print(f"generate_soft second token logits range: [{gen_soft.raw_lm_logits[0, 1].min().item():.2f}, {gen_soft.raw_lm_logits[0, 1].max().item():.2f}]")


if __name__ == "__main__":
    debug_incremental()