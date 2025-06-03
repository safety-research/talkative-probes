"""Native KV cache implementation using transformers' built-in support."""

from typing import List, Optional, Tuple
import torch


def generate_with_native_kv_cache(
    decoder,
    activation_input: torch.Tensor,
    max_length: int,
    gumbel_tau: float,
    use_projection: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate using native transformer KV caching with multi-layer patching.
    
    Returns:
        generated_text_embeddings, raw_lm_logits, hard_token_ids
    """
    # Setup
    main_base = decoder.base
    main_out = decoder.out
    prompt_left_emb = decoder.prompt_left_emb
    prompt_right_emb = decoder.prompt_right_emb
    
    # Get dtype from projection layer
    if decoder.config.per_layer_projections:
        activation_input = activation_input.to(decoder.proj_weight.dtype)
    else:
        activation_input = activation_input.to(decoder.proj.weight.dtype)
    
    B, d_model = activation_input.shape
    device = activation_input.device
    
    # Get embedding tables
    input_emb_table = main_base.get_input_embeddings().weight
    output_emb_table = main_base.get_output_embeddings().weight
    embeddings_tied = (input_emb_table.data_ptr() == output_emb_table.data_ptr())
    
    # Build initial sequence
    parts = []
    if prompt_left_emb is not None:
        parts.append(prompt_left_emb.expand(B, -1, -1))
    
    # Always insert activation as a token
    if use_projection:
        if decoder.config.patch_all_layers and decoder.config.per_layer_projections:
            a_proj = decoder._apply_projection(activation_input, layer_idx=0).unsqueeze(1)
        else:
            a_proj = decoder._apply_projection(activation_input).unsqueeze(1)
    else:
        a_proj = activation_input.unsqueeze(1)
    parts.append(a_proj)
    
    if prompt_right_emb is not None:
        parts.append(prompt_right_emb.expand(B, -1, -1))
    
    seq_embs = torch.cat(parts, dim=1)
    
    # Get transformer and detect architecture
    if hasattr(main_base, 'transformer'):
        # GPT-2 style model
        transformer = main_base.transformer
        layers = transformer.h
        final_norm = transformer.ln_f
    else:
        raise ValueError("Only GPT-2 style models supported for native KV cache")
    
    # For multi-layer patching, we need to do a custom forward pass
    if decoder.config.patch_all_layers:
        # Calculate embed position
        embed_pos = prompt_left_emb.size(0) if prompt_left_emb is not None else 0
        single_proj = decoder._apply_projection(activation_input) if not decoder.config.per_layer_projections else None
        
        # Initial forward pass with patching
        hidden_states = seq_embs
        seq_length = hidden_states.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        
        # Add position embeddings
        position_embeds = transformer.wpe(position_ids)
        hidden_states = transformer.drop(hidden_states + position_embeds)
        
        # Process through layers with activation patching and collect past_key_values
        past_key_values = []
        for layer_idx, layer in enumerate(layers):
            # Process layer with caching
            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids,
                use_cache=True
            )
            hidden_states = layer_outputs[0]
            
            # Store KV cache
            if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                past_key_values.append(layer_outputs[1])
            
            # Replace activation at the embed position
            if layer_idx > 0 or not decoder.config.per_layer_projections:
                if decoder.config.per_layer_projections:
                    a_proj_layer = decoder._apply_projection(activation_input, layer_idx=layer_idx)
                    hidden_states[:, embed_pos] = a_proj_layer
                else:
                    hidden_states[:, embed_pos] = single_proj
        
        # Final layer norm
        hidden_states = final_norm(hidden_states)
        
        # Convert to tuple format expected by transformers
        past_key_values = tuple(past_key_values)
    else:
        # Use standard forward pass
        outputs = main_base(
            inputs_embeds=seq_embs,
            use_cache=True,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        past_key_values = outputs.past_key_values
    
    # Storage for generation
    logits_list = []
    hard_ids_list = []
    output_embs_list = []
    
    # Get initial logits
    logits_t = main_out(hidden_states[:, -1])
    
    # Generation loop
    current_position = seq_embs.size(1)
    
    for step in range(max_length):
        # Gumbel-Softmax sampling
        logits_t_scaled = logits_t / 1.0  # T_sampling = 1.0
        with torch.amp.autocast('cuda', enabled=False):
            logits_f32 = logits_t_scaled.float()
            logits_f32 = logits_f32 - logits_f32.max(dim=-1, keepdim=True)[0].detach()
            ste_token_dist = torch.nn.functional.gumbel_softmax(
                logits_f32,
                tau=max(gumbel_tau, 0.1),
                hard=True
            ).to(logits_t.dtype)
        
        # Get embeddings
        emb_t_input = ste_token_dist @ input_emb_table
        if embeddings_tied:
            emb_t_output = emb_t_input
        else:
            emb_t_output = ste_token_dist @ output_emb_table
        
        # Store outputs
        logits_list.append(logits_t)
        output_embs_list.append(emb_t_output)
        hard_ids_list.append(ste_token_dist.argmax(dim=-1))
        
        # Process next token if needed
        if step < max_length - 1:
            # Prepare position IDs for the new token
            position_ids = torch.arange(
                current_position, current_position + 1,
                dtype=torch.long, device=device
            ).unsqueeze(0).expand(B, -1)
            
            # Add position embeddings
            position_embeds = transformer.wpe(position_ids)
            next_hidden = transformer.drop(emb_t_input.unsqueeze(1) + position_embeds)
            
            # Process through layers with cached KV
            new_past_key_values = []
            for layer_idx, (layer, past_kv) in enumerate(zip(layers, past_key_values)):
                layer_outputs = layer(
                    next_hidden,
                    past_key_value=past_kv,
                    position_ids=position_ids,
                    use_cache=True
                )
                next_hidden = layer_outputs[0]
                
                # Update cache
                if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                    new_past_key_values.append(layer_outputs[1])
            
            # Final norm
            next_hidden = final_norm(next_hidden)
            
            # Get logits for next iteration
            logits_t = main_out(next_hidden[:, -1])
            
            # Update state
            past_key_values = tuple(new_past_key_values)
            current_position += 1
    
    # Stack outputs
    logits_seq = torch.stack(logits_list, dim=1)
    hard_ids = torch.stack(hard_ids_list, dim=1)
    text_embs = torch.stack(output_embs_list, dim=1)
    
    return text_embs, logits_seq, hard_ids