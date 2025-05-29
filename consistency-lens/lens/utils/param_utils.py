import logging
import torch.nn as nn # For type hint
from lens.models.decoder import DecoderConfig # For type hint
from lens.models.encoder import EncoderConfig # For type hint
from lens.models.orig import OrigWrapper # For type hint

def log_parameter_counts(dec_raw: nn.Module, enc_raw: nn.Module, orig: OrigWrapper, 
                         decoder_config: DecoderConfig, encoder_config: EncoderConfig, 
                         log: logging.Logger) -> dict:
    """Log detailed parameter counts for all models."""
    
    # Ensure models passed are the raw nn.Module, not compiled versions for accurate naming.
    # If they might be compiled, access .module attribute if present.
    
    actual_dec_raw = dec_raw.module if hasattr(dec_raw, '_orig_mod') else dec_raw
    actual_enc_raw = enc_raw.module if hasattr(enc_raw, '_orig_mod') else enc_raw


    trainable_params_list = [p for p in actual_dec_raw.parameters() if p.requires_grad] + \
                            [p for p in actual_enc_raw.parameters() if p.requires_grad]
    total_trainable_params_val = sum(p.numel() for p in trainable_params_list)
    
    # num_params_orig_total = sum(p.numel() for p in orig.model.parameters())
    # Accessing orig.model might be an issue if it's not always an nn.Module (e.g. if loaded with bitsandbytes)
    # Let's assume orig.model is an nn.Module for parameter counting
    num_params_orig_total = 0
    if hasattr(orig, 'model') and isinstance(orig.model, nn.Module):
        num_params_orig_total = sum(p.numel() for p in orig.model.parameters())
    else:
        log.warning("Could not determine parameter count for original model (orig.model is not an nn.Module or not found).")


    log.info(f"Total trainable parameters (Decoder + Encoder combined, based on current requires_grad): {total_trainable_params_val:,}")

    num_params_dec_total = sum(p.numel() for p in actual_dec_raw.parameters())
    num_params_dec_base_trainable, num_params_dec_proj_trainable, num_params_dec_out_trainable, \
    num_params_dec_prompts_trainable, num_params_dec_embeddings_trainable, num_params_dec_other_trainable = 0, 0, 0, 0, 0, 0
    
    num_params_dec_base_frozen, num_params_dec_proj_frozen, num_params_dec_out_frozen, \
    num_params_dec_prompts_frozen, num_params_dec_embeddings_frozen, num_params_dec_other_frozen = 0, 0, 0, 0, 0, 0

    for n, p in actual_dec_raw.named_parameters():
        numel = p.numel()
        is_trainable = p.requires_grad

        if 'prompt_left_emb' in n or 'prompt_right_emb' in n:
            if is_trainable: num_params_dec_prompts_trainable += numel
            else: num_params_dec_prompts_frozen += numel
        elif n.startswith('proj.'):
            if is_trainable: num_params_dec_proj_trainable += numel
            else: num_params_dec_proj_frozen += numel
        elif n.startswith('out.'): # This refers to dec_raw.out
            if is_trainable: num_params_dec_out_trainable += numel
            else: num_params_dec_out_frozen += numel
        # Check for embeddings (wte, wpe) within the base model
        elif hasattr(actual_dec_raw, 'base') and actual_dec_raw.base and \
             (n.startswith('base.wte.') or n.startswith('base.wpe.') or \
              (hasattr(actual_dec_raw.base, 'shared') and actual_dec_raw.base.shared and 'base.shared.weight' in n) or \
              (n.startswith('base.embed_tokens.') or n.startswith('base.embed_positions.')) # More general names
             ):
            if is_trainable: num_params_dec_embeddings_trainable += numel
            else: num_params_dec_embeddings_frozen += numel
        # Check for lm_head within the base model if it's tied and considered part of "embeddings"
        elif hasattr(actual_dec_raw, 'base') and actual_dec_raw.base and \
             n.startswith('base.lm_head.'): # If lm_head is part of base and tied
            if is_trainable: num_params_dec_embeddings_trainable += numel # Counting as embedding
            else: num_params_dec_embeddings_frozen += numel
        elif 'base.' in n: 
            if is_trainable: num_params_dec_base_trainable += numel
            else: num_params_dec_base_frozen += numel
        else:
            if is_trainable: num_params_dec_other_trainable += numel
            else: num_params_dec_other_frozen += numel
            
    current_dec_trainable_total = (num_params_dec_base_trainable + num_params_dec_proj_trainable + 
                                   num_params_dec_out_trainable + num_params_dec_prompts_trainable +
                                   num_params_dec_embeddings_trainable + num_params_dec_other_trainable)
    current_dec_frozen_total = (num_params_dec_base_frozen + num_params_dec_proj_frozen +
                                num_params_dec_out_frozen + num_params_dec_prompts_frozen +
                                num_params_dec_embeddings_frozen + num_params_dec_other_frozen)
    
    log.info(f"Decoder - Total parameters: {num_params_dec_total:,}")
    log.info(f"  Decoder - Categorized Trainable parameters: {current_dec_trainable_total:,}")
    log.info(f"    Decoder base trainable: {num_params_dec_base_trainable:,} (Config: {decoder_config.base_model})")
    log.info(f"    Decoder proj trainable: {num_params_dec_proj_trainable:,} (Config: {decoder_config.projection_layer})")
    log.info(f"    Decoder out trainable: {num_params_dec_out_trainable:,} (Config: {decoder_config.output_head})") # dec.out layer
    log.info(f"    Decoder prompts trainable: {num_params_dec_prompts_trainable:,} (Config: {decoder_config.trainable_prompts})")
    log.info(f"    Decoder embeddings trainable: {num_params_dec_embeddings_trainable:,}")
    log.info(f"    Decoder other trainable: {num_params_dec_other_trainable:,}")
    log.info(f"  Decoder - Categorized Frozen parameters: {current_dec_frozen_total:,}")
    # ... (frozen logging details omitted for brevity but should be included) ...

    num_params_enc_total = sum(p.numel() for p in actual_enc_raw.parameters())
    num_params_enc_base_trainable, num_params_enc_proj_trainable, \
    num_params_enc_prompts_trainable, num_params_enc_embeddings_trainable, num_params_enc_other_trainable = 0, 0, 0, 0, 0

    num_params_enc_base_frozen, num_params_enc_proj_frozen, \
    num_params_enc_prompts_frozen, num_params_enc_embeddings_frozen, num_params_enc_other_frozen = 0, 0, 0, 0, 0

    for n, p in actual_enc_raw.named_parameters():
        numel = p.numel()
        is_trainable = p.requires_grad

        if 'soft_prompt_embeddings' in n:
            if is_trainable: num_params_enc_prompts_trainable += numel
            else: num_params_enc_prompts_frozen += numel
        elif n.startswith('proj.'):
            if is_trainable: num_params_enc_proj_trainable += numel
            else: num_params_enc_proj_frozen += numel
        elif hasattr(actual_enc_raw, 'base') and actual_enc_raw.base and \
             (n.startswith('base.wte.') or n.startswith('base.wpe.') or \
              (n.startswith('base.embed_tokens.') or n.startswith('base.embed_positions.'))
             ):
            if is_trainable: num_params_enc_embeddings_trainable += numel
            else: num_params_enc_embeddings_frozen += numel
        elif 'base.' in n:
            if is_trainable: num_params_enc_base_trainable += numel
            else: num_params_enc_base_frozen += numel
        else:
            if is_trainable: num_params_enc_other_trainable += numel
            else: num_params_enc_other_frozen += numel

    current_enc_trainable_total = (num_params_enc_base_trainable + num_params_enc_proj_trainable +
                                   num_params_enc_prompts_trainable + num_params_enc_embeddings_trainable + 
                                   num_params_enc_other_trainable)
    # ... (encoder logging details omitted for brevity) ...

    log.info(f"Encoder - Total parameters: {num_params_enc_total:,}")
    log.info(f"  Encoder - Categorized Trainable parameters: {current_enc_trainable_total:,}")
    log.info(f"    Encoder base trainable: {num_params_enc_base_trainable:,} (Config: {encoder_config.base_model}, present: {encoder_config.use_base_model})")
    # ...
    
    sum_of_categorized_trainable = current_dec_trainable_total + current_enc_trainable_total
    if total_trainable_params_val != sum_of_categorized_trainable:
        log.warning(
            f"Parameter count mismatch (trainable): total_trainable_params_val is {total_trainable_params_val:,}, "
            f"sum of categorized is {sum_of_categorized_trainable:,}."
        )
    
    log.info(f"Original LLM (frozen) parameters: {num_params_orig_total:,}")
    
    return {
        'total_trainable': total_trainable_params_val,
        'trainable_params_list': trainable_params_list, 
        'decoder_total': num_params_dec_total,
        'encoder_total': num_params_enc_total,
        'orig_total': num_params_orig_total
    }
