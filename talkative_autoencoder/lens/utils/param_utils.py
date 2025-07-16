import logging
import torch.nn as nn # For type hint
from lens.models.decoder import DecoderConfig # For type hint
from lens.models.encoder import EncoderConfig # For type hint
from lens.models.orig import OrigWrapper # For type hint
import torch
import torch.nn.functional as F

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



def log_parameter_drift(
    model: torch.nn.Module,
    initial_state: dict,
    model_prefix: str,
    step: int,
    logger_fn,  # e.g., log_metrics from lens.utils.logging (for W&B)
    log: logging.Logger, # Python logger
    is_main_process: bool
):
    """Calculates and logs parameter drift from their initial states."""
    metrics_to_log = {}
    param_group_drifts = {}  # To store sum of drifts and sum of initial norms per group

    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_state:
            initial_p = initial_state[name].to(param.device)
            current_p = param.data
            
            abs_drift = torch.norm(current_p - initial_p)
            norm_initial_p = torch.norm(initial_p)
            # rel_drift is calculated per parameter but not directly logged per parameter to avoid too many metrics.
            # We will calculate an overall relative drift for the group.

            group = "other_trainable"
            # Determine group for parameter based on its name
            if name == "soft_prompt_embeddings":  # Encoder specific
                group = "prompts"
            elif name == "prompt_left_emb" or name == "prompt_right_emb":  # Decoder specific
                group = "prompts"
            elif name.startswith("proj."):
                group = "projection"
            elif name.startswith("out."):  # Decoder specific output head
                group = "output_head"
            elif name =="activation_pos_embedder":  # Decoder specific
                group = "extra_pos_embeddings"
            elif name.startswith("base."):
                # Position embeddings
                if any(pos_emb in name for pos_emb in [".wpe.", ".position_embeddings."]):
                    group = "base_model_position_embeddings"
                # Input embeddings
                elif any(pos_emb in name for pos_emb in ["activation_pos_embedder"]):
                    group = "extra_pos_embeddings"
                elif any(emb_keyword in name for emb_keyword in [".wte.", ".embed_tokens.", ".word_embeddings."]):
                    group = "base_model_input_embeddings"
                # Output embeddings
                elif any(emb_keyword in name for emb_keyword in [".lm_head.", ".embed_out.", ".output_embeddings.m"]):
                    group = "base_model_output_embeddings"
                # LayerNorms (catch common LayerNorm naming)
                elif any(norm_keyword in name for norm_keyword in [".ln_", ".layernorm", ".norm."]):
                    group = "base_model_layernorms"
                # Transformer layers (blocks, attention, mlp, etc)
                elif any(layer_keyword in name for layer_keyword in [".h.", ".layers.", ".layer.", ".block.", ".attention.", ".mlp."]):
                    group = "base_model_transformer_layers"
                else:
                    group = "base_model_other"
            
            if group not in param_group_drifts:
                param_group_drifts[group] = {'sum_abs_drift': 0.0, 'sum_norm_initial': 0.0, 'count': 0}
            
            param_group_drifts[group]['sum_abs_drift'] += abs_drift.item()
            param_group_drifts[group]['sum_norm_initial'] += norm_initial_p.item()
            param_group_drifts[group]['count'] += 1
        elif param.requires_grad and name not in initial_state and is_main_process:
            log.warning(f"Trainable parameter {name} in {model_prefix} not found in initial_state for drift calculation.")

    for group, data in param_group_drifts.items():
        if data['count'] > 0:
            avg_abs_drift = data['sum_abs_drift'] / data['count']
            # Overall relative drift for the group: Sum(||curr-init||) / Sum(||init||)
            overall_rel_drift_group = data['sum_abs_drift'] / (data['sum_norm_initial'] + 1e-9)
                            
            metrics_to_log[f"drift/{model_prefix}/{group}/avg_abs_drift"] = avg_abs_drift
            metrics_to_log[f"drift/{model_prefix}/{group}/overall_rel_drift"] = overall_rel_drift_group
        elif is_main_process:
            log.warning(f"Parameter group {group} for {model_prefix} had 0 parameters for drift calculation. This is unexpected.")

    if metrics_to_log and is_main_process:
        logger_fn(metrics_to_log, step=step)
