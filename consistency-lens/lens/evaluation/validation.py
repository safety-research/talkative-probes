import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer # For type hint
from lens.models.orig import OrigWrapper # For type hint
from lens.training.loop import train_step # Assuming train_step is usable for eval
from lens.training.schedules import get_schedule_value
from lens.utils.logging import log_metrics # Assuming this is the W&B log_metrics

def run_validation_step(
    dec: nn.Module,
    enc: nn.Module,
    orig: OrigWrapper,
    val_loader: DataLoader,
    config: dict,
    tokenizer: AutoTokenizer, 
    cached_prefix_ids: torch.Tensor | None,
    device: torch.device,
    current_step: int,
    current_epoch: int, # 0-based epoch
    max_steps: int,
    steps_per_epoch: int,
    log: logging.Logger
) -> dict:
    """Runs a validation step and returns a dictionary of metrics."""
    dec.eval()
    enc.eval()
    val_loss = val_mse = val_lm = val_kl = 0.0
    val_seen = 0
    
    t_text = config['t_text']
    lm_base_weight = config['lm_base_weight']
    kl_base_weight = config['kl_base_weight']
    entropy_weight = config['entropy_weight']

    with torch.no_grad():
        for vbatch in val_loader:
            vbatch = {k: v.to(device) for k, v in vbatch.items()}
            sch_args = {
                "tau": get_schedule_value(config['gumbel_tau_schedule'], current_step, max_steps,
                                         current_epoch, steps_per_epoch),
                "T_text": t_text,
                "alpha": get_schedule_value(config['alpha_schedule'], current_step, max_steps,
                                           current_epoch, steps_per_epoch),
                "lm_base_weight": lm_base_weight,
                "kl_base_weight": kl_base_weight,
                "entropy_weight": entropy_weight,
                "mse_weight": config.get('mse_weight', 0.0),
            }
            # Assuming train_step can be used by passing eval versions of models
            # and it returns a dict of losses.
            v_losses = train_step(vbatch, {"dec": dec, "enc": enc, "orig": orig}, sch_args,
                                 lm_loss_natural_prefix=config.get('lm_loss_natural_prefix'),
                                 tokenizer=tokenizer,
                                 cached_prefix_ids=cached_prefix_ids) # train_step might not need tokenizer/cached_prefix_ids
            
            bsz = vbatch["A"].size(0) # Assuming "A" is a key in the batch
            val_loss += v_losses["total"].item() * bsz
            val_mse  += v_losses["mse"].item()   * bsz
            val_lm   += v_losses["lm"].item()    * bsz
            val_kl   += v_losses["kl"].item()    * bsz
            val_seen += bsz
            
    avg_val_loss = val_loss / val_seen if val_seen else float("nan")
    avg_val_mse  = val_mse  / val_seen if val_seen else float("nan")
    avg_val_lm   = val_lm   / val_seen if val_seen else float("nan")
    avg_val_kl   = val_kl   / val_seen if val_seen else float("nan")

    alpha_config = config.get('alpha_schedule', {})
    # Ensure 'value' exists for constant, 'end_value' for linear_warmup
    if alpha_config.get('type') == 'linear_warmup':
        final_alpha = alpha_config.get('end_value', 0.1) 
    elif alpha_config.get('type') == 'constant':
        final_alpha = alpha_config.get('value', 0.1)
    else: # Fallback or other types
        final_alpha = get_schedule_value(alpha_config, max_steps, max_steps, current_epoch, steps_per_epoch)


    normalized_val_loss = (lm_base_weight * final_alpha) * avg_val_lm + kl_base_weight * avg_val_kl
    
    log.info(
        f"Validation – loss {avg_val_loss:.4f}, mse {avg_val_mse:.4f}, lm {avg_val_lm:.4f}, kl {avg_val_kl:.4f}"
    )
    log.info(f"Normalized validation loss (for checkpointing): {normalized_val_loss:.4f}")
    
    metrics_to_log = {
        "eval/loss/total": avg_val_loss,
        "eval/loss/normalized": normalized_val_loss,
        "eval/loss/mse":    avg_val_mse,
        "eval/loss/lm":     avg_val_lm,
        "eval/loss/kl":     avg_val_kl,
    }
    log_metrics(metrics_to_log, step=current_step) # W&B logging

    dec.train()
    enc.train()
    
    return {
        "avg_val_loss": avg_val_loss,
        "normalized_val_loss": normalized_val_loss,
        "avg_val_mse": avg_val_mse,
        "avg_val_lm": avg_val_lm,
        "avg_val_kl": avg_val_kl,
    }
