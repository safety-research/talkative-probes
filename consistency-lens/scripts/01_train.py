"""Minimal single-process training loop.

The *real* entry-point will wire up DeepSpeed & read ``lens.yaml``.  For the MVP
we just train Decoder/Encoder for a handful of steps to prove gradients flow.
"""

from __future__ import annotations

import argparse
import logging
import math # For math.ceil if needed, or integer arithmetic for epochs

import torch
from lens.data.collate import collate
from lens.data.dataset import ActivationDataset
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from torch.utils.data import DataLoader, random_split
from lens.training.optim import param_groups
from lens.utils.logging import init as log_init, log as log_metrics
from lens.training.schedules import get_schedule_value
import yaml
# from transformers import AutoTokenizer # Not used in this selection
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from pathlib import Path


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Train Consistency-Lens MVP")
    parser.add_argument("--config_path", type=str, default="consistency-lens/config/lens_simple.yaml", help="Path to the simple lens config file.")
    # Allow overriding some key parameters from the command line
    parser.add_argument("--model_name", type=str, help="Override model_name from config.")
    parser.add_argument("--activation_dir", type=str, help="Override activation_dir from config.")
    parser.add_argument("--val_activation_dir", type=str, help="Path to validation activations directory (skip split)")
    parser.add_argument("--max_train_steps", type=int, help="Override max_train_steps from config.")
    parser.add_argument("--learning_rate", type=float, help="Override learning_rate from config.")
    parser.add_argument("--t_text", type=int, help="Override t_text from config.")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N steps.") # Default was 100
    parser.add_argument("--log_interval", type=int, help="Log metrics every N steps.")
    parser.add_argument("--wandb_log_interval", type=int, help="Log metrics to WandB every N steps.")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from.")
    parser.add_argument("--val_fraction", type=float, help="Fraction of dataset for validation")
    parser.add_argument("--split_seed", type=int,   help="Seed for train/val split")
    parser.add_argument("--val_interval", type=int, help="Validate every N steps")
    parser.add_argument("--max_train_samples", type=int, help="Maximum number of training samples to load.")
    parser.add_argument("--max_val_samples", type=int, help="Maximum number of validation samples to load.")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Logging setup (console). W&B handled via lens.utils.logging.
    # ---------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # Load config from YAML file
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with CLI arguments if provided
    if args.model_name: config['model_name'] = args.model_name
    if args.max_train_steps: config['max_train_steps'] = args.max_train_steps
    if args.learning_rate: config['learning_rate'] = args.learning_rate
    if args.t_text: config['t_text'] = args.t_text
    if args.log_interval: config['log_interval'] = args.log_interval
    if hasattr(args, 'wandb_log_interval') and args.wandb_log_interval is not None:
        config['wandb_log_interval'] = args.wandb_log_interval
    # val_fraction, split_seed, val_interval can also be overridden from CLI
    if args.val_fraction is not None: config['val_fraction'] = args.val_fraction
    if args.split_seed is not None: config['split_seed'] = args.split_seed
    if args.val_interval is not None: config['val_interval'] = args.val_interval


    # Use overridden values or defaults from config
    model_name = config['model_name']
    # Handle activation_dir override carefully: CLI takes precedence over config.
    activation_dir = args.activation_dir if args.activation_dir is not None else config['activation_dumper']['output_dir']
    
    # Handle val_activation_dir: CLI takes precedence over config.
    effective_val_activation_dir = args.val_activation_dir if args.val_activation_dir is not None else config.get('val_activation_dir')
    
    max_steps = config['max_train_steps']
    learning_rate = config['learning_rate']
    t_text = config['t_text']
    wandb_config = config.get('wandb', {}) # Ensure wandb_config is a dict
    wandb_log_interval = config['wandb_log_interval']
    ce_weight = config['ce_weight']
    kl_base_weight = config['kl_base_weight']
    entropy_weight = config['entropy_weight']
    log_interval = config['log_interval']
    if log_interval <= 0:
        log.warning(f"log_interval must be positive, got {log_interval}. Setting to 100.")
        log_interval = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize W&B logging (if enabled in config)
    # Pass the merged config (YAML + CLI overrides)
    log_init(project=wandb_config.get('project', 'consistency-lens'), # Add default project name
             config=config,  # Pass the potentially modified config dict
             mode=wandb_config.get('mode', 'online')) # Add default mode


    # Dataset loading parameters from args (CLI) and config
    max_train_samples_req = args.max_train_samples
    max_val_samples_req = args.max_val_samples
    
    split_seed = config['split_seed']
    val_interval = config['val_interval']

    train_ds, val_ds = None, None

    if effective_val_activation_dir and Path(effective_val_activation_dir).exists():
        log.info(f"Loading training data from {activation_dir} (limit: {max_train_samples_req if max_train_samples_req is not None else 'all'}).")
        train_ds = ActivationDataset(activation_dir, max_samples=max_train_samples_req, desc="Loading train activations")
        log.info(f"Loading validation data from {effective_val_activation_dir} (limit: {max_val_samples_req if max_val_samples_req is not None else 'all'}).")
        val_ds = ActivationDataset(effective_val_activation_dir, max_samples=max_val_samples_req, desc="Loading val activations")
        
        if not train_ds or len(train_ds) == 0:
            raise RuntimeError(
                f"No .pt files found or loaded from train directory {activation_dir} (limit: {max_train_samples_req})."
            )
        if val_ds is not None and len(val_ds) == 0: # Check if val_ds was attempted but empty
            log.warning(
                f"No .pt files found or loaded from validation directory {effective_val_activation_dir} (limit: {max_val_samples_req}). Validation will be skipped."
            )
            val_ds = None # Ensure val_loader becomes None later

    else:
        if effective_val_activation_dir and not Path(effective_val_activation_dir).exists():
            log.warning(f"Validation activations directory {effective_val_activation_dir} not found. Falling back to random split from {activation_dir}.")
        
        val_fraction = config.get('val_fraction', 0.1)
        log.info(f"Preparing to split data from {activation_dir} with val_fraction={val_fraction:.2f}, seed={split_seed}.")

        initial_load_n = None
        # Determine total number of samples to load initially to satisfy requests + fraction
        if max_train_samples_req is not None and max_val_samples_req is not None:
            initial_load_n = max_train_samples_req + max_val_samples_req
        elif max_train_samples_req is not None: # Only train count specified
            if 0.0 <= val_fraction < 1.0 and (1.0 - val_fraction) > 1e-9: # train_part > 0
                initial_load_n = math.ceil(max_train_samples_req / (1.0 - val_fraction))
            else: # val_fraction is 1.0 (all val) or train_part is 0. Load at least max_train_req.
                  # The split logic later will handle if max_train_req is incompatible with val_fraction.
                initial_load_n = max_train_samples_req 
        elif max_val_samples_req is not None: # Only val count specified
            if 0.0 < val_fraction <= 1.0 and val_fraction > 1e-9: # val_part > 0
                initial_load_n = math.ceil(max_val_samples_req / val_fraction)
            else: # val_fraction is 0.0 (all train) or val_part is 0. Load at least max_val_req.
                initial_load_n = max_val_samples_req
        
        log.info(f"Initial load limit for splitting: {initial_load_n if initial_load_n is not None else 'all available'}.")
        full_dataset_loaded = ActivationDataset(activation_dir, max_samples=initial_load_n, desc="Loading activations for split")

        if not full_dataset_loaded or len(full_dataset_loaded) == 0:
            raise RuntimeError(
                f"No .pt files found or loaded from {activation_dir} (limit: {initial_load_n}). Run scripts/00_dump_activations.py first."
            )

        available_total = len(full_dataset_loaded)
        log.info(f"Loaded {available_total} samples for splitting.")

        # Determine target sizes for train and val based on requests and available data
        final_val_size = 0
        if max_val_samples_req is not None:
            final_val_size = min(max_val_samples_req, available_total)
        else:
            final_val_size = int(available_total * val_fraction)
        final_val_size = max(0, min(final_val_size, available_total))

        final_train_size = 0
        if max_train_samples_req is not None:
            final_train_size = min(max_train_samples_req, available_total - final_val_size)
        else:
            final_train_size = available_total - final_val_size
        final_train_size = max(0, final_train_size)

        # Adjust if sum of targets differs from available_total or requested sum
        # This ensures the dataset to be split matches the sum of final train/val sizes.
        dataset_to_actually_split = full_dataset_loaded
        current_total_target = final_train_size + final_val_size

        if current_total_target > available_total:
            # This case implies requests were too high for available data.
            # Re-evaluate based on available_total, prioritizing val_fraction.
            log.warning(f"Sum of calculated train ({final_train_size}) and val ({final_val_size}) "
                        f"exceeds available ({available_total}). Re-adjusting based on val_fraction.")
            final_val_size = int(available_total * val_fraction)
            final_train_size = available_total - final_val_size
            # dataset_to_actually_split is already full_dataset_loaded (i.e., all available)
        elif current_total_target < available_total:
            # Loaded more than needed by final_train_size + final_val_size. Take a subset.
            log.info(f"Loaded {available_total} but target sum is {current_total_target}. "
                     f"Taking subset of {current_total_target} before splitting.")
            dataset_to_actually_split = torch.utils.data.Subset(full_dataset_loaded, range(current_total_target))
        # Else (current_total_target == available_total), dataset_to_actually_split is full_dataset_loaded.
        
        # Perform the split on dataset_to_actually_split
        # The lengths for random_split must be final_train_size and final_val_size.
        # Their sum is len(dataset_to_actually_split) due to the logic above.
        
        if final_val_size > 0 and final_train_size > 0:
            log.info(f"Splitting {len(dataset_to_actually_split)} samples into train: {final_train_size}, val: {final_val_size}.")
            train_ds, val_ds = random_split(
                dataset_to_actually_split,
                [final_train_size, final_val_size],
                generator=torch.Generator().manual_seed(split_seed),
            )
        elif final_train_size > 0: # Only train data
            train_ds = dataset_to_actually_split 
            val_ds = None
            log.info(f"Using all {len(train_ds)} samples from loaded/subsetted data for training, no validation split.")
        elif final_val_size > 0: # Only val data
            val_ds = dataset_to_actually_split
            train_ds = None
            log.warning(f"Training set is empty. Using all {len(val_ds)} samples from loaded/subsetted data for validation.")
        else: # Both are zero
            log.warning("Both train and validation target sizes are 0. No data to load.")
            train_ds = None
            val_ds = None

    # Create DataLoaders
    if train_ds and len(train_ds) > 0:
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate)
    else:
        # This path should ideally not be reached if checks above are robust,
        # but as a safeguard:
        log.error("Training dataset is empty or None after processing. Cannot create DataLoader.")
        raise RuntimeError("Training dataset is empty. Check data paths, limits, and split configuration.")

    if val_ds and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate)
    else:
        val_loader = None
        # This is an expected outcome if no validation data was configured or found.
        log.info("Validation dataset is empty or None. Validation will be skipped during training.")

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0: # Should not happen if dataset is not empty and batch_size > 0
        log.warning("DataLoader is empty or batch_size is too large for dataset; steps_per_epoch is 0.")
        # This implies an issue, but to prevent division by zero if max_steps > 0:
        num_epochs_total_approx = 0 if max_steps == 0 else 1 
    elif max_steps == 0:
        num_epochs_total_approx = 0
    else:
        num_epochs_total_approx = (max_steps - 1) // steps_per_epoch + 1


    log.info("Starting training run – Model: %s, Activations: %s", model_name, activation_dir)
    log.info(
        "Configuration: %d total steps, Batch Size: %d, Train Dataset Size: %d, Val Dataset Size: %d samples",
        max_steps, config['batch_size'], len(train_ds), len(val_ds)
    )
    if steps_per_epoch > 0 :
        log.info(
            "Derived: %d steps/epoch, Approx. %d total epochs",
            steps_per_epoch, num_epochs_total_approx
        )

    if val_loader:
        log.info("Dataset split – train: %d | val: %d", len(train_ds), len(val_ds))


    dec_raw = Decoder(DecoderConfig(model_name=model_name, n_prompt_tokens=config['decoder_n_prompt_tokens']))
    enc_raw = Encoder(model_name)

    if config.get('compile_models', True):
        log.info("Compiling models")
        dec = torch.compile(dec_raw).to(device)
        enc = torch.compile(enc_raw).to(device)
    else:
        log.info("Not compiling models")
        dec = dec_raw.to(device)
        enc = enc_raw.to(device)

    orig = OrigWrapper(model_name, load_in_8bit=False)
    orig.model.to(device)

    groups = param_groups(dec, learning_rate) + param_groups(enc, learning_rate)
    log.info(f"Total number of parameters: {sum(p['params'][0].numel() for p in groups)}") # Assuming p['params'] is a list with one tensor
    opt = torch.optim.AdamW(groups)
    # gradscaler takes device not device_type (Note: This comment is from original code. Standard GradScaler does not take 'device'.)
    scaler = GradScaler(enabled=device.type == "cuda") # Removed `device="cuda"` as it's not a standard param for GradScaler

    start_step = 0
    if args.resume:
        from lens.utils import checkpoint as ckpt
        # Checkpoint stores the last completed step
        rec = ckpt.load(args.resume, models={"dec": dec, "enc": enc}, optim=opt, map_location=device)
        start_step = int(rec.get("step", -1)) + 1 # Resume from the next step
        log.info(f"Resuming training from step {start_step}")


    step_iter = iter(train_loader)
    for step in range(start_step, max_steps):
        try:
            batch = next(step_iter)
        except StopIteration:
            # Epoch finished, re-initialize data loader
            step_iter = iter(train_loader)
            batch = next(step_iter)

        # Move batch tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}

        opt.zero_grad(set_to_none=True)

        cast_ctx = autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
        with cast_ctx:
            current_tau = get_schedule_value(config['gumbel_tau_schedule'], step, max_steps)
            current_alpha = get_schedule_value(config['alpha_schedule'], step, max_steps)

            losses = train_step(
                batch,
                {"dec": dec, "enc": enc, "orig": orig},
                {
                    "tau": current_tau,
                    "T_text": t_text,
                    "alpha": current_alpha,
                    "ce_weight": ce_weight,
                    "kl_base_weight": kl_base_weight,
                    "entropy_weight": entropy_weight,
                },
            )
            loss = losses["total"]

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        
        current_epoch_num = (step // steps_per_epoch) + 1 if steps_per_epoch > 0 else 1


        # Log metrics at specified interval or at the last step
        if step % log_interval == 0 or step == max_steps - 1:
            log_msg_parts = [
                f"Epoch {current_epoch_num}/{num_epochs_total_approx}" if steps_per_epoch > 0 else f"Step {step}",
                f"Step {step}/{max_steps-1}", # max_steps-1 is the last step index
                f"loss {loss.item():.4f}",
                f"mse {losses['mse'].item():.4f}",
                f"ce {losses['ce'].item():.4f}",
                f"kl {losses['kl'].item():.4f}",
                f"entropy {losses['entropy'].item():.4f}",
                f"tau {current_tau:.3f}",
                f"alpha {current_alpha:.3f}",
            ]
            log.info(" | ".join(log_msg_parts))
        if step % wandb_log_interval == 0 or step == max_steps - 1:
            metrics_to_log = {
                "loss/total": loss.item(),
                "loss/mse": losses["mse"].item(),
                "loss/ce": losses["ce"].item(),
                "loss/kl": losses["kl"].item(),
                "loss/entropy": losses["entropy"].item(),
                "params/tau": current_tau,
                "params/alpha": current_alpha,
                "params/ce_w": ce_weight,
                "params/kl_w": kl_base_weight,
                "params/entropy_w": entropy_weight,
            }
            if steps_per_epoch > 0:
                metrics_to_log["epoch"] = current_epoch_num
            
            log_metrics(metrics_to_log, step=step)

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        if args.save_every > 0 and step % args.save_every == 0 and step > 0: # step > 0 to avoid saving at step 0 if save_every is small
            from lens.utils import checkpoint

            checkpoint_path = f"outputs/ckpt_step_{step}.pt"
            checkpoint.save(
                path=checkpoint_path,
                models={"dec": dec, "enc": enc}, # Pass raw models if compiled
                optim=opt,
                step=step, # Save the completed step number
                tau=current_tau,
                alpha=current_alpha,
                # Optionally save config or other metadata
            )
            epoch_info_str = f"(Epoch {current_epoch_num})" if steps_per_epoch > 0 else ""
            log.info(f"Saved checkpoint to {checkpoint_path} at step {step} {epoch_info_str}")

        # run validation at interval
        if val_loader and val_interval > 0 and step > 0 and step % val_interval == 0:
            dec.eval()
            enc.eval()
            val_loss = val_mse = val_ce = val_kl = 0.0
            val_seen = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = {k: v.to(device) for k, v in vbatch.items()}
                    sch_args = {
                        "tau": get_schedule_value(config['gumbel_tau_schedule'], step, max_steps),
                        "T_text": t_text,
                        "alpha": get_schedule_value(config['alpha_schedule'], step, max_steps),
                        "ce_weight": ce_weight,
                        "kl_base_weight": kl_base_weight,
                        "entropy_weight": entropy_weight,
                    }
                    v_losses = train_step(vbatch, {"dec": dec, "enc": enc, "orig": orig}, sch_args)
                    bsz = vbatch["A"].size(0)
                    val_loss += v_losses["total"].item() * bsz
                    val_mse  += v_losses["mse"].item()   * bsz
                    val_ce   += v_losses["ce"].item()    * bsz
                    val_kl   += v_losses["kl"].item()    * bsz
                    val_seen += bsz
            avg_val_loss = val_loss / val_seen if val_seen else float("nan")
            avg_val_mse  = val_mse  / val_seen if val_seen else float("nan")
            avg_val_ce   = val_ce   / val_seen if val_seen else float("nan")
            avg_val_kl   = val_kl   / val_seen if val_seen else float("nan")
            log.info(
                f"Validation – loss {avg_val_loss:.4f}, mse {avg_val_mse:.4f}, ce {avg_val_ce:.4f}, kl {avg_val_kl:.4f}"
            )
            log_metrics({
                "eval/loss/total": avg_val_loss,
                "eval/loss/mse":    avg_val_mse,
                "eval/loss/ce":     avg_val_ce,
                "eval/loss/kl":     avg_val_kl,
            }, step=step)
            dec.train()
            enc.train()

    log.info("Training finished after %d steps.", max_steps)


if __name__ == "__main__":
    main()
