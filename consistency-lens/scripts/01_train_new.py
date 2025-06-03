#!/usr/bin/env python3
"""Training script for Consistency Lens MVP."""

import logging
import math
import os
import time
from collections import deque
from pathlib import Path
import argparse # Not used directly, but good to keep if Hydra uses it under the hood

import hydra
from omegaconf import DictConfig # Hydra specific
import torch
torch.set_float32_matmul_precision('high') # TF32 for Ampere
from torch.cuda.amp import GradScaler

from tqdm import tqdm

# Local (lens) imports - to be organized
from lens.utils.checkpoint_manager import CheckpointManager # Assuming this exists
from lens.utils.logging import log_metrics as log_wandb_metrics # For W&B
from lens.training.loop import train_step
from lens.training.schedules import (
    get_schedule_value, 
    should_unfreeze_any_component,
    apply_unfreeze_warmup,
    unfreeze_non_adapters,
    # parse_schedule_to_steps, # Now used via schedule_utils
    get_autocast_context,
    optimizer_step,
)
from lens.evaluation.verbose_samples import process_and_print_verbose_batch_samples # Used by verbose_logging
# from lens.evaluation.wandb_logger import verbose_samples_logger # Used by verbose_logging

# New utility imports
from lens.utils.config_utils import load_and_normalize_config
from lens.utils.path_utils import (
    _get_hydra_config_name, # Keep if needed directly, or if generate_run_name uses it
    extract_dataset_info,
    generate_run_name as generate_run_name_util,
    setup_activation_dirs,
    setup_checkpoint_dir_and_config,
)
from lens.utils.wandb_utils import init_wandb_run
from lens.utils.system_utils import get_system_metrics, format_time
from lens.utils.param_utils import log_parameter_counts as log_parameter_counts_util
from lens.data.loading import prepare_dataloaders
from lens.training.schedule_utils import derive_training_schedule_params
from lens.models.model_setup import initialize_raw_models_and_tokenizer
from lens.training.optim_setup import create_optimizer_and_scheduler
from lens.evaluation.validation import run_validation_step as run_validation_step_util
from lens.evaluation.metric_utils import log_epoch_token_statistics
from lens.evaluation.verbose_logging import log_verbose_samples_if_needed


# Helper for initial diagnostic prints (kept local for now as it's very specific)
def do_all_initial_validation(batch, orig, tokenizer, device, log, activation_dir):
    from lens.training.test import diagnose_activation_mismatch, diagnose_activation_save_load, check_dataset_activation_format, test_autocast_difference, check_layer_indexing # noqa
    diagnosis = diagnose_activation_mismatch(
        batch, orig, tokenizer, device, sample_idx=0, verbose=True
    )
    log.warning(diagnosis)
    print("\n=== Save/Load Cycle Diagnosis ===")
    i = 0 
    l = int(batch["layer_idx"][i].item())
    p = int(batch["token_pos_A"][i].item())
    input_ids = batch["input_ids_A"][i].unsqueeze(0).to(device)

    save_load_results, fresh_act = diagnose_activation_save_load(orig, input_ids, l, p, device)
    for k, v in save_load_results.items():
        print(f"{k}: {v}")

    print("\n=== Dataset Format Check ===")
    check_dataset_activation_format(activation_dir)

    print("\n=== Batch Activation Info ===")
    print(f"Batch A shape: {batch['A'].shape}")
    print(f"Batch A dtype: {batch['A'].dtype}")
    if batch['A'].numel() > 0: # Check if tensor is not empty
        print(f"Batch A[0] norm: {batch['A'][0].norm().item():.4f}")
    
    test_autocast_difference(orig, input_ids, l, p, device)
    check_layer_indexing(orig, input_ids, device)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ---------------------------------------------------------------
    # Initial Setup & Config Loading
    # ---------------------------------------------------------------
    config = load_and_normalize_config(cfg) # Includes OmegaConf.to_container and schedule parsing

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    log = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Path, Run Name, Dataset Info
    # ------------------------------------------------------------------
    activation_dir, effective_val_activation_dir = setup_activation_dirs(config, log)
    dataset_info = extract_dataset_info(activation_dir)
    hydra_config_name = _get_hydra_config_name() # From path_utils

    run_name_override = config.get('run_name')
    if run_name_override:
        run_name = run_name_override
        log.info(f"Using user-specified run name: {run_name}")
    else:
        run_name = generate_run_name_util(
            config, dataset_info, config.get('resume'), hydra_config_name, config.get('run_suffix')
        )
    
    run_checkpoint_dir = setup_checkpoint_dir_and_config(config, run_name, log) # Updates config inplace
    
    # Get gradient accumulation steps early
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

    log.info("=" * 60)
    log.info(f"Run Name: {run_name}")
    log.info(f"Dataset: {dataset_info.get('dataset', 'unknown')}")
    log.info(f"Model: {dataset_info.get('model_name', 'unknown')}") # model_name from config is used for loading
    log.info(f"Layer: {dataset_info.get('layer', config.get('layer_l', 'unknown'))}") # layer_l from config
    log.info(f"Checkpoint Dir: {run_checkpoint_dir}")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # W&B Initialization
    # ------------------------------------------------------------------
    resume_checkpoint_path = config.get('resume')
    current_wandb_run_id = init_wandb_run(
        config, run_name, dataset_info, resume_checkpoint_path, hydra_config_name, log
    )

    # ------------------------------------------------------------------
    # Device Setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    if device.type == 'cuda':
        log.info(f"Current CUDA devices: {torch.cuda.current_device()}")
        log.info(f"Current CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader, val_loader, train_ds, val_ds = prepare_dataloaders(
        config=config,
        activation_dir=activation_dir,
        effective_val_activation_dir=effective_val_activation_dir,
        max_train_samples_req=config.get('max_train_samples'),
        max_val_samples_req=config.get('max_val_samples'),
        log=log
    )
    if not train_loader or not train_ds : # Critical check
        log.error("Training DataLoader or Dataset could not be initialized. Exiting.")
        raise RuntimeError("Training data is not available.")
    
    val_loader_len = len(val_loader.dataset) if val_loader and val_loader.dataset else 0


    # ------------------------------------------------------------------
    # Training Schedule Parameters (steps, epochs, intervals)
    # ------------------------------------------------------------------
    steps_per_epoch, max_steps, num_epochs_total_approx, \
    wandb_log_interval, log_interval, val_interval = derive_training_schedule_params(
        config, len(train_loader), log # Pass len(train_loader) for steps_per_epoch calculation
    )
    config['max_train_steps'] = max_steps # Update config with derived max_steps

    # ------------------------------------------------------------------
    # Checkpoint Manager
    # ------------------------------------------------------------------
    # Pass the original hydra DictConfig 'cfg' if CheckpointManager parses schedule strings itself.
    # If CheckpointManager expects already parsed ints, pass the modified 'config' dict.
    # Assuming CheckpointManager can handle the main 'config' dict with parsed values.
    checkpoint_manager = CheckpointManager(cfg, log, steps_per_epoch)


    log.info("Starting training run – Model: %s, Activations: %s", config['model_name'], activation_dir)
    log.info(
        "Configuration: %d total steps, Batch Size: %d, Gradient Accumulation: %d, Train Dataset Size: %d, Val Dataset Size: %d samples",
        max_steps, config['batch_size'], gradient_accumulation_steps, len(train_ds), val_loader_len
    )
    if gradient_accumulation_steps > 1:
        log.info(
            "Effective batch size: %d (batch_size=%d × gradient_accumulation_steps=%d)",
            config['batch_size'] * gradient_accumulation_steps, config['batch_size'], gradient_accumulation_steps
        )
    if steps_per_epoch > 0:
        log.info(
            "Derived: %d steps/epoch, Approx. %d total epochs",
            steps_per_epoch, num_epochs_total_approx
        )

    # ------------------------------------------------------------------
    # Model Initialization (Raw models and Tokenizer)
    # ------------------------------------------------------------------
    dec_raw, enc_raw, orig, tokenizer, _ = initialize_raw_models_and_tokenizer(config, log)
    raise "error please check that we have tokenized"
    
    # Log parameter counts (BEFORE compilation)
    # Ensure correct model configs are passed if they were part of the initialize_raw_models_and_tokenizer return
    # For now, assuming DecoderConfig and EncoderConfig are constructed inside log_parameter_counts or passed correctly
    # The `log_parameter_counts_util` was written to take the raw model configs.
    from lens.models.decoder import DecoderConfig as DecConfigType # For type hint
    from lens.models.encoder import EncoderConfig as EncConfigType # For type hint

    param_stats = log_parameter_counts_util(
        dec_raw, enc_raw, orig, 
        DecConfigType(model_name=config['model_name'], **config.get('trainable_components', {}).get('decoder', {})), 
        EncConfigType(model_name=config['model_name'], **config.get('trainable_components', {}).get('encoder', {})), 
        log
    )
    initial_trainable_params_list = param_stats['trainable_params_list'] # Before any freeze logic changes requires_grad
    initial_total_trainable_val = param_stats['total_trainable']

    # Move orig model to device (dec_raw, enc_raw moved after potential compilation)
    if hasattr(orig, 'model') and orig.model is not None:
        orig.model.to(device)
    else:
        log.warning("orig.model not found or not initialized, cannot move to device.")


    # ------------------------------------------------------------------
    # Compile Models (Optional) & Move to Device
    # ------------------------------------------------------------------
    if config.get('compile_models', True):
        log.info("Compiling models...")
        # Before compiling, ensure models are on the correct device if compilation expects it.
        # Or, compile then move. PyTorch recommendation is usually model.to(device) then torch.compile(model)
        dec = torch.compile(dec_raw.to(device))
        enc = torch.compile(enc_raw.to(device))
    else:
        log.info("Not compiling models.")
        dec = dec_raw.to(device)
        enc = enc_raw.to(device)

    # ------------------------------------------------------------------
    # Freeze Schedule Setup
    # ------------------------------------------------------------------
    freeze_schedule_config = config.get('freeze_schedule', {})
    freeze_schedule_enabled = freeze_schedule_config.get('enabled', False)
    
    if freeze_schedule_enabled:
        log.info(f"Freeze schedule enabled. Global unfreeze: {freeze_schedule_config.get('unfreeze_at')}")
        # Initial freeze of non-adapter params
        for model_to_freeze in [dec_raw, enc_raw]: # Apply to raw models before optimizer creation
            for name, param in model_to_freeze.named_parameters():
                if 'base' in name or ('out' in name and model_to_freeze == dec_raw): # dec_raw.out
                    param.requires_grad = False
        log.info("Initial freeze applied to non-adapter parameters for freeze schedule.")
    else:
        log.info("Freeze schedule disabled.")

    # Update trainable_params list after initial freeze schedule modifications
    # This list will be used for gradient clipping.
    # The optimizer will be created based on current requires_grad status.
    current_trainable_params = [p for p in dec.parameters() if p.requires_grad] + \
                               [p for p in enc.parameters() if p.requires_grad]
    current_total_trainable_params_val = sum(p.numel() for p in current_trainable_params)
    
    log.info(f"Trainable params after initial freeze consideration: {current_total_trainable_params_val:,}")


    # ------------------------------------------------------------------
    # Optimizer and LR Scheduler
    # ------------------------------------------------------------------
    learning_rate = config['learning_rate']
    custom_lr_multipliers = config.get('custom_lr_multipliers', {})
    projection_lr_multiplier = custom_lr_multipliers.get('projection_layers', 1.0)
    embedding_lr_multiplier = custom_lr_multipliers.get('embedding_layers', 1.0)
    prompt_lr_multiplier = custom_lr_multipliers.get('prompt_layers', 1.0)

    start_step = 0
    resume_epoch_for_scheduler = 0
    scheduler_last_epoch_val = -1 # For PyTorch LR scheduler's last_epoch (step-based)
    opt_state_dict_to_load = None
    scheduler_state_dict_to_load = None
    checkpoint_data_for_resume = None  # Initialize for later reference

    if resume_checkpoint_path:
        # CheckpointManager.load_checkpoint should not modify dec/enc if they are already compiled
        # It should load into dec_raw/enc_raw if that's what was saved.
        # Let's assume CheckpointManager loads into the models passed to it.
        # If dec/enc are compiled, need to load into dec_raw/enc_raw.
        models_for_resume = {"dec": dec_raw, "enc": enc_raw}
        # Optimizer state is loaded *before* creating the new optimizer if unfreezing happens.
        # So, we load opt state here, and pass it to create_optimizer_and_scheduler
        
        # Peek into checkpoint for opt/scheduler state without creating optimizer yet
        if os.path.exists(resume_checkpoint_path):
            log.info(f"Attempting to load optimizer/scheduler state from: {resume_checkpoint_path}")
            # Create a dummy optimizer to load its state to avoid pre-maturely creating the final one.
            # This is a bit tricky. The original script loaded optimizer state *into an existing optimizer*.
            # If unfreezing happens, a new optimizer is created.
            # For now, let's fetch the state dicts.
            # CheckpointManager.load_checkpoint will need to be flexible.
            # For simplicity, let's assume we can get the state dicts.
            # This part needs careful handling of when the optimizer is created vs. when state is loaded,
            # especially with the freeze schedule.

            # Load checkpoint using CheckpointManager
            checkpoint_data_for_resume = checkpoint_manager.load_checkpoint(
                resume_checkpoint_path, 
                models=models_for_resume,
                optimizer=None,  # Will load optimizer state dict later
                map_location='cpu'
            )
            
            # Extract optimizer and scheduler states
            opt_state_dict_to_load = checkpoint_data_for_resume.get('optimizer')
            scheduler_state_dict_to_load = checkpoint_data_for_resume.get('scheduler')
            
            start_step = int(checkpoint_data_for_resume.get("step", -1)) + 1
            scheduler_last_epoch_val = int(checkpoint_data_for_resume.get("step", -1)) # for scheduler
            if steps_per_epoch > 0:
                resume_epoch_for_scheduler = start_step // steps_per_epoch
            log.info(f"Resuming training from step {start_step}")
        else:
            log.error(f"Resume checkpoint path specified but not found: {resume_checkpoint_path}")
            # Decide: raise error or continue without resuming? Original script raised FileNotFoundError earlier.
            # If we reach here, it implies an issue. For robustness, proceed without resume.
            resume_checkpoint_path = None # Nullify to prevent further resume attempts
            start_step = 0


    opt, lr_scheduler = create_optimizer_and_scheduler(
        dec, enc, config, learning_rate, 
        projection_lr_multiplier, embedding_lr_multiplier, prompt_lr_multiplier,
        max_steps, log,
        scheduler_last_epoch=scheduler_last_epoch_val,
        current_epoch_for_scheduler=resume_epoch_for_scheduler,
        steps_per_epoch_for_scheduler=steps_per_epoch,
        opt_state_to_load=opt_state_dict_to_load,
        scheduler_state_to_load=scheduler_state_dict_to_load
    )
    
    # GradScaler: Initialize after optimizer, and specific to device
    scaler = GradScaler(enabled=(device.type == "cuda"))
    if resume_checkpoint_path and scheduler_state_dict_to_load and 'scaler' in checkpoint_data_for_resume and device.type == "cuda":
        try:
            scaler.load_state_dict(checkpoint_data_for_resume['scaler'])
            log.info("Loaded GradScaler state from checkpoint.")
        except Exception as e:
            log.warning(f"Failed to load GradScaler state: {e}. Initializing new scaler.")


    # Log hyperparams
    log.info(f"Hyperparameters: lm_base_weight={config['lm_base_weight']}, kl_base_weight={config['kl_base_weight']}, entropy_weight={config['entropy_weight']}")
    log.info(f"Stop-grad on A′: {config['stop_grad_aprime']}") # From original
    log.info(f"Grad clip: {config['grad_clip']}")

    # Verify optimizer parameter counts if freeze schedule modified things
    num_params_in_optimizer_groups = sum(p.numel() for group in opt.param_groups for p in group['params'])
    if current_total_trainable_params_val != num_params_in_optimizer_groups:
        log_message = (
            f"Parameter count difference: current_total_trainable_params_val (after initial freeze) is {current_total_trainable_params_val:,}, "
            f"but optimizer groups sum to {num_params_in_optimizer_groups:,}."
        )
        if freeze_schedule_enabled: # Expected if params were frozen before opt creation
             log.info(f"{log_message} This is expected if freeze schedule made params non-trainable before optimizer creation.")
        else: # Not expected if no freeze schedule
             log.warning(f"{log_message} Check requires_grad flags and param grouping logic.")


    # ------------------------------------------------------------------
    # Training Loop Setup
    # ------------------------------------------------------------------
    epoch_decoded_tokens = []
    # train_loader should be an IterableDataset an not reset this way if using DDP with sharding.
    # For single GPU, this is fine.
    step_iter = iter(train_loader) 
    
    start_epoch_calc = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
    log.info(f"Effective Start step: {start_step}, Start epoch (0-indexed): {start_epoch_calc}, Max steps: {max_steps}")
    
    step_times = deque(maxlen=max(1, log_interval // 10 if log_interval > 10 else 10)) # Track recent step times for ETA
    
    cached_prefix_ids = None
    lm_loss_natural_prefix_text = config.get('lm_loss_natural_prefix')
    if lm_loss_natural_prefix_text and isinstance(lm_loss_natural_prefix_text, str):
        # Ensure tokenizer is available
        cached_prefix_ids = tokenizer(lm_loss_natural_prefix_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        log.info(f"Cached natural language prefix for LM loss: '{lm_loss_natural_prefix_text}' ({cached_prefix_ids.shape[1]} tokens)")
    
    # Freeze schedule state tracking
    current_epoch_for_freeze_check = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
    non_adapters_frozen_state = freeze_schedule_enabled and \
        not should_unfreeze_any_component(start_step, current_epoch_for_freeze_check, freeze_schedule_config, freeze_schedule_enabled)

    unfreeze_warmup_duration_str = freeze_schedule_config.get('warmup_duration', "100s")
    # _resolve_schedule_to_steps was moved to parse_schedule_to_steps in schedules.py
    from lens.training.schedules import parse_schedule_to_steps as parse_schedule_to_steps_fn
    unfreeze_warmup_steps_count = parse_schedule_to_steps_fn(unfreeze_warmup_duration_str, steps_per_epoch)

    newly_unfrozen_params_set = set()
    unfreeze_transition_step_marker = None


    # ------------------------------------------------------------------
    # Main Training Loop
    # ------------------------------------------------------------------
    # Initialize checkpoint_metrics before the loop
    checkpoint_metrics = {}
    
    pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="Training", dynamic_ncols=True, mininterval=1.0)

    for step in pbar:
        current_epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0
        current_epoch_display = current_epoch + 1 # 1-based for display
        epoch_just_ended = False

        # Gradient accumulation logic
        is_accumulation_cycle_start = (step % gradient_accumulation_steps == 0)
        is_optimizer_step_time = ((step + 1) % gradient_accumulation_steps == 0) or (step == max_steps - 1)

        # --- Unfreeze non-adapter parameters if scheduled ---
        if freeze_schedule_enabled and non_adapters_frozen_state and \
           should_unfreeze_any_component(step, current_epoch, freeze_schedule_config, freeze_schedule_enabled):
            log.info(f"Unfreezing non-adapter parameters at step {step}, epoch {current_epoch}")
            
            opt_state_before_recreate = opt.state_dict() # Save current optimizer state
            
            # unfreeze_non_adapters is from lens.training.schedules
            # It needs raw models to change requires_grad
            opt, current_trainable_params, newly_unfrozen_params_set = unfreeze_non_adapters(
                dec_raw, enc_raw, config, learning_rate, 
                projection_lr_multiplier, embedding_lr_multiplier, prompt_lr_multiplier,
                opt_state_before_recreate, step, current_epoch
            )
            
            # Recreate LR scheduler with new optimizer, preserving step count
            if lr_scheduler:
                scheduler_state_before_recreate = lr_scheduler.state_dict()
                lr_scheduler = create_optimizer_and_scheduler( # Simplified call, focus on scheduler part
                    dec, enc, config, learning_rate, 
                    projection_lr_multiplier, embedding_lr_multiplier, prompt_lr_multiplier,
                    max_steps, log,
                    scheduler_last_epoch=step -1, # current step is new "last_epoch" for scheduler
                    current_epoch_for_scheduler=current_epoch,
                    steps_per_epoch_for_scheduler=steps_per_epoch,
                    opt_state_to_load=None, # Optimizer already handled
                    scheduler_state_to_load=scheduler_state_before_recreate # Try to restore scheduler state
                )[1] # Get only the scheduler

            non_adapters_frozen_state = False
            unfreeze_transition_step_marker = step
            log_wandb_metrics({
                "freeze_schedule/transition_step": step,
                "freeze_schedule/non_adapters_frozen": 0,
            }, step=step)
        
        # --- Batch Loading ---
        try:
            batch = next(step_iter)
        except StopIteration:
            epoch_just_ended = True
            if epoch_decoded_tokens: # Log stats for the completed epoch
                log_epoch_token_statistics(epoch_decoded_tokens, tokenizer, current_epoch_display, step, log_interval, log)
            epoch_decoded_tokens = []
            step_iter = iter(train_loader) # Reset iterator for new epoch
            batch = next(step_iter)
            log.info(f"Epoch {current_epoch_display} finished. Starting epoch {current_epoch_display + 1}.")

        batch = {k: v.to(device) for k, v in batch.items()}
        
        if step == 0 and config.get('run_initial_diagnostics', False): # Optional initial diagnostics
            log.info("Running initial batch diagnostics...")
            do_all_initial_validation(batch, orig, tokenizer, device, log, activation_dir)


        # --- Forward and Backward Pass ---
        step_start_time_this_iter = time.time()

        if is_accumulation_cycle_start:
            opt.zero_grad(set_to_none=True)
        
        autocast_ctx = get_autocast_context(device) # From lens.training.schedules
        with autocast_ctx:
            current_gumbel_tau = get_schedule_value(config['gumbel_tau_schedule'], step, max_steps, current_epoch, steps_per_epoch)
            current_alpha_val = get_schedule_value(config['alpha_schedule'], step, max_steps, current_epoch, steps_per_epoch)

            loss_components = train_step(
                batch, {"dec": dec, "enc": enc, "orig": orig},
                {
                    "tau": current_gumbel_tau, "T_text": config['t_text'], "alpha": current_alpha_val,
                    "lm_base_weight": config['lm_base_weight'], "kl_base_weight": config['kl_base_weight'],
                    "entropy_weight": config['entropy_weight'], "mse_weight": config.get('mse_weight', 0.0),
                },
                lm_loss_natural_prefix=config.get('lm_loss_natural_prefix'), # From config
                tokenizer=tokenizer, cached_prefix_ids=cached_prefix_ids, # For train_step
                stop_grad_aprime=config.get('stop_grad_aprime', True) # From config
            )
            total_loss = loss_components["total"]
            if gradient_accumulation_steps > 1:
                total_loss = total_loss / gradient_accumulation_steps
        
        # Scale loss for AMP and backward
        if device.type == "cuda":
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        if "decoded_tokens_batch" in loss_components:
            epoch_decoded_tokens.extend(loss_components["decoded_tokens_batch"].tolist())

        # --- Optimizer Step & Gradient Clipping (if accumulation cycle ends) ---
        grad_norm_val = None
        update_norm_val = 0.0
        param_norm_val = 0.0
        update_ratio_val = 0.0

        if is_optimizer_step_time:
            if device.type == "cuda": scaler.unscale_(opt)
            grad_norm_val = torch.nn.utils.clip_grad_norm_(current_trainable_params, config['grad_clip'])
            
            # Store params before optimizer step for update ratio calculation
            params_before_step = [p.detach().clone() for p in current_trainable_params if p.grad is not None]

            # Unfreeze warmup application (from lens.training.schedules)
            unfreeze_transition_step_marker, should_clear_newly_unfrozen_set = apply_unfreeze_warmup(
                opt, newly_unfrozen_params_set, unfreeze_transition_step_marker,
                unfreeze_warmup_steps_count, step, freeze_schedule_config,
                log_interval, log
            )
            if should_clear_newly_unfrozen_set: newly_unfrozen_params_set.clear()

            optimizer_step(opt, scaler, device) # From lens.training.schedules (handles scaler.step/update)

            with torch.no_grad():
                upd_sq = 0.0
                param_sq = 0.0
                # Iterate over params_before_step and their corresponding current_trainable_params
                # This assumes order is preserved and param_groups in optimizer match current_trainable_params
                # This part needs to be careful if current_trainable_params changed due to unfreezing.
                # For simplicity, let's use opt.param_groups to get current params in optimizer.
                
                # A robust way: iterate through optimizer's param groups
                idx = 0
                for group in opt.param_groups:
                    for p_optim in group['params']:
                        if p_optim.grad is not None and idx < len(params_before_step): # Ensure we have a "before" state
                            diff = p_optim.data - params_before_step[idx].data
                            upd_sq += diff.pow(2).sum().item()
                            param_sq += p_optim.data.pow(2).sum().item()
                            idx +=1
                        elif p_optim.grad is None: # Param was not updated
                            param_sq += p_optim.data.pow(2).sum().item()


                update_norm_val = math.sqrt(upd_sq)
                param_norm_val = math.sqrt(param_sq)
                update_ratio_val = update_norm_val / (param_norm_val + 1e-12)


            if lr_scheduler: lr_scheduler.step()
        
        # --- Logging & Metrics ---
        step_time_this = time.time() - step_start_time_this_iter
        step_times.append(step_time_this)
        avg_recent_step_time = sum(step_times) / len(step_times)
        
        current_lr = opt.param_groups[0]["lr"]
        pbar.set_postfix({
            'loss': f'{loss_components["total"].item():.3f}', 'lr': f'{current_lr:.1e}',
            'sps': f'{1.0/avg_recent_step_time * config["batch_size"]:.1f}',
            'eta': format_time((max_steps - step - 1) * avg_recent_step_time),
            'tau': f'{current_gumbel_tau:.2f}', 'alpha': f'{current_alpha_val:.2f}',
            'acc': f'{(step % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}' if gradient_accumulation_steps > 1 else '',
        })

        if step % log_interval == 0 or step == max_steps - 1:
            log.info(f"Step {step}/{max_steps-1} | loss {loss_components['total'].item():.4f} | lr {current_lr:.2e} | {1.0/avg_recent_step_time * config['batch_size']:.1f} samples/s")
        
        if (step % wandb_log_interval == 0 or step == max_steps - 1) and current_wandb_run_id:
            metrics_dict = {
                "loss/total": loss_components["total"].item(), "loss/mse": loss_components["mse"].item(),
                "loss/lm": loss_components["lm"].item(), "loss/kl": loss_components["kl"].item(),
                "loss/entropy": loss_components["entropy"].item(),
                "params/tau": current_gumbel_tau, "params/alpha": current_alpha_val,
                "optim/lr": current_lr,
                "grads/norm": grad_norm_val.item() if isinstance(grad_norm_val, torch.Tensor) else grad_norm_val,
                "updates/norm": update_norm_val, "updates/ratio": update_ratio_val,
                "performance/avg_step_time_ms": avg_recent_step_time * 1000,
                "performance/samples_per_sec": (1.0/avg_recent_step_time * config['batch_size']) if avg_recent_step_time > 0 else 0,
                "gradient_accumulation/is_update_step": int(is_optimizer_step_time),
            }
            if steps_per_epoch > 0: metrics_dict["epoch"] = current_epoch_display
            if freeze_schedule_enabled: metrics_dict["freeze_schedule/non_adapters_frozen"] = 1 if non_adapters_frozen_state else 0
            
            if step % (wandb_log_interval * 10) == 0: # Less frequent system metrics
                 metrics_dict.update(get_system_metrics(device))
            
            log_wandb_metrics(metrics_dict, step=step)

        # --- Checkpointing ---
        # CheckpointManager determines if save is needed based on its internal logic (step/epoch interval)
        # Metrics for checkpoint metadata
        checkpoint_metrics = {
            "loss/total": loss_components["total"].item(),
            "loss/mse": loss_components["mse"].item(),
            "loss/lm": loss_components["lm"].item(),
            "loss/kl": loss_components["kl"].item(),
            "loss/entropy": loss_components["entropy"].item(),
            "params/tau": current_gumbel_tau,
            "params/alpha": current_alpha_val,
            "optim/lr": current_lr,
            "grads/norm": grad_norm_val.item() if isinstance(grad_norm_val, torch.Tensor) else (float(grad_norm_val) if grad_norm_val is not None else None),
        }
        if checkpoint_manager.should_save(step, current_epoch_display, epoch_just_ended):
            checkpoint_manager.save_checkpoint(
                step=step, epoch=current_epoch_display,
                models={"dec": dec_raw, "enc": enc_raw}, # Save raw models
                optimizer=opt, scheduler=lr_scheduler, metrics=checkpoint_metrics,
                config=cfg, # Pass original Hydra cfg for saving full config
                tau=current_gumbel_tau, alpha=current_alpha_val,
                wandb_run_id=current_wandb_run_id,
                scaler_state=scaler.state_dict() if device.type == "cuda" else None
            )
        
        # --- Validation ---
        if val_loader and val_interval > 0 and (step > 0 and step % val_interval == 0 or (val_interval == 1 and step ==0 ) or step == max_steps -1) :
            val_metrics_results = run_validation_step_util(
                dec, enc, orig, val_loader, config, tokenizer, cached_prefix_ids,
                device, step, current_epoch, max_steps, steps_per_epoch, log
            )
            # Save checkpoint if validation loss is best (CheckpointManager handles this logic)
            if checkpoint_manager.track_best_n > 0 and not math.isnan(val_metrics_results["normalized_val_loss"]):
                checkpoint_manager.save_checkpoint(
                    step=step, epoch=current_epoch_display,
                    models={"dec": dec_raw, "enc": enc_raw}, optimizer=opt, scheduler=lr_scheduler,
                    metrics={**checkpoint_metrics, **val_metrics_results}, config=cfg,
                    val_loss=val_metrics_results["normalized_val_loss"], # For best tracking
                    raw_val_loss=val_metrics_results["avg_val_loss"],
                    tau=current_gumbel_tau, alpha=current_alpha_val,
                    wandb_run_id=current_wandb_run_id,
                    scaler_state=scaler.state_dict() if device.type == "cuda" else None
                )
        
        # --- Verbose Samples ---
        if current_wandb_run_id: # Only if W&B is active for table logging
            log_verbose_samples_if_needed(
                batch, config, {"dec": dec, "enc": enc}, orig, tokenizer,
                step, current_epoch, max_steps, steps_per_epoch, epoch_just_ended,
                current_wandb_run_id, device, log
            )

    # --- End of Training ---
    pbar.close()
    if checkpoint_manager.save_at_end:
        log.info("Saving final checkpoint...")
        # Gather final metrics for checkpoint
        final_ckpt_metrics = {"loss/total": loss_components["total"].item() if 'loss_components' in locals() else float('nan')}
        checkpoint_manager.save_checkpoint(
            step=max_steps -1, epoch=num_epochs_total_approx,
            models={"dec": dec_raw, "enc": enc_raw}, optimizer=opt, scheduler=lr_scheduler,
            metrics=final_ckpt_metrics, config=cfg, 
            tau=current_gumbel_tau if 'current_gumbel_tau' in locals() else -1.0, 
            alpha=current_alpha_val if 'current_alpha_val' in locals() else -1.0,
            wandb_run_id=current_wandb_run_id,
            scaler_state=scaler.state_dict() if device.type == "cuda" else None
        )

    log.info("=" * 60)
    log.info(f"TRAINING COMPLETE. Run Name: {run_name}")
    log.info(f"Total Steps: {max_steps}, Final Step: {step if 'step' in locals() else 'N/A'}")
    best_ckpt = checkpoint_manager.get_best_checkpoint_path()
    if best_ckpt: log.info(f"Best Checkpoint: {best_ckpt.name}")
    if current_wandb_run_id: log.info(f"WandB Run ID: {current_wandb_run_id}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
