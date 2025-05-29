import logging
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler
from lens.training.optim import param_groups # Already exists
from lens.training.schedules import get_lr_scheduler # Already exists

def create_optimizer_and_scheduler(
    dec_model: torch.nn.Module, 
    enc_model: torch.nn.Module, 
    config: dict, 
    learning_rate: float,
    projection_lr_multiplier: float,
    embedding_lr_multiplier: float,
    prompt_lr_multiplier: float,
    max_steps: int, 
    log: logging.Logger,
    # For resuming:
    scheduler_last_epoch: int = -1, 
    current_epoch_for_scheduler: int = 0, # Actually current epoch for some schedulers
    steps_per_epoch_for_scheduler: int = 0,
    opt_state_to_load: dict | None = None,
    scheduler_state_to_load: dict | None = None
) -> tuple[AdamW, _LRScheduler | None]:
    """
    Creates optimizer (AdamW), learning rate scheduler, and GradScaler.
    Handles loading state for optimizer and scheduler if provided.
    """
    
    # Optimizer groups
    # Ensure we pass the raw model if it's compiled/DDP for param_groups
    dec_to_group = dec_model.module if hasattr(dec_model, 'module') else dec_model
    enc_to_group = enc_model.module if hasattr(enc_model, 'module') else enc_model

    optimizer_groups_list = param_groups(
        [dec_to_group, enc_to_group], 
        learning_rate, 
        projection_lr_multiplier, 
        embedding_lr_multiplier, 
        prompt_lr_multiplier
    )

    opt = AdamW(optimizer_groups_list)
    if opt_state_to_load:
        try:
            opt.load_state_dict(opt_state_to_load)
            log.info("Loaded optimizer state from checkpoint.")
        except Exception as e:
            log.warning(f"Failed to load optimizer state: {e}. Initializing new optimizer.")

    # GradScaler (enabled only when using CUDA, determined by caller via device)
    # The caller script (01_train.py) will create the scaler based on device.
    # This function will just return it if passed or create a new one.
    # For simplicity now, let's assume device check is done by caller.
    # scaler = GradScaler(enabled=(device.type == "cuda")) # Original script does this
    # For now, this function won't create the scaler, it will be passed or handled by caller.

    # LR Scheduler
    lr_scheduler_config = config.get('lr_scheduler', {'type': 'constant'})
    lr_scheduler = get_lr_scheduler(
        opt, 
        lr_scheduler_config, 
        max_steps,
        last_epoch=scheduler_last_epoch, # PyTorch's last_epoch is step-based for step schedulers
        current_epoch=current_epoch_for_scheduler,
        steps_per_epoch=steps_per_epoch_for_scheduler
    )

    if lr_scheduler:
        log.info(f"Using LR scheduler: {lr_scheduler_config['type']}")
        if lr_scheduler_config.get('warmup_steps', 0) > 0:
             # parse_schedule_to_steps is in schedules.py
            from lens.training.schedules import parse_schedule_to_steps
            warmup_val = lr_scheduler_config['warmup_steps']
            parsed_warmup_steps = parse_schedule_to_steps(warmup_val, steps_per_epoch_for_scheduler)
            log.info(f"  with {parsed_warmup_steps} warmup steps (raw config: {warmup_val})")
        
        if scheduler_state_to_load:
            try:
                lr_scheduler.load_state_dict(scheduler_state_to_load)
                log.info("Loaded scheduler state from checkpoint.")
            except Exception as e:
                log.warning(f"Failed to load scheduler state: {e}. Scheduler will start fresh or from last_epoch.")
    
    # GradScaler is typically device-dependent, so initialize in main script
    # For now, this function returns None for scaler, expecting main script to handle it.
    return opt, lr_scheduler
