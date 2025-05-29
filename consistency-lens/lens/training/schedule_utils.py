import logging
from lens.training.schedules import parse_schedule_to_steps

def derive_training_schedule_params(
    config: dict, 
    train_loader_len: int, 
    log: logging.Logger
) -> tuple[int, int, int, int, int, int]:
    """
    Calculates steps_per_epoch, max_steps, num_epochs_total_approx,
    and resolves various interval configurations.
    """
    max_steps_config = config['max_train_steps']
    
    steps_per_epoch = train_loader_len
    if steps_per_epoch == 0:
        log.warning("DataLoader is empty or batch_size is too large for dataset; steps_per_epoch is 0.")
        num_epochs_total_approx = 0 if max_steps_config == 0 else 1
        # If epoch-based training was intended, max_steps might become 0 here.
        # The training loop should handle max_steps = 0 by not running.
        if config.get('num_train_epochs', 0) > 0 and max_steps_config == 0:
             max_steps = 0 # Ensure it's 0 if epoch based and loader empty
        else:
            max_steps = max_steps_config # Use what's in config
    else:
        num_train_epochs = config.get('num_train_epochs', 0)
        if num_train_epochs > 0 and max_steps_config == 0:
            max_steps = steps_per_epoch * num_train_epochs
            num_epochs_total_approx = num_train_epochs
            log.info(f"Epoch-based training: {num_train_epochs} epochs × {steps_per_epoch} steps/epoch = {max_steps} total steps")
        elif max_steps_config > 0:
            max_steps = max_steps_config
            num_epochs_total_approx = (max_steps - 1) // steps_per_epoch + 1
        else:
            raise ValueError("Either 'num_train_epochs' or 'max_train_steps' must be > 0 in config if dataloader is not empty.")

    # Parse intervals
    # Using parse_schedule_to_steps from lens.training.schedules
    wandb_log_interval_str = config.get('wandb_log_interval', "100s") # Default if not present
    wandb_log_interval = parse_schedule_to_steps(wandb_log_interval_str, steps_per_epoch)
    if wandb_log_interval <= 0 :
        log.warning(f"wandb_log_interval parsed to {wandb_log_interval} from '{wandb_log_interval_str}'. Setting to 1.")
        wandb_log_interval = 1


    log_interval_str = config.get('log_interval', "100s") # Default if not present
    log_interval = parse_schedule_to_steps(log_interval_str, steps_per_epoch)
    if log_interval <= 0:
        log.warning(f"log_interval parsed to {log_interval} from '{log_interval_str}'. Setting to 1.")
        log_interval = 1
    
    val_interval_str = config.get('val_interval', "1e") # Default if not present
    val_interval = parse_schedule_to_steps(val_interval_str, steps_per_epoch)
    if val_interval <= 0 and train_loader_len > 0: # Only warn if validation is possible
        log.warning(f"val_interval parsed to {val_interval} from '{val_interval_str}'. Validation might not run as expected.")
        # Let it be non-positive if user set it that way, to disable validation by interval.
        # Or, enforce positive if val_loader exists? For now, allow disabling.


    return steps_per_epoch, max_steps, num_epochs_total_approx, wandb_log_interval, log_interval, val_interval
