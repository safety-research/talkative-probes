import logging
import torch
from transformers import AutoTokenizer # For type hint

from lens.models.decoder import Decoder # For type hint
from lens.models.encoder import Encoder # For type hint
from lens.models.orig import OrigWrapper # For type hint
from lens.evaluation.verbose_samples import process_and_print_verbose_batch_samples
from lens.evaluation.wandb_logger import verbose_samples_logger # For W&B table logging
from lens.training.schedules import get_schedule_value, parse_schedule_to_steps


def log_verbose_samples_if_needed(
    batch: dict, 
    config: dict, 
    models: dict, # {"dec": dec_model, "enc": enc_model}
    orig: OrigWrapper, 
    tokenizer: AutoTokenizer, 
    step: int, 
    current_epoch: int, # 0-based
    max_steps: int, 
    steps_per_epoch: int, 
    epoch_just_finished: bool,
    current_wandb_run_id: str | None, 
    device: torch.device, 
    log: logging.Logger
):
    """Checks if verbose samples should be generated and logged, then does so."""
    verbose_config = config.get('verbose_samples', {})
    if not verbose_config.get('enabled', False):
        return

    dec_model = models["dec"]
    enc_model = models["enc"]

    should_print = False
    
    verbose_interval_str = verbose_config.get('interval', "1000s") # Default from original
    try:
        # parse_schedule_to_steps is in lens.training.schedules
        verbose_interval = parse_schedule_to_steps(verbose_interval_str, steps_per_epoch)
        if step > 0 and step % verbose_interval == 0: # Ensure not on step 0 unless interval is 1
            should_print = True
    except Exception as e:
        log.warning(f"Failed to parse verbose_samples interval '{verbose_interval_str}': {e}")
    
    # Original script also had print_every_epoch and print_every_n_steps
    if verbose_config.get('print_every_epoch', False) and epoch_just_finished:
        should_print = True
    
    # This was an additional check in the original script
    # It might be redundant if 'interval' with 's' suffix is used correctly
    # For now, including it to match behavior.
    print_every_n_steps_val = verbose_config.get('print_every_n_steps', 0)
    if print_every_n_steps_val > 0:
        if step > 0 and step % print_every_n_steps_val == 0:
            should_print = True
            
    # Special case for step 0 from original script (for initial validation/test)
    # This was tied to `do_all_initial_validation` which is separate.
    # The verbose sample part for step 0 was:
    # if step == 0: should_print = True # (or based on some other condition for first batch)
    # For now, we'll rely on the interval logic. If interval is 1s, it will print on step 0.

    if should_print:
        log.info(f"Generating verbose samples at step {step}, epoch {current_epoch + 1}")
        dec_model.eval()
        enc_model.eval()
        
        sch_args = {
            "tau": get_schedule_value(config['gumbel_tau_schedule'], step, max_steps,
                                     current_epoch, steps_per_epoch),
            "T_text": config['t_text'],
            "alpha": get_schedule_value(config['alpha_schedule'], step, max_steps,
                                       current_epoch, steps_per_epoch),
            "lm_weight": config['lm_weight'],
            "kl_base_weight": config['kl_base_weight'],
            "entropy_weight": config['entropy_weight'],
            # mse_weight might be missing from original verbose call, adding if needed by process_and_print
            "mse_weight": config.get('mse_weight', 0.0), 
        }
        
        # process_and_print_verbose_batch_samples is in lens.evaluation.verbose_samples
        num_printed, captured_text = process_and_print_verbose_batch_samples(
            batch=batch,
            cfg=config, # Original script passed 'cfg', which is our 'config' dict
            models={"dec": dec_model, "enc": enc_model},
            orig=orig,
            tok=tokenizer,
            sch_args=sch_args,
            device=device,
            num_samples=verbose_config.get('num_samples', 2),
            top_n_analysis=verbose_config.get('top_n_predictions', 3),
            # printed_count_so_far=0, # This was for iterative printing, not needed here
            generate_continuation=verbose_config.get('generate_continuation', True),
            continuation_tokens=verbose_config.get('continuation_tokens', 30),
            return_structured_data=False, # Assuming we want text output for logging
            capture_output=True
        )
        
        if captured_text and current_wandb_run_id:
            # verbose_samples_logger is in lens.evaluation.wandb_logger
            verbose_samples_logger.log_verbose_samples(
                captured_text, 
                step=step,
                table_name="training_verbose_samples" # Default name from original
            )
        
        dec_model.train()
        enc_model.train()
