"""Minimal single-process training loop.

The *real* entry-point will wire up DeepSpeed & read ``lens.yaml``.  For the MVP
we just train Decoder/Encoder for a handful of steps to prove gradients flow.
"""

from __future__ import annotations

import argparse
import logging
import math # For math.ceil if needed, or integer arithmetic for epochs
from collections import Counter # Add this import

import torch
from lens.data.collate import collate
from lens.data.dataset import ActivationDataset
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.loop import train_step
from torch.utils.data import DataLoader, random_split
from lens.training.optim import param_groups
from lens.utils.logging import init as log_init, log as log_metrics
from lens.training.schedules import get_schedule_value
from lens.utils.checkpoint_manager import CheckpointManager
import yaml
from lens.evaluation.verbose_samples import process_and_print_verbose_batch_samples
from transformers import AutoTokenizer # Not used in this selection
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from pathlib import Path # This import is assumed to be at the module level. If not, it should be moved there.
import math # For math.ceil
import torch # For torch.utils.data, torch.Generator
from torch.utils.data import DataLoader, random_split, Dataset, Subset # Explicit imports for clarity
from lens.data.collate import collate
from lens.data.dataset import ActivationDataset
import logging # For type hinting Logger
import torch.nn as nn
from lens.utils.embedding_remap import remap_embeddings

# Helper function to prepare datasets and dataloaders
def _prepare_dataloaders(
    config: dict,
    activation_dir: str,
    effective_val_activation_dir: str | None,
    max_train_samples_req: int | None,
    max_val_samples_req: int | None,
    log: logging.Logger
) -> tuple[DataLoader | None, DataLoader | None, Dataset | None, Dataset | None]:
    """Loads, splits (if necessary), and creates DataLoaders for train/validation."""

    train_ds: Dataset | None = None
    val_ds: Dataset | None = None

    # Retrieve dataset configuration from the main config
    split_seed = config['split_seed']
    val_fraction = config.get('val_fraction', 0.1) # Default val_fraction if not in config
    batch_size = config['batch_size']


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
        dataset_to_actually_split: Dataset = full_dataset_loaded
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
            dataset_to_actually_split = Subset(full_dataset_loaded, range(current_total_target))
        # Else (current_total_target == available_total), dataset_to_actually_split is full_dataset_loaded.
        
        # Perform the split on dataset_to_actually_split
        # The lengths for random_split must be final_train_size and final_val_size.
        # Their sum is len(dataset_to_actually_split) due to the logic above.
        
        if final_val_size > 0 and final_train_size > 0:
            log.info(f"Splitting {len(dataset_to_actually_split)} samples into train: {final_train_size}, val: {final_val_size}.")
            # Type ignore below because random_split can return List[Subset[T]]
            # and we are destructuring it.
            train_ds, val_ds = random_split( # type: ignore[assignment]
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
    train_loader: DataLoader | None = None
    if train_ds and len(train_ds) > 0:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    else:
        # This path should ideally not be reached if checks above are robust,
        # but as a safeguard:
        log.error("Training dataset is empty or None after processing. Cannot create DataLoader.")
        raise RuntimeError("Training dataset is empty. Check data paths, limits, and split configuration.")

    val_loader: DataLoader | None = None
    if val_ds and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    else:
        # This is an expected outcome if no validation data was configured or found.
        log.info("Validation dataset is empty or None. Validation will be skipped during training.")
    
    return train_loader, val_loader, train_ds, val_ds


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
    parser.add_argument("--log_interval", type=int, help="Log metrics every N steps.")
    parser.add_argument("--wandb_log_interval", type=int, help="Log metrics to WandB every N steps.")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from.")
    parser.add_argument("--val_fraction", type=float, help="Fraction of dataset for validation")
    parser.add_argument("--split_seed", type=int,   help="Seed for train/val split")
    parser.add_argument("--val_interval", type=int, help="Validate every N steps") # Retained for use later in training loop
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
    tokenizer_name = config.get("tokenizer_name", model_name) # Moved here for earlier access
    
    # Handle activation_dir override carefully: CLI takes precedence over config.
    base_activation_dir_str = args.activation_dir if args.activation_dir is not None else config['activation_dumper']['output_dir']
    activation_dir = str(Path(Path(base_activation_dir_str).parent / tokenizer_name / Path(base_activation_dir_str).name))
    
    # Handle val_activation_dir: CLI takes precedence over config.
    base_val_activation_dir_str = args.val_activation_dir if args.val_activation_dir is not None else config.get('val_activation_dir')
    effective_val_activation_dir: str | None = None
    if base_val_activation_dir_str:
        effective_val_activation_dir = str(Path(Path(base_val_activation_dir_str).parent / tokenizer_name / Path(base_val_activation_dir_str).name))
    
    max_steps = config['max_train_steps']
    learning_rate = config['learning_rate']
    t_text = config['t_text']
    wandb_config = config.get('wandb', {}) # Ensure wandb_config is a dict
    wandb_log_interval = config['wandb_log_interval']
    lm_weight = config['lm_weight']
    kl_base_weight = config['kl_base_weight']
    entropy_weight = config['entropy_weight']
    log_interval = config['log_interval']
    if log_interval <= 0:
        log.warning(f"log_interval must be positive, got {log_interval}. Setting to 100.")
        log_interval = 100
    
    # val_interval is used by the training loop, not dataset prep, but initialized here from config
    val_interval = config['val_interval']

    # Extract trainable_components and custom_lr_multipliers from config
    trainable_components_config = config.get('trainable_components', {})
    decoder_train_cfg = trainable_components_config.get('decoder', {})
    encoder_train_cfg = trainable_components_config.get('encoder', {})
    custom_lr_multipliers = config.get('custom_lr_multipliers', {})
    projection_lr_multiplier = custom_lr_multipliers.get('projection_layers', 1.0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize W&B logging (if enabled in config)
    # Pass the merged config (YAML + CLI overrides)
    log_init(project=wandb_config.get('project', 'consistency-lens'), # Add default project name
             config=config,  # Pass the potentially modified config dict
             mode=wandb_config.get('mode', 'online')) # Add default mode

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(config, log)

    # Prepare DataLoaders by calling the helper function
    train_loader, val_loader, train_ds, val_ds = _prepare_dataloaders(
        config=config,
        activation_dir=activation_dir,
        effective_val_activation_dir=effective_val_activation_dir,
        max_train_samples_req=args.max_train_samples,
        max_val_samples_req=args.max_val_samples,
        log=log
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0: # Should not happen if dataset is not empty and batch_size > 0
        log.warning("DataLoader is empty or batch_size is too large for dataset; steps_per_epoch is 0.")
        # This implies an issue, but to prevent division by zero if max_steps > 0:
        num_epochs_total_approx = 0 if max_steps == 0 else 1 
    else:
        # Handle epoch-based training
        num_train_epochs = config.get('num_train_epochs', 0)
        
        if num_train_epochs > 0 and max_steps == 0:
            # Epoch-based training: calculate max_steps from num_train_epochs
            max_steps = steps_per_epoch * num_train_epochs
            num_epochs_total_approx = num_train_epochs
            log.info(f"Epoch-based training: {num_train_epochs} epochs × {steps_per_epoch} steps/epoch = {max_steps} total steps")
        elif max_steps > 0:
            # Step-based training: calculate approximate epochs from max_steps
            num_epochs_total_approx = (max_steps - 1) // steps_per_epoch + 1
        else:
            # Neither epochs nor steps specified
            raise ValueError("Either 'num_train_epochs' or 'max_train_steps' must be > 0 in config")


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

    # Initialize models using new config flags
    decoder_config = DecoderConfig(
        model_name=model_name,
        **decoder_train_cfg
    )
    dec_raw = Decoder(decoder_config)

    encoder_config = EncoderConfig(
        model_name=model_name,
        **encoder_train_cfg
    )
    enc_raw = Encoder(encoder_config)

    # ------------------------------------------------------------------
    # Tokenizer & vocab-size-based resizing (do this BEFORE optional torch.compile)
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    log.info(f"Tokenizer name: {tokenizer_name}")

    new_vocab_size = tokenizer.vocab_size

    from lens.utils.embedding_remap import remap_embeddings
    base_tok = AutoTokenizer.from_pretrained(model_name)

    if config["tokenizer_name"] != model_name:
        log.info(f"Remapping embeddings from {model_name} to {tokenizer_name}")
        remap_embeddings(dec_raw.base, base_tok, tokenizer)
        remap_embeddings(enc_raw.base, base_tok, tokenizer)
        log.info("Remapped Decoder & Encoder embedding matrices to new tokenizer")

    # Reinitialise prompt ids to ensure they are within new vocab
    dec_raw.set_prompt(config.get("decoder_prompt", "Explain: "), tokenizer)
    log.info("Prompt for decoder: %s",str([tokenizer.decode(d) for d in dec_raw.prompt_ids]))

    # Ensure Decoder's standalone LM head matches new vocab
    if dec_raw.out.weight.size(0) != new_vocab_size:
        d_model = dec_raw.base.config.hidden_size
        dec_raw.out = nn.Linear(d_model, new_vocab_size, bias=False)
        with torch.no_grad():
            dec_raw.out.weight.copy_(dec_raw.base.get_output_embeddings().weight)
        log.info("Resized Decoder.out to new vocab size")

    # Now compile models if requested
    if config.get('compile_models', True):
        log.info("Compiling models")
        dec = torch.compile(dec_raw).to(device)
        enc = torch.compile(enc_raw).to(device)
    else:
        log.info("Not compiling models")
        dec = dec_raw.to(device)
        enc = enc_raw.to(device)

    # Original model wrapper (remap after creation)
    orig = OrigWrapper(model_name, load_in_8bit=False)
    if config["tokenizer_name"] != model_name:
        remap_embeddings(orig.model, base_tok, tokenizer)
        log.info("Remapped Orig model embeddings to new tokenizer")
    orig.model.to(device)

    # Consolidate all parameters from dec and enc that require gradients.
    # This list is used for gradient clipping and for parameter counting.
    trainable_params = [p for p in dec.parameters() if p.requires_grad] + \
                       [p for p in enc.parameters() if p.requires_grad]
    
    # Calculate and log parameter counts
    total_trainable_params_val = sum(p.numel() for p in trainable_params)
    num_params_orig_total = sum(p.numel() for p in orig.model.parameters()) # Orig model is frozen

    log.info(f"Total trainable parameters (Decoder + Encoder combined): {total_trainable_params_val:,}")

    # Decoder parameter counts
    num_params_dec_total = sum(p.numel() for p in dec.parameters())
    num_params_dec_base_trainable = sum(p.numel() for n, p in dec.named_parameters() if p.requires_grad and 'base' in n)
    num_params_dec_proj_trainable = sum(p.numel() for n, p in dec.named_parameters() if p.requires_grad and 'proj' in n)
    num_params_dec_out_trainable = sum(p.numel() for n, p in dec.named_parameters() if p.requires_grad and 'out' in n)
    current_dec_trainable_total = num_params_dec_base_trainable + num_params_dec_proj_trainable + num_params_dec_out_trainable
    num_params_dec_frozen = num_params_dec_total - current_dec_trainable_total
    
    log.info(f"Decoder - Total parameters: {num_params_dec_total:,}")
    log.info(f"  Decoder - Trainable parameters: {current_dec_trainable_total:,}")
    log.info(f"    Decoder base trainable: {num_params_dec_base_trainable:,} (Config: {decoder_config.base_model})")
    log.info(f"    Decoder proj trainable: {num_params_dec_proj_trainable:,} (Config: {decoder_config.projection_layer})")
    log.info(f"    Decoder out trainable: {num_params_dec_out_trainable:,} (Config: {decoder_config.output_head})")
    log.info(f"  Decoder - Frozen parameters: {num_params_dec_frozen:,}")

    # Encoder parameter counts
    num_params_enc_total = sum(p.numel() for p in enc.parameters())
    num_params_enc_base_trainable = sum(p.numel() for n, p in enc.named_parameters() if p.requires_grad and 'base' in n)
    num_params_enc_proj_trainable = sum(p.numel() for n, p in enc.named_parameters() if p.requires_grad and 'proj' in n)
    current_enc_trainable_total = num_params_enc_base_trainable + num_params_enc_proj_trainable
    num_params_enc_frozen = num_params_enc_total - current_enc_trainable_total

    log.info(f"Encoder - Total parameters: {num_params_enc_total:,}")
    log.info(f"  Encoder - Trainable parameters: {current_enc_trainable_total:,}")
    log.info(f"    Encoder base trainable: {num_params_enc_base_trainable:,} (Config: {encoder_config.base_model}, present: {encoder_config.use_base_model})")
    log.info(f"    Encoder proj trainable: {num_params_enc_proj_trainable:,} (Config: {encoder_config.projection_layer})")
    log.info(f"  Encoder - Frozen parameters: {num_params_enc_frozen:,}")

    # Sanity check: sum of categorized trainable parameters should match total_trainable_params_val
    sum_of_categorized_trainable = current_dec_trainable_total + current_enc_trainable_total
    if total_trainable_params_val != sum_of_categorized_trainable:
        log.warning(
            f"Parameter count mismatch: total_trainable_params_val is {total_trainable_params_val:,}, "
            f"but sum of categorized trainable parameters (Decoder + Encoder) is {sum_of_categorized_trainable:,}. "
            "This might indicate that some trainable parameters are not covered by the 'base', 'proj', 'out' categorization."
        )

    log.info(f"Original LLM (frozen) parameters: {num_params_orig_total:,}")
    log.info(f"Hyperparameters: lm_weight={lm_weight}, kl_base_weight={kl_base_weight}, entropy_weight={entropy_weight}")
    log.info(f"Learning rate: {learning_rate}, Projection LR Multiplier: {projection_lr_multiplier}")
    log.info(f"Stop-grad on A′: {config['stop_grad_aprime']}")
    log.info(f"Grad clip: {config['grad_clip']}")
    
    # Create optimizer groups with potentially different LRs
    optimizer_groups = param_groups([dec, enc], learning_rate, projection_lr_multiplier)

    # Verify that the number of parameters in optimizer groups matches the count from trainable_params list.
    num_params_in_optimizer_groups = sum(p.numel() for group in optimizer_groups for p in group['params'])
    if total_trainable_params_val != num_params_in_optimizer_groups:
        log.warning(
            f"Parameter count mismatch: sum of p.numel() for trainable_params is {total_trainable_params_val:,}, "
            f"but optimizer groups sum to {num_params_in_optimizer_groups:,}. "
            "Check requires_grad flags and param grouping logic."
        )

    opt = torch.optim.AdamW(optimizer_groups)
    # GradScaler enabled only when using CUDA
    scaler = GradScaler(enabled=(device.type == "cuda"))

    start_step = 0
    if args.resume:
        # Checkpoint stores the last completed step
        rec = checkpoint_manager.load_checkpoint(args.resume, models={"dec": dec, "enc": enc}, optimizer=opt, map_location=device)
        start_step = int(rec.get("step", -1)) + 1 # Resume from the next step
        log.info(f"Resuming training from step {start_step}")

    epoch_decoded_tokens = []  # Initialize accumulator for decoded tokens per epoch
    step_iter = iter(train_loader)
    log.info(f"Start step: {start_step}, Max steps: {max_steps}")
    
    # Track validation loss for best checkpoint
    best_val_loss = float('inf')
    
    for step in range(start_step, max_steps):
        current_epoch_num = (step // steps_per_epoch) + 1 if steps_per_epoch > 0 else 1
        epoch_just_finished = False
        
        try:
            batch = next(step_iter)
        except StopIteration:
            # Epoch finished
            epoch_just_finished = True
            
            # Log token stats
            if epoch_decoded_tokens:
                token_counts = Counter(epoch_decoded_tokens)
                if token_counts:
                    most_common_token_id, most_common_count = token_counts.most_common(1)[0]
                    total_tokens_in_epoch = len(epoch_decoded_tokens)
                    frequency = most_common_count / total_tokens_in_epoch
                    if step % config['log_interval'] == 0:
                        log.info(
                            f"Epoch {current_epoch_num} most common token: ID {most_common_token_id} = `{tokenizer.decode([most_common_token_id])}` "
                            f"(Count: {most_common_count}/{total_tokens_in_epoch}, Freq: {frequency:.4f})"
                        )
                    log_metrics({
                        "epoch_stats/most_common_token_id": most_common_token_id,
                        "epoch_stats/most_common_token_count": most_common_count,
                        "epoch_stats/most_common_token_freq": frequency,
                        "epoch_stats/total_tokens_in_epoch": total_tokens_in_epoch,
                    }, step=step) # Log with the current step, which marks the end of the epoch
            
            epoch_decoded_tokens = [] # Reset for the next epoch
            step_iter = iter(train_loader)
            batch = next(step_iter)

        # Move batch tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}

        opt.zero_grad(set_to_none=True)

        if device.type == "cuda":
            preferred_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            cast_ctx = autocast(device_type="cuda", dtype=preferred_dtype)
        else:
            cast_ctx = nullcontext()
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
                    "lm_weight": lm_weight,
                    "kl_base_weight": kl_base_weight,
                    "entropy_weight": entropy_weight,
                },
            )
            loss = losses["total"]
            
            # Accumulate decoded tokens from the current step
            if "decoded_tokens_batch" in losses:
                epoch_decoded_tokens.extend(losses["decoded_tokens_batch"].tolist())

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config['grad_clip'])
            param_before = [p.detach().clone() for p in trainable_params]
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config['grad_clip'])
            param_before = [p.detach().clone() for p in trainable_params]
            opt.step()

        with torch.no_grad():
            upd_sq = 0.0
            param_sq = 0.0
            for p, prev in zip(trainable_params, param_before):
                diff = p.data - prev
                upd_sq += diff.pow(2).sum().item()
                param_sq += p.data.pow(2).sum().item()
            update_norm = math.sqrt(upd_sq)
            param_norm = math.sqrt(param_sq)
            update_ratio = update_norm / (param_norm + 1e-12)
        del param_before

        lr_current = opt.param_groups[0]["lr"]

        # Log metrics at specified interval or at the last step
        if step % log_interval == 0 or step == max_steps - 1:
            log_msg_parts = [
                f"Epoch {current_epoch_num}/{num_epochs_total_approx}" if steps_per_epoch > 0 else f"Step {step}",
                f"Step {step}/{max_steps-1}", # max_steps-1 is the last step index
                f"loss {loss.item():.4f}",
                f"mse {losses['mse'].item():.4f}",
                f"lm {losses['lm'].item():.4f}",
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
                "loss/lm": losses["lm"].item(),
                "loss/kl": losses["kl"].item(),
                "loss/entropy": losses["entropy"].item(),
                "params/tau": current_tau,
                "params/alpha": current_alpha,
                "params/lm_w": lm_weight,
                "params/kl_w": kl_base_weight,
                "params/entropy_w": entropy_weight,
                "optim/lr": lr_current,
                "grads/norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "updates/norm": update_norm,
                "updates/ratio": update_ratio,
            }
            if steps_per_epoch > 0:
                metrics_to_log["epoch"] = current_epoch_num
            
            log_metrics(metrics_to_log, step=step)

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        # Check if we should save based on step interval
        if checkpoint_manager.should_save_step(step):
            current_metrics = {
                "loss/total": loss.item(),
                "loss/mse": losses["mse"].item(),
                "loss/lm": losses["lm"].item(),
                "loss/kl": losses["kl"].item(),
                "loss/entropy": losses["entropy"].item(),
                "params/tau": current_tau,
                "params/alpha": current_alpha,
                "optim/lr": lr_current,
                "grads/norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            }
            
            checkpoint_manager.save_checkpoint(
                step=step,
                epoch=current_epoch_num,
                models={"dec": dec_raw if config.get('compile_models', True) else dec, 
                        "enc": enc_raw if config.get('compile_models', True) else enc},
                optimizer=opt,
                metrics=current_metrics,
                config=config,
                tau=current_tau,
                alpha=current_alpha,
            )
        
        # Check if we should save based on epoch completion
        if epoch_just_finished and checkpoint_manager.should_save_epoch(current_epoch_num, epoch_just_finished):
            current_metrics = {
                "loss/total": loss.item(),
                "loss/mse": losses["mse"].item(),
                "loss/lm": losses["lm"].item(),
                "loss/kl": losses["kl"].item(),
                "loss/entropy": losses["entropy"].item(),
                "params/tau": current_tau,
                "params/alpha": current_alpha,
                "optim/lr": lr_current,
                "grads/norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            }
            
            checkpoint_manager.save_checkpoint(
                step=step,
                epoch=current_epoch_num,
                models={"dec": dec_raw if config.get('compile_models', True) else dec, 
                        "enc": enc_raw if config.get('compile_models', True) else enc},
                optimizer=opt,
                metrics=current_metrics,
                config=config,
                tau=current_tau,
                alpha=current_alpha,
            )

        # run validation at interval
        if val_loader and val_interval > 0 and step > 0 and step % val_interval == 0:
            dec.eval()
            enc.eval()
            val_loss = val_mse = val_lm = val_kl = 0.0
            val_seen = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = {k: v.to(device) for k, v in vbatch.items()}
                    sch_args = {
                        "tau": get_schedule_value(config['gumbel_tau_schedule'], step, max_steps),
                        "T_text": t_text,
                        "alpha": get_schedule_value(config['alpha_schedule'], step, max_steps),
                        "lm_weight": lm_weight,
                        "kl_base_weight": kl_base_weight,
                        "entropy_weight": entropy_weight,
                    }
                    v_losses = train_step(vbatch, {"dec": dec, "enc": enc, "orig": orig}, sch_args)
                    bsz = vbatch["A"].size(0)
                    val_loss += v_losses["total"].item() * bsz
                    val_mse  += v_losses["mse"].item()   * bsz
                    val_lm   += v_losses["lm"].item()    * bsz
                    val_kl   += v_losses["kl"].item()    * bsz
                    val_seen += bsz
            avg_val_loss = val_loss / val_seen if val_seen else float("nan")
            avg_val_mse  = val_mse  / val_seen if val_seen else float("nan")
            avg_val_lm   = val_lm   / val_seen if val_seen else float("nan")
            avg_val_kl   = val_kl   / val_seen if val_seen else float("nan")
            log.info(
                f"Validation – loss {avg_val_loss:.4f}, mse {avg_val_mse:.4f}, lm {avg_val_lm:.4f}, kl {avg_val_kl:.4f}"
            )
            log_metrics({
                "eval/loss/total": avg_val_loss,
                "eval/loss/mse":    avg_val_mse,
                "eval/loss/lm":     avg_val_lm,
                "eval/loss/kl":     avg_val_kl,
            }, step=step)
            
            # Save checkpoint if validation loss tracking is enabled
            if checkpoint_manager.track_best_n > 0 and not math.isnan(avg_val_loss):
                val_metrics = {
                    "loss/total": loss.item(),
                    "loss/mse": losses["mse"].item(),
                    "loss/lm": losses["lm"].item(),
                    "loss/kl": losses["kl"].item(),
                    "loss/entropy": losses["entropy"].item(),
                    "eval/loss/total": avg_val_loss,
                    "eval/loss/mse": avg_val_mse,
                    "eval/loss/lm": avg_val_lm,
                    "eval/loss/kl": avg_val_kl,
                    "params/tau": current_tau,
                    "params/alpha": current_alpha,
                    "optim/lr": lr_current,
                    "grads/norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                }
                
                checkpoint_manager.save_checkpoint(
                    step=step,
                    epoch=current_epoch_num,
                    models={"dec": dec_raw if config.get('compile_models', True) else dec, 
                            "enc": enc_raw if config.get('compile_models', True) else enc},
                    optimizer=opt,
                    metrics=val_metrics,
                    config=config,
                    val_loss=avg_val_loss,
                    tau=current_tau,
                    alpha=current_alpha,
                )
                
            dec.train()
            enc.train()

        # Verbose sample printing
        verbose_config = config.get('verbose_samples', {})
        if verbose_config.get('enabled', False):
            should_print = False
            
            # Check if we should print based on interval type
            if verbose_config.get('interval_type', 'steps') == 'steps':
                # Print every N steps
                verbose_interval = verbose_config.get('interval', 1000)
                if step > 0 and step % verbose_interval == 0:
                    should_print = True
            else:  # interval_type == 'epochs'
                # Print at epoch boundaries
                if epoch_just_finished:
                    verbose_interval = verbose_config.get('interval', 1)
                    if current_epoch_num % verbose_interval == 0:
                        should_print = True
            
            # Also check for combined epoch+steps printing
            if verbose_config.get('print_every_epoch', False) and epoch_just_finished:
                should_print = True
            if verbose_config.get('print_every_n_steps', 0) > 0:
                if step > 0 and step % verbose_config['print_every_n_steps'] == 0:
                    should_print = True
            
            if should_print:
                log.info(f"Generating verbose samples at step {step}, epoch {current_epoch_num}")
                dec.eval()
                enc.eval()
                
                # Get schedule arguments
                sch_args = {
                    "tau": get_schedule_value(config['gumbel_tau_schedule'], step, max_steps),
                    "T_text": t_text,
                    "alpha": get_schedule_value(config['alpha_schedule'], step, max_steps),
                    "lm_weight": lm_weight,
                    "kl_base_weight": kl_base_weight,
                    "entropy_weight": entropy_weight,
                }
                
                # Process verbose samples
                num_printed = process_and_print_verbose_batch_samples(
                    batch=batch,  # Use current batch
                    cfg=config,
                    models={"dec": dec, "enc": enc},
                    orig=orig,
                    tok=tokenizer,
                    sch_args=sch_args,
                    device=device,
                    num_samples=verbose_config.get('num_samples', 2),
                    top_n_analysis=verbose_config.get('top_n_predictions', 3),
                    printed_count_so_far=0,
                    generate_continuation=verbose_config.get('generate_continuation', True),
                    continuation_tokens=verbose_config.get('continuation_tokens', 30)
                )
                
                dec.train()
                enc.train()

    # Save final checkpoint if enabled
    if checkpoint_manager.save_at_end:
        log.info("Saving final checkpoint...")
        final_metrics = {
            "loss/total": loss.item() if 'loss' in locals() else None,
            "step": max_steps - 1,
            "epoch": num_epochs_total_approx,
        }
        
        checkpoint_manager.save_checkpoint(
            step=max_steps - 1,
            epoch=num_epochs_total_approx,
            models={"dec": dec_raw if config.get('compile_models', True) else dec, 
                    "enc": enc_raw if config.get('compile_models', True) else enc},
            optimizer=opt,
            metrics=final_metrics,
            config=config,
        )

    log.info("Training finished after %d steps.", max_steps)


if __name__ == "__main__":
    main()
