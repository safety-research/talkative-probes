#!/usr/bin/env python3
"""
Standalone evaluation script for Talkative Autoencoders.
Supports standard validation and best-of-N sampling evaluation modes.
"""

import os
import json
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

# Core lens imports
from lens.analysis.analyzer_class import LensAnalyzer
from lens.data.collate import collate
from lens.training.train_aux import _prepare_dataloaders
from lens.models.decoder import Decoder, DecoderConfig
from lens.models.encoder import Encoder, EncoderConfig
from lens.models.orig import OrigWrapper
from lens.training.distributed import (
    init_distributed,
    is_main,
    setup_for_distributed,
    set_seed
)
from lens.training.fast_distributed_sampler import FastDistributedSampler
from lens.utils.checkpoint_manager import CheckpointManager
from lens.utils.logging import init as log_init

# Import validate_distributed by loading the training script as a module
import importlib.util
import sys


def import_training_functions():
    """Import functions from the training script."""
    train_script_path = Path(__file__).parent / "01_train_distributed.py"
    spec = importlib.util.spec_from_file_location("train_distributed", train_script_path)
    train_module = importlib.util.module_from_spec(spec)
    sys.modules["train_distributed"] = train_module
    spec.loader.exec_module(train_module)
    return train_module


def setup_distributed_eval(cfg: DictConfig):
    """Initialize distributed training environment."""
    # Use the same init_distributed from lens.training.distributed
    rank, world_size, local_rank = init_distributed()
    
    # Setup for distributed (handles printing etc)
    setup_for_distributed(rank == 0)
    
    # Set device
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    # Set seed
    if cfg.get('seed') is not None:
        set_seed(cfg.seed + rank)
    
    return rank, world_size, device, local_rank


def load_models_and_tokenizer_from_checkpoint(cfg: DictConfig, checkpoint_path: Path, device: torch.device):
    """Load encoder, decoder, orig model and tokenizer from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config from checkpoint
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
    else:
        raise ValueError("No config found in checkpoint")
    
    # Initialize tokenizer
    tokenizer_name = ckpt_config.get('tokenizer_name', ckpt_config.get('model_name', 'gpt2'))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create encoder with tokenizer
    encoder_config = EncoderConfig(**ckpt_config['trainable_components']['encoder'])
    encoder = Encoder(
        encoder_config,
        subject_model_name=ckpt_config['model_name'],
        tokenizer=tokenizer
    )
    
    # Create decoder with tokenizer
    decoder_config = DecoderConfig(**ckpt_config['trainable_components']['decoder'])
    decoder = Decoder(
        decoder_config,
        model_name=ckpt_config['model_name'],
        tokenizer=tokenizer
    )
    
    # Create orig model
    orig_model = OrigWrapper(
        ckpt_config.get('orig_model_name', ckpt_config['model_name']),
        load_in_8bit=False,
        base_to_use=None
    )
    
    # Load state dicts with strict=True for safety
    if 'models' in checkpoint:
        if 'encoder' in checkpoint['models']:
            encoder.load_state_dict(checkpoint['models']['encoder'], strict=True)
        if 'decoder' in checkpoint['models']:
            decoder.load_state_dict(checkpoint['models']['decoder'], strict=True)
    
    # Move to device
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    orig_model = orig_model.to(device).eval()
    
    # Get shared base model
    shared_base_model = encoder.base if hasattr(encoder, 'base') else None
    
    return encoder, decoder, orig_model, shared_base_model, tokenizer, ckpt_config


def prepare_val_dataloader(cfg: DictConfig, rank: int, world_size: int, log=None):
    """Prepare validation dataloader."""
    # Extract dataset paths from config
    activation_dir = cfg.dataset.activation_dir
    val_activation_dir = cfg.dataset.get('val_activation_dir', activation_dir)
    
    # Use _prepare_dataloaders to get datasets
    _, val_dataset = _prepare_dataloaders(
        config=cfg,
        activation_dir=activation_dir,
        effective_val_activation_dir=val_activation_dir,
        max_train_samples_req=None,
        max_val_samples_req=cfg.get('max_val_samples'),
        log=log,
        orig_model_for_gen=None,
        tokenizer_for_gen=None
    )
    
    if val_dataset is None or len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty")
    
    val_sampler = FastDistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=cfg.get('seed', 42),
        drop_last=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.eval.batch_size,
        sampler=val_sampler,
        num_workers=cfg.data.get('num_workers', 0),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate
    )
    
    return val_loader


def evaluate_with_best_of_n(
    val_loader,
    checkpoint_path: str,
    device: torch.device,
    eval_config: DictConfig,
    is_main_process: bool
) -> Dict[str, Any]:
    """Run evaluation using LensAnalyzer with best-of-N sampling.
    
    NOTE: This mode only provides MSE metrics. For comprehensive metrics including
    KL divergence, language modeling loss, variance explained, and intervention
    analysis, use the standard validation mode (use_best_of_n=false).
    """
    
    # Initialize analyzer
    analyzer = LensAnalyzer(
        checkpoint_path=checkpoint_path,
        device=device,
        use_bf16=eval_config.get('use_bf16', True)
    )
    
    total_mse = 0.0
    total_samples = 0
    all_results = []
    
    # Best-of-N config
    optimize_config = {
        "best_of_k": eval_config.best_of_n,
        "use_batched": True,
        "temperature": eval_config.temperature,
        "n_groups_per_rollout": eval_config.get('n_groups_per_rollout', 8)
    }
    
    # Process validation data
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Best-of-N Eval", disable=not is_main_process)):
            if eval_config.get('max_batches') and batch_idx >= eval_config.max_batches:
                break
                
            # Process each example in batch
            for i in range(batch['input_ids'].size(0)):
                text = analyzer.tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
                
                # Run analysis
                results_df = analyzer.analyze_all_tokens(
                    text,
                    optimize_explanations_config=optimize_config,
                    no_kl=True,
                    calculate_token_salience=eval_config.get('calculate_salience', False),
                    batch_size=eval_config.get('analyzer_batch_size', 32)
                )
                
                if results_df is not None and len(results_df) > 0:
                    example_mse = results_df['mse'].mean()
                    total_mse += example_mse
                    total_samples += 1
                    
                    if eval_config.get('save_detailed_results') and is_main_process:
                        all_results.append({
                            'text': text,
                            'avg_mse': example_mse,
                            'num_tokens': len(results_df)
                        })
    
    # Gather results across processes
    if dist.is_initialized():
        total_mse_tensor = torch.tensor(total_mse, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        dist.all_reduce(total_mse_tensor)
        dist.all_reduce(total_samples_tensor)
        total_mse = total_mse_tensor.item()
        total_samples = total_samples_tensor.item()
    
    avg_mse = total_mse / max(total_samples, 1)
    
    return {
        'avg_mse': avg_mse,
        'total_samples': total_samples,
        'detailed_results': all_results if eval_config.get('save_detailed_results') else None
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point."""
    
    # Import training functions
    train_module = import_training_functions()
    validate_distributed = train_module.validate_distributed
    
    # Setup distributed
    rank, world_size, device, local_rank = setup_distributed_eval(cfg)
    is_main_process = is_main()
    
    # Initialize logging
    log = None
    if is_main_process:
        log = log_init(cfg)
        log.info("Starting evaluation...")
        log.info(f"Checkpoint: {cfg.eval.checkpoint_path}")
        log.info(f"World size: {world_size}")
    
    # Load checkpoint path
    checkpoint_path = Path(cfg.eval.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Determine evaluation mode
    use_best_of_n = cfg.eval.get('use_best_of_n', False)
    
    if use_best_of_n:
        # For best-of-N, we only need the validation dataloader
        # Load minimal config from checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in ckpt:
            # Merge data config from checkpoint
            cfg.data = ckpt['config'].get('data', cfg.data)
            cfg.model_name = ckpt['config'].get('model_name', cfg.get('model_name', 'gpt2'))
            cfg.tokenizer_name = ckpt['config'].get('tokenizer_name', cfg.model_name)
        
        val_loader = prepare_val_dataloader(cfg, rank, world_size, log)
        
        results = evaluate_with_best_of_n(
            val_loader,
            str(checkpoint_path),
            device,
            cfg.eval,
            is_main_process
        )
        
        results['mode'] = 'best_of_n'
        
    else:
        # For standard validation, load models and use validate_distributed
        encoder, decoder, orig_model, shared_base_model, tokenizer, ckpt_config = load_models_and_tokenizer_from_checkpoint(
            cfg, checkpoint_path, device
        )
        
        # Update config with checkpoint config
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict.update(ckpt_config)
        cfg = OmegaConf.create(cfg_dict)
        
        # Prepare validation dataloader
        val_loader = prepare_val_dataloader(cfg, rank, world_size, log)
        
        # Wrap models with DDP if distributed
        if world_size > 1:
            encoder = DDP(encoder, device_ids=[local_rank])
            decoder = DDP(decoder, device_ids=[local_rank])
        
        # Get cached prefix ids if using special tokens
        cached_prefix_ids = None
        if hasattr(decoder.module if hasattr(decoder, 'module') else decoder, 'get_cached_prefix_ids'):
            cached_prefix_ids = (decoder.module if hasattr(decoder, 'module') else decoder).get_cached_prefix_ids()
        
        # Run validation
        avg_mse = validate_distributed(
            decoder=decoder,
            encoder=encoder,
            orig_model=orig_model,
            val_loader=val_loader,
            config=cfg,
            step=0,
            device=device,
            tokenizer=tokenizer,
            cached_prefix_ids=cached_prefix_ids,
            world_size=world_size,
            log=log,
            is_main_process=is_main_process,
            wandb_run_id=None,
            steps_per_epoch=None,
            current_epoch=None,
            max_steps=cfg.get('max_train_steps', 1),  # Provide a default
            gradient_accumulation_steps=1,
            val_interval=None,
            comparison_tuned_lens=None,
            should_print_val=True,
            shared_base_model=shared_base_model,
            should_run_interventions=cfg.eval.get('run_interventions', False)
        )
        
        # Note: validate_distributed only returns avg_mse, not full metrics
        # The full metrics are logged internally by the function
        results = {
            'avg_mse': avg_mse,
            'mode': 'standard_validation',
            'note': 'Full metrics are logged to console/wandb by validate_distributed'
        }
    
    # Save results
    if is_main_process:
        output_dir = Path(cfg.eval.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / cfg.eval.output_file
        
        # Add metadata
        results['checkpoint_path'] = str(checkpoint_path)
        results['eval_config'] = OmegaConf.to_container(cfg.eval, resolve=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        if use_best_of_n:
            print(f"Average MSE: {results['avg_mse']:.6f}")
            print(f"Total samples: {results['total_samples']}")
        else:
            print("\nValidation completed.")
            print(f"Average MSE: {results['avg_mse']:.6f}")
            if 'note' in results:
                print(f"Note: {results['note']}")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()