#!/usr/bin/env python3
"""Diagnose checkpoint resume issues - check what's missing when resuming."""

import torch
import sys
from pathlib import Path

def analyze_checkpoint(checkpoint_path):
    """Analyze what's in a checkpoint and what might be missing."""
    
    print(f"\n=== Analyzing checkpoint: {checkpoint_path} ===\n")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check top-level keys
    print("Top-level keys in checkpoint:")
    for key in sorted(ckpt.keys()):
        if key in ['models', 'optim', 'scheduler', 'rng_states']:
            print(f"  {key}: <complex object>")
        else:
            print(f"  {key}: {ckpt[key]}")
    
    # Check for potentially missing items
    print("\n=== Checking for potentially missing items ===")
    
    missing_items = []
    
    # Check scaler state
    if 'scaler' not in ckpt:
        missing_items.append("GradScaler state (scaler) - This can cause loss jumps!")
    else:
        print("✓ GradScaler state found")
    
    # Check RNG states
    if 'rng_states' not in ckpt:
        missing_items.append("RNG states - This can cause different random sampling")
    else:
        rng = ckpt['rng_states']
        print(f"✓ RNG states found: {list(rng.keys())}")
    
    # Check training state
    essential_keys = ['step', 'epoch', 'tau', 'alpha']
    for key in essential_keys:
        if key not in ckpt:
            missing_items.append(f"{key} - Training state variable")
        else:
            print(f"✓ {key}: {ckpt[key]}")
    
    # Check optimizer state
    if 'optim' in ckpt:
        optim = ckpt['optim']
        if 'state' in optim and optim['state']:
            first_state = list(optim['state'].values())[0]
            print(f"✓ Optimizer state: {len(optim['state'])} parameter states")
            print(f"  First state keys: {list(first_state.keys())}")
            if 'step' in first_state:
                print(f"  First param step count: {first_state['step']}")
        else:
            missing_items.append("Optimizer state - Can affect momentum/adaptive learning rates")
    
    # Check scheduler state  
    if 'scheduler' in ckpt:
        scheduler = ckpt['scheduler']
        print(f"✓ Scheduler state found")
        if isinstance(scheduler, dict):
            for k in ['last_epoch', '_step_count', 'base_lrs']:
                if k in scheduler:
                    print(f"  {k}: {scheduler[k]}")
    else:
        missing_items.append("LR Scheduler state - Can cause wrong learning rate")
    
    # Check model states
    if 'models' in ckpt:
        models = ckpt['models']
        print(f"\n✓ Models found: {list(models.keys())}")
        
        # Check for prompt embeddings in decoder
        if 'decoder' in models:
            dec_state = models['decoder']
            prompt_keys = [k for k in dec_state.keys() if 'prompt' in k]
            if prompt_keys:
                print(f"  Decoder prompt keys: {prompt_keys}")
                for k in prompt_keys:
                    if hasattr(dec_state[k], 'norm') and dec_state[k].dtype.is_floating_point:
                        print(f"    {k} norm: {dec_state[k].norm().item():.4f}")
                    elif k == 'prompt_ids':
                        print(f"    {k} shape: {dec_state[k].shape}")
        
        # Check for soft prompts in encoder
        if 'encoder' in models:
            enc_state = models['encoder']
            soft_prompt_keys = [k for k in enc_state.keys() if 'soft_prompt' in k or 'prompt' in k]
            if soft_prompt_keys:
                print(f"  Encoder prompt keys: {soft_prompt_keys}")
                for k in soft_prompt_keys:
                    if hasattr(enc_state[k], 'norm') and enc_state[k].dtype.is_floating_point:
                        print(f"    {k} norm: {enc_state[k].norm().item():.4f}")
    
    # Check for batch/epoch tracking
    tracking_keys = ['current_epoch', 'batch_within_epoch', 'steps_per_epoch', 'accumulation_step']
    print("\n✓ Training progress tracking:")
    for key in tracking_keys:
        if key in ckpt:
            print(f"  {key}: {ckpt[key]}")
        else:
            missing_items.append(f"{key} - May affect epoch boundaries")
    
    # Report missing items
    if missing_items:
        print("\n⚠️  MISSING ITEMS THAT COULD CAUSE ISSUES:")
        for item in missing_items:
            print(f"  - {item}")
    else:
        print("\n✅ All expected items found in checkpoint!")
    
    # Additional diagnostics
    print("\n=== Additional Info ===")
    if 'config' in ckpt and isinstance(ckpt['config'], dict):
        config = ckpt['config']
        print("Config keys (sample):", list(config.keys())[:10])
        if 'mixed_precision' in config:
            print(f"Mixed precision config: {config['mixed_precision']}")
    
    if 'wandb_run_id' in ckpt:
        print(f"WandB run ID: {ckpt['wandb_run_id']}")
    
    return ckpt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_resume_issue.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    analyze_checkpoint(checkpoint_path)