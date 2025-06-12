#!/usr/bin/env python3
"""Check if requires_grad state is correctly set after checkpoint loading."""

import torch
import sys
from pathlib import Path

def analyze_requires_grad(checkpoint_path):
    """Analyze requires_grad state in checkpoint and what it should be."""
    
    print(f"\n=== Analyzing checkpoint: {checkpoint_path} ===\n")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get config and current step
    config = ckpt.get('config', {})
    current_step = ckpt.get('step', 0)
    
    print(f"Current step: {current_step}")
    
    # Check freeze schedule
    freeze_cfg = config.get('freeze_schedule', {})
    if freeze_cfg.get('enabled'):
        unfreeze_at = freeze_cfg.get('unfreeze_at_parsed', {}).get('value', 0)
        warmup_duration = freeze_cfg.get('warmup_duration', '0')
        print(f"\nFreeze schedule:")
        print(f"  Enabled: True")
        print(f"  Unfreeze at step: {unfreeze_at}")
        print(f"  Warmup duration: {warmup_duration}")
        print(f"  Currently should be: {'UNFROZEN' if current_step >= unfreeze_at else 'FROZEN'}")
    
    # Check what's trainable according to config
    dec_cfg = config.get('trainable_components', {}).get('decoder', {})
    enc_cfg = config.get('trainable_components', {}).get('encoder', {})
    
    print(f"\nBase config (before progressive unfreeze):")
    print(f"  Decoder base_model trainable: {dec_cfg.get('base_model', False)}")
    print(f"  Encoder base_model trainable: {enc_cfg.get('base_model', False)}")
    
    # Analyze actual parameter states in checkpoint
    print(f"\n=== Parameter Analysis ===")
    
    # Note: The checkpoint doesn't save requires_grad state!
    # This is likely the problem - when loading, all parameters get their
    # requires_grad from the model definition, not from the training state
    
    print("\nWARNING: PyTorch checkpoints do not save requires_grad state!")
    print("This means after loading a checkpoint, requires_grad must be manually restored")
    print("based on the training step and freeze schedule.")
    
    # Count parameters by category
    for model_name in ['decoder', 'encoder']:
        if model_name not in ckpt.get('models', {}):
            continue
            
        print(f"\n{model_name.upper()}:")
        state_dict = ckpt['models'][model_name]
        
        base_params = sum(1 for k in state_dict.keys() if k.startswith('base.'))
        proj_params = sum(1 for k in state_dict.keys() if 'proj' in k and not k.startswith('base.'))
        prompt_params = sum(1 for k in state_dict.keys() if 'prompt' in k)
        other_params = len(state_dict) - base_params - proj_params - prompt_params
        
        print(f"  Base model parameters: {base_params}")
        print(f"  Projection parameters: {proj_params}")
        print(f"  Prompt parameters: {prompt_params}")
        print(f"  Other parameters: {other_params}")
        
        # Check if base model weights have changed from initialization
        if model_name == 'encoder' and enc_cfg.get('base_model', False):
            # Sample a few weights to check if they've been updated
            sample_keys = [k for k in state_dict.keys() if k.startswith('base.') and 'weight' in k][:3]
            print(f"\n  Sample base weights (should show training if unfrozen):")
            for k in sample_keys:
                if hasattr(state_dict[k], 'norm'):
                    print(f"    {k}: norm={state_dict[k].norm().item():.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_requires_grad_state.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    analyze_requires_grad(checkpoint_path)