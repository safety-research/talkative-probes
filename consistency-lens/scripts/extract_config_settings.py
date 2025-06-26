#!/usr/bin/env python3
"""
Extract settings from a YAML config file for the submit script.
This provides a clean interface to get all necessary settings without
mixing Python and bash logic.
"""

import argparse
import sys
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir


def extract_settings(config_path, overrides):
    """Extract all relevant settings from a config file using Hydra."""
    
    # Get config directory and name
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    config_name = config_path.stem
    
    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        # Compose the config, applying command-line overrides
        cfg = compose(config_name=config_name, overrides=overrides)
    
    # Extract key settings
    settings = {
        # Model and layer info
        'model_name': cfg.get('model_name', ''),
        'layer': cfg.get('layer_l', None),
        
        # Activation dumper settings
        'hf_dataset_name': cfg.get('activation_dumper', {}).get('hf_dataset_name', ''),
        'hf_split': cfg.get('activation_dumper', {}).get('hf_split', 'train'),
        'output_dir': cfg.get('activation_dumper', {}).get('output_dir', './data'),
        'val_output_dir': cfg.get('activation_dumper', {}).get('val_output_dir', ''),
        'use_pretokenized': cfg.get('activation_dumper', {}).get('use_pretokenized', False),
        'seq_len': cfg.get('activation_dumper', {}).get('seq_len', 64),
        
        # Pretokenize settings
        'pretokenize_num_proc': cfg.get('pretokenize', {}).get('num_proc', 8),
        
        # Training settings
        'num_train_epochs': cfg.get('num_train_epochs', 0),
        'max_train_steps': cfg.get('max_train_steps', 0),
        
        # Freeze schedule
        'freeze_enabled': cfg.get('freeze_schedule', {}).get('enabled', False),
        'unfreeze_at': cfg.get('freeze_schedule', {}).get('unfreeze_at', ''),
        
        # Determine training script based on config name and settings
        'training_script': determine_training_script(config_path, cfg),

        # On-the-fly dataset generation. Enabled if the section exists and is not null.
        'on_the_fly_enabled': 'on_the_fly' in cfg.get('dataset', {}) and cfg.dataset.on_the_fly is not None,
    }
    
    # Clean up model name for directory paths
    settings['model_name_clean'] = settings['model_name'].replace('/', '_')
    
    # Determine dataset name from HF dataset name or output dir
    if settings['hf_dataset_name']:
        # Extract dataset name from HF path (e.g., "SimpleStories/SimpleStories" -> "SimpleStories")
        settings['dataset_name'] = os.path.basename(settings['hf_dataset_name'])
    else:
        # Fall back to output dir name
        settings['dataset_name'] = os.path.basename(settings['output_dir'])
    
    return settings


def determine_training_script(config_path, cfg):
    """Determine the appropriate training script based on config."""
    
    config_name = Path(config_path).stem.lower()
    
    # Map config names to training scripts
    script_map = {
        'simplestories_frozen': 'scripts/slurm_simplestories_frozen.sh',
        'simplestories_frozen2': 'scripts/slurm_simplestories_frozen2.sh',
        'simplestories_unfreeze': 'scripts/slurm_simplestories_unfreeze.sh',
        'gpt2_frozen': 'scripts/slurm_gpt2_frozen.sh',
        'gpt2_unfreeze': 'scripts/slurm_gpt2_unfreeze.sh',
        'gpt2_pile_frozen': 'scripts/slurm_gpt2_pile.sh',
        'gpt2_pile_unfreeze': 'scripts/slurm_gpt2_pile_unfreeze.sh',
    }
    
    # Check if there's a direct mapping
    if config_name in script_map:
        return script_map[config_name]
    
    # Otherwise, try to infer from config contents
    model_name = cfg.get('model_name', '').lower()
    freeze_enabled = cfg.get('freeze_schedule', {}).get('enabled', False)
    
    if 'simplestories' in model_name:
        if freeze_enabled:
            return 'scripts/slurm_simplestories_unfreeze.sh'
        else:
            return 'scripts/slurm_simplestories_frozen.sh'
    elif 'gpt2' in model_name:
        dataset_name = cfg.get('activation_dumper', {}).get('hf_dataset_name', '').lower()
        if 'pile' in dataset_name:
            if freeze_enabled:
                return 'scripts/slurm_gpt2_pile_unfreeze.sh'
            else:
                return 'scripts/slurm_gpt2_pile.sh'
        else:
            if freeze_enabled:
                return 'scripts/slurm_gpt2_unfreeze.sh'
            else:
                return 'scripts/slurm_gpt2_frozen.sh'
    
    # Default fallback - will need to be specified manually
    return None


def print_setting(key):
    """Print a single setting value."""
    if key in settings:
        value = settings[key]
        # Convert booleans to bash-friendly strings
        if isinstance(value, bool):
            print('true' if value else 'false')
        else:
            print(value)
    else:
        print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract settings from YAML config')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--key', help='Specific key to extract')
    parser.add_argument('--all', action='store_true', help='Print all settings as KEY=VALUE')
    
    # Parse known arguments and collect the rest as overrides for Hydra
    args, overrides = parser.parse_known_args()
    
    try:
        settings = extract_settings(args.config, overrides)
        
        if args.all:
            # Print all settings as bash-friendly KEY=VALUE pairs
            for key, value in settings.items():
                # Convert booleans to bash-friendly strings
                if isinstance(value, bool):
                    value = 'true' if value else 'false'
                # Handle None values
                elif value is None:
                    value = ''
                # Quote strings that might have spaces
                if isinstance(value, str) and ' ' in value:
                    value = f'"{value}"'
                print(f'{key.upper()}={value}')
        elif args.key:
            print_setting(args.key)
        else:
            # Default: print key settings on separate lines
            print(settings['model_name'])
            print(settings['layer'])
            print(settings['dataset_name'])
            print(settings['output_dir'])
            print(settings['training_script'] or '')
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        # Check if it's a Hydra composition error
        if "Could not load" in str(e) or "defaults" in str(e):
            print("Note: This might be due to missing default configs referenced in the YAML file.", file=sys.stderr)
        sys.exit(1)