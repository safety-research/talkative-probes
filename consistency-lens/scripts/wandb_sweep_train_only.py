#!/usr/bin/env python3
"""
W&B Sweep wrapper for direct training (bypasses submit_with_config.sh)
Use this when activations are already dumped and data is pretokenized.
Supports both single-GPU and multi-GPU distributed training.
"""

import subprocess
import sys
import os
import wandb
import torch

def main():
    # Initialize wandb to get sweep parameters
    wandb.init()
    
    # Get the parameters from wandb config
    config = wandb.config
    
    # Get absolute path to config directory
    config_path = config.config
    if not config_path.startswith('/'):
        # Make relative path absolute
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, config_path)
    
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    
    # Detect number of GPUs available or use configured value
    num_gpus = int(config.get('num_gpus', torch.cuda.device_count()))
    
    # Ensure we don't exceed available GPUs
    available_gpus = torch.cuda.device_count()
    if num_gpus > available_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
        num_gpus = available_gpus
    
    # Build the base arguments
    base_args = [
        f"--config-path={config_dir}",
        f"--config-name={config_name}",
        f"run_suffix={config.run_suffix}{os.environ.get('SLURM_JOB_ID', 'local')}"
    ]
    
    # Add all sweep parameters
    for key, value in config.items():
        if key not in ["config", "num_gpus"]:
            base_args.append(f"{key}={value}")
    
    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Always use distributed script - it handles single GPU efficiently
    cmd = [
        "torchrun",
        "--nproc_per_node", str(num_gpus),
        "--nnodes", "1",
        "--node_rank", "0",
        "--master_addr", "127.0.0.1",
        "--master_port", os.environ.get("MASTER_PORT", "29500"),
        "scripts/01_train_distributed.py"
    ] + base_args
    
    print(f"Running with {num_gpus} GPU(s)")
    print(f"Executing: {' '.join(cmd)}")
    
    # Execute the command
    result = subprocess.run(cmd, cwd=project_root)
    
    # Exit with the same code as the subprocess
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 