#!/usr/bin/env python3
"""
W&B Sweep wrapper for direct training (bypasses submit_with_config.sh)
Use this when activations are already dumped and data is pretokenized.
"""

import subprocess
import sys
import os
import wandb

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
    
    # Build the command to call 01_train.py directly
    cmd = [
        "python",
        "scripts/01_train.py",
        f"--config-path={config_dir}",
        f"--config-name={config_name}",
        f"learning_rate={config.learning_rate}",
        f"num_train_epochs={config.num_train_epochs}"
    ]
    for key, value in config.items():
        if key not in ["config", "learning_rate", "num_train_epochs"]:
            cmd.append(f"{key}={value}")
    
    print(f"Executing: {' '.join(cmd)}")
    
    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Execute the command
    result = subprocess.run(cmd, cwd=project_root)
    
    # Exit with the same code as the subprocess
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 