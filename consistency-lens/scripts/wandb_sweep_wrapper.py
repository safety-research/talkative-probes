#!/usr/bin/env python3
"""
W&B Sweep wrapper for submit_with_config.sh
This script receives parameters from W&B sweep and calls the shell script with proper formatting.
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
    
    # Build the command to call submit_with_config.sh
    cmd = [
        "bash",
        "scripts/submit_with_config.sh",
        f"config={config.config}",
        f"learning_rate={config.learning_rate}",
        f"num_train_epochs={config.num_train_epochs}"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    # Execute the command
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    
    # Exit with the same code as the subprocess
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 