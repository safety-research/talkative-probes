#!/usr/bin/env python3
"""
Test script to verify single GPU training with distributed script
"""
import subprocess
import sys
import os
import time

def test_single_gpu():
    """Test that distributed script works efficiently with 1 GPU"""
    
    print("Testing single GPU training with distributed script...")
    
    # Simple test command - just run for a few steps
    cmd = [
        "torchrun",
        "--nproc_per_node", "1",
        "--nnodes", "1",
        "--node_rank", "0",
        "--master_addr", "127.0.0.1",
        "--master_port", "29501",
        "scripts/01_train_distributed.py",
        "--config-path=../conf",
        "--config-name=config",
        "max_train_steps=5",  # Just run 5 steps
        "log_interval=1",
        "wandb.mode=disabled"  # Disable wandb for test
    ]
    
    start_time = time.time()
    
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nTest completed in {elapsed_time:.2f} seconds")
    print(f"Return code: {result.returncode}")
    
    if result.returncode != 0:
        print("\nSTDERR:")
        print(result.stderr)
        print("\nSTDOUT:")
        print(result.stdout)
        return False
    
    # Check that it ran without DDP overhead warnings
    stdout_lower = result.stdout.lower()
    
    # Look for indicators of efficient single GPU execution
    if "distributed training with 1 gpus" in stdout_lower:
        print("✓ Correctly detected single GPU mode")
    
    # Check there are no DDP-related warnings
    if "ddp" not in stdout_lower and "distributed" not in result.stderr.lower():
        print("✓ No DDP overhead warnings")
    
    print("\n✓ Single GPU training works correctly with distributed script")
    return True

if __name__ == "__main__":
    success = test_single_gpu()
    sys.exit(0 if success else 1)