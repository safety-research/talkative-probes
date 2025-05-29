#!/usr/bin/env python3
"""Minimal test for distributed training setup without requiring activation data."""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lens.training.distributed import (
    init_distributed,
    cleanup_distributed,
    is_main,
    get_rank,
    get_world_size,
    reduce_dict,
)


def test_distributed_setup():
    """Test that distributed setup works correctly."""
    print("Testing distributed setup...")
    
    # Initialize distributed
    rank, world_size, local_rank = init_distributed()
    
    print(f"Process {rank}/{world_size} initialized successfully")
    print(f"  Rank: {rank}")
    print(f"  World Size: {world_size}")
    print(f"  Local Rank: {local_rank}")
    print(f"  Is Main: {is_main()}")
    
    # Test basic communication if multi-GPU
    if world_size > 1:
        # Create a test tensor with rank value
        test_tensor = torch.tensor(float(rank + 1), device='cuda' if torch.cuda.is_available() else 'cpu')
        test_dict = {'value': test_tensor}
        
        # Reduce across all processes
        reduced_dict = reduce_dict(test_dict)
        
        # Expected average
        expected = sum(range(1, world_size + 1)) / world_size
        
        if is_main():
            print(f"\nCommunication test:")
            print(f"  Original value (rank 0): {rank + 1}")
            print(f"  Reduced average: {reduced_dict['value'].item():.2f}")
            print(f"  Expected: {expected:.2f}")
            
            # Verify result
            assert abs(reduced_dict['value'].item() - expected) < 1e-6, "Communication test failed!"
            print("  ✓ Communication test PASSED!")
    else:
        print("\nSingle GPU mode - skipping communication test")
    
    # Test the optimized training script exists
    train_script_path = Path(__file__).parent / "01_train_distributed.py"
    if train_script_path.exists():
        print(f"\n✓ Distributed training script exists at {train_script_path}")
    else:
        print(f"\n✗ Distributed training script not found at {train_script_path}")
    
    cleanup_distributed()
    print("\n✓ Distributed cleanup successful")
    print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    test_distributed_setup()