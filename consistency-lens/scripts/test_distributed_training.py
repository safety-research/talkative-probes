#!/usr/bin/env python3
"""Test script for distributed training functionality."""

import os
import sys
import torch
import torch.distributed as dist
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


def test_basic_distributed():
    """Test basic distributed functionality."""
    print("Testing distributed initialization...")
    
    # Initialize distributed
    rank, world_size, local_rank = init_distributed()
    
    print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    print(f"Is main process: {is_main()}")
    print(f"get_rank(): {get_rank()}, get_world_size(): {get_world_size()}")
    
    # Test tensor reduction
    if world_size > 1:
        print("\nTesting tensor reduction...")
        test_tensor = torch.tensor(rank + 1.0, device='cuda')
        test_dict = {'value': test_tensor}
        
        reduced_dict = reduce_dict(test_dict)
        expected = sum(range(1, world_size + 1)) / world_size
        
        if is_main():
            print(f"Original value on rank 0: {test_dict['value'].item()}")
            print(f"Reduced value: {reduced_dict['value'].item()}")
            print(f"Expected: {expected}")
            assert abs(reduced_dict['value'].item() - expected) < 1e-6, "Reduction failed!"
            print("Reduction test PASSED!")
    
    # Test distributed barrier
    if world_size > 1:
        print(f"\nRank {rank} reached barrier")
        dist.barrier()
        print(f"Rank {rank} passed barrier")
    
    cleanup_distributed()
    print("\nDistributed test completed successfully!")


def test_model_distribution():
    """Test model distribution with DDP."""
    rank, world_size, local_rank = init_distributed()
    
    if is_main():
        print("\nTesting model distribution...")
    
    # Create a simple model
    model = torch.nn.Linear(10, 10).cuda()
    
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
        
        # Test forward pass
        x = torch.randn(5, 10).cuda()
        y = model(x)
        
        if is_main():
            print(f"Model output shape: {y.shape}")
            print("DDP model test PASSED!")
    else:
        if is_main():
            print("Single GPU mode - skipping DDP test")
    
    cleanup_distributed()


def test_data_distribution():
    """Test distributed data loading."""
    rank, world_size, local_rank = init_distributed()
    
    if is_main():
        print("\nTesting distributed data loading...")
    
    # Create dummy dataset
    dataset = torch.arange(100)
    
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        
        # Check that each rank gets different data
        indices = list(sampler)
        print(f"Rank {rank} got {len(indices)} samples: {indices[:5]}...")
        
        # Gather all indices to check coverage
        all_indices = [None] * world_size
        dist.all_gather_object(all_indices, indices)
        
        if is_main():
            # Flatten and check we got all data
            all_indices_flat = [idx for rank_indices in all_indices for idx in rank_indices]
            all_indices_flat.sort()
            
            print(f"Total samples distributed: {len(all_indices_flat)}")
            print(f"Unique samples: {len(set(all_indices_flat))}")
            assert len(set(all_indices_flat)) == len(dataset), "Not all data was distributed!"
            print("Distributed data loading test PASSED!")
    else:
        if is_main():
            print("Single GPU mode - skipping distributed sampler test")
    
    cleanup_distributed()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["basic", "model", "data", "all"], 
                       default="all", help="Which test to run")
    args = parser.parse_args()
    
    if args.test == "basic" or args.test == "all":
        test_basic_distributed()
    
    if args.test == "model" or args.test == "all":
        test_model_distribution()
    
    if args.test == "data" or args.test == "all":
        test_data_distribution()