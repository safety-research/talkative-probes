#!/usr/bin/env python3
"""Benchmark script to compare DistributedSampler vs FastDistributedSampler performance."""

import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lens.training.fast_distributed_sampler import FastDistributedSampler


class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return torch.randn(10)


def benchmark_sampler(sampler_class, dataset_size, num_epochs=5):
    """Benchmark a sampler class."""
    dataset = DummyDataset(dataset_size)
    
    # Simulate distributed environment
    num_replicas = 8  # Simulate 8 GPUs
    rank = 0
    
    times = []
    
    for epoch in range(num_epochs):
        sampler = sampler_class(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True
        )
        sampler.set_epoch(epoch)
        
        # Time the iterator creation (this is where the slowdown happens)
        start_time = time.time()
        indices = list(iter(sampler))
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
    avg_time = sum(times[1:]) / (len(times) - 1)  # Skip first epoch (warmup)
    return avg_time


def main():
    dataset_sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    
    print("Benchmarking DistributedSampler performance...")
    print(f"{'Dataset Size':>15} | {'Standard (s)':>12} | {'Fast (s)':>12} | {'Speedup':>8}")
    print("-" * 60)
    
    for size in dataset_sizes:
        # Standard DistributedSampler
        std_time = benchmark_sampler(DistributedSampler, size)
        
        # Fast DistributedSampler
        fast_time = benchmark_sampler(FastDistributedSampler, size)
        
        speedup = std_time / fast_time
        
        print(f"{size:>15,} | {std_time:>12.4f} | {fast_time:>12.4f} | {speedup:>7.1f}x")


if __name__ == "__main__":
    main() 