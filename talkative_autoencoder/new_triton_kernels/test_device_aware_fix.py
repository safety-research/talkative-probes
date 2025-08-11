#!/usr/bin/env python3
"""
Test script to verify that device-aware kernel caching and optimization fixes work correctly.
This script tests the critical fixes for multi-GPU corruption issues.
"""

import torch
import torch.multiprocessing as mp
import os

def test_device_isolation(rank, world_size):
    """Test that each rank uses correct device-specific kernels"""
    # Set up the device for this rank
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Create test tensors on the correct device
    x = torch.randn(128, 512, device=device, dtype=torch.float16)
    w = torch.randn(512, 256, device=device, dtype=torch.float16)
    
    print(f"Rank {rank}: Testing on device {device}")
    print(f"  x.device = {x.device}")
    print(f"  w.device = {w.device}")
    print(f"  current_device = {torch.cuda.current_device()}")
    
    # Import matmul_ogs to test kernel caching
    import sys
    sys.path.insert(0, '/root/.cache/uv/envs/consistency-lens/lib/python3.11/site-packages')
    from triton_kernels.matmul_ogs import matmul_ogs, get_kernels, _get_kernel_cache_key
    
    # Test 1: Verify cache key uses correct device
    cache_key = _get_kernel_cache_key("test", "test", device=x.device)
    print(f"  Cache key device: {cache_key[0]}")
    assert cache_key[0] == rank, f"Cache key should use device {rank}, got {cache_key[0]}"
    
    # Test 2: Verify kernel retrieval uses correct device
    kernels = get_kernels(device=x.device)
    print(f"  Kernel module retrieved successfully")
    
    # Test 3: Perform actual matmul operation
    try:
        result = matmul_ogs(x, w, bias=None)
        print(f"  Matmul result shape: {result.shape}")
        print(f"  Matmul result device: {result.device}")
        assert result.device == device, f"Result should be on {device}, got {result.device}"
        
        # Check for NaN or inf values
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"  Result has NaN: {has_nan}")
        print(f"  Result has Inf: {has_inf}")
        
        if not has_nan and not has_inf:
            print(f"Rank {rank}: ✅ ALL TESTS PASSED")
        else:
            print(f"Rank {rank}: ❌ Result contains invalid values")
            
    except Exception as e:
        print(f"Rank {rank}: ❌ Error during matmul: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test runner"""
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("This test requires at least 2 GPUs. Testing single GPU mode...")
        test_device_isolation(0, 1)
    else:
        print(f"Testing with {world_size} GPUs")
        
        # Test each device sequentially first
        print("\n=== Sequential Device Testing ===")
        for rank in range(world_size):
            test_device_isolation(rank, world_size)
            print()
        
        # Then test with multiprocessing to simulate DDP
        print("\n=== Parallel Device Testing (simulating DDP) ===")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        mp.spawn(test_device_isolation, args=(world_size,), nprocs=world_size, join=True)
        print("\n✅ All parallel tests completed")

if __name__ == "__main__":
    main()