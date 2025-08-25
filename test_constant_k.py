#!/usr/bin/env python3
"""Test script to verify constant-k mode implementation."""

import numpy as np
import torch


def verify_constant_k_logic():
    """Verify the logic of selecting best from K candidates for different k values."""

    # Simulate batch of 3 positions, each with K=8 candidates
    batch_size = 3
    k_max = 8
    hidden_dim = 4

    # Create mock MSE values for each candidate (batch_size, k_max)
    # Lower values are better
    torch.manual_seed(42)
    mses_per_pos = torch.rand(batch_size, k_max) * 10

    print("MSE values per position (rows) and candidate (columns):")
    print(mses_per_pos)
    print()

    # Test different k values
    k_values = [1, 2, 4, 8]

    for k in k_values:
        # Select best from first k candidates
        prefix_mses = mses_per_pos[:, :k]
        best_idx = torch.argmin(prefix_mses, dim=1)
        best_mses = prefix_mses[torch.arange(batch_size), best_idx]

        print(f"K={k}:")
        print(f"  Best indices: {best_idx.tolist()}")
        print(f"  Best MSEs: {best_mses.tolist()}")
        print(f"  Average MSE: {best_mses.mean().item():.4f}")
        print()

    # Verify that MSE decreases (or stays same) as K increases
    print("Verification:")
    avg_mses = []
    for k in k_values:
        prefix_mses = mses_per_pos[:, :k]
        best_idx = torch.argmin(prefix_mses, dim=1)
        best_mses = prefix_mses[torch.arange(batch_size), best_idx]
        avg_mses.append(best_mses.mean().item())

    for i in range(1, len(k_values)):
        if avg_mses[i] > avg_mses[i - 1]:
            print(f"❌ ERROR: Average MSE increased from K={k_values[i - 1]} to K={k_values[i]}")
        else:
            print(f"✓ Average MSE decreased or stayed same from K={k_values[i - 1]} to K={k_values[i]}")

    print(f"\nAverage MSEs by K: {dict(zip(k_values, avg_mses))}")

    # Test variance calculation logic
    print("\n" + "=" * 50)
    print("Testing variance calculation logic:")

    # Create mock activations and reconstructions
    N = 100  # number of samples
    hidden_dim = 10

    # Original activations
    A = torch.randn(N, hidden_dim)

    # Reconstructions with some error
    A_hat = A + torch.randn(N, hidden_dim) * 0.5

    # Calculate variance recovery
    total_variance = torch.var(A, dim=0, unbiased=True).sum().item()
    residuals = A - A_hat
    residual_variance = torch.var(residuals, dim=0, unbiased=True).sum().item()
    variance_recovery = 1 - (residual_variance / total_variance)

    print(f"Total variance: {total_variance:.4f}")
    print(f"Residual variance: {residual_variance:.4f}")
    print(f"Variance recovery: {variance_recovery:.4f}")

    # Verify with numpy for consistency
    total_variance_np = np.var(A.numpy(), axis=0, ddof=1).sum()
    residual_variance_np = np.var(residuals.numpy(), axis=0, ddof=1).sum()
    variance_recovery_np = 1 - (residual_variance_np / total_variance_np)

    print("\nNumpy verification:")
    print(f"Variance recovery (numpy): {variance_recovery_np:.4f}")

    if abs(variance_recovery - variance_recovery_np) < 1e-6:
        print("✓ PyTorch and NumPy calculations match")
    else:
        print("❌ PyTorch and NumPy calculations differ")


if __name__ == "__main__":
    verify_constant_k_logic()
