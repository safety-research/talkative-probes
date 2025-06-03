#!/usr/bin/env python3
"""Test if passing self to checkpoint causes gradient issues."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))
        self.bias = nn.Parameter(torch.randn(10))
    
    def forward_normal(self, x):
        # Normal forward
        y = x @ self.weight + self.bias
        return y.sum()
    
    def forward_checkpoint_bad(self, x):
        # Bad: passing self to checkpoint
        def inner(x, module):
            y = x @ module.weight + module.bias
            return y.sum()
        
        return checkpoint(inner, x, self, use_reentrant=False)
    
    def forward_checkpoint_good(self, x):
        # Good: passing only tensors
        def inner(x, weight, bias):
            y = x @ weight + bias
            return y.sum()
        
        return checkpoint(inner, x, self.weight, self.bias, use_reentrant=False)


def test_checkpoint_methods():
    """Test different checkpointing approaches."""
    
    print("Testing Checkpoint Methods")
    print("=" * 60)
    
    module = SimpleModule()
    x = torch.randn(5, 10, requires_grad=True)
    
    # Test 1: Normal vs bad checkpoint
    print("\nTest 1: Normal vs Bad Checkpoint (passing self)")
    
    # Normal
    module.zero_grad()
    x1 = x.clone().detach().requires_grad_(True)
    loss1 = module.forward_normal(x1)
    loss1.backward()
    
    grad_w1 = module.weight.grad.clone()
    grad_b1 = module.bias.grad.clone()
    grad_x1 = x1.grad.clone()
    
    # Bad checkpoint
    module.zero_grad()
    x2 = x.clone().detach().requires_grad_(True)
    loss2 = module.forward_checkpoint_bad(x2)
    loss2.backward()
    
    grad_w2 = module.weight.grad.clone()
    grad_b2 = module.bias.grad.clone()
    grad_x2 = x2.grad.clone()
    
    # Compare
    print(f"  Weight grad diff: {(grad_w1 - grad_w2).abs().max().item():.2e}")
    print(f"  Bias grad diff: {(grad_b1 - grad_b2).abs().max().item():.2e}")
    print(f"  Input grad diff: {(grad_x1 - grad_x2).abs().max().item():.2e}")
    
    # Test 2: Normal vs good checkpoint
    print("\nTest 2: Normal vs Good Checkpoint (passing tensors)")
    
    # Good checkpoint
    module.zero_grad()
    x3 = x.clone().detach().requires_grad_(True)
    loss3 = module.forward_checkpoint_good(x3)
    loss3.backward()
    
    grad_w3 = module.weight.grad.clone()
    grad_b3 = module.bias.grad.clone()
    grad_x3 = x3.grad.clone()
    
    # Compare
    print(f"  Weight grad diff: {(grad_w1 - grad_w3).abs().max().item():.2e}")
    print(f"  Bias grad diff: {(grad_b1 - grad_b3).abs().max().item():.2e}")
    print(f"  Input grad diff: {(grad_x1 - grad_x3).abs().max().item():.2e}")


if __name__ == "__main__":
    test_checkpoint_methods()