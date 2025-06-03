#!/bin/bash
# Install Flash Attention 2 with proper environment setup

set -e  # Exit on error

# Source the environment helper
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ensure_env.sh"

echo "Installing Flash Attention 2..."
echo "Note: This requires CUDA and may take several minutes to compile."

# Check CUDA availability
if ! uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'" 2>/dev/null; then
    echo "Error: CUDA is not available or PyTorch is not installed."
    echo "Please ensure the base environment is set up first with 'make'."
    exit 1
fi

# Get PyTorch and CUDA versions
TORCH_VERSION=$(uv run python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(uv run python -c "import torch; print(torch.version.cuda)")
echo "Detected PyTorch version: $TORCH_VERSION"
echo "Detected CUDA version: $CUDA_VERSION"

# Install build dependencies
echo "Installing build dependencies..."
uv pip install ninja packaging wheel setuptools

# Add flash-attn to dependencies and sync
echo "Adding flash-attn to project dependencies and building..."
uv add flash-attn --no-build-isolation-package flash-attn

# Verify installation
echo ""
if uv run python -c "import flash_attn; print(f'✓ Flash Attention {flash_attn.__version__} installed successfully!')" 2>/dev/null; then
    echo ""
    echo "To use Flash Attention in training, add to your config:"
    echo "  decoder:"
    echo "    use_flash_attention: true"
    echo ""
    echo "Or use the example config:"
    echo "  uv run python scripts/01_train.py --config conf/gpt2_frozen_flash.yaml"
else
    echo "Flash Attention installation may still be in progress or may have failed."
    echo "The code will automatically fall back to standard KV cache if Flash Attention is not available."
    echo ""
    echo "Common issues:"
    echo "  - Missing CUDA toolkit (nvcc)"
    echo "  - Incompatible CUDA/PyTorch versions"
    echo "  - Insufficient memory during compilation"
fi