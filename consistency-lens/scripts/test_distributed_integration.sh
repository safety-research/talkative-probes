#!/bin/bash
# Integration test for distributed training

set -e

echo "=== Distributed Training Integration Test ==="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Ensure environment is set up
source scripts/ensure_env.sh

# Test 1: Single GPU baseline
echo "Test 1: Single GPU training (baseline)"
echo "======================================"
uv_run python scripts/01_train.py --config-name=test_distributed max_train_steps=5 \
    wandb.mode=disabled batch_size=2 > /tmp/single_gpu_test.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Single GPU training works${NC}"
else
    echo -e "${RED}✗ Single GPU training failed${NC}"
    tail -20 /tmp/single_gpu_test.log
    exit 1
fi

# Test 2: Multi-GPU with torchrun
echo
echo "Test 2: Multi-GPU training with 2 GPUs"
echo "======================================="
CUDA_VISIBLE_DEVICES=0,1 uv_run torchrun --nproc_per_node=2 \
    scripts/01_train_distributed.py --config-name=test_distributed \
    max_train_steps=5 wandb.mode=disabled batch_size=2 > /tmp/multi_gpu_test.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Multi-GPU training works${NC}"
else
    echo -e "${RED}✗ Multi-GPU training failed${NC}"
    tail -40 /tmp/multi_gpu_test.log
    exit 1
fi

# Test 3: Submit script in direct mode
echo
echo "Test 3: Submit script (direct mode)"
echo "==================================="
FORCE_DIRECT=true ./scripts/submit_with_config.sh \
    config=conf/test_distributed.yaml \
    num_gpus_train=2 \
    max_train_steps=5 \
    force_redump=true > /tmp/submit_test.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Submit script works${NC}"
else
    echo -e "${RED}✗ Submit script failed${NC}"
    tail -40 /tmp/submit_test.log
    exit 1
fi

echo
echo -e "${GREEN}=== All tests passed! ===${NC}"
echo
echo "Summary:"
echo "- Single GPU training: ✓"
echo "- Multi-GPU training: ✓"  
echo "- Submit script: ✓"
echo
echo "The distributed training implementation is working correctly!"