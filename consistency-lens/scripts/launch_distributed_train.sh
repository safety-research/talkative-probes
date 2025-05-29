#!/bin/bash
# Launcher script for distributed training
# Handles both SLURM and standalone multi-GPU environments

set -e

# Navigate to project root and ensure environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source scripts/ensure_env.sh

# Default values
CONFIG_PATH=""
CONFIG_NAME=""
NUM_GPUS=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Auto-detect number of GPUs if not specified
if [ -z "$NUM_GPUS" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        IFS=',' read -ra GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
        NUM_GPUS=${#GPU_ARR[@]}
    else
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    fi
fi

echo "Launching distributed training with $NUM_GPUS GPUs"
echo "Config path: $CONFIG_PATH"
echo "Config name: $CONFIG_NAME"

# Check if we're in a SLURM environment
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running in SLURM environment"
    # SLURM provides these environment variables
    export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
    export MASTER_PORT=${MASTER_PORT:-29500}
    export WORLD_SIZE=$((SLURM_NNODES * NUM_GPUS))
    export RANK=$((SLURM_NODEID * NUM_GPUS))
    
    # Run with srun for proper SLURM integration
    srun --ntasks=$NUM_GPUS \
         --ntasks-per-node=$NUM_GPUS \
         --cpus-per-task=$((SLURM_CPUS_PER_TASK / NUM_GPUS)) \
         bash -c "source scripts/ensure_env.sh && uv_run python scripts/01_train_distributed.py \
         --config-path='$CONFIG_PATH' \
         --config-name='$CONFIG_NAME' \
         $EXTRA_ARGS"
else
    echo "Running in standalone mode"
    # Use torchrun for standalone multi-GPU
    uv_run torchrun --nproc_per_node=$NUM_GPUS \
             --master_port=${MASTER_PORT:-29500} \
             scripts/01_train_distributed.py \
             --config-path="$CONFIG_PATH" \
             --config-name="$CONFIG_NAME" \
             $EXTRA_ARGS
fi