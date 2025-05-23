#!/bin/bash

# Launch multi-GPU activation dumping with pretokenized data using Hydra
# Usage: ./launch_multigpu_dump_pretokenized_hydra.sh [hydra overrides]
# Example: ./launch_multigpu_dump_pretokenized_hydra.sh layer_l=7 activation_dumper.batch_size=4096

USE_SLURM=${USE_SLURM:-false}
# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # Go to project root

# Environment setup
# Only set CUDA_VISIBLE_DEVICES if not already set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES not set, using all available GPUs"
else
    echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# Optimized settings for pretokenized data
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}  # Less CPU needed for pretokenized
export TOKENIZERS_PARALLELISM=false  # Not needed for pretokenized data

echo "Environment:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"

# Default to using pretokenized data
PRETOKENIZED_ARGS="activation_dumper.use_pretokenized=true"

# Check if dataset is specified in args, otherwise use default from config
if ! echo "$@" | grep -q "pretokenized_path="; then
    # Assume pretokenized path follows the pattern from config
    PRETOKENIZED_ARGS="$PRETOKENIZED_ARGS"
fi

# Check if running on a SLURM cluster
if [ -n "$SLURM_JOB_ID" && $USE_SLURM = true ]; then
    echo "Running on SLURM cluster (job $SLURM_JOB_ID)"
    echo "Command: python scripts/00_dump_activations_multigpu.py $PRETOKENIZED_ARGS $@"
    
    srun --ntasks=$SLURM_NTASKS \
         --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
         --cpus-per-task=$SLURM_CPUS_PER_TASK \
         python scripts/00_dump_activations_multigpu.py $PRETOKENIZED_ARGS "$@"
else
    # Local multi-GPU launch with torchrun
    echo "Running locally with torchrun"
    
    # Detect number of GPUs
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    else
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    fi
    echo "Using $NUM_GPUS GPUs"
    
    echo "Command: torchrun --nproc_per_node=$NUM_GPUS scripts/00_dump_activations_multigpu.py $PRETOKENIZED_ARGS $@"
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=${MASTER_PORT:-29500} \
        scripts/00_dump_activations_multigpu.py $PRETOKENIZED_ARGS "$@"
fi 