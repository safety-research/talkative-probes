#!/bin/bash

# Launch multi-GPU activation dumping with Hydra configuration
# Usage: ./launch_multigpu_dump_hydra.sh [hydra overrides]
# Example: ./launch_multigpu_dump_hydra.sh activation_dumper.num_samples=10000 layer_l=7
# Example with pretokenized: ./launch_multigpu_dump_hydra.sh use_pretokenized=true pretokenized_path=data/pretokenized/SimpleStories

# Environment setup
# Only set CUDA_VISIBLE_DEVICES if not already set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES not set, using all available GPUs"
else
    echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# Set OMP_NUM_THREADS if not already set
if [ -z "$OMP_NUM_THREADS" ]; then
    export OMP_NUM_THREADS=16  # Default threads per process
fi
echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Check if running on a SLURM cluster
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running on SLURM cluster (job $SLURM_JOB_ID)"
    # Use SLURM's distributed launch
    srun --ntasks=$SLURM_NTASKS \
         --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
         --cpus-per-task=$SLURM_CPUS_PER_TASK \
         python scripts/00_dump_activations_multigpu.py "$@"
else
    # Local multi-GPU launch with torchrun
    echo "Running locally with torchrun"
    
    # Detect number of GPUs from CUDA_VISIBLE_DEVICES or nvidia-smi
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        # Count comma-separated GPU IDs
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    else
        # Fall back to nvidia-smi
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    fi
    echo "Using $NUM_GPUS GPUs"
    
    # Launch with torchrun
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        scripts/00_dump_activations_multigpu.py "$@"
fi 