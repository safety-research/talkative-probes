#!/bin/bash
# Optimized launch script for multi-GPU activation dumping with CPU threading

# Detect GPU count if NUM_GPUS not set
if [ -z "$NUM_GPUS" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        IFS=',' read -ra DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
        NUM_GPUS=${#DEV_ARR[@]}
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    else
        NUM_GPUS=1
    fi
fi

# Calculate optimal OMP threads per process
# Assuming a typical H100 node has 112-128 CPU cores
TOTAL_CPUS=$(nproc)
OMP_THREADS=$((TOTAL_CPUS / NUM_GPUS))
echo "Detected $TOTAL_CPUS CPU cores, allocating $OMP_THREADS threads per GPU process"

# Set OpenMP and MKL thread counts
export OMP_NUM_THREADS=$OMP_THREADS
export MKL_NUM_THREADS=$OMP_THREADS
export OPENBLAS_NUM_THREADS=$OMP_THREADS
export VECLIB_MAXIMUM_THREADS=$OMP_THREADS
export NUMEXPR_NUM_THREADS=$OMP_THREADS

# Determine base path for defaults based on current directory
# If running from consistency-lens/scripts, use relative paths
if [[ "$(pwd)" == */consistency-lens/scripts ]]; then
    DEFAULT_CONFIG_PATH="../config/lens_simple.yaml"
    DEFAULT_OUTPUT_DIR="../data/activations"
else
    DEFAULT_CONFIG_PATH="consistency-lens/config/lens_simple.yaml"
    DEFAULT_OUTPUT_DIR="data/activations"
fi

# Parse command line arguments
CONFIG_PATH=${1:-$DEFAULT_CONFIG_PATH}
OUTPUT_DIR=${2:-$DEFAULT_OUTPUT_DIR}
NUM_SAMPLES=${3:-""}  # Empty means use entire dataset
LAYER_IDX=${4:-""}  # Optional, will use config default if not provided

# Additional arguments
USE_HF_DATASET=${USE_HF_DATASET:-""}
HF_DATASET_NAME=${HF_DATASET_NAME:-""}
MODEL_PARALLEL=${MODEL_PARALLEL:-""}

echo "Launching multi-GPU activation dumping with $NUM_GPUS GPUs..."
echo "OMP_NUM_THREADS set to $OMP_THREADS per process"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
if [ -n "$NUM_SAMPLES" ]; then
    echo "Samples: $NUM_SAMPLES"
else
    echo "Samples: ALL (entire dataset)"
fi

# Build command
CMD="python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    consistency-lens/scripts/00_dump_activations_multigpu.py \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR"

# Only add num_samples if specified
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

# Add optional arguments
if [ -n "$LAYER_IDX" ]; then
    CMD="$CMD --layer_idx $LAYER_IDX"
fi

if [ -n "$USE_HF_DATASET" ]; then
    CMD="$CMD --use_hf_dataset"
fi

if [ -n "$HF_DATASET_NAME" ]; then
    CMD="$CMD --hf_dataset_name $HF_DATASET_NAME"
fi

if [ -n "$MODEL_PARALLEL" ]; then
    CMD="$CMD --model_parallel"
fi

# Execute
echo "Running: $CMD"
$CMD 