#!/bin/bash
# Launch script for multi-GPU activation dumping with pre-tokenized data

# Auto-detect GPUs
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

echo "Using $NUM_GPUS GPUs for pre-tokenized activation dumping"

# Auto-detect CPU cores and set thread counts
CPU_CORES=$(nproc)
echo "Detected $CPU_CORES CPU cores"

# Calculate threads per process
# Cap at 16 to avoid thrashing, ensure at least 1
THREADS_PER_PROC=$((CPU_CORES / NUM_GPUS))
if [ $THREADS_PER_PROC -gt 16 ]; then
    THREADS_PER_PROC=16
elif [ $THREADS_PER_PROC -lt 1 ]; then
    THREADS_PER_PROC=1
fi

echo "Allocating $THREADS_PER_PROC threads per GPU process"

# Set all thread-related environment variables
export OMP_NUM_THREADS=$THREADS_PER_PROC
export MKL_NUM_THREADS=$THREADS_PER_PROC
export OPENBLAS_NUM_THREADS=$THREADS_PER_PROC
export VECLIB_MAXIMUM_THREADS=$THREADS_PER_PROC
export NUMEXPR_NUM_THREADS=$THREADS_PER_PROC
export RAYON_NUM_THREADS=$THREADS_PER_PROC

# Determine base path for defaults based on current directory
if [[ "$(pwd)" == */consistency-lens/scripts ]]; then
    DEFAULT_CONFIG_PATH="../conf/config.yaml"
    DEFAULT_OUTPUT_DIR="../data/activations"
elif [[ "$(pwd)" == */consistency-lens ]]; then
    DEFAULT_CONFIG_PATH="conf/config.yaml"
    DEFAULT_OUTPUT_DIR="data/activations"
else
    DEFAULT_CONFIG_PATH="consistency-lens/conf/config.yaml"
    DEFAULT_OUTPUT_DIR="data/activations"
fi

# Parse command line arguments
CONFIG_PATH=${1:-$DEFAULT_CONFIG_PATH}
OUTPUT_DIR=${2:-$DEFAULT_OUTPUT_DIR}
NUM_SAMPLES=${3:-""}  # Empty means use entire dataset
LAYER_IDX=${4:-""}  # Optional
PRETOKENIZED_PATH=${5:-""}  # Optional, will auto-detect if not provided

# Additional arguments
MODEL_PARALLEL=${MODEL_PARALLEL:-""}

echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
if [ -n "$NUM_SAMPLES" ]; then
    echo "Samples: $NUM_SAMPLES"
else
    echo "Samples: ALL (entire dataset)"
fi

# Determine the correct script path based on current directory
SCRIPT_PATH="scripts/00_dump_activations_multigpu.py"
if [[ "$(pwd)" == */consistency-lens/scripts ]]; then
    SCRIPT_PATH="./00_dump_activations_multigpu.py"
elif [[ "$(pwd)" == */consistency-lens ]]; then
    SCRIPT_PATH="scripts/00_dump_activations_multigpu.py"
else
    SCRIPT_PATH="consistency-lens/scripts/00_dump_activations_multigpu.py"
fi

# Extract config name from path (remove .yaml extension)
CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
CONFIG_DIR=$(dirname "$CONFIG_PATH")

# Make config directory absolute if it's relative
if [[ ! "$CONFIG_DIR" = /* ]]; then
    CONFIG_DIR="$(pwd)/$CONFIG_DIR"
fi

# Build command with Hydra overrides
CMD="python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    $SCRIPT_PATH \
    --config-path=$CONFIG_DIR \
    --config-name=$CONFIG_NAME \
    activation_dumper.use_pretokenized=true"

# Add Hydra overrides for output directory
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD activation_dumper.output_dir=$OUTPUT_DIR"
fi

# Add optional arguments as Hydra overrides
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD activation_dumper.num_samples=$NUM_SAMPLES"
fi

if [ -n "$LAYER_IDX" ]; then
    CMD="$CMD layer_l=$LAYER_IDX"
fi

if [ -n "$PRETOKENIZED_PATH" ]; then
    CMD="$CMD activation_dumper.pretokenized_path=$PRETOKENIZED_PATH"
fi

if [ -n "$MODEL_PARALLEL" ]; then
    CMD="$CMD activation_dumper.model_parallel=true"
fi

echo "Note: Using pre-tokenized data eliminates tokenization bottleneck!"
echo "Running: $CMD"
$CMD 