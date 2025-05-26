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
TOTAL_CPUS=$(nproc)
THREADS_PER_GPU=$((TOTAL_CPUS / NUM_GPUS))

# Allow manual override
if [ -n "$FORCE_THREADS" ]; then
    OMP_THREADS=$FORCE_THREADS
    echo "Using forced thread count: $FORCE_THREADS"
else
# Cap threads to avoid diminishing returns and thrashing
# For tokenization, 8-16 threads per process is usually optimal
if [ $THREADS_PER_GPU -gt 16 ]; then
    echo "Warning: $THREADS_PER_GPU threads per GPU may cause thrashing. Capping at 16."
    OMP_THREADS=16
else
    OMP_THREADS=$THREADS_PER_GPU
fi
fi

echo "Detected $TOTAL_CPUS CPU cores, $NUM_GPUS GPUs"
echo "Allocating $OMP_THREADS threads per GPU process (calculated: $THREADS_PER_GPU)"
echo "Note: HF tokenizers may not use all threads depending on implementation"

# Set OpenMP and MKL thread counts
export OMP_NUM_THREADS=$OMP_THREADS
export MKL_NUM_THREADS=$OMP_THREADS
export OPENBLAS_NUM_THREADS=$OMP_THREADS
export VECLIB_MAXIMUM_THREADS=$OMP_THREADS
export NUMEXPR_NUM_THREADS=$OMP_THREADS
export RAYON_NUM_THREADS=$OMP_THREADS

# Debug: verify exports
echo "Thread settings: OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Determine base path for defaults based on current directory
# If running from consistency-lens/scripts, use relative paths
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

# Build command with Hydra overrides
CMD="python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    $SCRIPT_PATH \
    --config-path=$CONFIG_DIR \
    --config-name=$CONFIG_NAME"

# Add Hydra overrides for output directory
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD activation_dumper.output_dir=$OUTPUT_DIR"
fi

# Only add num_samples if specified
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD activation_dumper.num_samples=$NUM_SAMPLES"
fi

# Add optional arguments as Hydra overrides
if [ -n "$LAYER_IDX" ]; then
    CMD="$CMD layer_l=$LAYER_IDX"
fi

if [ -n "$USE_HF_DATASET" ]; then
    CMD="$CMD activation_dumper.use_hf_dataset=true"
fi

if [ -n "$HF_DATASET_NAME" ]; then
    CMD="$CMD activation_dumper.hf_dataset_name=$HF_DATASET_NAME"
fi

if [ -n "$MODEL_PARALLEL" ]; then
    CMD="$CMD activation_dumper.model_parallel=true"
fi

# Execute
echo "Running: $CMD"
$CMD 