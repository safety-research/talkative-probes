#!/bin/bash
# Launch script for multi-GPU activation dumping

# Set number of GPUs (default 8 for H100 node)
NUM_GPUS=${NUM_GPUS:-8}

# Parse command line arguments
CONFIG_PATH=${1:-"consistency-lens/config/lens_simple.yaml"}
OUTPUT_DIR=${2:-"data/activations"}
NUM_SAMPLES=${3:-10000}
LAYER_IDX=${4:-""}  # Optional, will use config default if not provided

# Additional arguments
USE_HF_DATASET=${USE_HF_DATASET:-""}
HF_DATASET_NAME=${HF_DATASET_NAME:-""}
MODEL_PARALLEL=${MODEL_PARALLEL:-""}

echo "Launching multi-GPU activation dumping with $NUM_GPUS GPUs..."
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"

# Build command
CMD="python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    consistency-lens/scripts/00_dump_activations_multigpu.py \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES"

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