#!/bin/bash
# Launch script for GPT-2 consistency lens experiments on 8xH100

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Experiment selection
EXPERIMENT="${1:-frozen}"  # Default to frozen experiment
CONFIG_NAME=""

case $EXPERIMENT in
    "frozen")
        CONFIG_NAME="gpt2_frozen.yaml"
        echo "Running GPT-2 FROZEN base model experiment with OpenWebText"
        ;;
    "unfreeze")
        CONFIG_NAME="gpt2_unfreeze.yaml"
        echo "Running GPT-2 PROGRESSIVE UNFREEZING experiment with OpenWebText"
        ;;
    "pile-frozen")
        CONFIG_NAME="gpt2_pile_frozen.yaml"
        echo "Running GPT-2 FROZEN base model experiment with The Pile"
        ;;
    *)
        echo "Usage: $0 [frozen|unfreeze|pile-frozen]"
        exit 1
        ;;
esac

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"

echo "=== Experiment: $EXPERIMENT ==="
echo "=== Config: $CONFIG_NAME ==="
echo "=== GPUs: 8xH100 ==="

# Step 1: Dump activations (if not already done)
ACTIVATION_DIR="${PROJECT_ROOT}/data/activations"
MODEL_NAME="openai-community/gpt2"
LAYER=6

# Check if we need to dump activations
if [[ $EXPERIMENT == "pile-frozen" ]]; then
    TRAIN_DIR="${ACTIVATION_DIR}/pile_train/${MODEL_NAME}/layer_${LAYER}/train"
    VAL_DIR="${ACTIVATION_DIR}/pile_val/${MODEL_NAME}/layer_${LAYER}/train"
else
    TRAIN_DIR="${ACTIVATION_DIR}/openwebtext_train/${MODEL_NAME}/layer_${LAYER}/train"
    VAL_DIR="${ACTIVATION_DIR}/openwebtext_val/${MODEL_NAME}/layer_${LAYER}/train"
fi

if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR 2>/dev/null)" ]; then
    echo "=== Dumping training activations ==="
    echo "This will take some time on first run..."
    
    # Use optimized multi-GPU dumping
    torchrun --nproc_per_node=8 \
        scripts/00_dump_activations_multigpu.py \
        --config_path "conf/${CONFIG_NAME}" \
        --layer_idx $LAYER
else
    echo "=== Training activations already exist, skipping dump ==="
fi

# Step 2: Train the model
echo "=== Starting training ==="

# For 8xH100, we can use larger batch sizes
# Adjust per_device_batch_size based on memory usage
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training using torchrun for DDP (recommended)
torchrun --nproc_per_node=8 \
    scripts/01_train.py \
    --config-name="${CONFIG_NAME%.yaml}" \
    batch_size=1024

# Alternative: You can override specific parameters
# torchrun --nproc_per_node=8 \
#     scripts/01_train.py \
#     --config-name="${CONFIG_NAME%.yaml}" \
#     batch_size=2048 \
#     learning_rate=5e-5

echo "=== Training complete ==="
echo "Check outputs/checkpoints for saved models"
echo "View results at https://wandb.ai/your-username/consistency-lens-gpt2"