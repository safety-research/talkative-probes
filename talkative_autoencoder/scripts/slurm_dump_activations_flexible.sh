#!/bin/bash
#SBATCH --job-name=dump
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --output=logs/dump_%j.out
#SBATCH --error=logs/dump_%j.err

# Set nodelist from environment variable or use default
if [ -n "${SLURM_NODELIST:-}" ]; then
    echo "Using nodelist from environment: ${SLURM_NODELIST}"
else
    # Default nodelist if not provided
    echo "Using default nodelist: 330702be7061"
fi

# Flexible activation dumping script
# Usage: sbatch slurm_dump_activations_flexible.sh <config_file> [layer_idx] [extra_args]

# Load modules if available
# module load cuda/12.4 || true
# module load python/3.11 || true

# Navigate to project root
cd /workspace/kitf/talkative-probes/consistency-lens

# Ensure environment is set up on this node
source scripts/ensure_env.sh

# Set environment variables
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true

# Create log directory
mkdir -p logs

# Parse arguments
CONFIG_FILE="${1:-conf/config.yaml}"
LAYER_IDX="${2:-}"
USE_PRETOKENIZED="${3:-}"
NUM_GPUS="${4:-8}"

# Detect number of GPUs allocated
#NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "Starting activation dumping on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs detected: $NUM_GPUS"
echo "Config: $CONFIG_FILE"
echo "Layer: ${LAYER_IDX:-from config}"
echo "Use pretokenized: ${USE_PRETOKENIZED:-no}"
echo "Start time: $(date)"

# Build command  
# Extract config directory and name
CONFIG_DIR=$(dirname $CONFIG_FILE)
CONFIG_NAME=$(basename $CONFIG_FILE .yaml)

# The python script is run from scripts/ directory, so config path needs to be relative from there
if [ "$CONFIG_DIR" = "conf" ]; then
    HYDRA_CONFIG_PATH="../conf"
else
    HYDRA_CONFIG_PATH="../$CONFIG_DIR"
fi

# Use detected number of GPUs with uv_run
CMD="uv_run torchrun --nproc_per_node=$NUM_GPUS scripts/00_dump_activations_multigpu.py --config-path $HYDRA_CONFIG_PATH --config-name $CONFIG_NAME"

if [ -n "$LAYER_IDX" ]; then
    CMD="$CMD layer_l=$LAYER_IDX"
fi

if [ "$USE_PRETOKENIZED" = "pretokenized" ]; then
    CMD="$CMD activation_dumper.use_pretokenized=true"
    echo "=== Using pretokenized data for faster dumping ==="
fi

# Override batch size for GPT-2 to avoid OOM
if [[ "$CONFIG_FILE" == *"gpt2"* ]]; then
    CMD="$CMD activation_dumper.batch_size=256"
    echo "=== Using reduced batch size (256) for GPT-2 ==="
fi

# Run multi-GPU activation dumping
echo "=== Dumping activations ==="
echo "Command: $CMD"
$CMD

echo "=== Dumping complete ==="
echo "End time: $(date)"