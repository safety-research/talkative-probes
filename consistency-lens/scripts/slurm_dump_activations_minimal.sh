#!/bin/bash
#SBATCH --job-name=dump
#SBATCH --gres=gpu:4
#SBATCH --nodelist=330702be7061
#SBATCH --time=4:00:00
#SBATCH --output=logs/dump_%j.out
#SBATCH --error=logs/dump_%j.err

# Minimal activation dumping script
# Usage: sbatch slurm_dump_activations_minimal.sh <config_file> [layer_idx]

# Load modules if available
# module load cuda/12.4 || true
# module load python/3.11 || true

# Activate virtual environment if needed
source .venv/bin/activate || true

# Set environment variables
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true

# Create log directory
mkdir -p logs

# Navigate to project root
cd /home/kitf/talkative-probes/consistency-lens

# Parse arguments
CONFIG_FILE="${1:-conf/config.yaml}"
LAYER_IDX="${2:-}"
USE_PRETOKENIZED="${3:-}"

echo "Starting activation dumping on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Config: $CONFIG_FILE"
echo "Layer: ${LAYER_IDX:-from config}"
echo "Use pretokenized: ${USE_PRETOKENIZED:-no}"
echo "Start time: $(date)"
echo "Debug: CONFIG_FILE='$CONFIG_FILE'"

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

CMD="torchrun --nproc_per_node=4 scripts/00_dump_activations_multigpu.py --config-path $HYDRA_CONFIG_PATH --config-name $CONFIG_NAME"

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