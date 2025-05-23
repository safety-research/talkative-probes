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

# Build command
CMD="torchrun --nproc_per_node=4 scripts/00_dump_activations_multigpu.py --config_path $CONFIG_FILE"

if [ -n "$LAYER_IDX" ]; then
    CMD="$CMD --layer_idx $LAYER_IDX"
fi

if [ "$USE_PRETOKENIZED" = "pretokenized" ]; then
    CMD="$CMD --activation_dumper.use_pretokenized true"
    echo "=== Using pretokenized data for faster dumping ==="
fi

# Run multi-GPU activation dumping
echo "=== Dumping activations ==="
echo "Command: $CMD"
$CMD

echo "=== Dumping complete ==="
echo "End time: $(date)"