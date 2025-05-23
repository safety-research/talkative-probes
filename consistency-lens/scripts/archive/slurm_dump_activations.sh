#!/bin/bash
#SBATCH --job-name=dump-activations
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=330702be7061
#SBATCH --time=4:00:00
#SBATCH --output=logs/dump_activations_%j.out
#SBATCH --error=logs/dump_activations_%j.err

# Script for dumping activations using 8 GPUs
# Usage: sbatch slurm_dump_activations.sh <config_file> [layer_idx]

# Load required modules
# module load cuda/12.4
# module load python/3.11

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

echo "Starting activation dumping on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Config: $CONFIG_FILE"
echo "Layer: ${LAYER_IDX:-from config}"
echo "Start time: $(date)"

# Run multi-GPU activation dumping
if [ -n "$LAYER_IDX" ]; then
    echo "=== Dumping activations for layer $LAYER_IDX ==="
    torchrun --nproc_per_node=8 \
        scripts/00_dump_activations_multigpu.py \
        --config_path "$CONFIG_FILE" \
        --layer_idx "$LAYER_IDX"
else
    echo "=== Dumping activations (layer from config) ==="
    torchrun --nproc_per_node=8 \
        scripts/00_dump_activations_multigpu.py \
        --config_path "$CONFIG_FILE"
fi

echo "=== Dumping complete ==="
echo "End time: $(date)"