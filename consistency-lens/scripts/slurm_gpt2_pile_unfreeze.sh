#!/bin/bash
#SBATCH --job-name=gpt2-pile-unfreeze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=330702be7061
#SBATCH --time=40:00:00
#SBATCH --output=logs/gpt2_pile_unfreeze_%j.out
#SBATCH --error=logs/gpt2_pile_unfreeze_%j.err

# Email notifications (optional - uncomment and add your email)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com

# Load required modules
# module load cuda/12.4
# module load python/3.11

# Activate virtual environment if needed
source .venv/bin/activate || true

# Set environment variables
# Note: SLURM manages GPU allocation, don't set CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=16
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"

# Create log directory
mkdir -p logs

# Navigate to project root
cd /home/kitf/talkative-probes/consistency-lens

echo "Starting GPT-2 Pile Unfreezing experiment on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi -L | head -n 1)"
echo "Start time: $(date)"

# Configuration variables
MODEL_NAME="gpt2"
LAYER=6
FORCE_REDUMP="${FORCE_REDUMP:-false}"

# Step 1: Check if we need to pretokenize
PRETOKENIZED_PATH="./data/corpus/pile/pretokenized"
if [ ! -d "$PRETOKENIZED_PATH" ] || [ "$FORCE_REDUMP" == "true" ]; then
    echo "=== Pretokenizing The Pile dataset ==="
    echo "=== NOTE: This will take longer due to larger dataset (10M samples) ==="
    python scripts/pretokenize_dataset.py \
        --config_path conf/gpt2_pile_unfreeze.yaml
else
    echo "=== Pretokenized data already exists, skipping ==="
fi

# Step 2: Check if we need to dump activations (same as pile frozen)
ACTIVATION_DIR="./data/activations"
TRAIN_DIR="${ACTIVATION_DIR}/pile/${MODEL_NAME}/layer_${LAYER}/pile_train"
VAL_DIR="${ACTIVATION_DIR}/pile/${MODEL_NAME}/layer_${LAYER}/pile_validation"

if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR 2>/dev/null)" ]; then
    echo "=== ERROR: Activations not found! ==="
    echo "=== Please submit a separate 8-GPU job for dumping: ==="
    echo "=== sbatch scripts/slurm_dump_activations.sh conf/gpt2_pile_unfreeze.yaml $LAYER ==="
    exit 1
else
    echo "=== Training activations already exist, skipping dump ==="
    echo "=== To force re-dump, set FORCE_REDUMP=true ==="
fi

# Step 3: Train the model with progressive unfreezing
echo "=== Starting training with PROGRESSIVE UNFREEZING ==="
echo "=== Base model frozen for first epoch, then unfrozen ==="
echo "=== Training on The Pile dataset ==="

# Note: The training script doesn't support distributed training yet
# Running on single GPU for now - you may want to reduce batch_size
echo "WARNING: Running on single GPU. Consider reducing batch_size in config if OOM occurs."

python scripts/01_train.py \
    --config-name=gpt2_pile_unfreeze \
    wandb.mode=online \
    wandb.project=consistency-lens-gpt2-pile \
    +wandb.name="gpt2_pile_unfreeze_${SLURM_JOB_ID}" \
    batch_size=256  # Reduced from 1024 for single GPU

echo "=== Training complete ==="
echo "End time: $(date)"
echo "Job completed successfully"