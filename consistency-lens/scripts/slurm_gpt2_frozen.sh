#!/bin/bash
#SBATCH --job-name=gpt2-frozen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=330702be7061
#SBATCH --time=24:00:00
#SBATCH --output=logs/gpt2_frozen_%j.out
#SBATCH --error=logs/gpt2_frozen_%j.err

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

# Create log directory
mkdir -p logs

# Navigate to project root
cd /home/kitf/talkative-probes/consistency-lens

echo "Starting GPT-2 Frozen experiment on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Start time: $(date)"

# Configuration variables
MODEL_NAME="openai-community/gpt2"
LAYER=6
FORCE_REDUMP="${FORCE_REDUMP:-false}"

# Step 1: Check if we need to pretokenize
PRETOKENIZED_PATH="./data/pretokenized/openwebtext"
if [ ! -d "$PRETOKENIZED_PATH" ] || [ "$FORCE_REDUMP" == "true" ]; then
    echo "=== Pretokenizing OpenWebText dataset ==="
    python scripts/pretokenize_dataset.py \
        --config_path conf/gpt2_frozen.yaml
else
    echo "=== Pretokenized data already exists, skipping ==="
fi

# Step 2: Check if we need to dump activations
ACTIVATION_DIR="./data/activations"
TRAIN_DIR="${ACTIVATION_DIR}/${MODEL_NAME}/layer_${LAYER}/openwebtext_train"
VAL_DIR="${ACTIVATION_DIR}/${MODEL_NAME}/layer_${LAYER}/openwebtext_validation"

if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR 2>/dev/null)" ] || [ "$FORCE_REDUMP" == "true" ]; then
    echo "=== Dumping training activations using pretokenized data ==="
    
    # Update config to use pretokenized data
    echo "=== ERROR: Activations not found! ==="
    echo "=== Please submit a separate 8-GPU job for dumping: ==="
    echo "=== sbatch scripts/slurm_dump_activations.sh conf/gpt2_frozen.yaml $LAYER ==="
    exit 1
else
    echo "=== Training activations already exist, skipping dump ==="
    echo "=== To force re-dump, set FORCE_REDUMP=true ==="
fi

# Step 2: Train the model
echo "=== Starting training ==="

# Note: The training script doesn't support distributed training yet
# Running on single GPU for now - you may want to reduce batch_size
echo "WARNING: Running on single GPU. Consider reducing batch_size in config if OOM occurs."

python scripts/01_train.py \
    --config-name=gpt2_frozen \
    wandb.mode=online \
    wandb.project=consistency-lens-gpt2 \
    +wandb.name="gpt2_frozen_${SLURM_JOB_ID}" \
    batch_size=256  # Reduced from 1024 for single GPU

echo "=== Training complete ==="
echo "End time: $(date)"
echo "Job completed successfully"