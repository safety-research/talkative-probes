#!/bin/bash
#SBATCH --job-name=gpt2-unfreeze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=330702be7061
#SBATCH --time=30:00:00
#SBATCH --output=logs/gpt2_unfreeze_%j.out
#SBATCH --error=logs/gpt2_unfreeze_%j.err

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

echo "Starting GPT-2 Unfreezing experiment on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Start time: $(date)"

# Step 1: Check if we need to dump activations (same as frozen)
ACTIVATION_DIR="./data/activations"
MODEL_NAME="openai-community/gpt2"
LAYER=6
TRAIN_DIR="${ACTIVATION_DIR}/${MODEL_NAME}/layer_${LAYER}/openwebtext_train"
VAL_DIR="${ACTIVATION_DIR}/${MODEL_NAME}/layer_${LAYER}/openwebtext_validation"

if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR 2>/dev/null)" ]; then
    echo "=== Dumping training activations ==="
    srun --ntasks=1 --exclusive torchrun --nproc_per_node=8 \
        scripts/00_dump_activations_multigpu.py \
        --config_path conf/gpt2_unfreeze.yaml \
        --layer_idx $LAYER
else
    echo "=== Training activations already exist, skipping dump ==="
fi

# Step 2: Train the model with progressive unfreezing
echo "=== Starting training with progressive unfreezing ==="
echo "=== Base model will be unfrozen after 10,000 steps ==="

# Note: The training script doesn't support distributed training yet
# Running on single GPU for now - you may want to reduce batch_size
echo "WARNING: Running on single GPU. Consider reducing batch_size in config if OOM occurs."

python scripts/01_train.py \
    --config-name=gpt2_unfreeze \
    wandb.mode=online \
    wandb.project=consistency-lens-gpt2 \
    +wandb.name="gpt2_unfreeze_${SLURM_JOB_ID}" \
    batch_size=256  # Reduced from 1024 for single GPU

echo "=== Training complete ==="
echo "End time: $(date)"
echo "Job completed successfully"