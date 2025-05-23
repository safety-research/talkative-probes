#!/bin/bash
#SBATCH --job-name=gpt2-pile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --output=logs/gpt2_pile_%j.out
#SBATCH --error=logs/gpt2_pile_%j.err

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

# Set nodelist from environment variable or use default
if [ -n "${SLURM_NODELIST:-}" ]; then
    echo "Using nodelist from environment: ${SLURM_NODELIST}"
    echo "Using nodelist from environment: ${SLURM_NODELIST}"
else
    # Default nodelist if not provided
    echo "Using default nodelist: 330702be7061"
    echo "Using default nodelist: 330702be7061"
fi

# Create log directory
mkdir -p logs

# Navigate to project root
cd /home/kitf/talkative-probes/consistency-lens

echo "Starting GPT-2 Pile experiment on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Start time: $(date)"

# Configuration variables
MODEL_NAME="openai-community/gpt2"
LAYER=6
FORCE_REDUMP="${FORCE_REDUMP:-false}"

# Step 1: Check if we need to pretokenize
PRETOKENIZED_PATH="./data/corpus/pile/pretokenized"
if [ ! -d "$PRETOKENIZED_PATH" ] || [ "$FORCE_REDUMP" == "true" ]; then
    echo "=== Pretokenizing The Pile dataset ==="
    echo "=== NOTE: This will take longer due to larger dataset (10M samples) ==="
    python scripts/pretokenize_dataset.py \
        --config_path conf/gpt2_pile_frozen.yaml
else
    echo "=== Pretokenized data already exists, skipping ==="
fi

# Step 2: Check if we need to dump activations
ACTIVATION_DIR="./data/activations"
TRAIN_DIR="${ACTIVATION_DIR}/pile/${MODEL_NAME}/layer_${LAYER}/pile_train"
VAL_DIR="${ACTIVATION_DIR}/pile/${MODEL_NAME}/layer_${LAYER}/pile_validation"

if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR 2>/dev/null)" ] || [ "$FORCE_REDUMP" == "true" ]; then
    echo "=== Dumping training activations from The Pile ==="
    echo "=== Using pretokenized data for faster processing ==="
    srun --ntasks=1 --exclusive torchrun --nproc_per_node=8 \
        scripts/00_dump_activations_multigpu.py \
        --config_path conf/gpt2_pile_frozen.yaml \
        --layer_idx $LAYER \
        activation_dumper.use_pretokenized=true \
        activation_dumper.pretokenized_path="$PRETOKENIZED_PATH"
else
    echo "=== Training activations already exist, skipping dump ==="
    echo "=== To force re-dump, set FORCE_REDUMP=true ==="
fi

# Step 2: Train the model
echo "=== Starting training with The Pile dataset ==="

# Note: The training script doesn't support distributed training yet
# Running on single GPU for now - you may want to reduce batch_size
echo "WARNING: Running on single GPU. Consider reducing batch_size in config if OOM occurs."

# Build training command with dynamic resume support
TRAIN_CMD="python scripts/01_train.py --config-name=gpt2_pile_frozen wandb.mode=online wandb.project=consistency-lens-gpt2-pile +wandb.name=\"gpt2_pile_frozen_${SLURM_JOB_ID}\" batch_size=256"

# Add resume parameters if provided
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    echo "=== Resuming from checkpoint: $RESUME_CHECKPOINT ==="
    TRAIN_CMD="$TRAIN_CMD resume=\"$RESUME_CHECKPOINT\""
fi

if [ -n "${WANDB_RESUME_ID:-}" ]; then
    echo "=== Resuming WandB run: $WANDB_RESUME_ID ==="
    TRAIN_CMD="$TRAIN_CMD wandb_resume_id=\"$WANDB_RESUME_ID\""
fi

echo "=== Running: $TRAIN_CMD ==="
eval $TRAIN_CMD

echo "=== Training complete ==="
echo "End time: $(date)"
echo "Job completed successfully"