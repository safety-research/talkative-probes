#!/bin/bash
#SBATCH --job-name=ss-frozen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=330702be7061
#SBATCH --time=12:00:00
#SBATCH --output=logs/simplestories_frozen_%j.out
#SBATCH --error=logs/simplestories_frozen_%j.err

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

echo "Starting SimpleStories Frozen experiment on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi -L | head -n 1)"
echo "Start time: $(date)"

# Configuration variables
MODEL_NAME="SimpleStories-5M"
LAYER=5
FORCE_REDUMP="${FORCE_REDUMP:-false}"

# Step 1: Check if we need to pretokenize
PRETOKENIZED_PATH="./data/corpus/SimpleStories/pretokenized"
if [ ! -d "$PRETOKENIZED_PATH" ] || [ "$FORCE_REDUMP" == "true" ]; then
    echo "=== Pretokenizing SimpleStories dataset ==="
    python scripts/pretokenize_dataset.py \
        --config_path conf/simplestories_frozen.yaml
else
    echo "=== Pretokenized data already exists, skipping ==="
fi

# Step 2: Check if we need to dump activations
ACTIVATION_DIR="./data/activations"
TRAIN_DIR="${ACTIVATION_DIR}/SimpleStories/${MODEL_NAME}/layer_${LAYER}/SimpleStories_train"
VAL_DIR="${ACTIVATION_DIR}/SimpleStories/${MODEL_NAME}/layer_${LAYER}/SimpleStories_test"

if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR 2>/dev/null)" ]; then
    echo "=== ERROR: Activations not found at $TRAIN_DIR ==="
    echo "=== Please use the submission wrapper to handle dependencies: ==="
    echo "=== ./scripts/submit_with_dumping.sh ss-frozen ==="
    echo "==="
    echo "=== This will automatically: ==="
    echo "=== 1. Check if activations exist ==="
    echo "=== 2. Submit dumping job if needed (8 GPUs) ==="
    echo "=== 3. Submit training job with dependency ==="
    exit 1
else
    echo "=== Activations found at: $TRAIN_DIR ==="
    echo "=== Proceeding with training ==="
fi

# Step 3: Train the model with frozen base
echo "=== Starting training with FROZEN base model ==="
echo "=== Using config.yaml with overrides ==="
echo "=== t_text=10 (width-10 explanations) ==="

# Training using the dedicated config
python scripts/01_train.py \
    --config-name=simplestories_frozen \
    +wandb.name="ss_frozen_t10_${SLURM_JOB_ID}" \
    run_name="ss_frozen_t10_${SLURM_JOB_ID}"

echo "=== Training complete ==="
echo "End time: $(date)"
echo "Job completed successfully"