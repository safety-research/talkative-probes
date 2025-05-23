#!/bin/bash
#SBATCH --job-name=ss-frozen
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodelist=330702be7061
#SBATCH --output=logs/simplestories_frozen_%j.out
#SBATCH --error=logs/simplestories_frozen_%j.err

# Optional parameters (uncomment if needed by your cluster):
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=16
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

# Load required modules
# module load cuda/12.4 || true
# module load python/3.11 || true

# Activate virtual environment if needed
source .venv/bin/activate || true

# Set environment variables
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

# Step 2: Check if activations exist
ACTIVATION_DIR="./data/activations"
TRAIN_DIR="${ACTIVATION_DIR}/SimpleStories/${MODEL_NAME}/layer_${LAYER}/SimpleStories_train"
VAL_DIR="${ACTIVATION_DIR}/SimpleStories/${MODEL_NAME}/layer_${LAYER}/SimpleStories_test"

if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR 2>/dev/null)" ]; then
    echo "=== ERROR: Activations not found at $TRAIN_DIR ==="
    echo "=== Please use the submission wrapper to handle dependencies: ==="
    echo "=== ./scripts/submit_with_dumping.sh ss-frozen ==="
    exit 1
else
    echo "=== Activations found at: $TRAIN_DIR ==="
    echo "=== Proceeding with training ==="
fi

# Step 3: Train the model with frozen base
echo "=== Starting training with FROZEN base model ==="

# Training using the dedicated config
python scripts/01_train.py \
    --config-name=simplestories_frozen \
    +wandb.name="ss_frozen_t10_${SLURM_JOB_ID}" \
    run_name="ss_frozen_t10_${SLURM_JOB_ID}"

echo "=== Training complete ==="
echo "End time: $(date)"
echo "Job completed successfully"
