#!/bin/bash
#SBATCH --job-name=ss-unfreeze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00
#SBATCH --output=logs/simplestories_unfreeze_%j.out
#SBATCH --error=logs/simplestories_unfreeze_%j.err

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

echo "Starting SimpleStories Progressive Unfreezing experiment on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi -L | head -n 1)"
echo "Start time: $(date)"

# Configuration variables
MODEL_NAME="SimpleStories-5M"
LAYER=5
FORCE_REDUMP="${FORCE_REDUMP:-false}"

# Step 1: Check if we need to pretokenize (same as frozen)
PRETOKENIZED_PATH="./data/corpus/SimpleStories/pretokenized"
if [ ! -d "$PRETOKENIZED_PATH" ] || [ "$FORCE_REDUMP" == "true" ]; then
    echo "=== Pretokenizing SimpleStories dataset ==="
    python scripts/pretokenize_dataset.py \
        --config-path=conf --config-name=simplestories_unfreeze
else
    echo "=== Pretokenized data already exists, skipping ==="
fi

# Step 2: Check if we need to dump activations (same data as frozen)
ACTIVATION_DIR="./data/activations"
TRAIN_DIR="${ACTIVATION_DIR}/SimpleStories/${MODEL_NAME}/layer_${LAYER}/SimpleStories_train"
VAL_DIR="${ACTIVATION_DIR}/SimpleStories/${MODEL_NAME}/layer_${LAYER}/SimpleStories_test"

if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A $TRAIN_DIR 2>/dev/null)" ] || [ "$FORCE_REDUMP" == "true" ]; then
    echo "=== Dumping activations using pretokenized data ==="
    
    # Create temporary config with pretokenization enabled
    cat > /tmp/ss_unfreeze_dump_${SLURM_JOB_ID}.yaml << EOF
# Temporary config for dumping with pretokenization
model_name: "SimpleStories/SimpleStories-5M"
tokenizer_name: "SimpleStories/SimpleStories-5M"
layer_l: $LAYER
t_text: 10  # Using width-10 as requested

activation_dumper:
  num_samples: -1
  seq_len: 64
  use_hf_dataset: true
  hf_dataset_name: "SimpleStories/SimpleStories"
  hf_split: "train"
  dataset_cache_dir: "./data/corpus/SimpleStories"
  output_dir: "./data/activations/SimpleStories"
  batch_size: 4096  # Larger batch with pretokenized data
  val_hf_split: "test"
  val_output_dir: "./data/activations/SimpleStories"
  val_num_samples: -1
  use_pretokenized: true
  pretokenized_path: "$PRETOKENIZED_PATH"
EOF

    # Use multi-GPU dumping for speed
    torchrun --nproc_per_node=8 \
        scripts/00_dump_activations_multigpu.py \
        --config_path /tmp/ss_unfreeze_dump_${SLURM_JOB_ID}.yaml \
        --layer_idx $LAYER
    
    # Cleanup temp config
    rm -f /tmp/ss_unfreeze_dump_${SLURM_JOB_ID}.yaml
else
    echo "=== Activations already exist, skipping dump ==="
fi

# Step 3: Train with progressive unfreezing
echo "=== Starting training with PROGRESSIVE UNFREEZING ==="
echo "=== Base model frozen for first epoch, then unfrozen ==="
echo "=== For tau decay after 10 epochs, we'll need to calculate steps ==="
echo "=== t_text=10 (width-10 explanations) ==="

# First, we need to increase epochs to >10 for tau decay
# And use a custom tau schedule that stays constant for 10 epochs

# Build training command with dynamic resume support
TRAIN_CMD="python scripts/01_train.py --config-name=simplestories_unfreeze +wandb.name=\"ss_unfreeze_t10_${SLURM_JOB_ID}\" run_name=\"ss_unfreeze_t10_${SLURM_JOB_ID}\""

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