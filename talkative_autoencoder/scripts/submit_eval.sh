#!/bin/bash
# Simple wrapper to run evaluation using submit_with_config.sh structure
# Usage: ./scripts/submit_eval.sh +eval=default eval.checkpoint_path=/path/to/checkpoint.pt

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check if we're on a SLURM system
if command -v sbatch &> /dev/null; then
    echo "SLURM detected. Submitting evaluation job..."
    
    # Default SLURM settings for evaluation (can be overridden)
    TIME="${TIME:-2:00:00}"
    NODES="${NODES:-1}"
    GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
    PARTITION="${PARTITION:-gpu}"
    
    # Build sbatch command
    sbatch_cmd="sbatch --job-name=eval_lens"
    sbatch_cmd="$sbatch_cmd --time=$TIME"
    sbatch_cmd="$sbatch_cmd --nodes=$NODES"
    sbatch_cmd="$sbatch_cmd --gpus-per-node=$GPUS_PER_NODE"
    sbatch_cmd="$sbatch_cmd --partition=$PARTITION"
    sbatch_cmd="$sbatch_cmd --output=$PROJECT_ROOT/logs/eval_%j.out"
    sbatch_cmd="$sbatch_cmd --error=$PROJECT_ROOT/logs/eval_%j.err"
    
    # Create logs directory if it doesn't exist
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Submit the job
    $sbatch_cmd <<EOF
#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

# Load modules if needed
module load cuda/11.8 2>/dev/null || true

# Setup environment
cd $PROJECT_ROOT
source scripts/setup_env.sh 2>/dev/null || true

# Run evaluation
if [ $GPUS_PER_NODE -gt 1 ]; then
    echo "Running distributed evaluation on $GPUS_PER_NODE GPUs..."
    torchrun --nproc_per_node=$GPUS_PER_NODE scripts/02_eval.py $@
else
    echo "Running single-GPU evaluation..."
    python scripts/02_eval.py $@
fi
EOF

else
    echo "No SLURM detected. Running evaluation locally..."
    
    cd "$PROJECT_ROOT"
    
    # Check if multiple GPUs are available
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
    
    if [ $GPU_COUNT -gt 1 ] && [ "${DISTRIBUTED:-true}" == "true" ]; then
        echo "Running distributed evaluation on $GPU_COUNT GPUs..."
        torchrun --nproc_per_node=$GPU_COUNT scripts/02_eval.py "$@"
    else
        echo "Running single-GPU evaluation..."
        python scripts/02_eval.py "$@"
    fi
fi