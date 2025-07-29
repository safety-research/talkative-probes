#!/bin/bash
# SLURM submission script for best-of-K sweep evaluation
# Supports both SLURM and direct execution, with distributed support via torchrun

set -e

# Change to script directory and then to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
PROJECT_ROOT=$(pwd)
echo "Running from $PROJECT_ROOT"

# Export environment variables
export OMP_NUM_THREADS=16
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

# Default values
NUM_GPUS="${NUM_GPUS:-1}"
TIME="${TIME:-24:00:00}"
JOB_NAME="${JOB_NAME:-k_sweep}"
NODES="${NODES:-1}"
NICE_JOB="${NICE:-false}"
FORCE_DIRECT="${FORCE_DIRECT:-false}"

# Store original command for logging
export SUBMIT_SCRIPT_COMMAND="$0 $@"

# Parse arguments
HYDRA_ARGS=""
for arg in "$@"; do
    case $arg in
        num_gpus=*)
            NUM_GPUS="${arg#*=}"
            ;;
        time=*)
            TIME="${arg#*=}"
            ;;
        job_name=*)
            JOB_NAME="${arg#*=}"
            ;;
        nodes=*)
            NODES="${arg#*=}"
            ;;
        nice=*)
            NICE_JOB="${arg#*=}"
            ;;
        force_direct=*)
            FORCE_DIRECT="${arg#*=}"
            ;;
        *)
            # All other arguments are passed to the script
            HYDRA_ARGS="$HYDRA_ARGS $arg"
            ;;
    esac
done

# Trim whitespace
HYDRA_ARGS=$(echo "$HYDRA_ARGS" | xargs)

# Detect SLURM availability
if [ "$FORCE_DIRECT" = "true" ]; then
    USE_SLURM=false
    echo "Forced direct execution mode (force_direct=true)"
elif command -v sbatch >/dev/null 2>&1; then
    USE_SLURM=true
    echo "SLURM environment detected"
else
    USE_SLURM=false
    echo "Non-SLURM environment detected"
fi

# Function to run the evaluation
run_evaluation() {
    cd "$PROJECT_ROOT"
    source scripts/ensure_env.sh
    ulimit -n 65536
    
    # Load .env file if it exists (for API keys, etc.)
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        # Multi-GPU: use torchrun
        # Generate random port to avoid conflicts
        MASTER_PORT=$((29500 + RANDOM % 500))
        echo "Running distributed evaluation on $NUM_GPUS GPUs (port $MASTER_PORT)..."
        uv_run torchrun \
            --nproc_per_node=$NUM_GPUS \
            --master_port=$MASTER_PORT \
            scripts/03_best_of_k_sweep.py $HYDRA_ARGS
    else
        # Single GPU
        echo "Running single-GPU evaluation..."
        uv_run python scripts/03_best_of_k_sweep.py $HYDRA_ARGS
    fi
}

# Main execution logic
if [ "$USE_SLURM" = true ]; then
    echo "Submitting K-sweep evaluation job via SLURM..."
    echo "GPUs: $NUM_GPUS, Time: $TIME"
    
    # Create logs directory
    mkdir -p logs
    
    # Build sbatch command
    SBATCH_ARGS=(
        --job-name="$JOB_NAME"
        --time="$TIME"
        --nodes="$NODES"
        --output="logs/k_sweep_%j.out"
        --error="logs/k_sweep_%j.err"
        --export="ALL,SUBMIT_SCRIPT_COMMAND=$SUBMIT_SCRIPT_COMMAND"
    )
    
    # GPU allocation
    if [ "$NUM_GPUS" -gt 1 ]; then
        SBATCH_ARGS+=(
            --gres=gpu:$NUM_GPUS
            --ntasks-per-node=$NUM_GPUS
            --cpus-per-task=8
        )
        # Request exclusive node for 8 GPUs
        if [ "$NUM_GPUS" -eq 8 ]; then
            SBATCH_ARGS+=(--exclusive)
            echo "Requesting exclusive node for 8-GPU evaluation"
        fi
    else
        SBATCH_ARGS+=(
            --gres=gpu:1
            --ntasks=1
            --cpus-per-task=16
        )
    fi
    
    # Add nice/requeueable options if requested
    if [ "$NICE_JOB" = "true" ] || [[ "$NICE_JOB" =~ ^[0-9]+$ ]]; then
        echo "Configuring as low-priority, requeueable job"
        SBATCH_ARGS+=(--requeue --qos=preemptable)
        if [[ "$NICE_JOB" =~ ^[0-9]+$ ]]; then
            SBATCH_ARGS+=(--nice="$NICE_JOB")
        else
            SBATCH_ARGS+=(--nice=10000)
        fi
        SBATCH_ARGS+=(--signal=B:TERM@120)
    fi
    
    # Create a temporary script file for SLURM
    TEMP_SCRIPT=$(mktemp /tmp/k_sweep_XXXXXX.sh)
    cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
cd $PROJECT_ROOT
source scripts/ensure_env.sh
ulimit -n 65536

# Load .env file if it exists (for API keys, etc.)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Run evaluation based on GPU count
if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU: use torchrun
    MASTER_PORT=\$((29500 + RANDOM % 500))
    echo "Running distributed evaluation on $NUM_GPUS GPUs (port \$MASTER_PORT)..."
    uv_run torchrun --nproc_per_node=$NUM_GPUS --master_port=\$MASTER_PORT scripts/03_best_of_k_sweep.py $HYDRA_ARGS
else
    # Single GPU
    echo "Running single-GPU evaluation..."
    uv_run python scripts/03_best_of_k_sweep.py $HYDRA_ARGS
fi
EOF
    chmod +x "$TEMP_SCRIPT"
    
    # Submit job with the script file
    JOB_ID=$(sbatch --parsable "${SBATCH_ARGS[@]}" "$TEMP_SCRIPT")
    
    if [ $? -eq 0 ]; then
        echo -e "\033[0;32mK-sweep job submitted with ID: $JOB_ID\033[0m"
        echo "Monitor with: squeue -j $JOB_ID"
        echo "View logs: tail -f logs/k_sweep_${JOB_ID}.out"
        
        # Log submission
        echo "$(date '+%Y-%m-%d %H:%M:%S'): k_sweep - job:$JOB_ID gpus:$NUM_GPUS args:[$HYDRA_ARGS]" >> submitted_jobs.log
    else
        echo -e "\033[0;31mERROR: Failed to submit job\033[0m"
        exit 1
    fi
else
    # Direct execution
    echo "Running K-sweep evaluation directly..."
    
    # Create logs directory
    mkdir -p logs
    
    # Run with output to logs
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_OUT="logs/k_sweep_direct_${TIMESTAMP}.out"
    LOG_ERR="logs/k_sweep_direct_${TIMESTAMP}.err"
    
    echo "Logs: $LOG_OUT and $LOG_ERR"
    
    if run_evaluation > "$LOG_OUT" 2> "$LOG_ERR"; then
        echo -e "\033[0;32mK-sweep evaluation completed successfully\033[0m"
        echo "Results saved in eval_results/"
    else
        echo -e "\033[0;31mERROR: K-sweep evaluation failed\033[0m"
        echo "Check logs: $LOG_OUT and $LOG_ERR"
        exit 1
    fi
fi

# Print usage examples if no arguments provided
if [ -z "$HYDRA_ARGS" ]; then
    echo ""
    echo "Usage examples:"
    echo "  # Basic single-GPU evaluation:"
    echo "  $0 +eval.checkpoint_path=/path/to/checkpoint.pt +eval.k_values=[1,2,4,8,16,32]"
    echo ""
    echo "  # Multi-GPU evaluation:"
    echo "  $0 num_gpus=4 +eval.checkpoint_path=/path/to/checkpoint.pt +eval.k_values=[1,2,4,8,16,32]"
    echo ""
    echo "  # With caching enabled:"
    echo "  $0 +eval.checkpoint_path=/path/to/checkpoint.pt +eval.k_values=[1,2,4,8,16,32] +eval.load_store=true"
    echo ""
    echo "  # Custom resource allocation:"
    echo "  $0 num_gpus=8 time=48:00:00 +eval.checkpoint_path=/path/to/checkpoint.pt"
    echo ""
    echo "Options:"
    echo "  num_gpus=N         Number of GPUs (default: 1)"
    echo "  time=HH:MM:SS      SLURM time limit (default: 24:00:00)"
    echo "  job_name=NAME      Job name (default: k_sweep)"
    echo "  nice=true/N        Submit as requeueable job with nice priority"
    echo "  force_direct=true  Force direct execution (no SLURM)"
    echo ""
    echo "All other arguments are passed directly to 03_best_of_k_sweep.py"
fi