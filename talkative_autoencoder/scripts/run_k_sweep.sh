#!/bin/bash
# Run best-of-K sweep analysis
# Usage: ./scripts/run_k_sweep.sh eval.checkpoint_path=/path/to/checkpoint.pt

## Get the directory where this script is located
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#  PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

set -e

# Change to script directory and then to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Running from $(pwd)"
PROJECT_ROOT=$(pwd)
echo "PROJECT_ROOT: $PROJECT_ROOT"



# Default settings
CONFIG="${CONFIG:-+eval=k_sweep}"
TIME="${TIME:-4:00:00}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
PARTITION="${PARTITION:-gpu}"

# Print usage
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 eval.checkpoint_path=/path/to/checkpoint.pt [additional_args]"
    echo ""
    echo "Environment variables:"
    echo "  CONFIG        - Config to use (default: +eval=k_sweep)"
    echo "  TIME          - SLURM time limit (default: 4:00:00)"
    echo "  NODES         - Number of nodes (default: 1)"
    echo "  GPUS_PER_NODE - GPUs per node (default: 1)"
    echo "  PARTITION     - SLURM partition (default: gpu)"
    echo ""
    echo "Examples:"
    echo "  # Quick test with 3 K values"
    echo "  CONFIG=+eval=k_sweep_quick $0 eval.checkpoint_path=/path/to/checkpoint.pt"
    echo ""
    echo "  # Full sweep on multiple GPUs"
    echo "  GPUS_PER_NODE=4 $0 eval.checkpoint_path=/path/to/checkpoint.pt"
    exit 0
fi

# Check if checkpoint path is provided
if [[ ! "$@" =~ eval.checkpoint_path= ]]; then
    echo "Error: Must provide eval.checkpoint_path=/path/to/checkpoint.pt"
    exit 1
fi

echo "Running best-of-K sweep analysis..."
echo "Config: $CONFIG"
echo "Arguments: $@"

# Check if we're on a SLURM system
if command -v sbatch &> /dev/null; then
    echo "SLURM detected. Submitting job..."
    
    # Build sbatch command
    sbatch_cmd="sbatch --job-name=k_sweep"
    sbatch_cmd="$sbatch_cmd --time=$TIME"
    sbatch_cmd="$sbatch_cmd --nodes=$NODES"
    sbatch_cmd="$sbatch_cmd --gpus-per-node=$GPUS_PER_NODE"
    sbatch_cmd="$sbatch_cmd --partition=$PARTITION"
    sbatch_cmd="$sbatch_cmd --output=$PROJECT_ROOT/logs/k_sweep_%j.out"
    sbatch_cmd="$sbatch_cmd --error=$PROJECT_ROOT/logs/k_sweep_%j.err"
    
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
source scripts/ensure_env.sh 2>/dev/null || true

# Run K-sweep
if [ $GPUS_PER_NODE -gt 1 ]; then
    echo "Running distributed K-sweep on $GPUS_PER_NODE GPUs..."
    uv run torchrun --nproc_per_node=$GPUS_PER_NODE scripts/03_best_of_k_sweep.py $CONFIG $@
else
    echo "Running single-GPU K-sweep..."
    uv run python scripts/03_best_of_k_sweep.py $CONFIG $@
fi
EOF

else
    echo "No SLURM detected. Running locally..."
    
    cd "$PROJECT_ROOT"
    source scripts/ensure_env.sh
    
    # Check if multiple GPUs are available
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
    
    if [ $GPU_COUNT -gt 1 ] && [ "${DISTRIBUTED:-true}" == "true" ]; then
        echo "Running distributed K-sweep on $GPU_COUNT GPUs..."
        uv run torchrun --nproc_per_node=$GPU_COUNT scripts/03_best_of_k_sweep.py $CONFIG "$@"
    else
        echo "Running single-GPU K-sweep..."
        uv run python scripts/03_best_of_k_sweep.py $CONFIG "$@"
    fi
fi

