#!/bin/bash
#SBATCH --job-name=wandb-sweep
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --nodelist=330702be7061

# =============================================================================
# SLURM W&B Sweep Agent Script
# =============================================================================
# 
# This script runs W&B sweep agents on SLURM following the recommended pattern
# of using --count 1 for predictable resource allocation.
#
# PREREQUISITES:
# 1. Activations must be already dumped for your config
# 2. Data must be pretokenized 
# 3. W&B sweep must be initialized
#
# WORKFLOW:
# 1. Prepare data (if not done):
#    ./scripts/submit_with_config.sh config=conf/your_config.yaml
#    # Cancel training job once dumping completes
#
# 2. Initialize sweep:
#    cd consistency-lens
#    wandb sweep sweeps/lr_sweep_slurm_train_only.yaml
#    # Note the sweep ID from output
#
# 3. Submit multiple SLURM jobs:
#    for i in {1..8}; do
#        sbatch scripts/slurm_sweep_agent.sh YOUR_SWEEP_ID 1
#    done
#
# USAGE:
#   sbatch scripts/slurm_sweep_agent.sh SWEEP_ID [COUNT]
#
# ARGUMENTS:
#   SWEEP_ID  - Full W&B sweep ID (e.g., user/project/abc123def)
#   COUNT     - Number of runs per job (default: 1, recommended for SLURM)
#
# EXAMPLES:
#   # Submit single job to run 1 sweep iteration
#   sbatch scripts/slurm_sweep_agent.sh user/consistency-lens-simplestories/abc123def 1
#
#   # Submit 8 jobs for 8-parameter grid search
#   for i in {1..8}; do
#       sbatch scripts/slurm_sweep_agent.sh user/consistency-lens-simplestories/abc123def 1
#   done
#
# MONITORING:
#   - SLURM jobs: squeue -u $USER
#   - W&B dashboard: Check sweep URL from initialization
#   - Logs: logs/sweep_*.out and logs/sweep_*.err
#
# =============================================================================
echo "Setting up environment"
set -euo pipefail
set -x
echo "Environment setup"

SWEEP_ID=$1
COUNT=${2:-1}  # Default to 1 as recommended for SLURM

if [ -z "$SWEEP_ID" ]; then
    echo "Error: SWEEP_ID required"
    echo ""
    echo "Usage: sbatch $0 SWEEP_ID [COUNT]"
    echo ""
    echo "Example:"
    echo "  sbatch $0 user/consistency-lens-simplestories/abc123def 1"
    echo ""
    echo "See script header for full workflow instructions."
    exit 1
fi

# Dynamically determine project root (parent of script's directory)
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Current directory: $(pwd)"

# Set up environment
export OMP_NUM_THREADS=16
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true

# Activate virtual environment
echo "Activating"
source .venv/bin/activate
echo "Activated virtual environment"

# Logging metadata
echo "=============================================="
echo "W&B Sweep Agent on SLURM"
echo "=============================================="
echo "Sweep ID: $SWEEP_ID"
echo "Count: $COUNT"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "Time: $(date)"
echo "=============================================="
# Assume ~/.netrc or WANDB_API_KEY already present; skip interactive login

# Run the sweep agent with --count flag for SLURM
wandb agent --count $COUNT $SWEEP_ID 
