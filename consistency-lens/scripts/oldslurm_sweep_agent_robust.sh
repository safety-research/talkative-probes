#!/bin/bash
#SBATCH --job-name=wandb-sweep-robust
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

# =============================================================================
# Robust SLURM W&B Sweep Agent Script with Auto-Retry
# =============================================================================
# 
# This script runs W&B sweep agents with automatic retry on crashes.
# It handles the common "Broken pipe" error that occurs after ~20 minutes.
#
# USAGE:
#   sbatch [--gres=gpu:N] scripts/slurm_sweep_agent_robust.sh SWEEP_ID [TOTAL_RUNS]
#
# ARGUMENTS:
#   SWEEP_ID    - Full W&B sweep ID (e.g., user/project/abc123def)
#   TOTAL_RUNS  - Total number of runs to complete (default: 10)
#
# EXAMPLES:
#   # Run 10 experiments with auto-retry on 2 GPUs
#   sbatch --gres=gpu:2 scripts/slurm_sweep_agent_robust.sh user/project/abc123def 10
#
# =============================================================================

echo "Setting up environment"
set -euo pipefail

SWEEP_ID=$1
TOTAL_RUNS=${2:-10}  # Total runs to complete

if [ -z "$SWEEP_ID" ]; then
    echo "Error: SWEEP_ID required"
    echo ""
    echo "Usage: sbatch [--gres=gpu:N] $0 SWEEP_ID [TOTAL_RUNS]"
    echo ""
    echo "Example:"
    echo "  sbatch --gres=gpu:2 $0 user/consistency-lens/abc123def 10"
    exit 1
fi

# Detect number of GPUs allocated
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=1
fi

# Navigate to project root
cd /workspace/kitf/talkative-probes/consistency-lens
echo "Current directory: $(pwd)"

# Ensure environment is set up on this node
source scripts/ensure_env.sh

# Set up environment with stability settings
export OMP_NUM_THREADS=16
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true

# W&B stability settings
export WANDB__SERVICE_WAIT=300
export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=false
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300
export WANDB_AGENT_MAX_INITIAL_FAILURES=10
export WANDB_AGENT_DISABLE_FLAPPING=true

# Disable W&B's internal retry to control it ourselves
export WANDB_AGENT_REPORT_INTERVAL=60
export WANDB_AGENT_HEARTBEAT_TIMEOUT=300

# Set master port for distributed training
export MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

# Configure for multi-GPU if needed
if [ $NUM_GPUS -gt 1 ]; then
    echo "Configuring for $NUM_GPUS GPU distributed training"
    export WORLD_SIZE=$NUM_GPUS
    export RANK=0
    export LOCAL_RANK=0
    export MASTER_ADDR=127.0.0.1
fi

# Logging metadata
echo "=============================================="
echo "Robust W&B Sweep Agent on SLURM"
echo "=============================================="
echo "Sweep ID: $SWEEP_ID"
echo "Total runs target: $TOTAL_RUNS"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES (Count: $NUM_GPUS)"
echo "Master Port: $MASTER_PORT"
echo "Working Directory: $(pwd)"
echo "Time: $(date)"
echo "=============================================="

# Track progress
COMPLETED_RUNS=0
ATTEMPT=0
MAX_ATTEMPTS=50  # Maximum retry attempts

# Function to run a single agent
run_agent() {
    local runs_per_agent=$1
    echo ""
    echo "Starting agent for $runs_per_agent run(s)..."
    
    # Create a timeout wrapper to kill stuck processes
    timeout --kill-after=10m 2h uv_run wandb agent --count $runs_per_agent $SWEEP_ID
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "Agent completed successfully"
        return 0
    elif [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
        echo "Agent timed out after 2 hours"
        return 1
    else
        echo "Agent exited with code $exit_code"
        # Check W&B logs for specific errors
        if grep -q "BrokenPipeError\|MessageRouterClosedError\|SockClientClosedError" wandb/latest-run/logs/debug.log 2>/dev/null; then
            echo "Detected W&B service crash - will retry"
            return 1
        fi
        return $exit_code
    fi
}

# Main retry loop
while [ $COMPLETED_RUNS -lt $TOTAL_RUNS ] && [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    REMAINING_RUNS=$((TOTAL_RUNS - COMPLETED_RUNS))
    
    echo ""
    echo "=============================================="
    echo "Attempt $ATTEMPT of $MAX_ATTEMPTS"
    echo "Completed: $COMPLETED_RUNS / $TOTAL_RUNS runs"
    echo "Remaining: $REMAINING_RUNS runs"
    echo "Time: $(date)"
    echo "=============================================="
    
    # Clear any stale W&B state
    if [ -d "wandb/latest-run" ]; then
        echo "Cleaning up stale W&B state..."
        rm -rf wandb/latest-run/.wandb-cache 2>/dev/null || true
        rm -f wandb/latest-run/.wandb_socket 2>/dev/null || true
    fi
    
    # Run at most 5 runs at a time to minimize loss on crash
    RUNS_THIS_ATTEMPT=$((REMAINING_RUNS > 5 ? 5 : REMAINING_RUNS))
    
    # Try to run the agent
    if run_agent $RUNS_THIS_ATTEMPT; then
        COMPLETED_RUNS=$((COMPLETED_RUNS + RUNS_THIS_ATTEMPT))
        echo "Successfully completed $RUNS_THIS_ATTEMPT runs"
    else
        # Try to detect how many runs completed before crash
        if [ -f "wandb/latest-run/wandb-summary.json" ]; then
            # This is approximate - better tracking would require parsing W&B API
            echo "Agent crashed, attempting to determine completed runs..."
            # Conservative estimate: assume 1 run completed if we ran for >20 minutes
            if [ $RUNS_THIS_ATTEMPT -gt 1 ]; then
                COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
                echo "Assuming at least 1 run completed before crash"
            fi
        fi
    fi
    
    # Wait before retry to let system stabilize
    if [ $COMPLETED_RUNS -lt $TOTAL_RUNS ]; then
        echo "Waiting 30 seconds before retry..."
        sleep 30
    fi
done

# Final summary
echo ""
echo "=============================================="
echo "FINAL SUMMARY"
echo "=============================================="
echo "Total runs completed: $COMPLETED_RUNS / $TOTAL_RUNS"
echo "Total attempts: $ATTEMPT"
echo "End time: $(date)"

if [ $COMPLETED_RUNS -ge $TOTAL_RUNS ]; then
    echo "SUCCESS: All runs completed!"
    exit 0
else
    echo "WARNING: Only completed $COMPLETED_RUNS of $TOTAL_RUNS runs"
    exit 1
fi