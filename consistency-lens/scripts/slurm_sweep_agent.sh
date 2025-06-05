#!/bin/bash
#SBATCH --job-name=wandb-sweep
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

# =============================================================================
# SLURM W&B Sweep Agent Script with Multi-GPU Support
# =============================================================================
# 
# This script runs W&B sweep agents on SLURM with flexible GPU allocation (1-8 GPUs)
# following the recommended pattern of using --count 1 for predictable resource allocation.
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
# 3. Submit SLURM jobs with desired GPU count:
#    # Single GPU jobs (default)
#    sbatch scripts/slurm_sweep_agent.sh YOUR_SWEEP_ID
#    
#    # Multi-GPU jobs (2-8 GPUs)
#    sbatch --gres=gpu:4 scripts/slurm_sweep_agent.sh YOUR_SWEEP_ID
#    sbatch --gres=gpu:8 scripts/slurm_sweep_agent.sh YOUR_SWEEP_ID
#
# USAGE:
#   sbatch [--gres=gpu:N] scripts/slurm_sweep_agent.sh SWEEP_ID [COUNT]
#
# ARGUMENTS:
#   SWEEP_ID  - Full W&B sweep ID (e.g., user/project/abc123def)
#   COUNT     - Number of runs per job (default: 1, recommended for SLURM)
#   
# GPU ALLOCATION:
#   Override GPU count with: sbatch --gres=gpu:N (where N is 1-8)
#   Default is 1 GPU if not specified
#
# EXAMPLES:
#   # Submit single-GPU job
#   sbatch scripts/slurm_sweep_agent.sh user/consistency-lens-simplestories/abc123def
#
#   # Submit 4-GPU job
#   sbatch --gres=gpu:4 scripts/slurm_sweep_agent.sh user/consistency-lens-simplestories/abc123def
#
#   # Submit multiple 2-GPU jobs for parameter sweep
#   for i in {1..4}; do
#       sbatch --gres=gpu:2 scripts/slurm_sweep_agent.sh user/consistency-lens-simplestories/abc123def
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
COUNT=${2:-auto}  # Default to "auto" for continuous until queue empty

if [ -z "$SWEEP_ID" ]; then
    echo "Error: SWEEP_ID required"
    echo ""
    echo "Usage: sbatch [--gres=gpu:N] $0 SWEEP_ID [COUNT]"
    echo ""
    echo "Arguments:"
    echo "  SWEEP_ID  - Full W&B sweep ID"
    echo "  COUNT     - Number of runs per agent, or 'auto' to run until queue empty (default: auto)"
    echo ""
    echo "Examples:"
    echo "  sbatch $0 user/project/abc123def         # Run until no experiments left"
    echo "  sbatch $0 user/project/abc123def 5       # Run exactly 5 experiments"
    echo "  sbatch $0 user/project/abc123def auto    # Explicit auto mode"
    echo ""
    echo "See script header for full workflow instructions."
    exit 1
fi

# Detect number of GPUs allocated
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Count comma-separated GPU indices
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    # Default to 1 if not set
    NUM_GPUS=1
fi

# Navigate to project root
cd /workspace/kitf/talkative-probes/consistency-lens
echo "Current directory: $(pwd)"

# Ensure environment is set up on this node
source scripts/ensure_env.sh

# Set up environment
export OMP_NUM_THREADS=16
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB__SERVICE_WAIT=300

# W&B stability settings to prevent crashes
export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=false
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300
export WANDB_AGENT_MAX_INITIAL_FAILURES=10
export WANDB_AGENT_DISABLE_FLAPPING=true

# Set master port for distributed training (avoid conflicts)
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
echo "W&B Sweep Agent on SLURM"
echo "=============================================="
echo "Sweep ID: $SWEEP_ID"
echo "Count: $COUNT"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES (Count: $NUM_GPUS)"
echo "Master Port: $MASTER_PORT"
echo "Working Directory: $(pwd)"
echo "Time: $(date)"
echo "=============================================="
# Assume ~/.netrc or WANDB_API_KEY already present; skip interactive login

# Determine mode and run accordingly
if [ "$COUNT" = "auto" ]; then
    echo "Running in AUTO mode - will continue until sweep queue is empty"
    
    # For auto mode, we run multiple rounds until no more experiments
    CONSECUTIVE_EMPTY=0
    ROUND=0
    
    while true; do
        ROUND=$((ROUND + 1))
        echo ""
        echo "=============================================="
        echo "Starting round $ROUND (auto mode)"
        echo "=============================================="
        
        # Run a single experiment with retry logic
        MAX_RETRIES=2
        RETRY_COUNT=0
        RUN_COMPLETED=0
        
        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            echo "Starting agent for 1 run (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)..."
            
            # Run agent for just 1 experiment at a time
            set +e
            uv_run wandb agent --count 1 $SWEEP_ID
            EXIT_CODE=$?
            set -e
            
            if [ $EXIT_CODE -eq 0 ]; then
                echo "Run completed successfully"
                RUN_COMPLETED=1
                CONSECUTIVE_EMPTY=0
                break
            else
                echo "Agent exited with code $EXIT_CODE"
                
                # Check if it's because the queue is empty
                # W&B returns specific exit codes, but we can also check logs
                if [ $EXIT_CODE -eq 0 ] || grep -q "No runs in queue" wandb/latest-run/logs/debug.log 2>/dev/null; then
                    CONSECUTIVE_EMPTY=$((CONSECUTIVE_EMPTY + 1))
                    echo "Sweep queue appears to be empty (check $CONSECUTIVE_EMPTY/3)"
                    
                    if [ $CONSECUTIVE_EMPTY -ge 3 ]; then
                        echo "Sweep queue confirmed empty after 3 checks - exiting"
                        exit 0
                    fi
                    
                    echo "Waiting 60 seconds before checking again..."
                    sleep 60
                    break
                fi
                
                RETRY_COUNT=$((RETRY_COUNT + 1))
                if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                    echo "Waiting 30 seconds before retry..."
                    sleep 30
                fi
            fi
        done
        
        if [ $RUN_COMPLETED -eq 0 ] && [ $CONSECUTIVE_EMPTY -eq 0 ]; then
            echo "WARNING: Failed to complete run after $MAX_RETRIES attempts"
            exit 1
        fi
        
        # Small delay between rounds
        sleep 5
    done
    
else
    # Fixed count mode - original behavior
    echo "Running in FIXED mode - will run exactly $COUNT experiment(s)"
    
    MAX_RETRIES=2  # Try at most 2 times (1 initial + 1 retry)
    RETRY_COUNT=0
    COMPLETED=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo ""
        echo "Starting sweep agent (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)..."
        
        # Run the agent and capture exit code
        set +e
        uv_run wandb agent --count $COUNT $SWEEP_ID
        EXIT_CODE=$?
        set -e
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Agent completed successfully"
            COMPLETED=1
            break
        else
            echo "Agent exited with code $EXIT_CODE"
            RETRY_COUNT=$((RETRY_COUNT + 1))
            
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                echo "Waiting 30 seconds before retry..."
                sleep 30
            fi
        fi
    done

    if [ $COMPLETED -eq 0 ]; then
        echo "WARNING: Agent failed after $MAX_RETRIES attempts"
        exit 1
    fi
fi 
