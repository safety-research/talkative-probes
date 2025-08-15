#!/bin/bash
# =============================================================================
# Local Multi-GPU W&B Sweep Launcher (Non-SLURM)
# =============================================================================
#
# This script launches W&B sweep agents on a local multi-GPU node without SLURM.
# Perfect for single-node setups with multiple GPUs (e.g., 8xH100).
#
# USAGE:
# Useful to do WANDB__SERVICE_WAIT=300 sometimes
#   ./scripts/launch_sweep_local.sh SWEEP_ID STRATEGY [STRATEGY_ARGS...]
#
# STRATEGIES:
#   simple N_AGENTS N_GPUS [RUNS_PER_AGENT]
#     - Launch N_AGENTS agents with N_GPUS each
#     - Each agent runs RUNS_PER_AGENT experiments (default: 1)
#
#   auto N_AGENTS [N_GPUS]
#     - Launch N_AGENTS that run until queue is empty
#     - Default: 1 GPU per agent
#
#   balanced TOTAL_EXPERIMENTS [N_GPUS]
#     - Automatically calculate optimal agent distribution
#     - Default: 2 GPUs per agent
#
# EXAMPLES:
#   # 4 agents with 2 GPUs each (4x2 on 8-GPU node)
#   ./scripts/launch_sweep_local.sh user/project/abc123def simple 4 2
#
#   # 8 agents with 1 GPU each (8x1 on 8-GPU node)
#   ./scripts/launch_sweep_local.sh user/project/abc123def simple 8 1
#
#   # Auto mode: 4 agents with 2 GPUs each, run until empty
#   ./scripts/launch_sweep_local.sh user/project/abc123def auto 4 2
#
# =============================================================================

set -euo pipefail

# Configuration
TOTAL_GPUS=${TOTAL_GPUS:-8}  # Total GPUs available on node
LOG_DIR="logs/sweep_local"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
SWEEP_ID=$1
STRATEGY=${2:-simple}

if [ -z "$SWEEP_ID" ]; then
    echo "Error: SWEEP_ID required"
    echo ""
    echo "Usage: $0 SWEEP_ID STRATEGY [STRATEGY_ARGS...]"
    echo ""
    echo "Strategies:"
    echo "  simple N_AGENTS N_GPUS [RUNS]     - Basic agent launch"
    echo "  auto N_AGENTS [N_GPUS]             - Run until queue empty"  
    echo "  balanced TOTAL_EXPERIMENTS [N_GPUS] - Auto-calculate distribution"
    echo ""
    echo "Examples (for 8-GPU node):"
    echo "  4x2 GPUs: $0 user/project/abc123def simple 4 2"
    echo "  8x1 GPUs: $0 user/project/abc123def simple 8 1"
    echo "  2x4 GPUs: $0 user/project/abc123def simple 2 4"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Function to launch a single agent
launch_agent() {
    local agent_id=$1
    local gpu_list=$2
    local runs=$3
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$LOG_DIR/agent_${agent_id}_${timestamp}.log"
    
    echo "Launching agent $agent_id on GPUs: $gpu_list"
    
    # Create agent script
    local agent_script="$LOG_DIR/agent_${agent_id}_${timestamp}.sh"
    cat > "$agent_script" << EOF
#!/bin/bash
set -euo pipefail

# Agent $agent_id configuration
export CUDA_VISIBLE_DEVICES=$gpu_list

# Navigate to project
cd /workspace/kitf/talkative-probes/consistency-lens

# Source environment
source scripts/ensure_env.sh

# Environment variables
export OMP_NUM_THREADS=16
export TORCHINDUCTOR_CACHE_DIR="\${HOME}/.cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB__SERVICE_WAIT=300
export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=false
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300
export WANDB_AGENT_MAX_INITIAL_FAILURES=10
export WANDB_AGENT_DISABLE_FLAPPING=true
export HYDRA_FULL_ERROR=1

# Count GPUs
NUM_GPUS=\$(echo "\$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# Configure for multi-GPU if needed
if [ \$NUM_GPUS -gt 1 ]; then
    export WORLD_SIZE=\$NUM_GPUS
    export RANK=0
    export LOCAL_RANK=0
    export MASTER_ADDR=127.0.0.1
    # Add random offset to avoid conflicts between runs
    RANDOM_OFFSET=\$((RANDOM % 1000))
    export MASTER_PORT=\$((29500 + RANDOM_OFFSET + $agent_id * 10))
    echo "Agent $agent_id using MASTER_PORT=\$MASTER_PORT (offset=\$RANDOM_OFFSET)"
fi

echo "=============================================="
echo "W&B Sweep Agent $agent_id"
echo "=============================================="
echo "Sweep ID: $SWEEP_ID"
echo "Count: $runs"
echo "GPUs: \$CUDA_VISIBLE_DEVICES (Count: \$NUM_GPUS)"
echo "Time: \$(date)"
echo "=============================================="

EOF

    # Add appropriate wandb agent command based on runs parameter
    if [ "$runs" = "auto" ]; then
        cat >> "$agent_script" << EOF
# Auto mode - run until queue empty
CONSECUTIVE_EMPTY=0
ROUND=0

while true; do
    ROUND=\$((ROUND + 1))
    echo ""
    echo "Round \$ROUND (auto mode)"
    
    set +e
    uv_run wandb agent --count 1 $SWEEP_ID
    EXIT_CODE=\$?
    set -e
    
    if [ \$EXIT_CODE -eq 0 ]; then
        CONSECUTIVE_EMPTY=0
    else
        CONSECUTIVE_EMPTY=\$((CONSECUTIVE_EMPTY + 1))
        if [ \$CONSECUTIVE_EMPTY -ge 3 ]; then
            echo "Queue empty - exiting"
            exit 0
        fi
        sleep 60
    fi
    
    sleep 5
done
EOF
    else
        cat >> "$agent_script" << EOF
# Fixed count mode
uv_run wandb agent --count $runs $SWEEP_ID
EOF
    fi
    
    chmod +x "$agent_script"
    
    # Launch in background
    nohup bash "$agent_script" > "$log_file" 2>&1 &
    local pid=$!
    echo "  PID: $pid"
    echo "  Log: $log_file"
    echo $pid >> "$LOG_DIR/pids_${timestamp}.txt"
    
    sleep 2  # Brief pause between launches
}

# Function to allocate GPUs to agents
allocate_gpus() {
    local n_agents=$1
    local gpus_per_agent=$2
    
    if [ $((n_agents * gpus_per_agent)) -gt $TOTAL_GPUS ]; then
        echo "Error: Requested $((n_agents * gpus_per_agent)) GPUs but only $TOTAL_GPUS available"
        exit 1
    fi
    
    for ((i=0; i<n_agents; i++)); do
        local start_gpu=$((i * gpus_per_agent))
        local gpu_list=""
        
        for ((j=0; j<gpus_per_agent; j++)); do
            if [ -n "$gpu_list" ]; then
                gpu_list="${gpu_list},"
            fi
            gpu_list="${gpu_list}$((start_gpu + j))"
        done
        
        echo "$gpu_list"
    done
}

# Main execution based on strategy
case "$STRATEGY" in
    simple)
        NUM_AGENTS=${3:-1}
        NUM_GPUS=${4:-1}
        RUNS_PER_AGENT=${5:-1}
        
        echo "=========================================="
        echo "Strategy: SIMPLE"
        echo "=========================================="
        echo "Sweep ID: $SWEEP_ID"
        echo "Configuration: ${NUM_AGENTS}x${NUM_GPUS} (agents x GPUs)"
        echo "Runs per agent: $RUNS_PER_AGENT"
        echo "Total GPUs needed: $((NUM_AGENTS * NUM_GPUS)) / $TOTAL_GPUS available"
        echo "=========================================="
        
        # Allocate GPUs
        gpu_allocations=($(allocate_gpus $NUM_AGENTS $NUM_GPUS))
        
        # Launch agents
        for i in $(seq 0 $((NUM_AGENTS - 1))); do
            launch_agent $i "${gpu_allocations[$i]}" $RUNS_PER_AGENT
        done
        ;;
        
    auto)
        NUM_AGENTS=${3:-8}
        NUM_GPUS=${4:-1}
        
        echo "=========================================="
        echo "Strategy: AUTO (Run Until Queue Empty)"
        echo "=========================================="
        echo "Sweep ID: $SWEEP_ID"
        echo "Configuration: ${NUM_AGENTS}x${NUM_GPUS} (agents x GPUs)"
        echo "Mode: Continuous until queue empty"
        echo "Total GPUs needed: $((NUM_AGENTS * NUM_GPUS)) / $TOTAL_GPUS available"
        echo "=========================================="
        
        # Allocate GPUs
        gpu_allocations=($(allocate_gpus $NUM_AGENTS $NUM_GPUS))
        
        # Launch agents
        for i in $(seq 0 $((NUM_AGENTS - 1))); do
            launch_agent $i "${gpu_allocations[$i]}" "auto"
        done
        ;;
        
    balanced)
        TOTAL_EXPERIMENTS=${3:-27}
        NUM_GPUS=${4:-2}
        
        # Calculate optimal distribution
        if [ $TOTAL_EXPERIMENTS -le 10 ]; then
            NUM_AGENTS=$TOTAL_EXPERIMENTS
            RUNS_PER_AGENT=1
        elif [ $TOTAL_EXPERIMENTS -le 30 ]; then
            NUM_AGENTS=$(( (TOTAL_EXPERIMENTS + 2) / 3 ))
            RUNS_PER_AGENT=$(( (TOTAL_EXPERIMENTS + NUM_AGENTS - 1) / NUM_AGENTS ))
        else
            NUM_AGENTS=$(( (TOTAL_EXPERIMENTS + 4) / 5 ))
            RUNS_PER_AGENT=$(( (TOTAL_EXPERIMENTS + NUM_AGENTS - 1) / NUM_AGENTS ))
        fi
        
        # Ensure we don't exceed available GPUs
        while [ $((NUM_AGENTS * NUM_GPUS)) -gt $TOTAL_GPUS ]; do
            NUM_AGENTS=$((NUM_AGENTS - 1))
            RUNS_PER_AGENT=$(( (TOTAL_EXPERIMENTS + NUM_AGENTS - 1) / NUM_AGENTS ))
        done
        
        echo "=========================================="
        echo "Strategy: BALANCED"
        echo "=========================================="
        echo "Sweep ID: $SWEEP_ID"
        echo "Total experiments: $TOTAL_EXPERIMENTS"
        echo "Configuration: ${NUM_AGENTS}x${NUM_GPUS} (agents x GPUs)"
        echo "Runs per agent: $RUNS_PER_AGENT"
        echo "Total GPUs needed: $((NUM_AGENTS * NUM_GPUS)) / $TOTAL_GPUS available"
        echo "=========================================="
        
        # Allocate GPUs
        gpu_allocations=($(allocate_gpus $NUM_AGENTS $NUM_GPUS))
        
        # Launch agents
        for i in $(seq 0 $((NUM_AGENTS - 1))); do
            launch_agent $i "${gpu_allocations[$i]}" $RUNS_PER_AGENT
        done
        ;;
        
    *)
        echo "Error: Unknown strategy '$STRATEGY'"
        echo "Valid strategies: simple, auto, balanced"
        exit 1
        ;;
esac

echo ""
echo "All agents launched!"
echo ""
echo "Monitor progress:"
echo "  - Logs: tail -f $LOG_DIR/agent_*.log"
echo "  - PIDs: cat $LOG_DIR/pids_*.txt"
echo "  - W&B: https://wandb.ai/$SWEEP_ID"
echo ""
echo "To stop all agents:"
echo "  pkill -f 'wandb agent.*$SWEEP_ID'" 