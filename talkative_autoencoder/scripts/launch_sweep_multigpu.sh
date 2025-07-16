#!/bin/bash
# =============================================================================
# Helper script to launch W&B sweeps with multi-GPU support
# =============================================================================
#
# This script helps launch W&B sweep agents with different GPU configurations
# on SLURM clusters.
#
# USAGE:
#   ./scripts/launch_sweep_multigpu.sh SWEEP_ID STRATEGY [STRATEGY_ARGS...]
#
# STRATEGIES:
#   simple N_AGENTS N_GPUS [RUNS_PER_AGENT]
#     - Launch N_AGENTS agents with N_GPUS each
#     - Each agent runs RUNS_PER_AGENT experiments (default: 1)
#
#   balanced TOTAL_EXPERIMENTS [N_GPUS]
#     - Automatically calculate optimal agent distribution
#     - Default: 2 GPUs per agent
#
#   spread TOTAL_EXPERIMENTS
#     - One agent per experiment (maximum parallelism)
#
#   efficient TOTAL_EXPERIMENTS AGENTS
#     - Run TOTAL_EXPERIMENTS across AGENTS agents evenly
#
#   auto N_AGENTS [N_GPUS]
#     - Launch N_AGENTS that run until queue is empty
#     - Perfect for grid search - just submit and forget
#     - Default: 1 GPU per agent
#
#   explore N_AGENTS [RUNS_EACH] [N_GPUS]
#     - For Bayesian/random search: launch N_AGENTS explorers
#     - Each runs RUNS_EACH experiments (default: 5)
#     - Default: 1 GPU per agent
#
#   continuous N_AGENTS [N_GPUS]
#     - Launch N_AGENTS that run indefinitely
#     - For long-running Bayesian optimization
#     - Default: 2 GPUs per agent
#
# EXAMPLES:
#   # Simple: Launch 4 agents with 2 GPUs, 3 runs each
#   ./scripts/launch_sweep_multigpu.sh user/project/abc123def simple 4 2 3
#
#   # Auto: Launch 8 agents (1 GPU each) that run until queue empty
#   ./scripts/launch_sweep_multigpu.sh user/project/abc123def auto 8
#
#   # Balanced: Run 27 experiments with auto-calculated distribution
#   ./scripts/launch_sweep_multigpu.sh user/project/abc123def balanced 27
#
#   # Spread: Launch 27 single-GPU agents (one per experiment)
#   ./scripts/launch_sweep_multigpu.sh user/project/abc123def spread 27
#
#   # Efficient: Run 27 experiments across 9 agents
#   ./scripts/launch_sweep_multigpu.sh user/project/abc123def efficient 27 9
#
# =============================================================================

set -euo pipefail

SWEEP_ID=$1
STRATEGY=${2:-simple}

if [ -z "$SWEEP_ID" ]; then
    echo "Error: SWEEP_ID required"
    echo ""
    echo "Usage: $0 SWEEP_ID STRATEGY [STRATEGY_ARGS...]"
    echo ""
    echo "Strategies:"
    echo "  simple N_AGENTS N_GPUS [RUNS_PER_AGENT]  - Basic agent launch (fixed runs)"
    echo "  balanced TOTAL_EXPERIMENTS [N_GPUS]       - Auto-calculate distribution (grid search)"
    echo "  spread TOTAL_EXPERIMENTS                  - One agent per experiment (grid search)"
    echo "  efficient TOTAL_EXPERIMENTS AGENTS        - Distribute experiments evenly (grid search)"
    echo "  auto N_AGENTS [N_GPUS]                    - Run until queue empty (grid search)"
    echo "  explore N_AGENTS [RUNS_EACH] [N_GPUS]    - For Bayesian/random search"
    echo "  continuous N_AGENTS [N_GPUS]              - Indefinite agents for long searches"
    echo ""
    echo "Examples:"
    echo "  Grid search (auto): $0 user/project/abc123def auto 8          # 8 agents, run until empty"
    echo "  Grid search:        $0 user/project/abc123def balanced 27     # auto-calculate distribution"
    echo "  Bayesian search:    $0 user/project/abc123def explore 4 10    # 4 agents, 10 runs each"
    echo "  Fixed runs:         $0 user/project/abc123def simple 4 2 5    # 4 agents, 2 GPUs, 5 runs"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$STRATEGY" in
    simple)
        NUM_AGENTS=${3:-1}
        NUM_GPUS=${4:-1}
        RUNS_PER_AGENT=${5:-1}
        
        echo "=========================================="
        echo "Strategy: SIMPLE"
        echo "=========================================="
        echo "Sweep ID: $SWEEP_ID"
        echo "Number of agents: $NUM_AGENTS"
        echo "GPUs per agent: $NUM_GPUS"
        echo "Runs per agent: $RUNS_PER_AGENT"
        echo "Total runs: $((NUM_AGENTS * RUNS_PER_AGENT))"
        echo "Total GPUs: $((NUM_AGENTS * NUM_GPUS))"
        echo "=========================================="
        
        for i in $(seq 1 $NUM_AGENTS); do
            echo "Submitting agent $i/$NUM_AGENTS..."
            if [ $NUM_GPUS -eq 1 ]; then
                sbatch "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID" "$RUNS_PER_AGENT"
            else
                sbatch --gres=gpu:$NUM_GPUS "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID" "$RUNS_PER_AGENT"
            fi
            sleep 1
        done
        ;;
        
    balanced)
        TOTAL_EXPERIMENTS=${3:-27}
        NUM_GPUS=${4:-2}
        
        # Calculate optimal distribution
        if [ $TOTAL_EXPERIMENTS -le 10 ]; then
            # Few experiments: one agent per experiment
            NUM_AGENTS=$TOTAL_EXPERIMENTS
            RUNS_PER_AGENT=1
        elif [ $TOTAL_EXPERIMENTS -le 30 ]; then
            # Medium: aim for ~3 runs per agent
            NUM_AGENTS=$(( (TOTAL_EXPERIMENTS + 2) / 3 ))
            RUNS_PER_AGENT=$(( (TOTAL_EXPERIMENTS + NUM_AGENTS - 1) / NUM_AGENTS ))
        else
            # Many: aim for ~5 runs per agent
            NUM_AGENTS=$(( (TOTAL_EXPERIMENTS + 4) / 5 ))
            RUNS_PER_AGENT=$(( (TOTAL_EXPERIMENTS + NUM_AGENTS - 1) / NUM_AGENTS ))
        fi
        
        echo "=========================================="
        echo "Strategy: BALANCED"
        echo "=========================================="
        echo "Sweep ID: $SWEEP_ID"
        echo "Total experiments: $TOTAL_EXPERIMENTS"
        echo "Calculated distribution:"
        echo "  Agents: $NUM_AGENTS"
        echo "  Runs per agent: $RUNS_PER_AGENT"
        echo "  GPUs per agent: $NUM_GPUS"
        echo "  Total GPUs: $((NUM_AGENTS * NUM_GPUS))"
        echo "=========================================="
        
        for i in $(seq 1 $NUM_AGENTS); do
            echo "Submitting agent $i/$NUM_AGENTS..."
            if [ $NUM_GPUS -eq 1 ]; then
                sbatch "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID" "$RUNS_PER_AGENT"
            else
                sbatch --gres=gpu:$NUM_GPUS "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID" "$RUNS_PER_AGENT"
            fi
            sleep 1
        done
        ;;
        
    spread)
        TOTAL_EXPERIMENTS=${3:-27}
        
        echo "=========================================="
        echo "Strategy: SPREAD (Maximum Parallelism)"
        echo "=========================================="
        echo "Sweep ID: $SWEEP_ID"
        echo "Total experiments: $TOTAL_EXPERIMENTS"
        echo "Agents: $TOTAL_EXPERIMENTS (1 per experiment)"
        echo "GPUs per agent: 1"
        echo "Total GPUs: $TOTAL_EXPERIMENTS"
        echo "=========================================="
        echo ""
        echo "WARNING: This will submit $TOTAL_EXPERIMENTS jobs!"
        echo "Press Ctrl+C to cancel, or Enter to continue..."
        read -r
        
        for i in $(seq 1 $TOTAL_EXPERIMENTS); do
            echo "Submitting agent $i/$TOTAL_EXPERIMENTS..."
            sbatch "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID" 1
            sleep 0.5
        done
        ;;
        
    efficient)
        TOTAL_EXPERIMENTS=${3:-27}
        NUM_AGENTS=${4:-9}
        
        # Calculate runs per agent (round up)
        RUNS_PER_AGENT=$(( (TOTAL_EXPERIMENTS + NUM_AGENTS - 1) / NUM_AGENTS ))
        
        echo "=========================================="
        echo "Strategy: EFFICIENT"
        echo "=========================================="
        echo "Sweep ID: $SWEEP_ID"
        echo "Total experiments: $TOTAL_EXPERIMENTS"
        echo "Number of agents: $NUM_AGENTS"
        echo "Runs per agent: $RUNS_PER_AGENT"
        echo "GPUs per agent: 1 (efficient mode)"
        echo "=========================================="
        
        for i in $(seq 1 $NUM_AGENTS); do
            # Last agent might need fewer runs
            if [ $i -eq $NUM_AGENTS ]; then
                REMAINING=$((TOTAL_EXPERIMENTS - (NUM_AGENTS - 1) * RUNS_PER_AGENT))
                if [ $REMAINING -gt 0 ] && [ $REMAINING -lt $RUNS_PER_AGENT ]; then
                    RUNS_THIS_AGENT=$REMAINING
                else
                    RUNS_THIS_AGENT=$RUNS_PER_AGENT
                fi
            else
                RUNS_THIS_AGENT=$RUNS_PER_AGENT
            fi
            
            echo "Submitting agent $i/$NUM_AGENTS (runs: $RUNS_THIS_AGENT)..."
            sbatch "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID" "$RUNS_THIS_AGENT"
            sleep 1
        done
        ;;
        
    explore)
        NUM_AGENTS=${3:-4}
        RUNS_EACH=${4:-5}
        NUM_GPUS=${5:-1}
        
        echo "=========================================="
        echo "Strategy: EXPLORE (Bayesian/Random Search)"
        echo "=========================================="
        echo "Sweep ID: $SWEEP_ID"
        echo "Number of agents: $NUM_AGENTS"
        echo "Runs per agent: $RUNS_EACH"
        echo "GPUs per agent: $NUM_GPUS"
        echo "Expected total runs: ~$((NUM_AGENTS * RUNS_EACH))"
        echo "Total GPUs: $((NUM_AGENTS * NUM_GPUS))"
        echo "=========================================="
        
        for i in $(seq 1 $NUM_AGENTS); do
            echo "Submitting explorer agent $i/$NUM_AGENTS..."
            if [ $NUM_GPUS -eq 1 ]; then
                sbatch "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID" "$RUNS_EACH"
            else
                sbatch --gres=gpu:$NUM_GPUS "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID" "$RUNS_EACH"
            fi
            sleep 1
        done
        ;;
        
    continuous)
        NUM_AGENTS=${3:-2}
        NUM_GPUS=${4:-2}
        
        echo "=========================================="
        echo "Strategy: CONTINUOUS (Long-running Search)"
        echo "=========================================="
        echo "Sweep ID: $SWEEP_ID"
        echo "Number of agents: $NUM_AGENTS"
        echo "Runs per agent: UNLIMITED"
        echo "GPUs per agent: $NUM_GPUS"
        echo "Total GPUs: $((NUM_AGENTS * NUM_GPUS))"
        echo "=========================================="
        echo ""
        echo "NOTE: These agents will run until SLURM time limit!"
        echo "Monitor and cancel manually when satisfied."
        echo ""
        
        # Don't pass count parameter - agents run indefinitely
        for i in $(seq 1 $NUM_AGENTS); do
            echo "Submitting continuous agent $i/$NUM_AGENTS..."
            if [ $NUM_GPUS -eq 1 ]; then
                sbatch "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID"
            else
                sbatch --gres=gpu:$NUM_GPUS "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID"
            fi
            sleep 1
        done
        ;;
        
    auto)
        NUM_AGENTS=${3:-8}
        NUM_GPUS=${4:-1}
        
        echo "=========================================="
        echo "Strategy: AUTO (Run Until Queue Empty)"
        echo "=========================================="
        echo "Sweep ID: $SWEEP_ID"
        echo "Number of agents: $NUM_AGENTS"
        echo "Runs per agent: AUTO (until queue empty)"
        echo "GPUs per agent: $NUM_GPUS"
        echo "Total GPUs: $((NUM_AGENTS * NUM_GPUS))"
        echo "=========================================="
        echo ""
        echo "Perfect for grid search - agents will automatically"
        echo "distribute work until all experiments complete."
        echo ""
        
        for i in $(seq 1 $NUM_AGENTS); do
            echo "Submitting auto agent $i/$NUM_AGENTS..."
            if [ $NUM_GPUS -eq 1 ]; then
                sbatch "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID" "auto"
            else
                sbatch --gres=gpu:$NUM_GPUS "$SCRIPT_DIR/slurm_sweep_agent.sh" "$SWEEP_ID" "auto"
            fi
            sleep 1
        done
        ;;
        
    *)
        echo "Error: Unknown strategy '$STRATEGY'"
        echo "Valid strategies: simple, balanced, spread, efficient, auto, explore, continuous"
        exit 1
        ;;
esac

echo ""
echo "All agents submitted. Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "View sweep progress at:"
echo "  https://wandb.ai/$SWEEP_ID"