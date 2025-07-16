#!/bin/bash
# =============================================================================
# Local Sweep Monitor - Helper for managing local W&B sweep agents
# =============================================================================

set -euo pipefail

LOG_DIR="logs/sweep_local"
ACTION=${1:-status}

case "$ACTION" in
    status)
        echo "=========================================="
        echo "W&B Sweep Agent Status"
        echo "=========================================="
        
        # Check for running agents
        AGENTS=$(pgrep -f "wandb agent" || true)
        if [ -z "$AGENTS" ]; then
            echo "No running agents found"
        else
            echo "Running agents (PIDs):"
            for pid in $AGENTS; do
                if ps -p $pid > /dev/null; then
                    CMD=$(ps -p $pid -o args= | head -n1)
                    echo "  PID $pid: $CMD"
                fi
            done
        fi
        
        echo ""
        echo "GPU Usage:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
        
        echo ""
        echo "Recent logs:"
        if [ -d "$LOG_DIR" ]; then
            ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -5 || echo "  No log files found"
        fi
        ;;
        
    logs)
        PATTERN=${2:-"*"}
        echo "Showing logs matching: $PATTERN"
        echo "=========================================="
        
        if [ -d "$LOG_DIR" ]; then
            for log in "$LOG_DIR"/*${PATTERN}*.log; do
                if [ -f "$log" ]; then
                    echo ""
                    echo "=== $(basename $log) ==="
                    tail -20 "$log"
                fi
            done
        else
            echo "Log directory not found: $LOG_DIR"
        fi
        ;;
        
    tail)
        AGENT_ID=${2:-"*"}
        echo "Tailing logs for agent: $AGENT_ID"
        echo "Press Ctrl+C to stop"
        echo "=========================================="
        
        if [ -d "$LOG_DIR" ]; then
            # Find most recent log for agent
            LOG_FILE=$(ls -t "$LOG_DIR"/agent_${AGENT_ID}_*.log 2>/dev/null | head -1)
            if [ -n "$LOG_FILE" ]; then
                tail -f "$LOG_FILE"
            else
                echo "No log found for agent $AGENT_ID"
                echo "Available logs:"
                ls "$LOG_DIR"/*.log 2>/dev/null || echo "  None"
            fi
        fi
        ;;
        
    stop)
        SWEEP_ID=${2:-""}
        if [ -z "$SWEEP_ID" ]; then
            echo "Error: SWEEP_ID required for stop action"
            echo "Usage: $0 stop SWEEP_ID"
            exit 1
        fi
        
        echo "Stopping agents for sweep: $SWEEP_ID"
        pkill -f "wandb agent.*$SWEEP_ID" || echo "No matching agents found"
        ;;
        
    stopall)
        echo "Stopping ALL wandb agents..."
        pkill -f "wandb agent" || echo "No agents found"
        ;;
        
    clean)
        echo "Cleaning old logs..."
        if [ -d "$LOG_DIR" ]; then
            # Remove logs older than 7 days
            find "$LOG_DIR" -name "*.log" -mtime +7 -delete
            find "$LOG_DIR" -name "*.sh" -mtime +7 -delete
            find "$LOG_DIR" -name "pids_*.txt" -mtime +7 -delete
            echo "Cleaned old files"
        fi
        ;;
        
    *)
        echo "Usage: $0 [ACTION] [ARGS]"
        echo ""
        echo "Actions:"
        echo "  status          - Show running agents and GPU usage (default)"
        echo "  logs [PATTERN]  - Show recent logs (optional pattern filter)"
        echo "  tail [AGENT_ID] - Tail logs for specific agent (default: most recent)"
        echo "  stop SWEEP_ID   - Stop all agents for a specific sweep"
        echo "  stopall         - Stop ALL wandb agents"
        echo "  clean           - Remove old log files (>7 days)"
        echo ""
        echo "Examples:"
        echo "  $0                    # Show status"
        echo "  $0 logs               # Show all recent logs"
        echo "  $0 logs agent_0       # Show logs for agent 0"
        echo "  $0 tail 2             # Tail logs for agent 2"
        echo "  $0 stop abc123def     # Stop specific sweep"
        exit 1
        ;;
esac 