#!/bin/bash
# Wrapper script to handle W&B agent crashes and auto-restart
# This helps with the "Broken pipe" error that occurs after ~20 minutes

set -euo pipefail

SWEEP_ID=$1
COUNT=${2:-1}
MAX_RETRIES=${3:-10}  # Maximum number of retries per agent

if [ -z "$SWEEP_ID" ]; then
    echo "Error: SWEEP_ID required"
    echo "Usage: $0 SWEEP_ID [COUNT] [MAX_RETRIES]"
    exit 1
fi

echo "Starting W&B sweep agent with auto-restart"
echo "Sweep ID: $SWEEP_ID"
echo "Count per attempt: $COUNT"
echo "Max retries: $MAX_RETRIES"

# Track total runs completed
TOTAL_RUNS=0
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo ""
    echo "=========================================="
    echo "Attempt $((RETRY_COUNT + 1)) of $MAX_RETRIES"
    echo "Time: $(date)"
    echo "=========================================="
    
    # Run the agent and capture exit code
    set +e
    wandb agent --count $COUNT $SWEEP_ID
    EXIT_CODE=$?
    set -e
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Agent completed successfully"
        TOTAL_RUNS=$((TOTAL_RUNS + COUNT))
    else
        echo "Agent crashed with exit code $EXIT_CODE"
        
        # Check if it was a broken pipe error
        if [ $EXIT_CODE -eq 1 ] || [ $EXIT_CODE -eq 141 ]; then
            echo "Likely a broken pipe error - will retry after delay"
            sleep 30  # Wait 30 seconds before retry
        else
            echo "Unknown error - exiting"
            exit $EXIT_CODE
        fi
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    # Small delay between attempts
    sleep 10
done

echo ""
echo "=========================================="
echo "Completed $TOTAL_RUNS runs across $RETRY_COUNT attempts"
echo "=========================================="