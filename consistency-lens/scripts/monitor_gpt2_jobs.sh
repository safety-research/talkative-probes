#!/bin/bash
# Monitor GPT-2 experiment jobs

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== GPT-2 Experiment Job Monitor ===${NC}"
echo "Time: $(date)"
echo ""

# Show current jobs
echo -e "${GREEN}Current Jobs:${NC}"
squeue -u $USER --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

echo -e "\n${GREEN}Job Details:${NC}"
# Get job IDs from squeue
JOB_IDS=$(squeue -u $USER -h -o "%i")

for JOB_ID in $JOB_IDS; do
    JOB_NAME=$(squeue -j $JOB_ID -h -o "%j")
    if [[ $JOB_NAME == gpt2-* ]]; then
        echo -e "\n${YELLOW}Job $JOB_ID ($JOB_NAME):${NC}"
        scontrol show job $JOB_ID | grep -E "JobState|RunTime|StartTime|NodeList"
    fi
done

# Show recent logs
echo -e "\n${GREEN}Recent Log Files:${NC}"
LOG_DIR="../logs"
if [ -d "$LOG_DIR" ]; then
    ls -lt $LOG_DIR/gpt2_*.out 2>/dev/null | head -5
fi

# Check for errors in recent logs
echo -e "\n${GREEN}Checking for errors in recent logs:${NC}"
if [ -d "$LOG_DIR" ]; then
    for log in $(ls -t $LOG_DIR/gpt2_*.err 2>/dev/null | head -3); do
        if [ -s "$log" ]; then
            echo -e "${RED}Errors found in $log:${NC}"
            tail -n 10 "$log"
            echo "---"
        fi
    done
fi

# Show training progress from most recent log
echo -e "\n${GREEN}Training Progress (most recent):${NC}"
LATEST_LOG=$(ls -t $LOG_DIR/gpt2_*.out 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    echo "From: $LATEST_LOG"
    # Look for training progress indicators
    tail -n 50 "$LATEST_LOG" | grep -E "(step|epoch|loss|Starting|complete)" | tail -10
fi

# Resource usage
echo -e "\n${GREEN}GPU Usage on Nodes:${NC}"
for JOB_ID in $JOB_IDS; do
    JOB_NAME=$(squeue -j $JOB_ID -h -o "%j")
    if [[ $JOB_NAME == gpt2-* ]]; then
        NODE=$(squeue -j $JOB_ID -h -o "%N")
        if [ ! -z "$NODE" ]; then
            echo "Job $JOB_ID on $NODE:"
            # This would need ssh access to nodes, commenting out for safety
            # ssh $NODE "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv"
        fi
    fi
done

echo -e "\n${BLUE}=== End of Monitor Report ===${NC}"