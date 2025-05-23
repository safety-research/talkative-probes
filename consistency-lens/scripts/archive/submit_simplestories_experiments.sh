#!/bin/bash
# Master script to submit SimpleStories experiments to SLURM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
EXPERIMENT="${1:-all}"
DRY_RUN="${2:-false}"
FORCE_REDUMP="${3:-false}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== SimpleStories Consistency Lens SLURM Submission ===${NC}"

# Function to submit a job
submit_job() {
    local script_name=$1
    local experiment_name=$2
    
    if [ "$DRY_RUN" == "true" ]; then
        echo -e "${YELLOW}[DRY RUN] Would submit: sbatch $script_name${NC}"
    else
        echo -e "${GREEN}Submitting $experiment_name experiment...${NC}"
        if [ "$FORCE_REDUMP" == "true" ]; then
            JOB_ID=$(sbatch --export=ALL,FORCE_REDUMP=true $script_name | awk '{print $4}')
        else
            JOB_ID=$(sbatch $script_name | awk '{print $4}')
        fi
        echo -e "${GREEN}Submitted with Job ID: $JOB_ID${NC}"
        echo "$JOB_ID:$experiment_name" >> submitted_jobs_ss.log
    fi
}

# Create logs directory
mkdir -p ../logs

# Initialize job log
if [ "$DRY_RUN" != "true" ]; then
    echo "# SimpleStories Experiment Jobs - $(date)" > submitted_jobs_ss.log
fi

case $EXPERIMENT in
    "frozen")
        submit_job "slurm_simplestories_frozen.sh" "SimpleStories Frozen (t_text=10)"
        ;;
    "unfreeze")
        submit_job "slurm_simplestories_unfreeze.sh" "SimpleStories Progressive Unfreezing (t_text=10)"
        ;;
    "all")
        echo -e "${BLUE}Submitting all SimpleStories experiments...${NC}"
        submit_job "slurm_simplestories_frozen.sh" "SimpleStories Frozen (t_text=10)"
        sleep 2
        submit_job "slurm_simplestories_unfreeze.sh" "SimpleStories Progressive Unfreezing (t_text=10)"
        ;;
    *)
        echo "Usage: $0 [frozen|unfreeze|all] [dry-run] [force-redump]"
        echo ""
        echo "Experiments:"
        echo "  frozen   - SimpleStories with frozen base model"
        echo "  unfreeze - SimpleStories with progressive unfreezing (frozen for 1st epoch)"
        echo "  all      - Submit both experiments"
        echo ""
        echo "Options:"
        echo "  dry-run      - Show what would be submitted without actually submitting"
        echo "  force-redump - Force re-dump of activations even if they exist"
        echo ""
        echo "Examples:"
        echo "  $0 frozen                    # Submit frozen experiment"
        echo "  $0 all                       # Submit all experiments"
        echo "  $0 all dry-run               # See what would be submitted"
        echo "  $0 frozen false force-redump # Force re-dump activations"
        echo ""
        echo "Key differences:"
        echo "  - Frozen: Base model stays frozen throughout training"
        echo "  - Unfreeze: Base model frozen for 1st epoch, then unfrozen"
        echo "  - Both use t_text=10 (width-10 explanations)"
        echo "  - Tau decay starts after 10 epochs in unfreezing experiment"
        exit 1
        ;;
esac

if [ "$DRY_RUN" != "true" ] && [ -f submitted_jobs_ss.log ]; then
    echo -e "\n${BLUE}=== Submitted Jobs ===${NC}"
    cat submitted_jobs_ss.log
    echo -e "\n${BLUE}Monitor with: squeue -u $USER${NC}"
    echo -e "${BLUE}Check logs in: logs/simplestories_*.out${NC}"
fi

echo -e "\n${GREEN}Done!${NC}"