#!/bin/bash
# Master script to submit GPT-2 experiments to SLURM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
EXPERIMENT="${1:-all}"
DRY_RUN="${2:-false}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== GPT-2 Consistency Lens SLURM Submission ===${NC}"

# Function to submit a job
submit_job() {
    local script_name=$1
    local experiment_name=$2
    
    if [ "$DRY_RUN" == "true" ]; then
        echo -e "${YELLOW}[DRY RUN] Would submit: sbatch $script_name${NC}"
    else
        echo -e "${GREEN}Submitting $experiment_name experiment...${NC}"
        JOB_ID=$(sbatch $script_name | awk '{print $4}')
        echo -e "${GREEN}Submitted with Job ID: $JOB_ID${NC}"
        echo "$JOB_ID:$experiment_name" >> submitted_jobs.log
    fi
}

# Create logs directory
mkdir -p ../logs

# Initialize job log
if [ "$DRY_RUN" != "true" ]; then
    echo "# GPT-2 Experiment Jobs - $(date)" > submitted_jobs.log
fi

case $EXPERIMENT in
    "frozen")
        submit_job "slurm_gpt2_frozen.sh" "GPT-2 Frozen (OpenWebText)"
        ;;
    "unfreeze")
        submit_job "slurm_gpt2_unfreeze.sh" "GPT-2 Progressive Unfreezing"
        ;;
    "pile")
        submit_job "slurm_gpt2_pile.sh" "GPT-2 Frozen (The Pile)"
        ;;
    "all")
        echo -e "${BLUE}Submitting all experiments...${NC}"
        submit_job "slurm_gpt2_frozen.sh" "GPT-2 Frozen (OpenWebText)"
        sleep 2
        submit_job "slurm_gpt2_unfreeze.sh" "GPT-2 Progressive Unfreezing"
        sleep 2
        submit_job "slurm_gpt2_pile.sh" "GPT-2 Frozen (The Pile)"
        ;;
    *)
        echo "Usage: $0 [frozen|unfreeze|pile|all] [dry-run]"
        echo ""
        echo "Experiments:"
        echo "  frozen   - GPT-2 with frozen base model (OpenWebText)"
        echo "  unfreeze - GPT-2 with progressive unfreezing"
        echo "  pile     - GPT-2 with frozen base model (The Pile)"
        echo "  all      - Submit all experiments"
        echo ""
        echo "Options:"
        echo "  dry-run  - Show what would be submitted without actually submitting"
        echo ""
        echo "Examples:"
        echo "  $0 frozen           # Submit frozen experiment"
        echo "  $0 all              # Submit all experiments"
        echo "  $0 all dry-run      # See what would be submitted"
        exit 1
        ;;
esac

if [ "$DRY_RUN" != "true" ] && [ -f submitted_jobs.log ]; then
    echo -e "\n${BLUE}=== Submitted Jobs ===${NC}"
    cat submitted_jobs.log
    echo -e "\n${BLUE}Monitor with: squeue -u $USER${NC}"
fi

echo -e "\n${GREEN}Done!${NC}"