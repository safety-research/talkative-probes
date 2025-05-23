#!/bin/bash
# Wrapper script to submit activation dumping followed by training with proper dependencies

set -e

# Change to script directory and then to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Running from $(pwd)"

# Parse arguments
EXPERIMENT="${1}"
FORCE_REDUMP="${2:-false}"

# Check if experiment name provided
if [ -z "$EXPERIMENT" ]; then
    echo "Error: No experiment specified"
    echo "Usage: $0 <experiment> [force-redump]"
    echo "Run with no arguments to see available experiments"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to check if activations exist
check_activations() {
    local model_name=$1
    local layer=$2
    local dataset=$3
    local split=$4
    
    local activation_dir="./data/activations/${model_name}/layer_${layer}/${dataset}"
    echo "Checking activations in $activation_dir"
    
    if [ -d "$activation_dir" ] && [ -n "$(ls -A $activation_dir 2>/dev/null)" ]; then
        return 0  # Activations exist
    else
        return 1  # Activations don't exist
    fi
}

# Function to submit pretokenization job
submit_pretokenize_job() {
    local config=$1
    local job_name=$2
    
    echo -e "${YELLOW}Submitting pretokenization job...${NC}"
    echo "Config: $config"
    
    # Create a simple sbatch script inline
    local result=$(sbatch --parsable \
        --job-name="${job_name}-pretok" \
        --gres=gpu:0 \
        --nodelist=330702be7061 \
        --time=2:00:00 \
        --output=logs/pretokenize_%j.out \
        --error=logs/pretokenize_%j.err \
        --wrap="bash -c 'cd /home/kitf/talkative-probes/consistency-lens && source .venv/bin/activate && python scripts/pretokenize_dataset.py --config_path $config --num_proc 32'" 2>&1)
    
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}ERROR: Failed to submit pretokenization job${NC}"
        echo -e "${RED}Error message: $result${NC}"
        exit 1
    fi
    
    local pretok_job_id="$result"
    echo -e "${GREEN}Pretokenization job submitted with ID: $pretok_job_id${NC}"
    echo "$pretok_job_id"
}

# Function to submit dumping job
submit_dump_job() {
    local config=$1
    local layer=$2
    local job_name=$3
    local use_pretokenized=${4:-false}
    local dependency=$5
    
    echo -e "${YELLOW}Submitting activation dumping job...${NC}"
    echo "Config: $config, Layer: $layer, Use pretokenized: $use_pretokenized"
    
    # Build sbatch command
    local sbatch_cmd="sbatch --parsable --job-name=${job_name}-dump"
    if [ -n "$dependency" ]; then
        sbatch_cmd="$sbatch_cmd --dependency=afterok:$dependency"
    fi
    
    # Add pretokenized flag if requested
    local extra_args=""
    if [ "$use_pretokenized" = "true" ]; then
        extra_args="pretokenized"
    fi
    
    # Submit job and capture both stdout and stderr
    local result=$($sbatch_cmd scripts/slurm_dump_activations_minimal.sh "$config" "$layer" "$extra_args" 2>&1)
    
    # Check if submission was successful
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}ERROR: Failed to submit dumping job${NC}"
        echo -e "${RED}Error message: $result${NC}"
        exit 1
    fi
    
    local dump_job_id="$result"
    if [[ -z "$dump_job_id" || ! "$dump_job_id" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}ERROR: Invalid job ID received: '$dump_job_id'${NC}"
        exit 1
    fi
    
    if [ -n "$dependency" ]; then
        echo -e "${GREEN}Dumping job submitted with ID: $dump_job_id (will start after job $dependency completes)${NC}"
    else
        echo -e "${GREEN}Dumping job submitted with ID: $dump_job_id${NC}"
    fi
    echo "$dump_job_id"
}

# Function to submit training job with optional dependency
submit_train_job() {
    local script=$1
    local job_name=$2
    local dependency=$3
    
    local sbatch_cmd
    if [ -n "$dependency" ]; then
        echo -e "${YELLOW}Submitting training job with dependency on job $dependency...${NC}"
        sbatch_cmd="sbatch --parsable --dependency=afterok:$dependency --job-name=$job_name $script"
    else
        echo -e "${YELLOW}Submitting training job...${NC}"
        sbatch_cmd="sbatch --parsable --job-name=$job_name $script"
    fi
    
    # Submit job and capture result
    local result=$($sbatch_cmd 2>&1)
    
    # Check if submission was successful
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}ERROR: Failed to submit training job${NC}"
        echo -e "${RED}Error message: $result${NC}"
        exit 1
    fi
    
    local train_job_id="$result"
    if [[ -z "$train_job_id" || ! "$train_job_id" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}ERROR: Invalid job ID received: '$train_job_id'${NC}"
        exit 1
    fi
    
    if [ -n "$dependency" ]; then
        echo -e "${GREEN}Training job submitted with ID: $train_job_id (will start after job $dependency completes)${NC}"
    else
        echo -e "${GREEN}Training job submitted with ID: $train_job_id${NC}"
    fi
    echo "$train_job_id"
}

# Main logic
case $EXPERIMENT in
    "ss-frozen"|"simplestories-frozen")
        echo -e "${BLUE}=== SimpleStories Frozen Experiment ===${NC}"
        
        # Check if activations exist
        if check_activations "SimpleStories/SimpleStories-5M" 5 "SimpleStories_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_simplestories_frozen.sh" "ss-frozen" "")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            dump_job=$(submit_dump_job "conf/simplestories_frozen.yaml" 5 "ss-frozen")
            train_job=$(submit_train_job "scripts/slurm_simplestories_frozen.sh" "ss-frozen" "$dump_job")
        fi
        ;;
        
    "ss-unfreeze"|"simplestories-unfreeze")
        echo -e "${BLUE}=== SimpleStories Unfreeze Experiment ===${NC}"
        
        # Same activations as frozen
        if check_activations "SimpleStories/SimpleStories-5M" 5 "SimpleStories_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_simplestories_unfreeze.sh" "ss-unfreeze" "")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            dump_job=$(submit_dump_job "conf/simplestories_unfreeze.yaml" 5 "ss-unfreeze")
            train_job=$(submit_train_job "scripts/slurm_simplestories_unfreeze.sh" "ss-unfreeze" "$dump_job")
        fi
        ;;
        
    "gpt2-frozen")
        echo -e "${BLUE}=== GPT-2 Frozen Experiment (OpenWebText) ===${NC}"
        echo -e "${BLUE}Training GPT-2 on OpenWebText (its original training data)${NC}"
        echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"
        
        if check_activations "openai-community/gpt2" 6 "openwebtext_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_gpt2_frozen.sh" "gpt2-frozen" "")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            # First pretokenize, then dump with pretokenized data
            pretok_job=$(submit_pretokenize_job "conf/gpt2_frozen.yaml" "gpt2-frozen")
            dump_job=$(submit_dump_job "conf/gpt2_frozen.yaml" 6 "gpt2-frozen" true "$pretok_job")
            train_job=$(submit_train_job "scripts/slurm_gpt2_frozen.sh" "gpt2-frozen" "$dump_job")
        fi
        ;;
        
    "gpt2-unfreeze")
        echo -e "${BLUE}=== GPT-2 Progressive Unfreeze Experiment (OpenWebText) ===${NC}"
        echo -e "${BLUE}Training GPT-2 on OpenWebText with unfreezing after 10k steps${NC}"
        echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"
        
        # Same activations as frozen
        if check_activations "openai-community/gpt2" 6 "openwebtext_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_gpt2_unfreeze.sh" "gpt2-unfreeze" "")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            # First pretokenize, then dump with pretokenized data
            pretok_job=$(submit_pretokenize_job "conf/gpt2_unfreeze.yaml" "gpt2-unfreeze")
            dump_job=$(submit_dump_job "conf/gpt2_unfreeze.yaml" 6 "gpt2-unfreeze" true "$pretok_job")
            train_job=$(submit_train_job "scripts/slurm_gpt2_unfreeze.sh" "gpt2-unfreeze" "$dump_job")
        fi
        ;;
        
    "gpt2-pile")
        echo -e "${BLUE}=== GPT-2 Pile Frozen Experiment ===${NC}"
        echo -e "${BLUE}Training GPT-2 on The Pile (diverse text dataset)${NC}"
        echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"
        
        if check_activations "openai-community/gpt2" 6 "pile_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_gpt2_pile.sh" "gpt2-pile" "")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            # First pretokenize, then dump with pretokenized data
            pretok_job=$(submit_pretokenize_job "conf/gpt2_pile_frozen.yaml" "gpt2-pile")
            dump_job=$(submit_dump_job "conf/gpt2_pile_frozen.yaml" 6 "gpt2-pile" true "$pretok_job")
            train_job=$(submit_train_job "scripts/slurm_gpt2_pile.sh" "gpt2-pile" "$dump_job")
        fi
        ;;
        
    "gpt2-pile-unfreeze")
        echo -e "${BLUE}=== GPT-2 Pile Progressive Unfreeze Experiment ===${NC}"
        echo -e "${BLUE}Training GPT-2 on The Pile with unfreezing after 1st epoch${NC}"
        echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"
        
        # Same activations as pile frozen
        if check_activations "openai-community/gpt2" 6 "pile_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_gpt2_pile_unfreeze.sh" "gpt2-pile-unfreeze" "")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            # First pretokenize, then dump with pretokenized data
            pretok_job=$(submit_pretokenize_job "conf/gpt2_pile_unfreeze.yaml" "gpt2-pile-unfreeze")
            dump_job=$(submit_dump_job "conf/gpt2_pile_unfreeze.yaml" 6 "gpt2-pile-unfreeze" true "$pretok_job")
            train_job=$(submit_train_job "scripts/slurm_gpt2_pile_unfreeze.sh" "gpt2-pile-unfreeze" "$dump_job")
        fi
        ;;
        
    *)
        echo "Usage: $0 <experiment> [force-redump]"
        echo ""
        echo "SimpleStories Experiments (5M params, faster):"
        echo "  ss-frozen        - SimpleStories data, frozen base model"
        echo "  ss-unfreeze      - SimpleStories data, unfreeze after epoch 1"
        echo ""
        echo "GPT-2 Experiments (124M params, slower):"
        echo "  gpt2-frozen        - OpenWebText data, frozen base model"
        echo "  gpt2-unfreeze      - OpenWebText data, unfreeze after 1st epoch"
        echo "  gpt2-pile          - The Pile data, frozen base model"
        echo "  gpt2-pile-unfreeze - The Pile data, unfreeze after 1st epoch"
        echo ""
        echo "Datasets:"
        echo "  - SimpleStories: Simple children's stories (for 5M model)"
        echo "  - OpenWebText: GPT-2's original training data"
        echo "  - The Pile: Diverse 800GB dataset from EleutherAI"
        echo ""
        echo "Options:"
        echo "  force-redump     - Force re-dumping even if activations exist"
        echo ""
        echo "Examples:"
        echo "  $0 ss-frozen              # SimpleStories experiment"
        echo "  $0 gpt2-frozen            # GPT-2 on OpenWebText"
        echo "  $0 gpt2-pile force-redump # GPT-2 on Pile, force re-dump"
        exit 1
        ;;
esac

echo -e "\n${BLUE}=== Job Summary ===${NC}"
echo "Experiment: $EXPERIMENT"
if [ -n "${dump_job:-}" ]; then
    echo "Dumping Job ID: $dump_job"
fi
echo "Training Job ID: $train_job"
echo -e "\n${BLUE}Monitor with:${NC} squeue -u $USER"
echo -e "${BLUE}Cancel with:${NC} scancel $train_job"

# Save job IDs to log
echo "$(date): $EXPERIMENT - dump:${dump_job:-none} train:$train_job" >> submitted_jobs_with_deps.log