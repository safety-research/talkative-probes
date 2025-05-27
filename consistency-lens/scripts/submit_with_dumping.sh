#!/bin/bash
# Wrapper script to submit activation dumping followed by training with proper dependencies
# Works on both SLURM and non-SLURM environments

set -e

# Change to script directory and then to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Running from $(pwd)"

# Parse arguments
EXPERIMENT="${1}"
FORCE_REDUMP="${2:-false}"
RESUME_CHECKPOINT="${3:-}"
WANDB_RESUME_ID="${4:-}"
NODELIST="${5:-330702be7061}"  # Default to current cluster node (only used for SLURM)
NUM_GPUS="${6:-}"  # Number of GPUs to use (auto-detect if not specified)

# Detect if SLURM is available (not necessarily if we're in a SLURM job)
# Can be overridden with FORCE_DIRECT=true environment variable
if [ "${FORCE_DIRECT:-false}" = "true" ]; then
    USE_SLURM=false
    echo "Forced direct execution mode (FORCE_DIRECT=true)"
elif command -v sbatch >/dev/null 2>&1; then
    USE_SLURM=true
    echo "SLURM environment detected"
else
    USE_SLURM=false
    echo "Non-SLURM environment detected"
fi

# Auto-detect GPU count if not specified
if [ -z "$NUM_GPUS" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        IFS=',' read -ra DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
        NUM_GPUS=${#DEV_ARR[@]}
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    else
        NUM_GPUS=1
    fi
fi
echo "Using $NUM_GPUS GPUs"

# Check if experiment name provided
if [ -z "$EXPERIMENT" ]; then
    echo "Error: No experiment specified"
    echo "Usage: $0 <experiment> [force-redump] [resume_checkpoint] [wandb_resume_id] [nodelist] [num_gpus]"
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
    
    # Check multiple possible activation locations
    local model_name_clean=$(echo "$model_name" | tr '/' '_')
    local activation_dirs=(
        "./data/activations/${model_name_clean}/layer_${layer}/${dataset}"  # New clean format
        "./data/activations/${model_name}/layer_${layer}/${dataset}"        # Existing format with slashes
        "./data/activations/${dataset}"                                      # Direct dataset path
    )
    
    for activation_dir in "${activation_dirs[@]}"; do
        echo "Checking activations in $activation_dir"
        if [ -d "$activation_dir" ]; then
            # Check if directory has content (including rank_* subdirectories)
            if [ -n "$(find "$activation_dir" -name "*.pt" -o -name "rank_*" 2>/dev/null | head -1)" ]; then
                echo "Found activations in $activation_dir"
                return 0  # Activations exist
            fi
        fi
    done
    
    return 1  # Activations don't exist
}

# Function to submit pretokenization job
submit_pretokenize_job() {
    local config=$1
    local job_name=$2
    
    if [ "$USE_SLURM" = true ]; then
        echo -e "${YELLOW}Submitting pretokenization job via SLURM...${NC}" >&2
        echo "Config: $config" >&2
        
        # Create a simple sbatch script inline
        local result=$(sbatch --parsable \
            --job-name="${job_name}-pretok" \
            --gres=gpu:0 \
            --nodelist="$NODELIST" \
            --time=2:00:00 \
            --output=logs/pretokenize_%j.out \
            --error=logs/pretokenize_%j.err \
            --wrap="bash -c 'cd /home/kitf/talkative-probes/consistency-lens && source .venv/bin/activate && python scripts/pretokenize_dataset.py --config-path=/home/kitf/talkative-probes/consistency-lens/conf --config-name=$(basename $config .yaml)'" 2>&1)
        
        if [[ $? -ne 0 ]]; then
            echo -e "${RED}ERROR: Failed to submit pretokenization job${NC}" >&2
            echo -e "${RED}Error message: $result${NC}" >&2
            exit 1
        fi
        
        local pretok_job_id="$result"
        echo -e "${GREEN}Pretokenization job submitted with ID: $pretok_job_id${NC}" >&2
        echo "$pretok_job_id"
    else
        echo -e "${YELLOW}Running pretokenization directly...${NC}" >&2
        echo "Config: $config" >&2
        
        # Run pretokenization directly
        # Get absolute path to config directory
        local config_dir="$(pwd)/$(dirname "$config")"
        if source .venv/bin/activate && python scripts/pretokenize_dataset.py --config-path="$config_dir" --config-name=$(basename "$config" .yaml) >&2; then
            echo -e "${GREEN}Pretokenization completed successfully${NC}" >&2
            echo "completed"
        else
            echo -e "${RED}ERROR: Pretokenization failed${NC}" >&2
            exit 1
        fi
    fi
}

# Function to submit dumping job
submit_dump_job() {
    local config=$1
    local layer=$2
    local job_name=$3
    local use_pretokenized=${4:-false}
    local dependency=$5
    
    if [ "$USE_SLURM" = true ]; then
        echo -e "${YELLOW}Submitting activation dumping job via SLURM...${NC}" >&2
        echo "Config: $config, Layer: $layer, Use pretokenized: $use_pretokenized" >&2
        
        # Build sbatch command
        local sbatch_cmd="sbatch --parsable --job-name=${job_name}-dump --export=SLURM_NODELIST='$NODELIST'"
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            sbatch_cmd="$sbatch_cmd --dependency=afterok:$dependency"
        fi
        
        # Add pretokenized flag if requested
        local extra_args=""
        if [ "$use_pretokenized" = "true" ]; then
            extra_args="pretokenized"
        fi
        
        # Submit job and capture both stdout and stderr
        echo "submitting job" >&2
        local result=$($sbatch_cmd scripts/slurm_dump_activations_minimal.sh "$config" "$layer" "$extra_args" 2>&1)
        
        # Check if submission was successful
        if [[ $? -ne 0 ]]; then
            echo -e "${RED}ERROR: Failed to submit dumping job${NC}" >&2
            echo -e "${RED}Error message: $result${NC}" >&2
            exit 1
        fi
        
        local dump_job_id="$result"
        if [[ -z "$dump_job_id" || ! "$dump_job_id" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}ERROR: Invalid job ID received: '$dump_job_id'${NC}" >&2
            exit 1
        fi
        
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            echo -e "${GREEN}Dumping job submitted with ID: $dump_job_id (will start after job $dependency completes)${NC}" >&2
        else
            echo -e "${GREEN}Dumping job submitted with ID: $dump_job_id${NC}" >&2
        fi
        echo "$dump_job_id"
    else
        echo -e "${YELLOW}Running activation dumping directly...${NC}" >&2
        echo "Config: $config, Layer: $layer, Use pretokenized: $use_pretokenized, GPUs: $NUM_GPUS" >&2
        echo "DEBUG: Dependency value: '$dependency'" >&2
        
        # In non-SLURM mode, dependency="completed" means the previous step succeeded
        # Only fail if dependency exists but is NOT "completed" 
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            echo -e "${RED}ERROR: Previous step (pretokenization) did not complete successfully${NC}" >&2
            echo "DEBUG: Dependency was '$dependency', expected 'completed' or empty" >&2
            exit 1
        fi
        
        # Run activation dumping directly
        export NUM_GPUS="$NUM_GPUS"
        if [ "$use_pretokenized" = "true" ]; then
            if ./scripts/launch_multigpu_dump_pretokenized.sh "$config" "" "" "$layer"; then
                echo -e "${GREEN}Activation dumping completed successfully${NC}" >&2
                echo "completed"
            else
                echo -e "${RED}ERROR: Activation dumping failed${NC}" >&2
                exit 1
            fi
        else
            if ./scripts/launch_multigpu_dump_optimized.sh "$config" "" "" "$layer"; then
                echo -e "${GREEN}Activation dumping completed successfully${NC}" >&2
                echo "completed"
            else
                echo -e "${RED}ERROR: Activation dumping failed${NC}" >&2
                exit 1
            fi
        fi
    fi
}

# Function to submit training job with optional dependency
submit_train_job() {
    local script=$1
    local job_name=$2
    local dependency=$3
    local resume_checkpoint=$4
    local wandb_resume_id=$5
    
    if [ "$USE_SLURM" = true ]; then
        # Build environment variables for resume parameters
        local env_vars=""
        if [ -n "$resume_checkpoint" ]; then
            env_vars="--export=RESUME_CHECKPOINT='$resume_checkpoint'"
            echo -e "${GREEN}Will resume from checkpoint: $resume_checkpoint${NC}" >&2
        fi
        if [ -n "$wandb_resume_id" ]; then
            if [ -n "$env_vars" ]; then
                env_vars="$env_vars,WANDB_RESUME_ID='$wandb_resume_id'"
            else
                env_vars="--export=WANDB_RESUME_ID='$wandb_resume_id'"
            fi
            echo -e "${GREEN}Will resume WandB run: $wandb_resume_id${NC}" >&2
        fi
        echo -e "${BLUE}Using nodelist: $NODELIST${NC}" >&2
        
        local sbatch_cmd
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            echo -e "${YELLOW}Submitting training job with dependency on job $dependency...${NC}" >&2
            sbatch_cmd="sbatch --parsable --dependency=afterok:$dependency --job-name=$job_name --nodelist=$NODELIST $env_vars $script"
        else
            echo -e "${YELLOW}Submitting training job...${NC}" >&2
            sbatch_cmd="sbatch --parsable --job-name=$job_name --nodelist=$NODELIST $env_vars $script"
        fi
        
        # Submit job and capture result
        local result=$($sbatch_cmd 2>&1)
        
        # Check if submission was successful
        if [[ $? -ne 0 ]]; then
            echo -e "${RED}ERROR: Failed to submit training job${NC}" >&2
            echo -e "${RED}Error message: $result${NC}" >&2
            exit 1
        fi
        
        local train_job_id="$result"
        if [[ -z "$train_job_id" || ! "$train_job_id" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}ERROR: Invalid job ID received: '$train_job_id'${NC}" >&2
            exit 1
        fi
        
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            echo -e "${GREEN}Training job submitted with ID: $train_job_id (will start after job $dependency completes)${NC}" >&2
        else
            echo -e "${GREEN}Training job submitted with ID: $train_job_id${NC}" >&2
        fi
        echo "$train_job_id"
    else
        echo -e "${YELLOW}Running training directly...${NC}" >&2
        
        # In non-SLURM mode, dependency="completed" means the previous step succeeded
        # Only fail if dependency exists but is NOT "completed"
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            echo -e "${RED}ERROR: Previous step did not complete successfully${NC}" >&2
            exit 1
        fi
        
        # Set up environment variables for resume parameters
        if [ -n "$resume_checkpoint" ]; then
            export RESUME_CHECKPOINT="$resume_checkpoint"
            echo -e "${GREEN}Will resume from checkpoint: $resume_checkpoint${NC}" >&2
        fi
        if [ -n "$wandb_resume_id" ]; then
            export WANDB_RESUME_ID="$wandb_resume_id"
            echo -e "${GREEN}Will resume WandB run: $wandb_resume_id${NC}" >&2
        fi
        
        # Extract config from script and run training directly
        local config_file=""
        case "$job_name" in
            "ss-frozen") config_file="conf/simplestories_frozen.yaml" ;;
            "ss-unfreeze") config_file="conf/simplestories_unfreeze.yaml" ;;
            "gpt2-frozen") config_file="conf/gpt2_frozen.yaml" ;;
            "gpt2-unfreeze") config_file="conf/gpt2_unfreeze.yaml" ;;
            "gpt2-pile") config_file="conf/gpt2_pile_frozen.yaml" ;;
            "gpt2-pile-unfreeze") config_file="conf/gpt2_pile_unfreeze.yaml" ;;
            *) echo -e "${RED}ERROR: Unknown job name: $job_name${NC}" >&2; exit 1 ;;
        esac
        
        echo -e "${GREEN}Running training with config: $config_file${NC}" >&2
        
        # Run training with resume parameters if set
        # The training script expects configs to be in ../conf relative to the script
        local config_dir="$(dirname "$config_file")"
        local config_name="$(basename "$config_file" .yaml)"
        
        # Make config directory absolute if it's relative
        if [[ ! "$config_dir" = /* ]]; then
            config_dir="$(pwd)/$config_dir"
        fi
        
        local train_cmd="python scripts/01_train.py --config-path=$config_dir --config-name=$config_name"
        if [ -n "$resume_checkpoint" ]; then
            train_cmd="$train_cmd resume=\"$resume_checkpoint\""
        fi
        if [ -n "$wandb_resume_id" ]; then
            train_cmd="$train_cmd wandb_resume_id=\"$wandb_resume_id\""
        fi
        
        if eval "$train_cmd"; then
            echo -e "${GREEN}Training completed successfully${NC}" >&2
            echo "completed"
        else
            echo -e "${RED}ERROR: Training failed${NC}" >&2
            exit 1
        fi
    fi
}

# Main logic
case $EXPERIMENT in
    "ss-frozen"|"simplestories-frozen")
        echo -e "${BLUE}=== SimpleStories Frozen Experiment ===${NC}"
        echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"
        
        # Extract layer from config (defaults to 5)
        LAYER=$(python -c "import yaml; cfg=yaml.safe_load(open('conf/simplestories_frozen.yaml')); print(cfg.get('layer_l', 5))" 2>/dev/null || echo "5")
        
        # Check if activations exist (using the actual path structure)
        if check_activations "SimpleStories/SimpleStories-5M" "$LAYER" "SimpleStories_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_simplestories_frozen.sh" "ss-frozen" "" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            # First pretokenize, then dump with pretokenized data
            pretok_job=$(submit_pretokenize_job "conf/simplestories_frozen.yaml" "ss-frozen")
            echo "DEBUG: pretok_job returned: '$pretok_job'" >&2
            dump_job=$(submit_dump_job "conf/simplestories_frozen.yaml" "$LAYER" "ss-frozen" true "$pretok_job")
            train_job=$(submit_train_job "scripts/slurm_simplestories_frozen.sh" "ss-frozen" "$dump_job" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        fi
        ;;
        
    "ss-unfreeze"|"simplestories-unfreeze")
        echo -e "${BLUE}=== SimpleStories Unfreeze Experiment ===${NC}"
        echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"
        
        # Extract layer from config (defaults to 5)
        LAYER=$(python -c "import yaml; cfg=yaml.safe_load(open('conf/simplestories_unfreeze.yaml')); print(cfg.get('layer_l', 5))" 2>/dev/null || echo "5")
        
        # Same activations as frozen (using the actual path structure)
        if check_activations "SimpleStories/SimpleStories-5M" "$LAYER" "SimpleStories_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_simplestories_unfreeze.sh" "ss-unfreeze" "" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            # First pretokenize, then dump with pretokenized data
            pretok_job=$(submit_pretokenize_job "conf/simplestories_unfreeze.yaml" "ss-unfreeze")
            dump_job=$(submit_dump_job "conf/simplestories_unfreeze.yaml" "$LAYER" "ss-unfreeze" true "$pretok_job")
            train_job=$(submit_train_job "scripts/slurm_simplestories_unfreeze.sh" "ss-unfreeze" "$dump_job" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        fi
        ;;
        
    "gpt2-frozen")
        echo -e "${BLUE}=== GPT-2 Frozen Experiment (OpenWebText) ===${NC}"
        echo -e "${BLUE}Training GPT-2 on OpenWebText (its original training data)${NC}"
        echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"
        
        # Extract layer from config (GPT-2 uses layer 6)
        LAYER=$(python -c "import yaml; cfg=yaml.safe_load(open('conf/gpt2_frozen.yaml')); print(cfg.get('layer_l', 6))" 2>/dev/null || echo "6")
        
        if check_activations "openai-community/gpt2" "$LAYER" "openwebtext_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_gpt2_frozen.sh" "gpt2-frozen" "" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            # First pretokenize, then dump with pretokenized data
            pretok_job=$(submit_pretokenize_job "conf/gpt2_frozen.yaml" "gpt2-frozen")
            dump_job=$(submit_dump_job "conf/gpt2_frozen.yaml" "$LAYER" "gpt2-frozen" true "$pretok_job")
            train_job=$(submit_train_job "scripts/slurm_gpt2_frozen.sh" "gpt2-frozen" "$dump_job" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        fi
        ;;
        
    "gpt2-unfreeze")
        echo -e "${BLUE}=== GPT-2 Progressive Unfreeze Experiment (OpenWebText) ===${NC}"
        echo -e "${BLUE}Training GPT-2 on OpenWebText with unfreezing after 10k steps${NC}"
        echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"
        
        # Extract layer from config (GPT-2 uses layer 6)
        LAYER=$(python -c "import yaml; cfg=yaml.safe_load(open('conf/gpt2_unfreeze.yaml')); print(cfg.get('layer_l', 6))" 2>/dev/null || echo "6")
        
        # Same activations as frozen
        if check_activations "openai-community/gpt2" "$LAYER" "openwebtext_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_gpt2_unfreeze.sh" "gpt2-unfreeze" "" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            # First pretokenize, then dump with pretokenized data
            pretok_job=$(submit_pretokenize_job "conf/gpt2_unfreeze.yaml" "gpt2-unfreeze")
            dump_job=$(submit_dump_job "conf/gpt2_unfreeze.yaml" "$LAYER" "gpt2-unfreeze" true "$pretok_job")
            train_job=$(submit_train_job "scripts/slurm_gpt2_unfreeze.sh" "gpt2-unfreeze" "$dump_job" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        fi
        ;;
        
    "gpt2-pile")
        echo -e "${BLUE}=== GPT-2 Pile Frozen Experiment ===${NC}"
        echo -e "${BLUE}Training GPT-2 on The Pile (diverse text dataset)${NC}"
        echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"
        
        # Extract layer from config (GPT-2 uses layer 6)
        LAYER=$(python -c "import yaml; cfg=yaml.safe_load(open('conf/gpt2_pile_frozen.yaml')); print(cfg.get('layer_l', 6))" 2>/dev/null || echo "6")
        
        if check_activations "openai-community/gpt2" "$LAYER" "pile_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_gpt2_pile.sh" "gpt2-pile" "" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            # First pretokenize, then dump with pretokenized data
            pretok_job=$(submit_pretokenize_job "conf/gpt2_pile_frozen.yaml" "gpt2-pile")
            dump_job=$(submit_dump_job "conf/gpt2_pile_frozen.yaml" "$LAYER" "gpt2-pile" true "$pretok_job")
            train_job=$(submit_train_job "scripts/slurm_gpt2_pile.sh" "gpt2-pile" "$dump_job" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        fi
        ;;
        
    "gpt2-pile-unfreeze")
        echo -e "${BLUE}=== GPT-2 Pile Progressive Unfreeze Experiment ===${NC}"
        echo -e "${BLUE}Training GPT-2 on The Pile with unfreezing after 1st epoch${NC}"
        echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"
        
        # Extract layer from config (GPT-2 uses layer 6)
        LAYER=$(python -c "import yaml; cfg=yaml.safe_load(open('conf/gpt2_pile_unfreeze.yaml')); print(cfg.get('layer_l', 6))" 2>/dev/null || echo "6")
        
        # Same activations as pile frozen
        if check_activations "openai-community/gpt2" "$LAYER" "pile_train" "train" && [ "$FORCE_REDUMP" != "true" ]; then
            echo -e "${GREEN}Activations already exist, submitting training only${NC}"
            train_job=$(submit_train_job "scripts/slurm_gpt2_pile_unfreeze.sh" "gpt2-pile-unfreeze" "" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        else
            echo -e "${YELLOW}Activations not found or force redump requested${NC}"
            # First pretokenize, then dump with pretokenized data
            pretok_job=$(submit_pretokenize_job "conf/gpt2_pile_unfreeze.yaml" "gpt2-pile-unfreeze")
            dump_job=$(submit_dump_job "conf/gpt2_pile_unfreeze.yaml" "$LAYER" "gpt2-pile-unfreeze" true "$pretok_job")
            train_job=$(submit_train_job "scripts/slurm_gpt2_pile_unfreeze.sh" "gpt2-pile-unfreeze" "$dump_job" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID")
        fi
        ;;
        
    *)
        echo "Usage: $0 <experiment> [force-redump] [resume_checkpoint] [wandb_resume_id] [nodelist] [num_gpus]"
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
        echo "  force-redump       - Force re-dumping even if activations exist"
        echo "  resume_checkpoint  - Path to checkpoint file to resume from"
        echo "  wandb_resume_id    - WandB run ID to resume"
        echo "  nodelist           - SLURM node list (default: 330702be7061, ignored on non-SLURM)"
        echo "  num_gpus           - Number of GPUs to use (auto-detected if not specified)"
        echo ""
        echo "Examples:"
        echo "  $0 ss-frozen                                         # New SimpleStories experiment"
        echo "  $0 gpt2-frozen                                       # New GPT-2 on OpenWebText"
        echo "  $0 gpt2-pile force-redump                            # GPT-2 on Pile, force re-dump"
        echo "  $0 ss-frozen false outputs/ckpt_step_1000.pt        # Resume from checkpoint"
        echo "  $0 ss-frozen false outputs/ckpt_step_1000.pt abc123xyz # Resume with WandB ID"
        echo "  $0 ss-frozen false \"\" \"\" node001,node002             # Use different nodes (SLURM)"
        echo "  $0 ss-frozen false \"\" \"\" \"\" 4                        # Use 4 GPUs (non-SLURM)"
        exit 1
        ;;
esac

echo -e "\n${BLUE}=== Job Summary ===${NC}"
echo "Experiment: $EXPERIMENT"
echo "Environment: $([ "$USE_SLURM" = true ] && echo "SLURM" || echo "Direct execution")"
echo "GPUs: $NUM_GPUS"
if [ "$USE_SLURM" = true ]; then
    echo "Nodelist: $NODELIST"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume Checkpoint: $RESUME_CHECKPOINT"
fi
if [ -n "$WANDB_RESUME_ID" ]; then
    echo "WandB Resume ID: $WANDB_RESUME_ID"
fi
if [ -n "${dump_job:-}" ]; then
    if [ "$USE_SLURM" = true ]; then
        echo "Dumping Job ID: $dump_job"
    else
        echo "Dumping: $dump_job"
    fi
fi
if [ "$USE_SLURM" = true ]; then
    echo "Training Job ID: $train_job"
    echo -e "\n${BLUE}Monitor with:${NC} squeue -u $USER"
    echo -e "${BLUE}Cancel with:${NC} scancel $train_job"
else
    echo "Training: $train_job"
    echo -e "\n${BLUE}Execution completed in foreground${NC}"
fi

# Save job IDs to log with resume info
resume_info=""
if [ -n "$RESUME_CHECKPOINT" ]; then
    resume_info=" resume:$RESUME_CHECKPOINT"
fi
if [ -n "$WANDB_RESUME_ID" ]; then
    resume_info="$resume_info wandb:$WANDB_RESUME_ID"
fi
echo "$(date): $EXPERIMENT - dump:${dump_job:-none} train:$train_job$resume_info" >> submitted_jobs_with_deps.log
