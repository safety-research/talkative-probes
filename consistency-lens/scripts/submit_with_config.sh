#!/bin/bash
# Extensible wrapper script for submitting activation dumping followed by training
# Takes a config YAML file and extracts all settings from it
# Works on both SLURM and non-SLURM environments

set -e

# Change to script directory and then to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Running from $(pwd)"

# Parse arguments
CONFIG_FILE="${1}"
FORCE_REDUMP="${2:-false}"
RESUME_CHECKPOINT="${3:-}"
WANDB_RESUME_ID="${4:-}"
NODELIST="${5:-330702be7061}"  # Default to current cluster node (only used for SLURM)
NUM_GPUS="${6:-}"  # Number of GPUs to use (auto-detect if not specified)

# Check if config file provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified"
    echo "Usage: $0 <config.yaml> [force-redump] [resume_checkpoint] [wandb_resume_id] [nodelist] [num_gpus]"
    echo ""
    echo "Examples:"
    echo "  $0 conf/simplestories_frozen.yaml                    # New experiment"
    echo "  $0 conf/gpt2_frozen.yaml force-redump                # Force re-dump activations"
    echo "  $0 conf/simplestories_frozen.yaml false outputs/ckpt_step_1000.pt  # Resume"
    echo ""
    echo "Available configs:"
    ls -1 conf/*.yaml | grep -E "(simplestories|gpt2)" | sed 's/^/  /'
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract settings from config using Python helper
echo "Extracting settings from $CONFIG_FILE..."
# Activate virtual environment if it exists
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
if ! source <(python scripts/extract_config_settings.py "$CONFIG_FILE" --all); then
    echo "Error: Failed to extract settings from config file"
    exit 1
fi

# Verify we got the required settings
if [ -z "$MODEL_NAME" ] || [ -z "$LAYER" ]; then
    echo "Error: Could not extract model_name or layer from config"
    exit 1
fi

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
else
    # If NUM_GPUS is specified and CUDA_VISIBLE_DEVICES is set, use the lesser
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        IFS=',' read -ra DEV_ARR <<< "$CUDA_VISIBLE_DEVICES"
        CUDA_GPU_COUNT=${#DEV_ARR[@]}
        if [ "$CUDA_GPU_COUNT" -lt "$NUM_GPUS" ]; then
            NUM_GPUS=$CUDA_GPU_COUNT
        else
            # Re-set CUDA_VISIBLE_DEVICES to only use the first NUM_GPUS devices
            NEW_CUDA_DEVICES=""
            for ((i=0; i<NUM_GPUS; i++)); do
                if [ $i -gt 0 ]; then
                    NEW_CUDA_DEVICES="$NEW_CUDA_DEVICES,"
                fi
                NEW_CUDA_DEVICES="$NEW_CUDA_DEVICES${DEV_ARR[$i]}"
            done
            export CUDA_VISIBLE_DEVICES="$NEW_CUDA_DEVICES"
        fi
    fi
fi
echo "Using $NUM_GPUS GPUs"

# Display extracted settings
echo ""
echo "=== Extracted Configuration ==="
echo "Model: $MODEL_NAME"
echo "Layer: $LAYER"
echo "Dataset: $DATASET_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "Use Pretokenized: $USE_PRETOKENIZED (always true for 5x speedup)"
echo "Freeze Schedule Enabled: $FREEZE_ENABLED"
if [ "$FREEZE_ENABLED" = "true" ]; then
    echo "Unfreeze At: $UNFREEZE_AT"
fi
echo "=============================="
echo ""

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
    local output_dir=$3
    
    # Clean model name for directory paths
    local model_name_clean=$(echo "$model_name" | tr '/' '_')
    
    # List of possible activation directory patterns based on your data tree
    local activation_dirs=(
        "$output_dir"  # e.g., ./data/SimpleStories_train
        "./data/${model_name_clean}/layer_${layer}/${DATASET_NAME}_train"  # e.g., ./data/SimpleStories_SimpleStories-5M/layer_5/SimpleStories_train
        "./data/${model_name_clean}/layer_${layer}/${DATASET_NAME}_test"   # Check test too
        "./data/activations/${DATASET_NAME}/${model_name}/layer_${layer}/${DATASET_NAME}_train"  # Your tree structure
        "./data/activations/${DATASET_NAME}/${model_name}/layer_${layer}/${DATASET_NAME}_test"
    )
    
    for dir in "${activation_dirs[@]}"; do
        echo "Checking: $dir"
        if [ -d "$dir" ]; then
            # Check for rank_* subdirectories (multi-GPU dumps) or .pt files
            if [ -n "$(find "$dir" -maxdepth 2 -name "*.pt" -o -type d -name "rank_*" 2>/dev/null | head -1)" ]; then
                echo -e "${GREEN}Found activations in $dir${NC}"
                return 0  # Activations exist
            fi
        fi
    done
    
    return 1  # Activations don't exist
}

# Function to determine training script if not already set
# NO LONGER USED - we run training directly with the config

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
            --wrap="bash -c 'cd /home/kitf/talkative-probes/consistency-lens && source .venv/bin/activate && python scripts/pretokenize_dataset.py --config-path=/home/kitf/talkative-probes/consistency-lens/$(dirname $config) --config-name=$(basename $config .yaml)'" 2>&1)
        
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
        echo "Submitting dump job..." >&2
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
        
        # In non-SLURM mode, dependency="completed" means the previous step succeeded
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            echo -e "${RED}ERROR: Previous step (pretokenization) did not complete successfully${NC}" >&2
            exit 1
        fi
        
        # Run activation dumping directly
        export NUM_GPUS="$NUM_GPUS"
        if [ "$use_pretokenized" = "true" ]; then
            if ./scripts/launch_multigpu_dump_pretokenized.sh "$config" "" "" "$layer" >&2; then
                echo -e "${GREEN}Activation dumping completed successfully${NC}" >&2
                echo "completed"
            else
                echo -e "${RED}ERROR: Activation dumping failed${NC}" >&2
                exit 1
            fi
        else
            if ./scripts/launch_multigpu_dump_optimized.sh "$config" "" "" "$layer" >&2; then
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
    local job_name=$1
    local dependency=$2
    local resume_checkpoint=$3
    local wandb_resume_id=$4
    local config_file=$5
    
    if [ "$USE_SLURM" = true ]; then
        echo -e "${YELLOW}Submitting training job via SLURM...${NC}" >&2
        echo "Config: $config_file" >&2
        
        # Build environment variables
        local env_vars="--export=ALL"
        if [ -n "$resume_checkpoint" ]; then
            env_vars="$env_vars,RESUME_CHECKPOINT='$resume_checkpoint'"
            echo -e "${GREEN}Will resume from checkpoint: $resume_checkpoint${NC}" >&2
        fi
        if [ -n "$wandb_resume_id" ]; then
            env_vars="$env_vars,WANDB_RESUME_ID='$wandb_resume_id'"
            echo -e "${GREEN}Will resume WandB run: $wandb_resume_id${NC}" >&2
        fi
        
        # Build the training command
        local config_dir="$(dirname "$config_file")"
        local config_name="$(basename "$config_file" .yaml)"
        if [[ ! "$config_dir" = /* ]]; then
            config_dir="$(pwd)/$config_dir"
        fi
        
        local train_cmd="cd /home/kitf/talkative-probes/consistency-lens && source .venv/bin/activate && python scripts/01_train.py --config-path=$config_dir --config-name=$config_name"
        if [ -n "$resume_checkpoint" ]; then
            train_cmd="$train_cmd resume='$resume_checkpoint'"
        fi
        if [ -n "$wandb_resume_id" ]; then
            train_cmd="$train_cmd wandb_resume_id='$wandb_resume_id'"
        fi
        
        # Build sbatch command
        local sbatch_cmd="sbatch --parsable --job-name=${job_name}-train"
        sbatch_cmd="$sbatch_cmd --gres=gpu:1 --nodelist=$NODELIST"
        sbatch_cmd="$sbatch_cmd --time=24:00:00"
        sbatch_cmd="$sbatch_cmd --output=logs/${job_name}_train_%j.out"
        sbatch_cmd="$sbatch_cmd --error=logs/${job_name}_train_%j.err"
        sbatch_cmd="$sbatch_cmd $env_vars"
        
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            sbatch_cmd="$sbatch_cmd --dependency=afterok:$dependency"
            echo -e "${YELLOW}Will start after job $dependency completes${NC}" >&2
        fi
        
        sbatch_cmd="$sbatch_cmd --wrap=\"$train_cmd\""
        
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
        
        echo -e "${GREEN}Training job submitted with ID: $train_job_id${NC}" >&2
        echo "$train_job_id"
    else
        echo -e "${YELLOW}Running training directly...${NC}" >&2
        
        # In non-SLURM mode, dependency="completed" means the previous step succeeded
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
        
        echo -e "${GREEN}Running training with config: $config_file${NC}" >&2
        
        # Run training with resume parameters if set
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
echo -e "${BLUE}=== Running experiment from $CONFIG_FILE ===${NC}"

# Determine job name from config file
JOB_NAME=$(basename "$CONFIG_FILE" .yaml)

# Always use pretokenization for efficiency (5x faster)
# The config setting is ignored - we always pretokenize
USE_PRETOKENIZED="true"
echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"

# Check if activations exist
if check_activations "$MODEL_NAME" "$LAYER" "$OUTPUT_DIR" && [ "$FORCE_REDUMP" != "true" ]; then
    echo -e "${GREEN}Activations already exist, submitting training only${NC}"
    train_job=$(submit_train_job "$JOB_NAME" "" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID" "$CONFIG_FILE")
else
    echo -e "${YELLOW}Activations not found or force redump requested${NC}"
    # Always pretokenize first for efficiency
    pretok_job=$(submit_pretokenize_job "$CONFIG_FILE" "$JOB_NAME")
    dump_job=$(submit_dump_job "$CONFIG_FILE" "$LAYER" "$JOB_NAME" true "$pretok_job")
    train_job=$(submit_train_job "$JOB_NAME" "$dump_job" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID" "$CONFIG_FILE")
fi

# Summary
echo -e "\n${BLUE}=== Job Summary ===${NC}"
echo "Config: $CONFIG_FILE"
echo "Experiment: $JOB_NAME"
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
echo "$(date): $JOB_NAME (from $CONFIG_FILE) - dump:${dump_job:-none} train:$train_job$resume_info" >> submitted_jobs_with_deps.log