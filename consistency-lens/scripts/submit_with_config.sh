#!/bin/bash
# Extensible wrapper script for submitting activation dumping followed by training
# Takes a config YAML file and extracts all settings from it
# Works on both SLURM and non-SLURM environments

set -e

# Change to script directory and then to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Running from $(pwd)"
CONSISTENCY_LENS_DIR=$(pwd)
echo "CONSISTENCY_LENS_DIR: $CONSISTENCY_LENS_DIR"

export OMP_NUM_THREADS=16
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true


# Parse arguments
CONFIG_FILE="${1}"
FORCE_REDUMP="${2:-false}"
FORCE_RETOKENIZE="${3:-false}"
RESUME_CHECKPOINT="${4:-}"
WANDB_RESUME_ID="${5:-}"
NODELIST="${6:-330702be7061}"  # Default to current cluster node (only used for SLURM)
NUM_GPUS="${7:-}"  # Number of GPUs to use (auto-detect if not specified)
RUN_SUFFIX="${8:-}"  # Optional suffix to add to run name

# Check if config file provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified"
    echo "Usage: $0 <config.yaml> [force-redump] [resume_checkpoint] [wandb_resume_id] [nodelist] [num_gpus] [run_suffix]"
    echo ""
    echo "Examples:"
    echo "  $0 conf/simplestories_frozen.yaml                    # New experiment"
    echo "  $0 conf/gpt2_frozen.yaml force-redump                # Force re-dump activations"
    echo "  $0 conf/simplestories_frozen.yaml false outputs/ckpt_step_1000.pt  # Resume"
    echo "  $0 conf/simplestories_frozen.yaml false \"\" \"\" \"\" \"\" exp1  # Add suffix 'exp1'"
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
    
    # Use the already extracted PRETOKENIZE_NUM_PROC variable
    local num_proc="${PRETOKENIZE_NUM_PROC:-8}"
    
    if [ "$USE_SLURM" = true ]; then
        echo -e "${YELLOW}Submitting pretokenization job via SLURM...${NC}" >&2
        echo "Config: $config" >&2
        echo "Using $num_proc CPUs for pretokenization" >&2
        
        # Create a simple sbatch script inline
        local result=$(sbatch --parsable \
            --job-name="${job_name}-pretok" \
            --gres=gpu:0 \
            --nodelist="$NODELIST" \
            --cpus-per-task=$num_proc \
            --time=2:00:00 \
            --output=logs/pretokenize_%j.out \
            --error=logs/pretokenize_%j.err \
            --wrap="bash -c 'cd $CONSISTENCY_LENS_DIR && source .venv/bin/activate && export OMP_NUM_THREADS=1 && export TOKENIZERS_PARALLELISM=false && python scripts/pretokenize_dataset.py --config-path=$CONSISTENCY_LENS_DIR/$(dirname $config) --config-name=$(basename $config .yaml)'" 2>&1)
        
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
    local dependency=$4
    
    if [ "$USE_SLURM" = true ]; then
        echo -e "${YELLOW}Submitting activation dumping job via SLURM...${NC} with dependency $dependency" >&2
        echo "Config: $config, Layer: $layer" >&2
        
        # Build sbatch command
        local sbatch_cmd="sbatch --parsable --job-name=${job_name}-dump --gres=gpu:$NUM_GPUS --nodelist=$NODELIST --export=SLURM_NODELIST='$NODELIST'"
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            sbatch_cmd="$sbatch_cmd --dependency=afterok:$dependency"
        fi
        
        # Always use pretokenized for efficiency
        local extra_args="pretokenized"
        
        # Submit job and capture both stdout and stderr
        echo "Submitting dump job..." >&2
        local result=$($sbatch_cmd scripts/slurm_dump_activations_flexible.sh "$config" "$layer" "$extra_args" 2>&1)
        
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
        echo "Config: $config, Layer: $layer, GPUs: $NUM_GPUS" >&2
        
        # In non-SLURM mode, dependency="completed" means the previous step succeeded
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            echo -e "${RED}ERROR: Previous step (pretokenization) did not complete successfully${NC}" >&2
            exit 1
        fi
        
        # Run activation dumping directly (always pretokenized)
        export NUM_GPUS="$NUM_GPUS"
        if ./scripts/launch_multigpu_dump.sh "$config" "" "" "$layer" >&2; then
            echo -e "${GREEN}Activation dumping completed successfully${NC}" >&2
            echo "completed"
        else
            echo -e "${RED}ERROR: Activation dumping failed${NC}" >&2
            exit 1
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
    local run_suffix=$6
    
    if [ "$USE_SLURM" = true ]; then
        echo -e "${YELLOW}Submitting training job via SLURM...${NC}" >&2
        echo "Config: $config_file" >&2
        
        # Show resume info
        if [ -n "$resume_checkpoint" ]; then
            echo -e "${GREEN}Will resume from checkpoint: $resume_checkpoint${NC}" >&2
        fi
        if [ -n "$wandb_resume_id" ]; then
            echo -e "${GREEN}Will resume WandB run: $wandb_resume_id${NC}" >&2
        fi
        
        # Build the training command
        local config_dir="$(dirname "$config_file")"
        local config_name="$(basename "$config_file" .yaml)"
        if [[ ! "$config_dir" = /* ]]; then
            config_dir="$(pwd)/$config_dir"
        fi
        
        local train_cmd="cd $CONSISTENCY_LENS_DIR && . .venv/bin/activate && python scripts/01_train.py --config-path=$config_dir --config-name=$config_name"
        if [ -n "$resume_checkpoint" ]; then
            train_cmd="$train_cmd resume=\"$resume_checkpoint\""
        fi
        if [ -n "$wandb_resume_id" ]; then
            train_cmd="$train_cmd wandb_resume_id=\"$wandb_resume_id\""
        fi
        if [ -n "$run_suffix" ]; then
            train_cmd="$train_cmd run_suffix=\"$run_suffix\""
        fi
        
        # Submit job using array to avoid quote issues
        local sbatch_args=(
            --parsable
            --job-name="${job_name}-train"
            --gres=gpu:1
            --nodelist="$NODELIST"
            --time=24:00:00
            --output="logs/${job_name}_train_%j.out"
            --error="logs/${job_name}_train_%j.err"
        )
        
        # Add environment variables
        local export_vars="ALL,TORCHINDUCTOR_CACHE_DIR=${HOME}/.cache/torchinductor"
        if [ -n "$resume_checkpoint" ]; then
            export_vars="$export_vars,RESUME_CHECKPOINT=$resume_checkpoint"
        fi
        if [ -n "$wandb_resume_id" ]; then
            export_vars="$export_vars,WANDB_RESUME_ID=$wandb_resume_id"
        fi
        sbatch_args+=(--export="$export_vars")
        
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            sbatch_args+=(--dependency=afterok:"$dependency")
            echo -e "${YELLOW}Will start after job $dependency completes${NC}" >&2
        fi
        
        sbatch_args+=(--wrap "$train_cmd")
        
        # Submit job and capture result
        local result=$(sbatch "${sbatch_args[@]}" 2>&1)
        
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
        export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
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
        if [ -n "$run_suffix" ]; then
            train_cmd="$train_cmd run_suffix=\"$run_suffix\""
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

# Determine job name from config file and experiment type
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)

# Create a more informative job name based on config
if [[ "$CONFIG_BASENAME" == *"unfreeze"* ]]; then
    FREEZE_TYPE="unfreeze"
elif [[ "$FREEZE_ENABLED" == "true" ]]; then
    FREEZE_TYPE="prog"
else
    FREEZE_TYPE="frozen"
fi

# Extract key info for job name
MODEL_SHORT="${MODEL_NAME##*/}"  # Get last part after /
if [[ "$MODEL_SHORT" == "SimpleStories-5M" ]]; then
    MODEL_SHORT="5M"
elif [[ "$MODEL_SHORT" == "gpt2" ]]; then
    MODEL_SHORT="GPT2"
fi

# Build job name: config_dataset_model_layer_freeze[_suffix]
# Use the config basename as prefix for better tracking
JOB_NAME="${CONFIG_BASENAME}_${DATASET_NAME}_${MODEL_SHORT}_L${LAYER}_${FREEZE_TYPE}"
if [ -n "$RUN_SUFFIX" ]; then
    JOB_NAME="${JOB_NAME}_${RUN_SUFFIX}"
fi

# Always use pretokenization for efficiency (5x faster)
# The config setting is ignored - we always pretokenize
USE_PRETOKENIZED="true"
echo -e "${BLUE}Using pretokenization for 5x faster dumping${NC}"

# Check if activations exist
if check_activations "$MODEL_NAME" "$LAYER" "$OUTPUT_DIR" && [ "$FORCE_REDUMP" != "true" ]; then
    echo -e "${GREEN}Activations already exist, submitting training only${NC}"
    train_job=$(submit_train_job "$JOB_NAME" "" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID" "$CONFIG_FILE" "$RUN_SUFFIX")
else
    echo -e "${YELLOW}Activations not found or force redump requested${NC}"
    # Only force pretokenization if FORCE_RETOKENIZE is true
    if [ "$FORCE_RETOKENIZE" = "true" ]; then
        pretok_job=$(submit_pretokenize_job "$CONFIG_FILE" "$JOB_NAME")
        dump_job=$(submit_dump_job "$CONFIG_FILE" "$LAYER" "$JOB_NAME" "$pretok_job")
    else
        dump_job=$(submit_dump_job "$CONFIG_FILE" "$LAYER" "$JOB_NAME")
    fi
    train_job=$(submit_train_job "$JOB_NAME" "$dump_job" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID" "$CONFIG_FILE" "$RUN_SUFFIX")
fi

# Summary
echo -e "\n${BLUE}=== Job Summary ===${NC}"
echo "Config: $CONFIG_FILE"
echo "Config Name: $CONFIG_BASENAME"
echo "Job Name: $JOB_NAME"
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_NAME (${MODEL_SHORT})"
echo "Layer: $LAYER"
echo "Freeze: $FREEZE_TYPE"
echo "Environment: $([ "$USE_SLURM" = true ] && echo "SLURM" || echo "Direct execution")"
echo "GPUs: $NUM_GPUS"
if [ "$USE_SLURM" = true ]; then
    echo "Nodelist: $NODELIST"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume Checkpoint: $(basename "$RESUME_CHECKPOINT")"
fi
if [ -n "$WANDB_RESUME_ID" ]; then
    echo "WandB Resume ID: $WANDB_RESUME_ID"
fi
if [ -n "$RUN_SUFFIX" ]; then
    echo "Run Suffix: $RUN_SUFFIX"
fi
echo ""
echo "Job Pipeline:"
if [ -n "${pretok_job:-}" ]; then
    if [ "$USE_SLURM" = true ]; then
        echo "  1. Pretokenization: Job $pretok_job"
    else
        echo "  1. Pretokenization: $pretok_job"
    fi
fi
if [ -n "${dump_job:-}" ]; then
    if [ "$USE_SLURM" = true ]; then
        echo "  2. Activation Dumping: Job $dump_job"
    else
        echo "  2. Activation Dumping: $dump_job"
    fi
fi
if [ "$USE_SLURM" = true ]; then
    echo "  3. Training: Job $train_job"
    echo -e "\n${BLUE}Monitor jobs:${NC} squeue -u $USER"
    echo -e "${BLUE}Cancel all:${NC} scancel $train_job"
else
    echo "  3. Training: $train_job"
    echo -e "\n${BLUE}All steps completed${NC}"
fi

# Save job IDs to log with detailed info
log_entry="$(date '+%Y-%m-%d %H:%M:%S'): $JOB_NAME"
log_entry="$log_entry [${DATASET_NAME}/${MODEL_SHORT}/L${LAYER}/${FREEZE_TYPE}]"
log_entry="$log_entry - config:$CONFIG_FILE"
if [ -n "${pretok_job:-}" ]; then
    log_entry="$log_entry pretok:$pretok_job"
fi
log_entry="$log_entry dump:${dump_job:-none} train:$train_job"
if [ -n "$RESUME_CHECKPOINT" ]; then
    log_entry="$log_entry resume:$(basename "$RESUME_CHECKPOINT")"
fi
if [ -n "$WANDB_RESUME_ID" ]; then
    log_entry="$log_entry wandb:$WANDB_RESUME_ID"
fi
echo "$log_entry" >> submitted_jobs_with_deps.log