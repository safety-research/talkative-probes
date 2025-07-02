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
export HYDRA_FULL_ERROR=1


# Default values
CONFIG_FILE=""
FORCE_REDUMP="false"
FORCE_RETOKENIZE="false"
RESUME_CHECKPOINT=""
WANDB_RESUME_ID=""
NODELIST="" # Default to empty, letting SLURM choose a node (only used for SLURM)
NUM_GPUS=""  # Number of GPUs to use (auto-detect if not specified)
NUM_GPUS_TRAIN="1"  # Number of GPUs for training (defaults to 1)
USE_DISTRIBUTED="false"  # Whether to use distributed training
RUN_SUFFIX=""  # Optional suffix to add to run name
HYDRA_OVERRIDES=""
ALWAYS_DISTRIBUTED="true"
NICE_JOB="false" # Default for nice job (requeueable SLURM job)

# Store the original command for logging
export SUBMIT_SCRIPT_COMMAND="$0 $@"

# Parse Hydra-style arguments
for arg in "$@"; do
    case $arg in
        config=*)
            CONFIG_FILE="${arg#*=}"
            ;;
        force_redump=*)
            FORCE_REDUMP="${arg#*=}"
            ;;
        force_retokenize=*)
            FORCE_RETOKENIZE="${arg#*=}"
            ;;
        resume_checkpoint=*)
            RESUME_CHECKPOINT="${arg#*=}"
            ;;
        wandb_resume_id=*)
            WANDB_RESUME_ID="${arg#*=}"
            ;;
        nodelist=*)
            NODELIST="${arg#*=}"
            ;;
        num_gpus=*)
            NUM_GPUS="${arg#*=}"
            ;;
        num_gpus_train=*)
            NUM_GPUS_TRAIN="${arg#*=}"
            # Auto-enable distributed if more than 1 GPU requested
            if [ "${arg#*=}" -gt 1 ]; then
                USE_DISTRIBUTED="true"
            fi
            ;;
        use_distributed=*)
            USE_DISTRIBUTED="${arg#*=}"
            ;;
        run_suffix=*)
            RUN_SUFFIX="${arg#*=}"
            ;;
        nice=*) # New argument for nice/requeueable jobs
            NICE_JOB="${arg#*=}"
            ;;
        # Special handling for positional config file (backwards compatibility)
        *.yaml)
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$arg"
            else
                # Treat as Hydra override
                HYDRA_OVERRIDES="$HYDRA_OVERRIDES $arg"
            fi
            ;;
        *)
            # All other arguments are Hydra overrides
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES $arg"
            ;;
    esac
done

# Trim leading/trailing whitespace from HYDRA_OVERRIDES
HYDRA_OVERRIDES=$(echo "$HYDRA_OVERRIDES" | xargs)

# Check if config file provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified"
    echo "Usage: $0 config=<config.yaml> [options...]"
    echo ""
    echo "Options (all use Hydra-style key=value syntax):"
    echo "  config=<path>             Config file (required, or just pass <config.yaml> as first arg)"
    echo "  force_redump=true         Force re-dump activations even if they exist"
    echo "  force_retokenize=true     Force re-tokenize dataset"
    echo "  resume_checkpoint=<path>  Resume from checkpoint"
    echo "  wandb_resume_id=<id>      Resume specific WandB run"
    echo "  nodelist=<nodes>          SLURM nodes to use (comma-separated)"
    echo "  num_gpus=<n>              Number of GPUs for dumping (auto-detected if not set)"
    echo "  num_gpus_train=<n>        Number of GPUs for training (default: 1)"
    echo "  use_distributed=true      Force distributed training (auto-enabled if num_gpus_train>1)"
    echo "  run_suffix=<suffix>       Suffix to add to run name"
    echo "  nice=true                 Submit as a low-priority, requeueable SLURM job (sets --requeue, --qos=preemptable, --nice, --signal)"
    echo ""
    echo "Examples:"
    echo "  $0 conf/simplestories_frozen.yaml                    # New experiment (backwards compatible)"
    echo "  $0 config=conf/simplestories_frozen.yaml             # New experiment (Hydra style)"
    echo "  $0 config=conf/gpt2_frozen.yaml force_redump=true    # Force re-dump activations"
    echo "  $0 config=conf/simplestories_frozen.yaml resume_checkpoint=outputs/ckpt_step_1000.pt"
    echo "  $0 config=conf/gpt2_frozen.yaml learning_rate=1e-3 batch_size=16  # With Hydra overrides"
    echo "  $0 config=conf/gpt2_frozen.yaml num_gpus_train=8                  # Multi-GPU training"
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
# Ensure environment is set up
source scripts/ensure_env.sh
ulimit -n 65536
if ! source <(uv_run python scripts/extract_config_settings.py "$CONFIG_FILE" --all $HYDRA_OVERRIDES); then
    echo "Error: Failed to extract settings from config file"
    exit 1
fi

# New variable from extract_config_settings.py
ON_THE_FLY_ENABLED="${ON_THE_FLY_ENABLED:-false}"

# Verify we got the required settings
if [ -z "$MODEL_NAME" ]; then
    echo "Error: Could not extract model_name from config"
    exit 1
fi

# Layer is not needed for on-the-fly, but is needed for dumping
if [ "$ON_THE_FLY_ENABLED" != "true" ] && [ -z "$LAYER" ]; then
    echo "Error: Could not extract layer from config (required for activation dumping)"
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
echo "Using $NUM_GPUS GPUs for dumping and $NUM_GPUS_TRAIN GPUs for training"

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
            --cpus-per-task=$num_proc \
            --time=2:00:00 \
            --output=logs/pretokenize_%j.out \
            --error=logs/pretokenize_%j.err \
            --export="ALL,SUBMIT_SCRIPT_COMMAND=$SUBMIT_SCRIPT_COMMAND" \
            --wrap="bash -c 'cd $CONSISTENCY_LENS_DIR && source scripts/ensure_env.sh && export OMP_NUM_THREADS=1 && export TOKENIZERS_PARALLELISM=false && uv_run python scripts/pretokenize_dataset.py --config-path=$CONSISTENCY_LENS_DIR/$(dirname $config) --config-name=$(basename $config .yaml)'" 2>&1)
        
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
        
        # Ensure logs directory exists
        mkdir -p logs
        
        # Run pretokenization directly with logging
        # Get absolute path to config directory
        local config_dir="$(pwd)/$(dirname "$config")"
        local log_out="logs/${job_name}_pretok_direct.out"
        local log_err="logs/${job_name}_pretok_direct.err"
        
        echo -e "${BLUE}Logs: $log_out and $log_err${NC}" >&2
        
        if uv_run python scripts/pretokenize_dataset.py --config-path="$config_dir" --config-name=$(basename "$config" .yaml) > "$log_out" 2> "$log_err"; then
            echo -e "${GREEN}Pretokenization completed successfully${NC}" >&2
            echo "completed"
        else
            echo -e "${RED}ERROR: Pretokenization failed${NC}" >&2
            echo -e "${RED}Check logs: $log_out and $log_err${NC}" >&2
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
        echo -e "${YELLOW}Submitting activation dumping job via SLURM...${NC} with dependency <$dependency>" >&2
        echo "Config: $config, Layer: $layer" >&2
        
        # Build sbatch command
        local sbatch_cmd="sbatch --parsable --job-name=${job_name}-dump --gres=gpu:$NUM_GPUS"
        if [ -n "$dependency" ] && [ "$dependency" != "completed" ]; then
            echo -e "${YELLOW}Adding dependency: <$dependency>${NC}" >&2
            sbatch_cmd="$sbatch_cmd --dependency=afterok:$dependency"
        fi
        
        # Always use pretokenized for efficiency
        local extra_args="pretokenized"
        
        # Submit job and capture both stdout and stderr
        echo "Submitting dump job..." >&2
        echo "Config: $config" >&2
        echo "Extra args: $extra_args" >&2
        echo "command to be run: $sbatch_cmd scripts/slurm_dump_activations_flexible.sh "$config" "$layer" "$extra_args" "$NUM_GPUS" 2>&1" >&2
        local result=$($sbatch_cmd scripts/slurm_dump_activations_flexible.sh "$config" "$layer" "$extra_args" "$NUM_GPUS" 2>&1)
        
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
        
        # Ensure logs directory exists
        mkdir -p logs
        
        # Run activation dumping directly (always pretokenized) with logging
        export NUM_GPUS="$NUM_GPUS"
        local log_out="logs/${job_name}_dump_direct.out"
        local log_err="logs/${job_name}_dump_direct.err"
        
        echo -e "${BLUE}Logs: $log_out and $log_err${NC}" >&2
        
        if ./scripts/launch_multigpu_dump.sh "$config_dir" "" "" "$layer" > "$log_out" 2> "$log_err"; then
            echo -e "${GREEN}Activation dumping completed successfully${NC}" >&2
            echo "completed"
        else
            echo -e "${RED}ERROR: Activation dumping failed${NC}" >&2
            echo -e "${RED}Check logs: $log_out and $log_err${NC}" >&2
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
    local nice_flag=$7 # Added nice_flag argument
    
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
        
        # Choose training script based on distributed mode
        local train_script="01_train.py"
        if [ "$USE_DISTRIBUTED" = "true" ] || [ "$NUM_GPUS_TRAIN" -gt 1 ] || [ "$ALWAYS_DISTRIBUTED" = "true" ]; then
            train_script="01_train_distributed.py"
        fi
        
        # Build command based on distributed mode
        local base_cmd="cd $CONSISTENCY_LENS_DIR && source scripts/ensure_env.sh && ulimit -n 65536 && "
        if [ "$USE_DISTRIBUTED" = "true" ] || [ "$NUM_GPUS_TRAIN" -gt 1 ] || [ "$ALWAYS_DISTRIBUTED" = "true" ]; then
            # Use torchrun for distributed training with random port to avoid conflicts
            # Generate random port between 29500-29999
            local random_port=$((29500 + RANDOM % 500))
            echo "Using random port: $random_port"
            
            base_cmd="${base_cmd}uv_run torchrun --nproc_per_node=$NUM_GPUS_TRAIN --master_port=$random_port scripts/$train_script --config-path=$config_dir --config-name=$config_name"
        else
            # Regular single-GPU training
            base_cmd="${base_cmd}uv_run python scripts/$train_script --config-path=$config_dir --config-name=$config_name"
        fi
        if [ -n "$resume_checkpoint" ]; then
            base_cmd="$base_cmd resume=\"$resume_checkpoint\""
        fi
        if [ -n "$wandb_resume_id" ]; then
            base_cmd="$base_cmd wandb_resume_id=\"$wandb_resume_id\""
        fi
        if [ -n "$run_suffix" ]; then
            base_cmd="$base_cmd run_suffix=\"$run_suffix\""
        fi
        # Add any additional Hydra overrides
        if [ -n "$HYDRA_OVERRIDES" ]; then
            base_cmd="$base_cmd $HYDRA_OVERRIDES"
        fi
        
        # Wrap in bash -c for proper shell execution
        local train_cmd="bash -c '$base_cmd'"
        
        # Submit job using array to avoid quote issues
        local sbatch_args=(
            --parsable
            --job-name="${job_name}-train"
            --time=168:00:00 # Consider if nice jobs need different default time
            --output="logs/${job_name}_train_%j.out"
            --error="logs/${job_name}_train_%j.err"
        )
        
        # Configure GPU allocation based on number requested
        if [ "$NUM_GPUS_TRAIN" -gt 1 ]; then
            # Multi-GPU: ensure all GPUs are on the same node
            sbatch_args+=(--gres=gpu:$NUM_GPUS_TRAIN)
            sbatch_args+=(--nodes=1)  # Force single node
            sbatch_args+=(--ntasks-per-node=$NUM_GPUS_TRAIN)  # One task per GPU
            
            # Request exclusive node for 8 GPUs (full node)
            if [ "$NUM_GPUS_TRAIN" -eq 8 ]; then
                sbatch_args+=(--exclusive)
                echo -e "${YELLOW}Requesting exclusive node access for 8 GPU training${NC}" >&2
            else
                echo -e "${GREEN}Requesting $NUM_GPUS_TRAIN GPUs on a single node${NC}" >&2
            fi
            
            # Only use nodelist if specified and it's a single node
            if [ -n "$NODELIST" ] && [ "$NODELIST" != "$(hostname)" ]; then
                if [[ "$NODELIST" == *","* ]]; then
                    echo -e "${RED}WARNING: Multi-node nodelist specified but distributed training requires single node${NC}" >&2
                    echo -e "${RED}Ignoring nodelist for optimal performance. Use SLURM scheduler to find suitable node.${NC}" >&2
                else
                    sbatch_args+=(--nodelist="$NODELIST")
                fi
            fi
        else
            # Single GPU: simpler allocation
            sbatch_args+=(--gres=gpu:1)
            if [ -n "$NODELIST" ] && [ "$NODELIST" != "$(hostname)" ]; then
                sbatch_args+=(--nodelist="$NODELIST")
            fi
        fi
        
        # Add environment variables
        local export_vars="ALL,TORCHINDUCTOR_CACHE_DIR=${HOME}/.cache/torchinductor,SUBMIT_SCRIPT_COMMAND=$SUBMIT_SCRIPT_COMMAND"
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

        # Add SLURM options for nice/requeueable jobs
        if [[ "$nice_flag" =~ ^[0-9]+$ ]] || [ "$nice_flag" = "true" ]; then
            echo -e "${YELLOW}Configuring SLURM job as low-priority and requeueable.${NC}" >&2
            sbatch_args+=(--requeue --qos=preemptable --open-mode=append)
            if [[ "$nice_flag" =~ ^[0-9]+$ ]]; then
                sbatch_args+=(--nice="$nice_flag")
            else
                sbatch_args+=(--nice=10000)
            fi
            sbatch_args+=(--signal=B:TERM@120) #Request SIGTERM 120 seconds before preemption/kill, to allow checkpointing
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
        
        # Ensure logs directory exists
        mkdir -p logs
        
        # Run training with resume parameters if set
        local config_dir="$(dirname "$config_file")"
        local config_name="$(basename "$config_file" .yaml)"
        
        # Make config directory absolute if it's relative
        if [[ ! "$config_dir" = /* ]]; then
            config_dir="$(pwd)/$config_dir"
        fi
        
        # Choose training script and command based on distributed mode
        local train_script="01_train.py"
        if [ "$USE_DISTRIBUTED" = "true" ] || [ "$NUM_GPUS_TRAIN" -gt 1 ] || [ "$ALWAYS_DISTRIBUTED" = "true" ]; then
            train_script="01_train_distributed.py"
        fi
        
        # Build command based on distributed mode
        if [ "$USE_DISTRIBUTED" = "true" ] || [ "$NUM_GPUS_TRAIN" -gt 1 ] || [ "$ALWAYS_DISTRIBUTED" = "true" ]; then
            # Use torchrun for distributed training
            random_port=$((29500 + RANDOM % 500))
            echo "Using random port: $random_port"
            local train_cmd="uv_run torchrun --nproc_per_node=$NUM_GPUS_TRAIN --master_port=$random_port scripts/$train_script --config-path=$config_dir --config-name=$config_name"
            echo -e "${GREEN}Running distributed training with $NUM_GPUS_TRAIN GPUs${NC}" >&2
        else
            # Regular single-GPU training
            local train_cmd="uv_run python scripts/$train_script --config-path=$config_dir --config-name=$config_name"
        fi
        if [ -n "$resume_checkpoint" ]; then
            train_cmd="$train_cmd resume=\"$resume_checkpoint\""
        fi
        if [ -n "$wandb_resume_id" ]; then
            train_cmd="$train_cmd wandb_resume_id=\"$wandb_resume_id\""
        fi
        if [ -n "$run_suffix" ]; then
            train_cmd="$train_cmd run_suffix=\"$run_suffix\""
        fi
        # Add any additional Hydra overrides
        if [ -n "$HYDRA_OVERRIDES" ]; then
            train_cmd="$train_cmd $HYDRA_OVERRIDES"
        fi
        timestamp=$(date +%Y%m%d_%H%M%S)
        local log_out="logs/${job_name}_train_direct_${timestamp}d.out"
        local log_err="logs/${job_name}_train_direct_${timestamp}d.err"
        
        echo -e "${BLUE}Logs: $log_out and $log_err${NC}" >&2
        
        if eval "$train_cmd" > "$log_out" 2> "$log_err"; then
            echo -e "${GREEN}Training completed successfully${NC}" >&2
            echo "completed"
        else
            echo -e "${RED}ERROR: Training failed${NC}" >&2
            echo -e "${RED}Check logs: $log_out and $log_err${NC}" >&2
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
if [ "$ON_THE_FLY_ENABLED" = "true" ]; then
    echo -e "${GREEN}On-the-fly generation enabled, skipping activation check/dump and submitting training only${NC}"
    train_job=$(submit_train_job "$JOB_NAME" "" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID" "$CONFIG_FILE" "$RUN_SUFFIX" "$NICE_JOB")
elif check_activations "$MODEL_NAME" "$LAYER" "$OUTPUT_DIR" && [ "$FORCE_REDUMP" != "true" ]; then
    echo -e "${GREEN}Activations already exist, submitting training only${NC}"
    train_job=$(submit_train_job "$JOB_NAME" "" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID" "$CONFIG_FILE" "$RUN_SUFFIX" "$NICE_JOB")
else
    echo -e "${YELLOW}Activations not found or force redump requested${NC}"
    # Only force pretokenization if FORCE_RETOKENIZE is true
    if [ "$FORCE_RETOKENIZE" = "true" ]; then
        pretok_job=$(submit_pretokenize_job "$CONFIG_FILE" "$JOB_NAME")
        dump_job=$(submit_dump_job "$CONFIG_FILE" "$LAYER" "$JOB_NAME" "$pretok_job")
    else
        dump_job=$(submit_dump_job "$CONFIG_FILE" "$LAYER" "$JOB_NAME")
    fi
    train_job=$(submit_train_job "$JOB_NAME" "$dump_job" "$RESUME_CHECKPOINT" "$WANDB_RESUME_ID" "$CONFIG_FILE" "$RUN_SUFFIX" "$NICE_JOB")
fi

# Summary
echo -e "\n${BLUE}=== Job Summary ===${NC}"
echo "Command: $0 $@"
echo "Config: $CONFIG_FILE"
echo "Config Name: $CONFIG_BASENAME"
echo "Job Name: $JOB_NAME"
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_NAME (${MODEL_SHORT})"
echo "Layer: $LAYER"
echo "Freeze: $FREEZE_TYPE"
echo "Environment: $([ "$USE_SLURM" = true ] && echo "SLURM" || echo "Direct execution")"
echo "GPUs (dumping): $NUM_GPUS"
echo "GPUs (training): $NUM_GPUS_TRAIN"
if [ "$USE_DISTRIBUTED" = "true" ] || [ "$NUM_GPUS_TRAIN" -gt 1 ]; then
    echo "Training mode: Distributed (DDP)"
else  
    echo "Training mode: Single GPU"
fi
if [ "$NICE_JOB" = "true" ]; then
    echo "SLURM Nice Job: Yes (requeueable, low priority) ${NICE_JOB}"
fi
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
if [ -n "$HYDRA_OVERRIDES" ]; then
    echo "Hydra Overrides: $HYDRA_OVERRIDES"
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
log_entry="$log_entry - cmd:[$0 $@]"
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