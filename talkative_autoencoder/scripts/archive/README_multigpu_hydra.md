# Multi-GPU Activation Dumping with Hydra

This directory contains scripts for multi-GPU activation dumping using Hydra configuration.

## Scripts

- `00_dump_activations_multigpu.py`: The main Python script using Hydra for configuration
- `launch_multigpu_dump_hydra.sh`: General launcher for multi-GPU dumping
- `launch_multigpu_dump_pretokenized_hydra.sh`: Specialized launcher for pretokenized data

## Usage Examples

### Basic Usage

```bash
# Use default configuration from conf/config.yaml
./scripts/launch_multigpu_dump_hydra.sh

# Override specific parameters
./scripts/launch_multigpu_dump_hydra.sh \
    layer_l=7 \
    activation_dumper.num_samples=50000 \
    activation_dumper.batch_size=2048
```

### Using Pretokenized Data (Recommended for Performance)

```bash
# Use pretokenized dataset (much faster!)
./scripts/launch_multigpu_dump_pretokenized_hydra.sh

# Override parameters for pretokenized run
./scripts/launch_multigpu_dump_pretokenized_hydra.sh \
    layer_l=10 \
    activation_dumper.batch_size=4096 \
    pretokenized_path=data/pretokenized/MyDataset
```

### GPU Configuration

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/launch_multigpu_dump_hydra.sh

# Adjust CPU threads per process
OMP_NUM_THREADS=32 ./scripts/launch_multigpu_dump_hydra.sh
```

### Common Hydra Overrides

```bash
# Change model and tokenizer
./scripts/launch_multigpu_dump_hydra.sh \
    model_name="meta-llama/Llama-2-7b-hf" \
    tokenizer_name="meta-llama/Llama-2-7b-hf"

# Change dataset
./scripts/launch_multigpu_dump_hydra.sh \
    activation_dumper.hf_dataset_name="openwebtext" \
    activation_dumper.hf_split="train"

# Process entire dataset
./scripts/launch_multigpu_dump_hydra.sh \
    activation_dumper.num_samples=-1 \
    activation_dumper.val_num_samples=-1

# Custom output directories
./scripts/launch_multigpu_dump_hydra.sh \
    activation_dumper.output_dir="./data/activations/custom_train" \
    activation_dumper.val_output_dir="./data/activations/custom_val"
```

### SLURM Usage

The scripts automatically detect SLURM environments. Example SLURM job:

```bash
#!/bin/bash
#SBATCH --job-name=dump_activations
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00

module load cuda/11.8

./scripts/launch_multigpu_dump_pretokenized_hydra.sh \
    layer_l=12 \
    activation_dumper.num_samples=1000000
```

## Configuration

All configuration is managed through Hydra. The main config file is at `conf/config.yaml`.

Key configuration sections:
- `model_name`: The model to use for activation extraction
- `tokenizer_name`: Tokenizer (defaults to model_name if not specified)
- `layer_l`: Target layer for activation extraction
- `activation_dumper`: Contains all dumping-specific settings
  - `num_samples`: Number of samples to process (-1 for all)
  - `seq_len`: Sequence length
  - `batch_size`: Batch size per GPU
  - `use_hf_dataset`: Whether to use HuggingFace datasets
  - `hf_dataset_name`: Name of HuggingFace dataset
  - `output_dir`: Where to save activations

## Performance Tips

1. **Use Pretokenized Data**: The pretokenized workflow is significantly faster
2. **Batch Size**: Larger batch sizes improve throughput. Try 2048-4096 for pretokenized data
3. **CPU Threads**: Adjust `OMP_NUM_THREADS` based on your CPU cores
4. **Storage**: Use fast SSD storage for output directories

## Troubleshooting

- If you see NCCL errors, ensure all GPUs are visible and accessible
- For out-of-memory errors, reduce batch_size
- Check that your dataset exists and is accessible
- Ensure the output directory has sufficient space 