# Multi-GPU Training for Consistency Lens

This document describes the multi-GPU training implementation for the Consistency Lens project.

## Overview

The project now supports distributed data-parallel training across multiple GPUs, enabling faster training through parallel processing of batches.

## Key Features

- **Data Parallelism**: Each GPU processes different batches of data
- **Automatic Scaling**: Effective batch size scales with number of GPUs
- **SLURM Integration**: Works seamlessly with SLURM job schedulers
- **Flexible Launch**: Supports both torchrun and SLURM environments
- **Backward Compatible**: Single-GPU training still works as before

## Quick Start

### Using submit_with_config.sh (Recommended)

```bash
# Train with 8 GPUs
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml num_gpus_train=8

# Different GPU counts for dumping vs training
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml num_gpus=8 num_gpus_train=4
```

### Manual Launch

```bash
# Using torchrun
torchrun --nproc_per_node=8 scripts/01_train_distributed.py \
    --config-path=../conf --config-name=gpt2_frozen

# Using launcher script
./scripts/launch_distributed_train.sh \
    --config-path=../conf --config-name=gpt2_frozen --num-gpus=8
```

## Implementation Details

### Architecture

1. **Distributed Initialization**: Each process initializes with its rank and connects via NCCL
2. **Model Wrapping**: Models are wrapped with DistributedDataParallel (DDP)
3. **Data Distribution**: DistributedSampler ensures each GPU sees different data
4. **Gradient Synchronization**: DDP automatically synchronizes gradients across GPUs
5. **Metrics Aggregation**: Loss and metrics are averaged across all processes

### Files Modified/Added

- `scripts/01_train_distributed.py`: Main distributed training script
- `scripts/submit_with_config.sh`: Updated to support multi-GPU training
- `scripts/launch_distributed_train.sh`: Launcher for distributed training
- `lens/training/distributed.py`: Enhanced distributed utilities
- `conf/distributed.yaml`: Configuration for distributed settings

### Key Functions

- `init_distributed()`: Initialize distributed process group
- `setup_distributed_models()`: Wrap models with DDP
- `get_dataloader_for_distributed()`: Create distributed data loaders
- `sync_metrics()`: Synchronize metrics across processes

## Configuration

Add to your config YAML:

```yaml
# Include distributed defaults
defaults:
  - distributed

# Custom settings
distributed:
  backend: nccl
  mixed_precision:
    enabled: true
    dtype: bfloat16
```

## Performance Tips

1. **Batch Size**: Keep per-GPU batch size moderate (4-8)
2. **Gradient Accumulation**: Adjust based on memory constraints
3. **Network**: Fast interconnect (NVLink) improves scaling
4. **GPUs**: Best results with homogeneous GPU types

## SLURM Allocation

When using SLURM, the submit script now intelligently handles GPU allocation:

### Single Node Optimization (Recommended)
For best performance, all GPUs should be on the same node:
- **Automatic**: Script requests `--nodes=1` for multi-GPU training
- **8 GPUs**: Automatically requests `--exclusive` for full node access
- **NVLink**: Ensures high-bandwidth GPU communication

### Examples
```bash
# Let SLURM find the best node with 8 GPUs
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml num_gpus_train=8

# Request specific node (only if you know it has enough GPUs)
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml num_gpus_train=4 nodelist=node001

# Warning shown if multi-node list provided for distributed training
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml num_gpus_train=8 nodelist=node001,node002
# ^ This will show a warning and ignore the nodelist for optimal single-node allocation
```

## Troubleshooting

### Common Issues

1. **NCCL Errors**: Ensure all GPUs are visible and on same node
2. **Memory**: Each GPU needs full model copy - reduce batch size if OOM
3. **Hanging**: Check if GPUs are split across nodes (use `nvidia-smi`)
4. **Uneven Performance**: Verify all GPUs have same specs (no mixing GPU types)

### Debug Commands

```bash
# Test basic distributed functionality
python scripts/test_distributed_training.py --test basic

# Test with 2 GPUs
torchrun --nproc_per_node=2 scripts/test_distributed_training.py --test all

# Run integration tests
./scripts/test_distributed_integration.sh
```

## Future Enhancements

- Model parallelism for larger models
- DeepSpeed integration for additional optimizations
- FSDP (Fully Sharded Data Parallel) support
- Gradient checkpointing for memory efficiency