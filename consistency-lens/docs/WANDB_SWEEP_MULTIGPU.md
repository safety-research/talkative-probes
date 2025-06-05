# W&B Sweeps with Multi-GPU Support

This document describes how to run W&B hyperparameter sweeps with multi-GPU distributed training support.

## Overview

The updated sweep infrastructure supports flexible GPU allocation from 1-8 GPUs per sweep agent. The system automatically detects the number of allocated GPUs and uses either single-GPU training (`01_train.py`) or distributed training (`01_train_distributed.py`) as appropriate.

## Prerequisites

1. **Data Preparation**: Activations must be dumped and data pretokenized
   ```bash
   ./scripts/submit_with_config.sh config=conf/your_config.yaml
   # Cancel the training job once dumping completes
   ```

2. **Environment Setup**: Ensure you have the required dependencies
   ```bash
   make  # If not already done
   ```

## Quick Start

### 1. Initialize a W&B Sweep

```bash
cd consistency-lens
wandb sweep sweeps/lr_sweep_multigpu_example.yaml
# Note the sweep ID from output (e.g., user/project/abc123def)
```

### 2. Launch Sweep Agents

**Option A: Using the helper script (recommended)**
```bash
# Launch 4 single-GPU agents
./scripts/launch_sweep_multigpu.sh user/project/abc123def 4 1

# Launch 2 agents with 4 GPUs each
./scripts/launch_sweep_multigpu.sh user/project/abc123def 2 4

# Launch 1 agent with 8 GPUs
./scripts/launch_sweep_multigpu.sh user/project/abc123def 1 8
```

**Option B: Manual submission**
```bash
# Single GPU agent
sbatch scripts/slurm_sweep_agent.sh user/project/abc123def

# 4-GPU agent
sbatch --gres=gpu:4 scripts/slurm_sweep_agent.sh user/project/abc123def

# Multiple 2-GPU agents
for i in {1..4}; do
    sbatch --gres=gpu:2 scripts/slurm_sweep_agent.sh user/project/abc123def
done
```

## Configuration

### Sweep Configuration

Add `num_gpus` to your sweep parameters to control GPU allocation:

```yaml
parameters:
  num_gpus:
    values: [1, 2, 4, 8]  # Test different GPU counts
  
  # Adjust batch size based on GPU count
  batch_size:
    values: [8, 16, 32]  # Per-GPU batch size
```

### Effective Batch Size

The effective batch size scales with the number of GPUs:
- Effective batch size = `batch_size` × `num_gpus` × `gradient_accumulation_steps`

### Example Configurations

**Memory-intensive model (large batch sizes)**
```yaml
parameters:
  num_gpus:
    value: 4
  batch_size:
    value: 8  # 32 effective with 4 GPUs
  gradient_accumulation_steps:
    value: 2  # Total effective: 64
```

**Compute-intensive sweep (many small experiments)**
```yaml
parameters:
  num_gpus:
    value: 1
  batch_size:
    values: [4, 8, 16, 32]
  learning_rate:
    values: [1e-5, 1e-4, 1e-3, 1e-2]
```

## Monitoring

### SLURM Jobs
```bash
# View all your jobs
squeue -u $USER

# View detailed job info
scontrol show job <job_id>

# Check job output
tail -f logs/sweep_<job_id>.out
```

### W&B Dashboard
- Navigate to your sweep URL: `https://wandb.ai/<sweep_id>`
- Monitor metrics across different GPU configurations
- Compare efficiency: samples/sec, GPU utilization

## Best Practices

1. **GPU Scaling**
   - Start with single GPU to verify configuration
   - Scale up for final training or large sweeps
   - Monitor GPU utilization to ensure efficient scaling

2. **Batch Size Tuning**
   - Adjust per-GPU batch size based on memory usage
   - Use gradient accumulation for larger effective batches
   - Consider memory vs. compute trade-offs

3. **Cost Efficiency**
   - Use fewer GPUs with larger batch sizes for parameter sweeps
   - Use more GPUs for final training runs
   - Monitor samples/sec to find optimal configuration

4. **Distributed Training Considerations**
   - Communication overhead increases with more GPUs
   - Larger batch sizes are more efficient for multi-GPU
   - Consider gradient accumulation vs. data parallelism trade-offs

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` per GPU
   - Increase `gradient_accumulation_steps`
   - Use fewer GPUs with larger memory

2. **Slow Multi-GPU Training**
   - Check network bandwidth between GPUs
   - Increase batch size to amortize communication
   - Verify GPUs are on the same node

3. **Port Conflicts**
   - The script automatically randomizes MASTER_PORT
   - If issues persist, manually set: `export MASTER_PORT=<port>`

### Debug Commands

```bash
# Check GPU visibility
echo $CUDA_VISIBLE_DEVICES

# Test GPU count detection
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Verify distributed setup
torchrun --nproc_per_node=2 --nnodes=1 scripts/01_train_distributed.py --help
```

## Advanced Usage

### Custom GPU Allocation

For non-standard GPU configurations:

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,2,4,6 sbatch scripts/slurm_sweep_agent.sh <sweep_id>

# Override detected GPU count
sbatch --export=ALL,NUM_GPUS=2 --gres=gpu:2 scripts/slurm_sweep_agent.sh <sweep_id>
```

### Integration with Existing Sweeps

To add multi-GPU support to existing sweeps:

1. Update your sweep YAML to include `num_gpus` parameter
2. Ensure your config supports distributed training
3. Test with a small sweep first

## Performance Guidelines

| GPUs | Recommended Batch Size | Gradient Accumulation | Use Case |
|------|----------------------|---------------------|----------|
| 1    | 16-32               | 4-8                 | Debugging, small models |
| 2    | 8-16                | 2-4                 | Standard training |
| 4    | 4-8                 | 1-2                 | Large models, fast iteration |
| 8    | 2-4                 | 1                   | Maximum throughput |

## Example Workflow

```bash
# 1. Prepare data
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml

# 2. Create sweep config
cat > my_sweep.yaml << EOF
program: scripts/wandb_sweep_train_only.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  config:
    value: conf/gpt2_frozen.yaml
  learning_rate:
    min: 1e-5
    max: 1e-2
  num_gpus:
    values: [1, 2, 4]
  batch_size:
    values: [8, 16]
EOF

# 3. Initialize sweep
SWEEP_ID=$(wandb sweep my_sweep.yaml 2>&1 | grep "Created sweep with ID" | awk '{print $NF}')

# 4. Launch agents
./scripts/launch_sweep_multigpu.sh $SWEEP_ID 8 2  # 8 agents with 2 GPUs each

# 5. Monitor
watch squeue -u $USER
```