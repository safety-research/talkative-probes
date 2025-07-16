






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

















OLD DOCS



## W&B Hyperparameter Sweeps

We provide comprehensive support for hyperparameter sweeps using Weights & Biases, with optimized workflows for both local development and SLURM clusters.

### Overview

The sweep system supports:
- **Grid search** and **Bayesian optimization** methods
- **SLURM integration** following W&B's recommended `--count 1` pattern
- **Automatic data reuse** (activations and pre-tokenized datasets)
- **Direct training** (bypasses complex submission scripts for efficiency)

### Prerequisites

Before running sweeps, ensure your data is prepared:

```bash
# 1. Dump activations and pre-tokenize data (one-time setup)
./scripts/submit_with_config.sh config=conf/simplestories_frozenl4e40p1largebatch3072.yaml

# 2. Cancel the training job once dumping completes (we only need the data)
# Check SLURM queue: squeue -u $USER
# Cancel training: scancel JOB_ID
```

### Local Development Sweeps

For quick experimentation on local machines or single nodes:

```bash
# 1. Initialize the sweep
cd consistency-lens
wandb sweep sweeps/lr_sweep_simplestories_frozen.yaml

# 2. Run agents (can run multiple in parallel)
wandb agent your-entity/consistency-lens-simplestories/SWEEP_ID
```

**Example sweep configuration** (`sweeps/lr_sweep_simplestories_frozen.yaml`):
```yaml
program: scripts/wandb_sweep_wrapper.py
method: grid
metric:
  goal: minimize
  name: train/loss
parameters:
  learning_rate:
    values: [1.0e-3, 7.0e-4, 5.0e-4, 3.0e-4, 1.0e-4, 7.0e-5, 5.0e-5, 1.0e-5]
  config:
    value: conf/simplestories_frozenl4e40p1largebatch3072.yaml
  num_train_epochs:
    value: 5
```

### SLURM Cluster Sweeps

For production-scale hyperparameter optimization on SLURM clusters:

```bash
# 1. Initialize the sweep
cd consistency-lens
wandb sweep sweeps/lr_sweep_slurm_train_only.yaml

# 2. Submit multiple SLURM jobs (following W&B's --count 1 recommendation)
for i in {1..8}; do
    sbatch scripts/slurm_sweep_agent.sh your-entity/consistency-lens-simplestories/SWEEP_ID 1
done

# Alternative: Submit jobs individually for better control
sbatch scripts/slurm_sweep_agent.sh your-entity/consistency-lens-simplestories/SWEEP_ID 1
sbatch scripts/slurm_sweep_agent.sh your-entity/consistency-lens-simplestories/SWEEP_ID 1
# ... repeat for desired parallelism
```

**SLURM sweep configuration** (`sweeps/lr_sweep_slurm_train_only.yaml`):
```yaml
program: scripts/wandb_sweep_train_only.py
method: grid
metric:
  goal: minimize
  name: train/loss
parameters:
  learning_rate:
    values: [1.0e-3, 7.0e-4, 5.0e-4, 3.0e-4, 1.0e-4, 7.0e-5, 5.0e-5, 1.0e-5]
  config:
    value: conf/simplestories_frozenl4e40p1largebatch3072.yaml
  num_train_epochs:
    value: 5
```

### Key Differences: Local vs SLURM

| Aspect | Local Sweeps | SLURM Sweeps |
|--------|-------------|--------------|
| **Wrapper Script** | `wandb_sweep_wrapper.py` | `wandb_sweep_train_only.py` |
| **Job Submission** | Direct `wandb agent` | `sbatch slurm_sweep_agent.sh` |
| **Resource Management** | Manual GPU allocation | SLURM scheduler |
| **Parallelism** | Multiple agents per node | One agent per SLURM job |
| **Data Pipeline** | Full submission script | Direct training only |

### Sweep Workflow Details

#### 1. Data Preparation (One-time)

```bash
# Ensure activations are dumped and data is pre-tokenized
./scripts/submit_with_config.sh config=conf/your_config.yaml

# This creates:
# - data/activations/model_name/layer_X/dataset_train/
# - data/pretokenized/dataset_name/
```

#### 2. Sweep Initialization

```bash
cd consistency-lens
wandb sweep sweeps/your_sweep_config.yaml
```

**Output example:**
```
wandb: Created sweep with ID: abc123def
wandb: View sweep at: https://wandb.ai/your-entity/consistency-lens-simplestories/sweeps/abc123def
wandb: Run sweep agent with: wandb agent your-entity/consistency-lens-simplestories/abc123def
```

#### 3. Agent Execution

**Local (development):**
```bash
# Single agent
wandb agent your-entity/consistency-lens-simplestories/abc123def

# Multiple agents (parallel)
wandb agent your-entity/consistency-lens-simplestories/abc123def &
wandb agent your-entity/consistency-lens-simplestories/abc123def &
wandb agent your-entity/consistency-lens-simplestories/abc123def &
```

**SLURM (production):**
```bash
# Submit 8 jobs for 8-parameter grid search
for i in {1..8}; do
    sbatch scripts/slurm_sweep_agent.sh your-entity/consistency-lens-simplestories/abc123def 1
done

# Monitor jobs
squeue -u $USER
```

### Monitoring and Management

#### W&B Dashboard
- **Sweep overview**: View progress, best runs, parameter importance
- **Real-time metrics**: Loss curves, validation metrics, training speed
- **Run comparison**: Side-by-side parameter and metric comparison

#### SLURM Monitoring
```bash
# Check job status
squeue -u $USER

# View job logs
tail -f logs/sweep_*.out

# Cancel all sweep jobs
scancel -u $USER --name=wandb-sweep
```

#### Log Files
- **SLURM logs**: `logs/sweep_*.out` and `logs/sweep_*.err`
- **Training logs**: Captured in W&B runs
- **Debug info**: Check wrapper script output for parameter passing

### Advanced Sweep Configurations

#### Bayesian Optimization
```yaml
method: bayes
metric:
  goal: minimize
  name: train/loss
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1.0e-5
    max: 1.0e-3
  batch_size:
    values: [16, 32, 64, 128]
```

#### Multi-parameter Grid Search
```yaml
method: grid
parameters:
  learning_rate:
    values: [1.0e-3, 5.0e-4, 1.0e-4]
  alpha_schedule.value:
    values: [0.05, 0.1, 0.2]
  t_text:
    values: [4, 6, 8]
```

#### Random Search with Early Termination
```yaml
method: random
early_terminate:
  type: hyperband
  min_iter: 3
  eta: 2
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1.0e-5
    max: 1.0e-3
```

### Performance Expectations

| Configuration | Jobs | Time per Job | Total Time | GPU Hours |
|--------------|------|--------------|------------|-----------|
| 8-param grid (local) | 8 sequential | 4 hours | 32 hours | 32 |
| 8-param grid (SLURM) | 8 parallel | 4 hours | 4 hours | 32 |
| Bayesian (20 trials) | 20 parallel | 4 hours | 4 hours | 80 |

### Troubleshooting

#### Common Issues

**Parameter substitution errors:**
```bash
# Error: learning_rate= (empty value)
# Solution: Check sweep config parameter names match wrapper script
```

**SLURM job failures:**
```bash
# Check logs
cat logs/sweep_*.err

# Common fixes:
# 1. Ensure environment is set up: make
# 2. Check GPU availability: nvidia-smi
# 3. Verify sweep ID format: user/project/sweep_id
```

**Data not found errors:**
```bash
# Ensure activations are dumped
ls data/activations/model_name/layer_X/

# Re-run data preparation if needed
./scripts/submit_with_config.sh config=conf/your_config.yaml force_redump=true
```

#### Best Practices

1. **Start small**: Test with 2-3 parameter values before large sweeps
2. **Use SLURM for production**: Much faster than sequential local runs
3. **Monitor early**: Check first few runs to catch configuration issues
4. **Resource planning**: Each training run needs ~40GB GPU memory
5. **Data reuse**: Prepare activations once, use for multiple sweeps

### Example: Complete Learning Rate Sweep

```bash
# 1. Prepare data (one-time)
./scripts/submit_with_config.sh config=conf/simplestories_frozenl4e40p1largebatch3072.yaml
# Cancel training once dumping completes

# 2. Initialize sweep
cd consistency-lens
wandb sweep sweeps/lr_sweep_slurm_train_only.yaml
# Note the sweep ID: your-entity/consistency-lens-simplestories/abc123def

# 3. Submit SLURM jobs
for i in {1..8}; do
    sbatch scripts/slurm_sweep_agent.sh your-entity/consistency-lens-simplestories/abc123def 1
done

# 4. Monitor progress
squeue -u $USER                    # SLURM jobs
# Check W&B dashboard for metrics

# 5. Results available in ~4 hours
# Best learning rate will be highlighted in W&B sweep dashboard
```

This workflow efficiently explores hyperparameter space while maximizing resource utilization and minimizing manual intervention.