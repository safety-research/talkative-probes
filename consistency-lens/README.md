# Consistency Lenses: Scalable LLM Interpretation via Textual Bottlenecks

This document outlines the architectural plan for training talkative probes as autoencoders to interpretthe internal states of Large Language Models (LLMs) by forcing them through a textual, human-readable bottleneck.

```bash
make
```
The main entrypoint is scripts/submit_with_config.sh, which torchruns 01_distributed_train.py. We configure hyperparameters via conf/



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