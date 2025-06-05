# W&B Sweep Quick Start Guide

This guide provides step-by-step instructions for launching W&B hyperparameter sweeps with multi-GPU support.

## Prerequisites

1. **W&B Account**: Sign up at https://wandb.ai if you don't have an account
2. **W&B Login**: Run `wandb login` and enter your API key (found at https://wandb.ai/authorize)
3. **Data Prepared**: Activations must be dumped and data pretokenized

## Step-by-Step Instructions

### 1. Prepare Your Data (if not already done)

```bash
# Dump activations for your config
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml

# Wait for dumping to complete, then cancel the training job
# Check logs to confirm activations are saved
```

### 2. Create Your Sweep Configuration

Create a sweep YAML file (or use an existing one):

```bash
# Example: Create a learning rate sweep
cat > sweeps/my_lr_sweep.yaml << 'EOF'
program: scripts/wandb_sweep_train_only.py
method: grid
metric:
  name: loss/kl
  goal: minimize

parameters:
  config:
    value: conf/gpt2_frozen.yaml  # Your config file
  
  learning_rate:
    values: [1e-4, 3e-4, 1e-3]
  
  batch_size:
    values: [8, 16]
  
  num_gpus:
    value: 2  # Number of GPUs per run
  
  run_suffix:
    value: "_mysweep_"
  
  max_train_steps:
    value: 10000
EOF
```

### 3. Initialize the W&B Sweep

```bash
cd consistency-lens  # Make sure you're in the project root

# Initialize sweep and get the sweep ID
wandb sweep sweeps/my_lr_sweep.yaml

# You'll see output like:
# Created sweep with ID: username/project-name/abc123xyz
# Sweep URL: https://wandb.ai/username/project-name/sweeps/abc123xyz
```

Copy the sweep ID (e.g., `username/project-name/abc123xyz`) - you'll need it!

### 4. Launch Sweep Agents

**Option A: Using the helper script (recommended)**

```bash
# Launch 4 agents with 2 GPUs each
./scripts/launch_sweep_multigpu.sh username/project-name/abc123xyz 4 2

# Launch 8 single-GPU agents
./scripts/launch_sweep_multigpu.sh username/project-name/abc123xyz 8 1

# Launch 1 agent with 8 GPUs
./scripts/launch_sweep_multigpu.sh username/project-name/abc123xyz 1 8
```

**Option B: Manual SLURM submission**

```bash
# Single GPU agents
for i in {1..4}; do
    sbatch scripts/slurm_sweep_agent.sh username/project-name/abc123xyz
done

# Multi-GPU agents (4 GPUs each)
for i in {1..2}; do
    sbatch --gres=gpu:4 scripts/slurm_sweep_agent.sh username/project-name/abc123xyz
done
```

**Option C: Direct execution (non-SLURM)**

```bash
# Run directly without SLURM
wandb agent username/project-name/abc123xyz
```

### 5. Monitor Your Sweep

**Check SLURM jobs:**
```bash
squeue -u $USER
```

**View logs:**
```bash
# List recent log files
ls -lt logs/sweep_*.out | head -10

# Watch a specific job's output
tail -f logs/sweep_<job_id>.out
```

**W&B Dashboard:**
- Go to the sweep URL shown when you initialized the sweep
- Watch real-time metrics, compare runs, and analyze results

## Common Sweep Configurations

### Learning Rate Sweep (Grid Search)
```yaml
program: scripts/wandb_sweep_train_only.py
method: grid
parameters:
  config:
    value: conf/your_config.yaml
  learning_rate:
    values: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
  run_suffix:
    value: "_lr_sweep_"
```

### Hyperparameter Optimization (Bayesian)
```yaml
program: scripts/wandb_sweep_train_only.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  config:
    value: conf/your_config.yaml
  learning_rate:
    min: 1e-5
    max: 1e-2
    distribution: log_uniform
  batch_size:
    values: [4, 8, 16, 32]
  gumbel_tau_schedule:
    values: ["2.0->0.5@linear", "1.0->0.1@cosine", "0.5"]
  alpha_schedule:
    values: ["0.0->1.0@linear", "0.0->0.5@linear", "0.1->1.0@linear"]
```

### Multi-GPU Scaling Test
```yaml
program: scripts/wandb_sweep_train_only.py
method: grid
parameters:
  config:
    value: conf/your_config.yaml
  num_gpus:
    values: [1, 2, 4, 8]
  batch_size:
    values: [8, 16]  # Per-GPU batch size
  run_suffix:
    value: "_gpu_scaling_"
```

## Tips and Best Practices

1. **Start Small**: Test with a few runs before launching large sweeps
   ```bash
   # Test sweep with 2 agents for 100 steps
   wandb sweep --count 2 sweeps/test_sweep.yaml
   ```

2. **Use Early Stopping**: Add to your sweep config:
   ```yaml
   early_terminate:
     type: hyperband
     s: 2
     eta: 3
     max_iter: 27
   ```

3. **Resource Planning**: 
   - Single GPU: More experiments, slower each
   - Multi-GPU: Fewer experiments, faster each
   - Balance based on your parameter space

4. **Checkpoint Resume**: If a run fails, it can resume from checkpoint:
   ```yaml
   parameters:
     resume_checkpoint:
       value: outputs/previous_run/checkpoint_step5000.pt
   ```

5. **Custom Metrics**: Track what matters:
   ```yaml
   metric:
     name: val/improvement_over_mean  # Custom validation metric
     goal: maximize
   ```

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` per GPU
- Increase `gradient_accumulation_steps`
- Use fewer GPUs with checkpointing

### "Sweep not starting"
- Check W&B login: `wandb login --verify`
- Ensure data is prepared: check activation directory
- Verify config path is correct

### "Port already in use"
- The scripts automatically randomize ports
- If still issues: `export MASTER_PORT=$((30000 + RANDOM % 1000))`

### View detailed logs
```bash
# Check SLURM error logs
cat logs/sweep_<job_id>.err

# Check Python traceback
grep -A 20 "Traceback" logs/sweep_<job_id>.out
```

## Example: Complete Learning Rate Sweep

```bash
# 1. Prepare data
./scripts/submit_with_config.sh config=conf/gpt2_frozen.yaml

# 2. Create sweep config
cat > sweeps/lr_sweep_example.yaml << 'EOF'
program: scripts/wandb_sweep_train_only.py
method: grid
metric:
  name: val/loss
  goal: minimize

parameters:
  config:
    value: conf/gpt2_frozen.yaml
  learning_rate:
    values: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
  num_gpus:
    value: 2
  batch_size:
    value: 16
  max_train_steps:
    value: 5000
  val_interval:
    value: "500s"
  run_suffix:
    value: "_lr_search_"
EOF

# 3. Initialize sweep
SWEEP_ID=$(wandb sweep sweeps/lr_sweep_example.yaml 2>&1 | grep "Created sweep with ID" | awk '{print $NF}')
echo "Sweep ID: $SWEEP_ID"

# 4. Launch 5 agents (one for each LR value) with 2 GPUs each
./scripts/launch_sweep_multigpu.sh $SWEEP_ID 5 2

# 5. Monitor
echo "Monitor at: https://wandb.ai/$SWEEP_ID"
watch -n 10 squeue -u $USER
```

## Advanced: Distributed Sweep Across Multiple Nodes

For very large sweeps, distribute agents across nodes:

```bash
# Node 1
sbatch --nodelist=node001 --gres=gpu:8 scripts/slurm_sweep_agent.sh $SWEEP_ID

# Node 2
sbatch --nodelist=node002 --gres=gpu:8 scripts/slurm_sweep_agent.sh $SWEEP_ID

# Or use array jobs
sbatch --array=1-10 --gres=gpu:2 scripts/slurm_sweep_agent.sh $SWEEP_ID
```

Remember: Each agent will pull the next set of hyperparameters from the sweep queue automatically!