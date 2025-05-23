# Configuration Files Overview

## Base Configuration
- **`config.yaml`** - Base configuration for SimpleStories-5M model

## SimpleStories Experiments (5M parameters)
- **`simplestories_frozen.yaml`** - SimpleStories data, base model frozen, t_text=10
- **`simplestories_unfreeze.yaml`** - SimpleStories data, unfreeze after epoch 1, t_text=10

## GPT-2 Experiments (124M parameters)  
- **`gpt2_frozen.yaml`** - OpenWebText data, base model frozen, t_text=10
- **`gpt2_unfreeze.yaml`** - OpenWebText data, unfreeze after 1st epoch, t_text=10
- **`gpt2_pile_frozen.yaml`** - The Pile data, base model frozen, t_text=10
- **`gpt2_pile_unfreeze.yaml`** - The Pile data, unfreeze after 1st epoch, t_text=10

## Other Configs
- **`debug.yaml`** - Debug configuration (small batches, few steps)
- **`high_lr.yaml`** - Higher learning rate experiment
- **`larger_model.yaml`** - Configuration for larger models

## Key Differences

### Model-Dataset Pairings
- SimpleStories-5M → SimpleStories dataset
- GPT-2 → OpenWebText or The Pile dataset

### Unfreezing Strategies
- SimpleStories: Unfreeze after 1 epoch
- GPT-2: Unfreeze after 10,000 steps

### Common Settings
- All use t_text=10 (width-10 explanations)
- All use same layer (5 for SimpleStories, 6 for GPT-2)
- Unfrozen models use 0.1x LR for base model