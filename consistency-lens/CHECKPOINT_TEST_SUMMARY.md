# Checkpoint Save/Load Test Summary

## Tests Created

### 1. `test_checkpoint_save_load.py`
This comprehensive test verifies core checkpoint functionality:

- ✅ **Model Parameter Preservation**: All decoder and encoder parameters are exactly preserved
- ✅ **Optimizer State Preservation**: Learning rates, momentum, and other optimizer states maintained
- ✅ **Scheduler State Preservation**: Learning rate scheduler state correctly saved/loaded
- ✅ **Metadata Preservation**: Step, epoch, metrics, config, and custom data saved/loaded correctly
- ✅ **Random State Preservation**: PyTorch, CUDA, and NumPy random states are restored
- ✅ **Compiled Model Compatibility**: Non-compiled checkpoints can be loaded into compiled models

### 2. `test_checkpoint_with_prompts.py`
This test specifically verifies prompt-related checkpoint functionality:

- ✅ **Prompt Preservation**: Soft prompts for both decoder and encoder are saved and restored correctly
- ✅ **Prompt Initialization**: Prompts initialized from text are preserved with correct values
- ✅ **Configuration Compatibility**: System correctly handles (rejects) loading checkpoints into models with:
  - No prompts when checkpoint has prompts
  - Different prompt lengths than checkpoint

## Key Findings

1. **Decoder Prompts**: Stored as `prompt_left_emb` and `prompt_right_emb` parameters
2. **Encoder Prompts**: Stored as `soft_prompt_embeddings` parameter
3. **Prompt Size Matching**: Models must have matching prompt sizes to load checkpoints with prompts
4. **Random State Format**: Random states need careful handling to ensure correct dtype (ByteTensor)

## Running the Tests

```bash
cd /workspace/kitf/talkative-probes/consistency-lens
source scripts/ensure_env.sh

# Run checkpoint save/load test
uv run python test_checkpoint_save_load.py

# Run prompt preservation test
uv run python test_checkpoint_with_prompts.py
```

Both tests pass successfully, confirming that the checkpoint system correctly preserves all training state including model parameters, optimizer state, prompts, and random number generator states.