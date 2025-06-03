# KV Cache Bug Fix Summary

## Issue
The user encountered an error accessing `layer.ln_2` when using KV cache, occurring "some number of steps after" epoch boundaries during training.

## Root Cause
The `generate_soft_kv_cached` method in `decoder.py` had incorrect handling of the `override_model_base_and_out` parameter:

1. When `override_model_base_and_out` is an `OrigWrapper` (used in evaluation), it has a `.model` attribute
2. The code assumed `main_model.model` would always exist
3. But when a raw model is passed, it doesn't have a `.model` attribute
4. Additionally, the transformer extraction logic was incorrect for override models

## Fix Applied
Updated the model and transformer extraction logic in `generate_soft_kv_cached` to:

1. Check if the override model has a `.model` attribute (OrigWrapper case)
2. Handle both OrigWrapper and raw model cases correctly
3. Extract the transformer component properly in both cases

## Changes Made
- Modified `lens/models/decoder.py` lines 551-565 to add proper attribute checking
- Fixed transformer extraction to work correctly for both regular and override models

## Testing
- Created comprehensive tests covering all scenarios
- Verified KV cache works with:
  - Regular decoder generation
  - OrigWrapper override (as used in evaluation)
  - Raw model override
  - Training loop integration
  - Prompt changes (simulating epoch boundaries)
  - Memory cleanup scenarios

## Result
The KV cache now works correctly in all scenarios and the `ln_2` error should no longer occur.
