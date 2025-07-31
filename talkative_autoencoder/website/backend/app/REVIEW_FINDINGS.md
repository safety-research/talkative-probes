# Review Findings - Grouped Model Implementation

## Critical Issues Found and Fixed

### 1. **Incorrect shared_base_model handling**
- **Issue**: Original code tried to extract shared_base_model from the analyzer or store the analyzer itself
- **Fix**: Properly extract the actual transformer model (`analyzer.shared_base_model`) and pass it to subsequent analyzers
- **Why it matters**: The shared base model is the large transformer that takes up most GPU memory

### 2. **Missing orig_model caching**
- **Issue**: Each model would create its own orig_model wrapper, even when sharing the same base
- **Fix**: Cache orig_model wrappers (OrigWrapper instances) and reuse them between analyzers
- **Why it matters**: Avoids duplicate model loading and memory usage

### 3. **Incorrect old_lens usage**
- **Issue**: old_lens was only passed within a switch, not utilized properly
- **Fix**: Pass the current analyzer as old_lens to potentially reuse encoder/decoder components
- **Why it matters**: LensAnalyzer can reuse components from previous instances for efficiency

### 4. **Incomplete CPU/GPU movement**
- **Issue**: Only moved analyzers, not the underlying models and orig_model wrappers
- **Fix**: Move shared_base_model, all analyzers, and associated orig_models together
- **Why it matters**: Ensures all components stay on the same device, avoiding device mismatch errors

### 5. **Different activations model handling**
- **Issue**: Didn't properly handle cases where models in a group use different orig models
- **Fix**: Track orig_model wrappers by path and reuse when possible
- **Why it matters**: Some models might use different activation sources while sharing the base

## Key Implementation Details

### Correct Pattern (from sandbagging script):
```python
# First analyzer - creates everything
analyzer1 = LensAnalyzer(checkpoint1, device)
shared_base = analyzer1.shared_base_model
orig_wrapper = analyzer1.orig_model

# Second analyzer - reuses shared resources
analyzer2 = LensAnalyzer(
    checkpoint2,
    shared_base_model=shared_base,      # Pass actual model
    different_activations_orig=orig_wrapper,  # Pass wrapper or path
    old_lens=analyzer1,                  # Pass previous analyzer
)
```

### Memory Savings Example:
- Gemma 2 9B base model: ~20GB
- Each lens checkpoint: ~1-2GB
- Switching within group: Only loads 1-2GB instead of 21-22GB

## Frontend Considerations

The grouped UI implementation looks good, but consider:
1. Add visual indicators for which groups are fully loaded (all models cached)
2. Show estimated switch time based on whether it's within-group or cross-group
3. Add tooltips explaining the grouping concept to users

## API Design

The API design is solid with good separation between v1 (legacy) and v2 (grouped). Consider:
1. Add a `/api/v2/groups/{group_id}/status` endpoint to check group loading status
2. Add memory usage estimates to the model list response
3. Consider WebSocket events for preload progress updates

## Integration Recommendations

1. **Test incrementally**: Start with a single group of 2-3 models
2. **Monitor memory**: Log GPU memory before/after switches to verify sharing works
3. **Add metrics**: Track switch times within vs across groups
4. **Gradual rollout**: Keep legacy endpoints active during transition

## Performance Optimization Opportunities

1. **Parallel preloading**: When preloading a group, could create multiple analyzers in parallel
2. **Predictive loading**: Track usage patterns to preload likely next models
3. **Memory pressure handling**: Add logic to automatically move groups to CPU under memory pressure

## Edge Cases to Handle

1. **Checkpoint corruption**: Handle cases where a checkpoint in a group fails to load
2. **Memory exhaustion**: Gracefully handle OOM by moving groups to CPU
3. **Concurrent switches**: Ensure proper locking if multiple switch requests arrive
4. **Different layer counts**: Validate that models in a group are compatible

## Overall Assessment

The implementation is well-structured and follows good patterns, but needs the fixes in `model_manager_grouped_fixed.py` to properly implement memory sharing. The UI and API design are solid and provide a good user experience with the grouped model concept.