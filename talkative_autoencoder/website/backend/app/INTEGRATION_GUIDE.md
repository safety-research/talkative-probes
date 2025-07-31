# Model Groups Integration Guide

This guide explains how to integrate the grouped model management system that allows models sharing a base to be switched efficiently.

## Overview

The grouped model system allows:
- Models sharing the same base model (e.g., all Gemma 2 9B variants) to be grouped together
- Fast switching within a group (only lens weights change, base stays in GPU)
- Efficient memory management by caching base models
- Hierarchical dropdown UI showing model groups

## Backend Integration

### 1. Update main.py to use the grouped model manager

```python
# In main.py, replace the existing model manager import and initialization:

# Option A: Use grouped manager directly
from .model_manager_grouped import GroupedModelManager

# In lifespan function:
model_manager = GroupedModelManager(settings)

# Option B: Support both old and new endpoints
from .model_manager import ModelManager  # Keep existing
from . import api_grouped  # Import new grouped API

# In lifespan, after creating model_manager:
grouped_manager = api_grouped.setup_grouped_model_manager(app, settings)
```

### 2. Update WebSocket handlers

Add handlers for grouped model operations in your WebSocket message handler:

```python
# In handle_websocket_message function:

elif message_type == "list_model_groups":
    # Use grouped manager
    groups = grouped_manager.get_model_list()
    current_info = grouped_manager.get_current_model_info()
    
    await websocket.send_json({
        "type": "model_groups_list",
        "groups": groups,
        "current_model": current_info.get("model_id"),
        "current_group": current_info.get("group_id"),
        "is_switching": current_info.get("is_switching", False),
        "model_status": current_info
    })

elif message_type == "switch_model_grouped":
    model_id = data.get("model_id")
    if model_id:
        result = await grouped_manager.switch_model(model_id)
        # Existing switch handling code...

elif message_type == "preload_group":
    group_id = data.get("group_id")
    if group_id:
        await grouped_manager.preload_group(group_id)
        await websocket.send_json({
            "type": "group_preload_complete",
            "group_id": group_id
        })
```

### 3. Update inference service

Modify `inference_service.py` to use the grouped manager:

```python
# Replace model_manager with grouped_manager
# Or add a check to use the appropriate manager:

if hasattr(self.model_manager, 'get_analyzer'):
    analyzer = await self.model_manager.get_analyzer()
else:
    # Legacy support
    analyzer = self.model_manager.current_analyzer
```

## Frontend Integration

### 1. Update index.html

Include the new grouped model switcher script:

```html
<!-- Replace or add after existing model-switcher.js -->
<script src="model-switcher-grouped.js"></script>
```

### 2. Update app.js

Replace the ModelSwitcher initialization with GroupedModelSwitcher:

```javascript
// In the main initialization function:

// Replace:
// state.modelSwitcher = new ModelSwitcher(state.ws, modelSwitcherContainer);

// With:
state.modelSwitcher = new GroupedModelSwitcher(state.ws, modelSwitcherContainer, 'v2');
// Use 'v1' for backward compatibility with flat model list
```

### 3. Update WebSocket message handling

The grouped model switcher expects different message types:

```javascript
// Update the message type checks in handleWebSocketMessage:

if (state.modelSwitcher && [
    'model_groups_list',      // New grouped format
    'models_list',            // Legacy support
    'model_switch_status',
    'group_preload_complete', // New preload feature
    // ... other types
].includes(data.type)) {
    state.modelSwitcher.handleMessage(data);
}
```

## Configuration

### 1. Create/Update model_groups.json

The `model_groups.json` file defines your model groups. Example structure:

```json
{
  "model_groups": [
    {
      "group_id": "gemma2-9b-it",
      "group_name": "Gemma 2 9B IT",
      "base_model_path": "google/gemma-2-9b-it",
      "models": [
        {
          "id": "gemma2-9b-wildchat",
          "name": "WildChat L30",
          "lens_checkpoint_path": "/path/to/checkpoint",
          "layer": 30,
          "batch_size": 48
        }
      ]
    }
  ]
}
```

### 2. Environment Variables

You may want to add:

```bash
# Maximum number of model groups to keep in CPU cache
MAX_CPU_CACHED_GROUPS=2

# Whether to preload all models in a group when switching
PRELOAD_GROUP_ON_SWITCH=false
```

## Migration Path

### Phase 1: Parallel deployment
1. Keep existing endpoints working
2. Add new `/api/v2/` endpoints
3. Deploy grouped UI as opt-in feature

### Phase 2: Migration
1. Update all clients to use grouped API
2. Migrate model configurations to groups
3. Test memory efficiency improvements

### Phase 3: Cleanup
1. Remove legacy model manager
2. Remove flat model list support
3. Optimize memory settings based on usage

## Benefits

1. **Memory Efficiency**: 
   - Gemma 2 9B base model: ~20GB
   - Each lens checkpoint: ~1-2GB
   - Switching within group: Save 18GB per switch

2. **Speed**:
   - Cross-group switch: 1-2 minutes
   - Within-group switch: ~10 seconds

3. **User Experience**:
   - Clear organization of model variants
   - Visual indicators for cached models
   - Preload option for frequently used groups

## Troubleshooting

### Models not appearing in groups
- Check `model_groups.json` syntax
- Verify checkpoint paths exist
- Check logs for loading errors

### Slow switching within groups
- Ensure shared base model is properly detected
- Check if models have different `different_activations_orig_path`
- Verify GPU memory is available

### Memory errors
- Reduce `max_cpu_groups` setting
- Check total GPU memory with `nvidia-smi`
- Consider using smaller batch sizes

## Next Steps

1. Test with a small set of models first
2. Monitor GPU memory usage during switches
3. Gather user feedback on grouped UI
4. Optimize preloading strategy based on usage patterns