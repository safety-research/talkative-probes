# Unified Multi-GPU Architecture Summary

## Overview

The backend has been refactored to use a unified architecture where single-GPU deployments are treated as a special case of multi-GPU, eliminating code duplication and complexity.

## Key Changes

### 1. Unified Model Manager (`model_manager.py`)

**Before**: Separate `ModelManager` (single-GPU) and `GroupedModelManager` (multi-GPU) with backward compatibility shims.

**After**: Single `UnifiedModelManager` that:
- Treats all models as part of groups (single-model groups for standalone models)
- All devices have a current group (which may contain one or many models)
- All operations are inherently device-aware
- Single-GPU is just `num_devices=1` with no special handling needed

### 2. Clean API Design (`unified_api.py`)

**Before**: Multiple API versions with compatibility layers (`/api/v1/` for single-GPU, grouped endpoints for multi-GPU).

**After**: Unified API (`/api/v2/`) where:
- `device_id` is optional in all requests (system auto-selects if not provided)
- Single-GPU clients just omit `device_id` or use `device_id=0`
- Multi-GPU clients can specify devices or let the system optimize placement

### 3. Core API Endpoints

#### Model Loading
```python
POST /api/v2/models/load
{
    "model_id": "model-name",
    "device_id": 0  # Optional - omit for auto-selection
}
```

#### Processing
```python
POST /api/v2/models/process
{
    "model_id": "model-name",
    "prompt": "text to analyze",
    "num_predictions": 3,
    "device_id": null  # Optional
}
```

#### System State
```python
GET /api/v2/system/state
# Returns full system state with all devices and loaded models
```

### 4. WebSocket Updates

The WebSocket now provides unified real-time updates:
- `system_state`: Complete view of all devices and models
- `state_update`: Broadcast when any device/model state changes
- No distinction between single/multi GPU in the protocol

### 5. Model Groups

- Single models automatically get wrapped in single-model groups
- Groups are the primary organizational unit
- Memory estimates and device assignment work at the group level
- Switching models within a group is instant (no memory operations)

## Benefits

1. **Simplicity**: One code path for all deployments
2. **Scalability**: Easy to go from 1 to N GPUs without code changes
3. **Flexibility**: Models can be loaded on any device dynamically
4. **Efficiency**: Smart routing and placement without special cases

## Migration Guide

### For Single-GPU Clients

No changes needed! The API is designed so single-GPU usage is natural:

```python
# Old single-GPU code still works
response = requests.post("/api/v2/models/load", json={
    "model_id": "gemma-2b"
})

# Process a prompt (device auto-selected)
response = requests.post("/api/v2/models/process", json={
    "model_id": "gemma-2b",
    "prompt": "Hello world"
})
```

### For Multi-GPU Clients

Simply add device specifications where needed:

```python
# Load model on specific device
response = requests.post("/api/v2/models/load", json={
    "model_id": "gemma-27b",
    "device_id": 1
})

# Or let the system optimize placement
response = requests.post("/api/v2/models/load", json={
    "model_id": "gemma-27b"
    # device_id omitted - system chooses best device
})
```

## Implementation Status

- ✅ UnifiedModelManager implemented
- ✅ Unified API with clean design
- ✅ Backward compatibility via deprecated endpoints
- ✅ WebSocket support for real-time updates
- ✅ Smart device selection and routing
- ✅ Group-based model organization

## Next Steps

1. Update frontend to use the new unified API
2. Remove legacy code once migration is complete
3. Add advanced features like model pinning and explicit device affinity
4. Implement cross-device model sharing for very large models