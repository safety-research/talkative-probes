# Multi-GPU Backend Setup

This backend now supports multiple GPUs for improved throughput and model capacity.

## Configuration

### Environment Variables

```bash
# Specify multiple GPUs (comma-separated)
DEVICES=cuda:0,cuda:1,cuda:2

# Number of workers per GPU (default: 1)
NUM_WORKERS_PER_GPU=2

# Default model group to preload on startup (overrides model_groups.json)
DEFAULT_GROUP=gemma3-27b-it

# Backward compatibility - single device
DEVICE=cuda:0  # Still supported, will be converted to DEVICES=["cuda:0"]
```

### Example .env file

```bash
# Multi-GPU configuration
DEVICES=cuda:0,cuda:1,cuda:2,cuda:3
NUM_WORKERS_PER_GPU=1

# Other settings
DEFAULT_GROUP=gemma3-27b-it  # Default model group to preload (overrides JSON config)
MAX_CPU_CACHED_MODELS=4
LAZY_LOAD_MODEL=true
```

## Architecture Overview

### Per-Device State Management
- Each GPU maintains its own state and model cache
- Groups can be loaded on different GPUs simultaneously
- Smart routing minimizes group switches

### Worker Pool
- Each GPU has `NUM_WORKERS_PER_GPU` workers
- Workers process requests independently
- Central dispatcher routes requests optimally

### Request Routing Strategy
1. **Exclusive GPU**: Requests can specify `exclusive_gpu` for dedicated processing
2. **Group Affinity**: Requests route to GPUs with matching groups already loaded
3. **Load Balancing**: Otherwise, routes to least busy GPU

## API Usage

### Exclusive GPU Assignment
```javascript
// Request exclusive use of a specific GPU
const options = {
  model_id: "gemma3-chat",
  exclusive_gpu: "cuda:2",  // Use only GPU 2
  // ... other options
};
```

### Multi-GPU Status
```javascript
// GET /api/gpu_stats returns:
{
  "available": true,
  "devices": {
    "cuda:0": {
      "utilization": 45,
      "memory_used": 12.5,
      "memory_total": 80.0,
      "memory_percent": 15.6,
      "peak_utilization": 95
    },
    "cuda:1": {
      "utilization": 0,
      "memory_used": 8.2,
      "memory_total": 80.0,
      "memory_percent": 10.3,
      "peak_utilization": 12
    }
  },
  // Backward compatibility fields for first device
  "utilization": 45,
  "memory_used": 12.5,
  "memory_total": 80.0,
  "memory_percent": 15.6,
  "peak_utilization": 95
}
```

### Model/Device Status
```javascript
// GET /api/models returns enhanced info:
{
  "model_groups": [{
    "group_id": "gemma3-27b-it",
    "device_info": [
      {"device": "cuda:0", "has_group": true, "is_current": true},
      {"device": "cuda:1", "has_group": false, "is_current": false}
    ],
    "models": [{
      "id": "gemma3-chat",
      "loaded_on_devices": ["cuda:0"],
      // ... other model info
    }]
  }]
}
```

## Benefits

1. **Increased Throughput**: Multiple requests processed in parallel
2. **Better Memory Utilization**: Different groups on different GPUs
3. **Fault Tolerance**: If one GPU fails, others continue working
4. **Flexibility**: Mix of exclusive and shared GPU usage

## Monitoring

The backend provides comprehensive monitoring:
- Per-device GPU utilization and memory
- Worker queue sizes and busy status
- Group loading status per device
- Request routing decisions in logs

## Best Practices

1. **Group Distribution**: Spread frequently used groups across GPUs
2. **Worker Count**: Start with 1 worker per GPU, increase if needed
3. **Exclusive GPU**: Use for long-running or memory-intensive tasks
4. **CPU Cache**: Set `MAX_CPU_CACHED_MODELS` based on system RAM

## Troubleshooting

### All GPUs Busy
- Check worker status: `GET /api/queue`
- Consider increasing `NUM_WORKERS_PER_GPU`
- Monitor group switching frequency

### Out of Memory
- Reduce concurrent models per GPU
- Use exclusive GPU for large models
- Increase `MAX_CPU_CACHED_MODELS` to offload more

### Suboptimal Routing
- Check logs for routing decisions
- Ensure popular groups are pre-loaded
- Consider group affinity in request patterns