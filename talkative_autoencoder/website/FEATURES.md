# Talkative Autoencoder Website Features

This document describes the key features implemented in the Talkative Autoencoder web interface.

## Table of Contents
- [Model Management](#model-management)
- [Request Logging](#request-logging)
- [Queue Management](#queue-management)
- [Real-time Updates](#real-time-updates)
- [Debug Mode](#debug-mode)

## Model Management

### Multiple Model Support
The system supports multiple LensAnalyzer models that can be switched dynamically:

- **Available Models**: Qwen 2.5 14B, Gemma 2 9B, Gemma 3 27B
- **Model Registry**: Centralized configuration in `backend/app/model_registry.py`
- **Hot Swapping**: Models can be switched without restarting the server

### CPU Offloading
- Models are moved to CPU memory instead of being deleted when switching
- Enables faster switching back to previously used models
- LRU cache evicts least recently used models when CPU memory limit is reached
- Configure max CPU models with `max_cpu_cached_models` setting

### Model Switch Warnings
- **Global Notifications**: All connected users see a yellow warning bar during model switches
- **Queue Warnings**: Red warning with active request counts when switching with pending requests
- **User Confirmation**: Requires explicit confirmation before switching models

## Request Logging

### Backend Logging

#### Stdout Logging (Always Enabled)
Every request prints to stdout with the format:
```
================================================================================
[ANALYSIS REQUEST 438d2fa0-6828-4d57-b78a-72f1e7a7ae62]
Timestamp: 2025-07-29T16:49:32.000000
Text length: 150 characters
Input text: <first 500 chars>...
================================================================================
```

#### File Logging (Always Enabled)
- Location: `website/logs/requests_YYYY-MM-DD.jsonl`
- Format: JSON Lines (one JSON object per line)
- Includes: request ID, type, timestamp, text preview (1000 chars), options, process ID
- Rotation: New file created daily

Example log entry:
```json
{
  "request_id": "438d2fa0-6828-4d57-b78a-72f1e7a7ae62",
  "type": "analysis",
  "timestamp": "2025-07-29T16:49:32.000000",
  "text_length": 150,
  "text_preview": "What is the meaning of life?",
  "options": {"batch_size": 32, "temperature": 1.0},
  "pid": 12345
}
```

### Frontend Logging (Debug Mode Only)

When in debug mode, requests are logged to browser localStorage:
- **Storage Key**: `talkative_autoencoder_requests`
- **Limit**: Last 1000 requests
- **Access**: Use `window.RequestLogger` in browser console

Commands:
```javascript
// View all logs
RequestLogger.getLogs()

// Export logs as JSON file
RequestLogger.export()

// Clear all logs
RequestLogger.clear()
```

## Queue Management

### Queue Status Updates
- **Periodic Broadcasting**: Queue status sent to all clients every 2 seconds
- **Real-time Position**: Users see their position in the queue
- **Active Request Counts**: Shows both queued and processing requests

### Queue Indicators
```
Queue: 3 (1 processing)
Your position: 2/3
```

### Request Queuing
- Requests are queued during model switches
- Queue processes in FIFO order
- WebSocket updates keep users informed of progress

## Real-time Updates

### WebSocket Features
- **Progress Updates**: Live progress bars during analysis (batch 1/8, 2/8, etc.)
- **Model Switch Status**: All users notified of model changes
- **Queue Updates**: Real-time queue position updates
- **Connection Status**: Shows current model and connection state

### Progress Tracking
- Analysis shows batch progress with ETA
- Generation shows "Processing..." state
- Interrupt button available during processing

## Debug Mode

Debug mode is automatically enabled when:
- Running on `localhost` or `127.0.0.1`
- URL contains `?debug` parameter

### Debug Features
- Frontend request logging to localStorage
- Additional console logging
- Request history export functionality

### Accessing Debug Info
```javascript
// In browser console:

// View request logs
RequestLogger.getLogs()

// Export request history
RequestLogger.export()

// Check if debug mode is active
isDebugMode()
```

## Configuration

### Environment Variables
- `LAZY_LOAD_MODEL`: Load model on first request vs startup
- `MAX_CPU_CACHED_MODELS`: Number of models to keep in CPU memory
- `RATE_LIMIT_PER_MINUTE`: API rate limiting

### Model Configuration (Hot-Reloadable!)
Models are configured in `backend/app/models.json`:
```python
"model-id": ModelConfig(
    name="model-id",
    display_name="Model Display Name",
    checkpoint_path="/path/to/checkpoint",
    model_name="huggingface/model-name",
    batch_size=32,
    layer=45,
    # ... other settings
)
```

## Security Considerations

- Request logging excludes sensitive information beyond 1000 character preview
- Log files are gitignored to prevent accidental commits
- Frontend logging only in debug mode to prevent data leakage
- WebSocket broadcasts don't include request content, only metadata
- Model switching is available to all users (no authentication required)
- Only model IDs are sent from frontend, never file paths
- Model registry reload requires API key authentication

### API Key Authentication

Some sensitive operations require authentication:
- **Registry Reload**: Requires API key to prevent unauthorized configuration changes
- **REST API Model Switch**: The `/api/models/switch` endpoint requires API key

Note: WebSocket model switching is intentionally open to all users for ease of use.

### Path Validation

Model configuration paths are validated to prevent path traversal attacks:
- Checkpoint paths must be absolute paths
- No `..` path components allowed
- Validation applies to `checkpoint_path`, `tuned_lens_dir`, and `comparison_tl_checkpoint`

### Thread Safety

The ModelRegistry is thread-safe for concurrent access:
- Uses RLock (reentrant lock) for all public methods
- Safe hot-reloading while serving requests
- Atomic updates when reloading configuration