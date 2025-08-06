# Natural Languuage Autoencoder Backend API Documentation

This document provides comprehensive API documentation for the Natural Language Autoencoder backend service, designed for developing automated auditing pipelines.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URLs](#base-urls)
4. [Rate Limiting](#rate-limiting)
5. [REST API Endpoints](#rest-api-endpoints)
6. [WebSocket API](#websocket-api)
7. [Data Models](#data-models)
8. [Error Handling](#error-handling)
9. [Examples](#examples)

## Overview

The NLAE API provides endpoints for:
- Text analysis using neural language models
- Model management and switching between model groups
- Real-time processing via WebSocket connections
- Queue management for concurrent requests
- GPU resource monitoring

## Authentication

### API Key Authentication (Optional)

When `API_KEY` is configured in the environment, all endpoints require Bearer token authentication:

```http
Authorization: Bearer your-api-key-here
```

**Note**: Authentication is disabled in development mode (localhost) if no API key is configured.

## Base URLs

```
HTTP API: http://localhost:8000
WebSocket: ws://localhost:8000/ws
```

Production URLs will vary based on deployment.

## Rate Limiting

Default rate limit: **60 requests per minute** per IP address

Configurable via `RATE_LIMIT_PER_MINUTE` environment variable.

## REST API Endpoints

### 1. Root Endpoint

**GET /**

Returns API status and version information.

**Response:**
```json
{
  "name": "Consistency Lens API",
  "version": "1.0.0",
  "status": "running"
}
```

### 2. Health Check

**GET /health**

Check service health and GPU availability.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "queue_size": 0
}
```

### 3. Analyze Text

**POST /analyze**

Queue a text analysis request.

**Request Body:**
```json
{
  "text": "The quick brown fox jumps over the lazy dog",
  "options": {
    "batch_size": 32,
    "temperature": 1.0,
    "seed": 42,
    "no_eval": false,
    "tuned_lens": false,
    "logit_lens_analysis": false,
    "do_hard_tokens": false,
    "return_structured": true,
    "move_devices": false,
    "no_kl": false,
    "calculate_token_salience": true,
    "add_tokens": null,
    "replace_left": null,
    "replace_right": null,
    "optimize_explanations_config": {
      "use_batched": true,
      "best_of_k": 8,
      "n_groups_per_rollout": 8,
      "temperature": 1.0,
      "num_samples_per_iteration": 16,
      "salience_pct_threshold": 0.0
    }
  }
}
```

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "queue_position": 1,
  "queue_size": 1
}
```

**Parameters:**
- `text` (string, required): Text to analyze (max length configurable via `MAX_TEXT_LENGTH`)
- `options` (object, optional): Analysis configuration
  - `batch_size` (int): Processing batch size (auto-calculated if not provided)
  - `temperature` (float): Sampling temperature [0.1-2.0]
  - `seed` (int): Random seed for reproducibility
  - `no_eval` (bool): Skip evaluation metrics (MSE, KL)
  - `tuned_lens` (bool): Include TunedLens predictions
  - `logit_lens_analysis` (bool): Add logit-lens predictions
  - `calculate_salience` (bool): Calculate salience scores
  - `optimize_explanations_config` (object): Optimization settings
    - `best_of_k` (int): Number of rollouts [1-64]
    - `n_groups_per_rollout` (int): Batch size for rollouts [1-32]
    - `temperature` (float): Generation temperature [0.1-2.0]

### 4. Get Request Status

**GET /status/{request_id}**

Get the status of a queued or completed request.

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": "2024-01-20T10:30:00Z",
  "started_at": "2024-01-20T10:30:01Z",
  "completed_at": "2024-01-20T10:30:05Z",
  "processing_time": 4.2,
  "result": {
    "tokens": [...],
    "metadata": {...}
  },
  "error": null
}
```

**Status Values:**
- `queued`: Request is waiting to be processed
- `processing`: Request is currently being processed
- `completed`: Request completed successfully
- `failed`: Request failed with error
- `cancelled`: Request was cancelled

### 5. GPU Statistics

**GET /api/gpu_stats**

Get current GPU utilization and memory statistics.

**Response:**
```json
{
  "available": true,
  "utilization": 45,
  "memory_used": 12.5,
  "memory_total": 80.0,
  "memory_percent": 15.6,
  "peak_utilization": 85
}
```

### 6. Metrics

**GET /metrics**

Get service metrics for monitoring.

**Response:**
```json
{
  "total_requests": 150,
  "completed_requests": 145,
  "failed_requests": 2,
  "queue_size": 3,
  "average_processing_time": 3.7,
  "active_websocket_connections": 5
}
```

### 7. Reset Service

**POST /reset**

Reset the model and clear GPU memory.

**Response:**
```json
{
  "status": "reset",
  "message": "Model cleared and reloaded."
}
```

## Model Management API (v2)

### 8. List Model Groups

**GET /api/v2/models**

List all available model groups and their models.

**Response:**
```json
{
  "groups": [
    {
      "group_id": "gemma3-27b-it",
      "group_name": "Gemma-3 27B",
      "description": "Instruction-tuned Gemma-3 27B models",
      "base_model": "google/gemma-2-27b-it",
      "models": [
        {
          "id": "gemma3-27b-it-layer12",
          "name": "Layer 12",
          "layer": 12,
          "description": "Early reasoning layer",
          "batch_size": 32
        }
      ],
      "is_loaded": true,
      "estimated_memory": "52GB"
    }
  ],
  "current_model": "gemma3-27b-it-layer12",
  "current_group": "gemma3-27b-it",
  "is_switching": false,
  "model_status": {
    "model_id": "gemma3-27b-it-layer12",
    "group_id": "gemma3-27b-it",
    "is_loaded": true,
    "cache_info": {
      "groups_loaded": ["gemma3-27b-it"],
      "models_cached": ["gemma3-27b-it-layer12"],
      "base_locations": {}
    }
  }
}
```

### 9. Switch Model

**POST /api/v2/models/switch**

Switch to a different model, potentially in a different group.

**Request Body:**
```json
{
  "model_id": "qwen2.5-14b-instruct-layer20"
}
```

**Response:**
```json
{
  "status": "success",
  "model_id": "qwen2.5-14b-instruct-layer20",
  "group_id": "qwen2.5-14b-instruct",
  "message": "Switched to model qwen2.5-14b-instruct-layer20",
  "group_switched": true,
  "model_info": {
    "model_id": "qwen2.5-14b-instruct-layer20",
    "group_id": "qwen2.5-14b-instruct",
    "is_loaded": true
  }
}
```

### 10. Preload Model Group

**POST /api/v2/groups/{group_id}/preload**

Preload all models in a group for fast switching.

**Response:**
```json
{
  "status": "success",
  "message": "Preloaded all models in group gemma3-27b-it",
  "group_id": "gemma3-27b-it"
}
```

### 11. Get Memory Status

**GET /api/v2/models/memory**

Get detailed memory usage information.

**Response:**
```json
{
  "memory": {
    "allocated_gb": 52.3,
    "reserved_gb": 58.0,
    "free_gb": 22.0,
    "total_gb": 80.0
  },
  "cache_info": {
    "groups_loaded": ["gemma3-27b-it"],
    "models_cached": ["gemma3-27b-it-layer12", "gemma3-27b-it-layer20"],
    "base_model_cache": {
      "gemma3-27b-it": "/dev/shm/model_cache/..."
    }
  },
  "current_model": "gemma3-27b-it-layer12",
  "current_group": "gemma3-27b-it"
}
```

### 12. Get Model Info

**GET /api/v2/models/{model_id}/info**

Get detailed information about a specific model.

**Response:**
```json
{
  "model_id": "gemma3-27b-it-layer12",
  "group_id": "gemma3-27b-it",
  "group_name": "Gemma-3 27B",
  "model_name": "Layer 12",
  "description": "Early reasoning layer",
  "layer": 12,
  "batch_size": 32,
  "base_model": "google/gemma-2-27b-it",
  "is_loaded": true,
  "is_current": true
}
```

## WebSocket API

### Connection

**URL:** `ws://localhost:8000/ws`

### Message Types

#### 1. Analyze Request

**Client sends:**
```json
{
  "type": "analyze",
  "text": "Text to analyze",
  "model_id": "gemma3-27b-it-layer12",
  "client_request_id": "client-generated-id",
  "options": {
    "temperature": 1.0,
    "optimize_explanations_config": {
      "best_of_k": 8
    }
  }
}
```

**Server responses:**

**Queued:**
```json
{
  "type": "queued",
  "request_id": "server-generated-id",
  "client_request_id": "client-generated-id",
  "queue_position": 1,
  "queue_size": 3,
  "context": "analysis"
}
```

**Processing:**
```json
{
  "type": "processing",
  "request_id": "server-generated-id",
  "client_request_id": "client-generated-id",
  "message": "Processing batch 1/5",
  "batch_idx": 0,
  "num_batches": 5,
  "stage": "explanation_generation"
}
```

**Result:**
```json
{
  "type": "result",
  "request_id": "server-generated-id",
  "client_request_id": "client-generated-id",
  "tokens": [
    {
      "position": 0,
      "token": "The",
      "explanation": "Definite article indicating a specific noun",
      "explanation_structured": ["Determiner", "Definite", "Singular"],
      "token_salience": [0.85, 0.12, 0.03],
      "mse": 0.0023,
      "kl_divergence": 0.0012,
      "relative_rmse": 0.048
    }
  ],
  "metadata": {
    "model_id": "gemma3-27b-it-layer12",
    "processing_time": 3.2,
    "batch_size": 32
  }
}
```

#### 2. Generate Request

**Client sends:**
```json
{
  "type": "generate",
  "text": "Once upon a time",
  "options": {
    "max_length": 100,
    "temperature": 0.8,
    "top_p": 0.9,
    "model_id": "gemma3-27b-it-layer12"
  }
}
```

**Server streams tokens:**
```json
{
  "type": "token",
  "token": " there",
  "position": 4,
  "finish_reason": null
}
```

#### 3. List Model Groups

**Client sends:**
```json
{
  "type": "list_model_groups",
  "refresh": false
}
```

**Server responds with model groups list (same format as REST endpoint)**

#### 4. Switch Model

**Client sends:**
```json
{
  "type": "switch_model_grouped",
  "model_id": "qwen2.5-14b-instruct-layer20"
}
```

**Server responses:**

**Group switch queued:**
```json
{
  "type": "group_switch_queued",
  "request_id": "switch-request-id",
  "model_id": "qwen2.5-14b-instruct-layer20",
  "target_group_id": "qwen2.5-14b-instruct",
  "queue_position": 1,
  "active_requests": 2,
  "queued_ahead": 0,
  "message": "Group switch queued at position 1. Will start after 2 request(s) complete."
}
```

**Switch complete:**
```json
{
  "type": "model_switch_complete",
  "model_id": "qwen2.5-14b-instruct-layer20",
  "message": "Successfully switched to qwen2.5-14b-instruct-layer20",
  "model_info": {
    "model_id": "qwen2.5-14b-instruct-layer20",
    "display_name": "Qwen2.5 14B - Layer 20",
    "layer": 20,
    "auto_batch_size_max": 64,
    "generation_config": {
      "max_new_tokens": 512,
      "temperature": 0.7
    }
  }
}
```

#### 5. Queue Updates

**Server broadcasts periodically:**
```json
{
  "type": "queue_update",
  "queue_size": 5,
  "queued_requests": 3,
  "processing_requests": 2,
  "total_active": 5,
  "queued_ids": ["req-1", "req-2", "req-3"]
}
```

#### 6. Cancel/Interrupt Request

**Client sends:**
```json
{
  "type": "cancel_request",
  "request_id": "request-to-cancel"
}
```

**Server responds:**
```json
{
  "type": "request_cancelled",
  "request_id": "request-to-cancel",
  "message": "Request cancelled successfully"
}
```

## Data Models

### AnalyzeRequest

```typescript
interface AnalyzeRequest {
  text: string;  // Required, min length 1, max configurable
  options?: AnalyzeOptions;
}
```

### AnalyzeOptions

```typescript
interface AnalyzeOptions {
  // Processing
  batch_size?: number;  // Auto-calculated if not provided
  seed?: number;  // Default: 42
  
  // Analysis features
  no_eval?: boolean;  // Skip evaluation metrics
  tuned_lens?: boolean;  // Include TunedLens predictions
  logit_lens_analysis?: boolean;  // Add logit-lens predictions
  
  // Generation parameters
  temperature?: number;  // Range: 0.1-2.0, default: 1.0
  do_hard_tokens?: boolean;
  return_structured?: boolean;  // Default: true
  move_devices?: boolean;
  
  // Evaluation options
  no_kl?: boolean;  // Skip KL divergence
  calculate_salience?: boolean;  // Default: true
  calculate_token_salience?: boolean;  // Default: true
  
  // Token manipulation
  add_tokens?: string[];
  replace_left?: string;
  replace_right?: string;
  
  // Optimization
  optimize_explanations_config?: OptimizeExplanationsConfig;
}
```

### OptimizeExplanationsConfig

```typescript
interface OptimizeExplanationsConfig {
  use_batched?: boolean;  // Default: true
  best_of_k?: number;  // Range: 1-64, default: 8
  n_groups_per_rollout?: number;  // Range: 1-32, default: 8
  temperature?: number;  // Range: 0.1-2.0, default: 1.0
  num_samples_per_iteration?: number;  // Range: 1-64, default: 16
  salience_pct_threshold?: number;  // Range: 0.0-1.0, default: 0.0
}
```

### TokenAnalysis

```typescript
interface TokenAnalysis {
  position: number;
  token: string;
  explanation: string;
  explanation_structured?: string[];
  token_salience?: number[];
  mse: number;
  kl_divergence: number;
  relative_rmse?: number;
  tuned_lens_top?: string;
  logit_lens_top?: string;
  layer?: number;
}
```

## Error Handling

### HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Invalid or missing API key
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### WebSocket Error Messages

```json
{
  "type": "error",
  "error": "Detailed error message",
  "request_id": "associated-request-id"
}
```

## Slim API

In addition to the main API endpoints documented above, there is also a **Slim API** available at `/api/slim` that provides a simplified interface with minimal configuration options.

### Key Differences from Main API:

1. **Simplified Parameters**: Only essential parameters are exposed
2. **Chat Format**: All endpoints use standard chat message format
3. **Automatic Defaults**: All processing options use sensible defaults
4. **Additional Endpoints**: Includes a `send_message` endpoint for single-turn conversations

### Available Slim API Endpoints:

- **POST /api/slim/generate** - Generate text continuations
- **POST /api/slim/analyze** - Analyze text with token salience
- **POST /api/slim/send_message** - Send a message and get a single response
- **GET /api/slim/status/{request_id}** - Check request status
- **GET /api/slim/models** - List available model groups
- **GET /api/slim/models/{group_id}/layers** - List layers for a model group

For detailed Slim API documentation, see [SLIM_API_DOCUMENTATION.md](./SLIM_API_DOCUMENTATION.md).

## Examples

### Python Client Example

```python
import asyncio
import websockets
import json

async def analyze_text():
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        # Send analysis request
        request = {
            "type": "analyze",
            "text": "The capital of France is Paris.",
            "options": {
                "temperature": 1.0,
                "optimize_explanations_config": {
                    "best_of_k": 8
                }
            }
        }
        
        await websocket.send(json.dumps(request))
        
        # Receive responses
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            
            print(f"Received: {data['type']}")
            
            if data['type'] == 'result':
                # Process results
                for token in data['tokens']:
                    print(f"{token['token']}: {token['explanation']}")
                break
            elif data['type'] == 'error':
                print(f"Error: {data['error']}")
                break

# Run the client
asyncio.run(analyze_text())
```

### cURL REST API Example

```bash
# Analyze text
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "text": "Hello world",
    "options": {
      "temperature": 1.0,
      "batch_size": 32
    }
  }'

# Check status
curl "http://localhost:8000/status/550e8400-e29b-41d4-a716-446655440000" \
  -H "Authorization: Bearer your-api-key"

# Get GPU stats
curl "http://localhost:8000/api/gpu_stats"
```

### JavaScript WebSocket Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
    // Send analysis request
    ws.send(JSON.stringify({
        type: 'analyze',
        text: 'The quick brown fox',
        options: {
            temperature: 1.0,
            calculate_salience: true
        }
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch (data.type) {
        case 'queued':
            console.log(`Queued at position ${data.queue_position}`);
            break;
        case 'processing':
            console.log(`Processing: ${data.message}`);
            break;
        case 'result':
            console.log('Analysis complete:', data.tokens);
            break;
        case 'error':
            console.error('Error:', data.error);
            break;
    }
};
```

## Environment Variables Reference

```bash
# Model Configuration
CHECKPOINT_PATH=/path/to/checkpoint.pt
DEVICE=cuda
BATCH_SIZE=32
USE_BF16=true
LAZY_LOAD_MODEL=false

# API Configuration
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
API_KEY=your-secret-key
MAX_QUEUE_SIZE=100
MAX_TEXT_LENGTH=1000
RATE_LIMIT_PER_MINUTE=60

# Server Configuration
HOST=0.0.0.0
PORT=8000
REQUEST_TIMEOUT=300
ENABLE_DOCS=false  # Set to true to enable /docs endpoint

# Optional Features
TUNED_LENS_DIR=/path/to/tuned/lens
ENABLE_TRUSTED_HOST=true
ENVIRONMENT=production  # or development
```

## Monitoring and Observability

The API provides several endpoints for monitoring:

1. `/health` - Basic health check
2. `/metrics` - Detailed service metrics
3. `/api/gpu_stats` - Real-time GPU utilization
4. WebSocket queue updates - Real-time queue status

These endpoints can be integrated with monitoring tools like Prometheus, Grafana, or custom dashboards.

## Security Considerations

1. **API Key**: Always use API keys in production
2. **CORS**: Configure `ALLOWED_ORIGINS` appropriately
3. **Rate Limiting**: Adjust limits based on your needs
4. **Input Validation**: Text length and parameter ranges are validated
5. **Trusted Host**: Enable `ENABLE_TRUSTED_HOST` in production

## Performance Tips

1. **Batch Size**: Larger batch sizes improve throughput but increase latency
2. **Model Groups**: Preload frequently used model groups
3. **Queue Management**: Monitor queue size and adjust `MAX_QUEUE_SIZE`
4. **GPU Memory**: Use `/api/gpu_stats` to monitor memory usage
5. **WebSocket**: Use WebSocket for real-time requirements

## Troubleshooting

Common issues and solutions:

1. **Connection Refused**: Check if the service is running on the correct port
2. **401 Unauthorized**: Verify API key is correct
3. **429 Too Many Requests**: Implement client-side rate limiting
4. **WebSocket Disconnects**: Implement reconnection logic
5. **Out of Memory**: Reduce batch size or switch to smaller model

For detailed logs, set `LOG_LEVEL=debug` in environment variables.