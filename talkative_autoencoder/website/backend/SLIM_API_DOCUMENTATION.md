# Slim API Documentation

The Slim API provides a simplified interface for text generation and analysis with minimal configuration options. It runs in parallel with the main API at `/api/slim`.

**Important**: The Slim API assumes that entire model groups are pre-loaded into memory. Before using the Slim API, ensure all models in your target group (e.g., all layers of "gemma3-27b-it") are loaded. This allows instant switching between different models within a group without loading delays.

## Endpoints

### 1. Generate Text

**POST /api/slim/generate**

Generate text continuations from chat-formatted messages.

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about the weather"}
  ],
  "temperature": 0.8,
  "n_continuations": 1,
  "n_tokens": 100,
  "model_group": "gemma3-27b-it"
}
```

**Parameters:**
- `messages` (required): List of chat messages with `role` and `content`
- `model_group` (required): Model group ID (e.g., "gemma3-27b-it")
- `temperature` (optional): Sampling temperature 0.1-2.0 (default: 1.0)
- `n_continuations` (optional): Number of continuations 1-10 (default: 1)
- `n_tokens` (optional): Max tokens to generate 1-512 (default: 100)
- `model_id` (optional): Specific model ID within the group (e.g., "gemma3-27b-it-layer20")

### 2. Analyze Text

**POST /api/slim/analyze**

Analyze text with token salience scores.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "The capital of France is Paris."}
  ],
  "calculate_token_salience": true,
  "best_of_k": 8,
  "model_group": "gemma3-27b-it",
  "last_n_messages": 2
}
```

**Parameters:**
- `messages` (required): List of chat messages to analyze
- `model_group` (required): Model group ID
- `calculate_token_salience` (optional): Calculate salience scores (default: true)
- `best_of_k` (optional): Number of explanation rollouts 1-32 (default: 8)
- `model_id` (optional): Specific model ID within the group (e.g., "gemma3-27b-it-layer20")
- `last_n_messages` (optional): Only analyze the last N messages (e.g., 2 for last user/assistant turn). This significantly improves performance for long conversations by skipping analysis of earlier messages.

### 3. Send Message

**POST /api/slim/send_message**

Send a message and get a single assistant response. Unlike generate, this endpoint returns only the assistant's response text, not the full conversation. Supports prefill.

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.8,
  "max_tokens": 1024,
  "model_group": "gemma3-27b-it"
}
```

**Parameters:**
- `messages` (required): List of chat messages (conversation history)
- `model_group` (required): Model group ID (e.g., "gemma3-27b-it")
- `temperature` (optional): Sampling temperature 0.1-2.0 (default: 1.0)
- `max_tokens` (optional): Maximum tokens to generate 1-4096 (default: 1024)
- `model_id` (optional): Specific model ID within the group

**Response:**
The response will contain only the assistant's reply text, automatically stopping at the model's end-of-turn marker.

### 4. Check Status

**GET /api/slim/status/{request_id}**

Get the status of a generation, analysis, or send_message request.

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {...},
  "processing_time": 3.2
}
```

### 4. List Model Groups

**GET /api/slim/models**

List available model groups.

**Response:**
```json
{
  "model_groups": [
    {
      "id": "gemma3-27b-it",
      "name": "Gemma-3 27B Instruct",
      "description": "Instruction-tuned Gemma-3 27B",
      "is_loaded": true,
      "default_layer": "gemma3-27b-it-layer12"
    }
  ],
  "current_model_id": "gemma3-27b-it-layer12"
}
```

### 5. List Model Layers

**GET /api/slim/models/{group_id}/layers**

List available layers/sub-models for a specific model group.

**Response:**
```json
{
  "group_id": "gemma3-27b-it",
  "group_name": "Gemma-3 27B Instruct",
  "is_loaded": true,
  "layers": [
    {
      "id": "gemma3-27b-it-layer12",
      "name": "Layer 12",
      "layer": 12,
      "description": "Early reasoning layer",
      "is_current": true
    },
    {
      "id": "gemma3-27b-it-layer20",
      "name": "Layer 20",
      "layer": 20,
      "description": "Mid-level reasoning",
      "is_current": false
    }
  ],
  "default_layer": "gemma3-27b-it-layer12"
}
```

## Examples

### Python Client

```python
import requests

# Generate text with default model (first in group)
response = requests.post("http://localhost:8000/api/slim/generate", json={
    "messages": [
        {"role": "user", "content": "Write a haiku about coding"}
    ],
    "temperature": 0.7,
    "n_tokens": 50,
    "model_group": "gemma3-27b-it"
})

# Generate text with specific model/layer
response = requests.post("http://localhost:8000/api/slim/generate", json={
    "messages": [
        {"role": "user", "content": "Write a haiku about coding"}
    ],
    "temperature": 0.7,
    "n_tokens": 50,
    "model_group": "gemma3-27b-it",
    "model_id": "gemma3-27b-it-layer20"  # Use layer 20 specifically
})

request_id = response.json()["request_id"]

# Check status
status = requests.get(f"http://localhost:8000/api/slim/status/{request_id}")
print(status.json())

# Analyze text with specific model/layer
analysis = requests.post("http://localhost:8000/api/slim/analyze", json={
    "messages": [
        {"role": "user", "content": "The quick brown fox"}
    ],
    "calculate_token_salience": True,
    "best_of_k": 16,
    "model_group": "gemma3-27b-it",
    "model_id": "gemma3-27b-it-layer12",  # Analyze with layer 12
    "last_n_messages": 2  # Only analyze last 2 messages for efficiency
})

# Send a message and get response
response = requests.post("http://localhost:8000/api/slim/send_message", json={
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "model_group": "gemma3-27b-it"
})

request_id = response.json()["request_id"]

# Check status and get the response
import time
while True:
    status = requests.get(f"http://localhost:8000/api/slim/status/{request_id}")
    result = status.json()
    if result["status"] == "completed":
        print("Assistant:", result["result"]["response"])
        break
    time.sleep(0.5)
```

### cURL Examples

```bash
# Generate text
curl -X POST http://localhost:8000/api/slim/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "model_group": "gemma3-27b-it",
    "temperature": 0.8
  }'

# Analyze text
curl -X POST http://localhost:8000/api/slim/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Test text"}],
    "model_group": "gemma3-27b-it",
    "best_of_k": 8
  }'

# Send message
curl -X POST http://localhost:8000/api/slim/send_message \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "model_group": "gemma3-27b-it",
    "temperature": 0.5,
    "max_tokens": 50
  }'
```

## Key Features

1. **Simplified Interface**: Only essential parameters exposed
2. **Chat Format**: Uses standard chat message format for all endpoints
3. **Send Message**: Get single assistant responses with automatic end-of-turn detection
4. **Automatic Defaults**: All processing options use sensible defaults
5. **Model Groups**: Easy selection of model groups without layer specification
6. **Queued Processing**: Requests are queued and processed asynchronously

## Default Settings

- Model Selection: Uses the first available model in the selected model group (typically the lowest layer)
- Batch size: Auto-calculated based on GPU memory
- Seed: 42 for reproducibility
- All evaluation metrics enabled for analysis
- Structured explanations enabled
- Batched optimization enabled

## Notes

- **Important**: The slim API assumes the entire model group is already loaded into memory
- When you select a model group (e.g., "gemma3-27b-it"), all layers are available for immediate use
- The API defaults to the first model/layer in the group if not specified
- All requests are queued and processed asynchronously
- Use the status endpoint to check when your request is complete
- Authentication follows the same rules as the main API (Bearer token if configured)