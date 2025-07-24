# Consistency Lens API Backend

FastAPI backend for the Talkative Autoencoder (Consistency Lens) web interface.

## Quick Start

### Local Development

1. **Prerequisites**
   - Python 3.10+
   - CUDA-capable GPU
   - Model checkpoint file

2. **Setup**
   ```bash
   # From the talkative_autoencoder directory
   make  # Install uv and dependencies
   
   # Navigate to backend
   cd website/backend
   
   # Install Python dependencies
   uv pip install -r requirements.txt
   ```

3. **Configuration**
   ```bash
   # Copy example env file
   cp .env.example .env
   
   # Edit .env with your settings
   ```

4. **Run the server**
   ```bash
   uv run uvicorn app.main:app --reload
   ```

   The API will be available at http://localhost:8000

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `GET /metrics` - Server metrics
- `POST /analyze` - Queue text analysis
- `GET /status/{request_id}` - Check request status
- `WS /ws` - WebSocket for real-time updates

## WebSocket Usage

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

// Send analysis request
ws.send(JSON.stringify({
  type: 'analyze',
  text: 'Your text here',
  options: { batch_size: 32 }
}));

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

## Testing

```bash
uv run pytest tests/
```

## Deployment

See [deployment scripts](./scripts/) for RunPod deployment instructions.

## Architecture

- **FastAPI**: Modern async web framework
- **WebSocket**: Real-time bidirectional communication
- **Queue System**: Handles concurrent requests
- **Lazy Loading**: Model loads on first request to reduce startup time