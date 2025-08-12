# Talkative Autoencoder Backend

FastAPI backend server for the Talkative Autoencoder web interface.

## Overview

This backend provides:
- WebSocket endpoint for real-time analysis
- Model loading and inference
- Queue management for multiple concurrent requests
- Rate limiting and security features

## Prerequisites

1. **Set up the environment** using uv:
   ```bash
   cd /workspace/kitf/talkative-probes/talkative_autoencoder
   source scripts/ensure_env.sh
   ```

2. **GPU Requirements**: The backend requires a GPU with sufficient memory to load the model. Tested on NVIDIA A100 and H100.

## Running the Backend

### Development Mode

```bash
cd /workspace/kitf/talkative-probes/talkative_autoencoder
uv run uvicorn website.backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
or just 
```bash
make run
```

This starts the server with:
- HTTP endpoint: http://localhost:8000
- WebSocket endpoint: ws://localhost:8765
- API documentation: http://localhost:8000/docs

### Production Mode

For production deployment with multiple workers (not implemented yet):

```bash
uv run uvicorn website.backend.app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
```

## Configuration

### Environment Variables

Create a `.env` file in `website/backend/`. Backend `.env` overrides the repo root `.env` for overlapping keys.

For complete details (keys, types, precedence) see `ENVIRONMENT.md`.

Minimal example:
```env
# Startup default group (env overrides JSON settings.default_group)
DEFAULT_GROUP=gemma3-27b-it

# Infrastructure
DEVICES=cuda:0
NUM_WORKERS_PER_GPU=1
ALLOWED_ORIGINS=http://localhost:3000
API_KEY=
MAX_QUEUE_SIZE=100
MAX_TEXT_LENGTH=20000
HOST=0.0.0.0
PORT=8000
REQUEST_TIMEOUT=300
RATE_LIMIT_PER_MINUTE=60
LAZY_LOAD_MODEL=false
TUNED_LENS_DIR=

# Global model/group behavior overrides
GLOBAL_USE_BF16=true
GLOBAL_NO_ORIG=true
GLOBAL_BATCH_SIZE=48
GLOBAL_AUTO_BATCH_SIZE_MAX=768
```

### Model Loading

At startup in non-lazy mode, the default group is selected by:
1. `DEFAULT_GROUP` env (if set)
2. Otherwise `website/backend/model_groups.json` → `settings.default_group`
3. Otherwise no default is loaded

#### CORS
- `ALLOWED_ORIGINS` can be a comma-separated list or `*` to allow any origin.
- If not `*`, common localhost origins are auto-added.

## API Documentation

For comprehensive API documentation including all endpoints, parameters, request/response formats, and examples, see:

**[API_DOCUMENTATION.md](./API_DOCUMENTATION.md)**

### Quick Reference

#### Main Endpoints
- **WebSocket**: `ws://localhost:8000/ws` - Real-time analysis and model management
- **REST API**: `http://localhost:8000` - HTTP endpoints for analysis, status, and monitoring

#### Key Features
- Text analysis with neural language models
- Model group management and switching
- Real-time processing via WebSocket
- Queue management for concurrent requests
- GPU resource monitoring
- Rate limiting and optional API key authentication

#### Common Operations
- `POST /analyze` - Queue text for analysis
- `GET /status/{request_id}` - Check request status
- `GET /api/v2/models` - List available model groups
- `POST /api/v2/models/switch` - Switch to a different model
- `GET /api/gpu_stats` - Monitor GPU utilization

## Development

### Dependencies

The backend dependencies are specified in `pyproject.toml`:
- FastAPI for the web framework
- PyTorch for model inference
- Transformers for model loading
- Additional ML libraries as needed

### Running Tests

```bash
cd /workspace/kitf/talkative-probes/talkative_autoencoder
uv run pytest website/backend/tests/ -v
```

### Code Quality

```bash
# Format code
uv run black website/backend/

# Lint code
uv run ruff website/backend/

# Type checking
uv run mypy website/backend/
```

## Deployment

### Docker

A Dockerfile is provided for containerized deployment:

```bash
docker build -t talkative-backend ./website/backend
docker run -p 8000:8000 -p 8765:8765 --gpus all talkative-backend
```

### RunPod

For RunPod deployment:
1. Use the provided Docker image
2. Set environment variables in RunPod configuration
3. Ensure GPU is allocated
4. Use the RunPod proxy URL for frontend configuration

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use a smaller model
2. **WebSocket Connection Failed**: Check CORS settings and allowed origins
3. **Model Loading Failed**: Verify checkpoint path and GPU availability

### Logs

Enable debug logging:
```bash
LOG_LEVEL=debug uv run python -m website.backend.app.main
```

## Architecture

```
backend/
├── app/
│   ├── main.py           # FastAPI application and lifecycle
│   ├── websocket.py      # WebSocket connection handling
│   ├── inference.py      # Model loading and analysis
│   ├── models.py         # Pydantic data models
│   └── config.py         # Configuration management
├── tests/                # Test suite
├── pyproject.toml        # Dependencies and tool configuration
└── README.md            # This file
```