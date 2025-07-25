# Talkative Autoencoder Website

Web interface and API for the Talkative Autoencoder project.

## Components

### 1. Frontend (`frontend/`)
- **Main Application**: Real-time analysis interface with WebSocket connection
- **Standalone Viewer**: Offline data visualization tool
- **Shared Components**: Modular visualization library used by both

See [frontend/README.md](frontend/README.md) for detailed documentation.

### 2. Backend (`backend/`)
- **FastAPI Server**: Handles model inference and WebSocket connections
- **Queue Management**: Manages concurrent analysis requests
- **Model Integration**: Loads and runs the Talkative Autoencoder model

See [backend/README.md](backend/README.md) for detailed documentation.

## Quick Start

### Prerequisites

```bash
cd /workspace/kitf/talkative-probes/talkative_autoencoder
source scripts/ensure_env.sh
```

This sets up the Python environment with uv and installs all dependencies.

### Where to Run

#### For Development/Testing

**Local Testing (Recommended)**:
- Run everything on your local machine for fast iteration
- Works well if you have any GPU (even small ones)
- Best for frontend development and debugging

**RunPod Testing**:
- Use when you need powerful GPUs (H100/A100)
- Required for large model testing
- Good for production-like testing

**Hybrid Approach (Often Best)**:
- Backend on RunPod (for GPU power)
- Frontend on local machine (for fast iteration)
- Just update `API_URL` in `frontend/app.js` to your RunPod URL

### Running Everything Locally

The easiest way to run both frontend and backend:

```bash
cd /workspace/kitf/talkative-probes/talkative_autoencoder/website
make demo
```

This will:
1. Start the backend server on http://localhost:8000
2. Start the frontend server on http://localhost:8080
3. Open your browser to the frontend

### Individual Components

**Backend only**:
```bash
cd /workspace/kitf/talkative-probes/talkative_autoencoder
uv run python -m website.backend.app.main
```

**Frontend only**:
```bash
cd website/frontend
python -m http.server 8080
```

**Standalone viewer** (no backend needed):
```bash
cd website/frontend
python -m http.server 8081
# Open http://localhost:8081/visualizer-standalone.html
```

### Running with RunPod

When you need more GPU power:

1. **On your RunPod instance**:
   ```bash
   # SSH into RunPod
   ssh root@[your-runpod-ip] -p [port]
   
   # Setup and run backend
   cd /workspace/kitf/talkative-probes/talkative_autoencoder
   source scripts/ensure_env.sh
   uv run python -m website.backend.app.main
   maybe just make run
   ```

2. **On your local machine**:
   ```bash
   # Edit frontend/app.js to use RunPod URL:
   # const API_URL = 'https://[your-pod-id]-8000.proxy.runpod.net';
   
   # Run frontend locally
   cd website/frontend
   python -m http.server 8080
   ```

## Deployment

### GitHub Pages

The frontend is automatically deployed to GitHub Pages:
- Main app: https://kitft.github.io/talkative-autoencoder/
- Data viewer: https://kitft.github.io/data-viewer/

### Backend Deployment

The backend can be deployed to:
- RunPod (GPU instances)
- Any cloud provider with GPU support
- Docker containers

See backend/README.md for deployment details.

## Development Workflow

1. **Make changes** to frontend or backend code
2. **Test locally** using `make demo`
3. **Run tests**: `uv run pytest`
4. **Check code quality**: 
   ```bash
   uv run ruff .
   uv run black .
   ```
5. **Commit and push** to deploy frontend automatically

## Architecture

```
website/
├── frontend/               # Web UI
│   ├── index.html         # Main app
│   ├── app.js            # WebSocket client
│   ├── visualizer-*.js   # Visualization components
│   └── README.md         # Frontend docs
├── backend/               # API server
│   ├── app/              # FastAPI application
│   ├── tests/            # Backend tests
│   └── README.md         # Backend docs
├── Makefile              # Convenience commands
└── README.md            # This file
```

## Key Features

- **Real-time Analysis**: WebSocket connection for live updates
- **Modular Visualization**: Shared components between online and offline viewers
- **Queue Management**: Handle multiple concurrent requests
- **Parameter Control**: Fine-tune analysis with k-rollouts, temperature, etc.
- **Data Export/Import**: Save and share analysis results
- **Responsive Design**: Works on desktop and mobile

## Troubleshooting

### Frontend Issues
- Check browser console for errors
- Verify WebSocket connection to backend
- Ensure correct API_URL in app.js

### Backend Issues
- Check GPU availability: `nvidia-smi`
- Verify model checkpoint exists
- Check logs for memory errors

### Common Problems

1. **"Disconnected" status**: Backend not running or wrong URL
2. **Analysis stuck in queue**: GPU memory full, restart backend
3. **Visualization not loading**: Check browser compatibility

For more help, see the individual README files or open an issue.