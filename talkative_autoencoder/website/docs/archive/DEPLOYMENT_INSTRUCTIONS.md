# Talkative Autoencoder Web Interface - Deployment Instructions

## Overview

This web interface provides a real-time visualization tool for the Talkative Autoencoder (Consistency Lens) model. It consists of:
- A FastAPI backend with WebSocket support for real-time updates
- A web frontend with parameter controls and visualization
- Support for deployment on RunPod with H100 GPUs

## Prerequisites

1. **RunPod Account** with access to H100 GPUs
2. **GitHub Repository** (public): https://github.com/kitft/talkative-probes
3. **Domain/GitHub Pages** for hosting the frontend (e.g., kitft.github.io)
4. **Model Checkpoint**: The system expects a checkpoint at `/workspace/checkpoints/qwen2_5_WCHAT_14b_frozen_nopostfix.pt`

## Critical Issues Already Fixed

The following critical issues have been addressed in the current code:

1. ✅ **Frontend State Management**: Added `currentResultData` variable to properly store and re-render results
2. ✅ **Backend Race Condition**: Added `_load_lock` to prevent concurrent model loading
3. ✅ **API URL Configuration**: Frontend now uses `window.location.origin` for production
4. ✅ **Datetime Handling**: Fixed datetime serialization in the status endpoint
5. ✅ **CORS Security**: Restricted allowed methods to ["GET", "POST"] and headers to ["Content-Type", "Authorization"]

## Deployment Configuration

### Frontend API URL Configuration

The frontend is configured to automatically use the same domain for API requests (`window.location.origin`). This works perfectly when:
- Both frontend and backend are served from the same domain
- You're using a reverse proxy to route requests

**⚠️ Cross-Domain Deployment (e.g., GitHub Pages + RunPod):**

If you're hosting the frontend on a different domain than your API (e.g., frontend on GitHub Pages, API on RunPod), you MUST update the API URL:

**File:** `website/frontend/app.js`

```javascript
// Change from:
const API_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : window.location.origin;

// To:
const API_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : 'https://YOUR-RUNPOD-POD-ID-8000.proxy.runpod.net';
```

Replace `YOUR-RUNPOD-POD-ID` with your actual RunPod pod ID.

## Environment Management

### Understanding the Project Structure

This project has two interconnected parts:
1. **Main Project** (`/talkative-probes/talkative_autoencoder/`): The core ML model and training code
2. **Web Backend** (`/talkative-probes/talkative_autoencoder/website/backend/`): The FastAPI web interface

### The ensure_env.sh Script

The main project uses `scripts/ensure_env.sh` for environment management across multiple nodes in distributed computing environments. This script:

- Installs `uv` if not present
- Sets up a shared UV cache at `/workspace/.uv_cache`
- Creates node-local virtual environments for performance
- Exports helper functions for running commands

**For the web backend**, we use standard `uv` commands directly since it's a simpler, single-node deployment.

### Backend-Specific Environment

The web backend has its own `pyproject.toml` with:
- Web framework dependencies (FastAPI, uvicorn, etc.)
- ML dependencies that match the main project versions
- Optional dev/test dependencies

To work with the backend:
```bash
cd website/backend
uv sync              # Install all dependencies
uv sync --all-extras # Include dev dependencies
uv run python ...    # Run commands in the environment
```

## RunPod Deployment Steps

### 1. Create RunPod Pod

1. Log into RunPod
2. Create a new pod with:
   - **GPU:** H100 (1x)
   - **Container Image:** `runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04`
   - **Disk Space:** At least 100GB
   - **Exposed Ports:** 8000 (HTTP)

### 2. Setup Script

Create and run this setup script on the RunPod pod:

```bash
#!/bin/bash
# Save as setup_talkative_web.sh

# Update system
apt-get update && apt-get install -y git curl

# Clone the repository
cd /workspace
git clone https://github.com/kitft/talkative-probes.git
cd talkative-probes/talkative_autoencoder

# Setup Python environment
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install dependencies
cd website/backend
uv sync  # This creates venv and installs all dependencies

# Create checkpoint directory
mkdir -p /workspace/checkpoints

# IMPORTANT: Upload your checkpoint file to /workspace/checkpoints/
echo "Please upload your checkpoint to /workspace/checkpoints/qwen2_5_WCHAT_14b_frozen_nopostfix.pt"

# Create .env file
cat > .env << EOF
CHECKPOINT_PATH=/workspace/checkpoints/qwen2_5_WCHAT_14b_frozen_nopostfix.pt
DEVICE=cuda
ALLOWED_ORIGINS=https://kitft.github.io,http://localhost:3000
HOST=0.0.0.0
PORT=8000
EOF

# Start the server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Upload Model Checkpoint

Use RunPod's file upload feature or `curl`/`wget` to download your checkpoint to:
```
/workspace/checkpoints/qwen2_5_WCHAT_14b_frozen_nopostfix.pt
```

### 4. Get Your RunPod URL

Your API will be available at:
```
https://YOUR-POD-ID-8000.proxy.runpod.net
```

Copy this URL - you'll need it for the frontend configuration if hosting on a different domain.

## Frontend Deployment (GitHub Pages)

### 1. Update Frontend Files (if needed)

If hosting on GitHub Pages (different domain from API):
1. Fork or clone the repository
2. Update `website/frontend/app.js` with your RunPod API URL
3. Copy the frontend files to your GitHub Pages repository:

```bash
# In your kitft.github.io repository
mkdir -p talkative-autoencoder
cp -r path/to/talkative-probes/talkative_autoencoder/website/frontend/* talkative-autoencoder/
```

### 2. Create an Index or Redirect

In your main `index.html` at the root of kitft.github.io, add a link or redirect:

```html
<!-- Option 1: Link -->
<a href="/talkative-autoencoder/">Talkative Autoencoder Demo</a>

<!-- Option 2: JavaScript Redirect -->
<script>
// Redirect to the demo if accessing a specific path
if (window.location.pathname === '/talkative-demo') {
    window.location.href = '/talkative-autoencoder/';
}
</script>
```

### 3. Commit and Push

```bash
git add .
git commit -m "Add Talkative Autoencoder web interface"
git push origin main
```

## Testing

1. **Local Testing:**
   ```bash
   # Backend
   cd website/backend
   uv sync  # First time setup
   uv run uvicorn app.main:app --reload

   # Run tests
   uv run pytest

   # Frontend - serve with any HTTP server
   cd website/frontend
   python -m http.server 3000
   ```

2. **Production Testing:**
   - Visit: https://kitft.github.io/talkative-autoencoder/
   - Enter some text and adjust parameters
   - Click "Analyze" and watch for real-time results

## Monitoring & Maintenance

### Health Checks

- API Health: `https://YOUR-POD-ID-8000.proxy.runpod.net/health`
- Metrics: `https://YOUR-POD-ID-8000.proxy.runpod.net/metrics`

### Logs

On RunPod, check logs with:
```bash
# If using systemd
journalctl -u talkative-autoencoder -f

# If running directly
# Check the terminal output
```

### Common Issues

1. **Model fails to load:**
   - Check checkpoint path exists
   - Verify GPU memory (H100 should be sufficient)
   - Check CUDA is available

2. **WebSocket connection fails:**
   - Ensure RunPod proxy supports WebSockets
   - Check CORS settings include your domain
   - Verify the WebSocket URL is correctly formed

3. **Slow inference:**
   - Monitor GPU utilization
   - Adjust batch sizes in frontend controls
   - Check the queue status via `/metrics`

## Security Considerations

1. **API Key (Optional):** Add API key authentication by setting `API_KEY` environment variable
2. **CORS:** Already restricted to specific origins, methods, and headers
3. **Rate Limiting:** Built-in rate limiting at 60 requests/minute per IP
4. **HTTPS:** RunPod proxy provides HTTPS - ensure frontend uses https:// URLs

## Cost Optimization

1. **Auto-shutdown:** Configure RunPod pod to shut down after inactivity
2. **Model Optimization:** Consider using 8-bit quantization (set `LOAD_IN_8BIT=true`)
3. **Batch Processing:** Use larger batch sizes for better GPU utilization

## Advanced Configuration

### Environment Variables

The backend supports these environment variables:

- `CHECKPOINT_PATH`: Path to model checkpoint
- `DEVICE`: CUDA device (default: "cuda")
- `BATCH_SIZE`: Default batch size (default: 32)
- `USE_BF16`: Use bfloat16 precision (default: true)
- `ALLOWED_ORIGINS`: Comma-separated list of allowed origins
- `HOST`: Server host (default: "0.0.0.0")
- `PORT`: Server port (default: 8000)
- `MAX_QUEUE_SIZE`: Maximum request queue size (default: 100)
- `MAX_TEXT_LENGTH`: Maximum input text length (default: 1000)
- `API_KEY`: Optional API key for authentication
- `LOAD_IN_8BIT`: Load model in 8-bit quantization (default: false)
- `MODEL_NAME`: HuggingFace model name (default: "Qwen/Qwen2.5-14B-Instruct")

### Production Deployment with Docker

For a more robust deployment, use the provided Dockerfile:

```bash
# Build the image
docker build -t talkative-autoencoder-api website/backend/

# Run with environment variables
docker run -p 8000:8000 \
  -e CHECKPOINT_PATH=/workspace/checkpoints/your_model.pt \
  -e ALLOWED_ORIGINS=https://yourdomain.com \
  -v /path/to/checkpoints:/workspace/checkpoints \
  --gpus all \
  talkative-autoencoder-api
```

## Support

For issues with:
- **Model/ML Code:** Check the main talkative-probes repository
- **Web Interface:** File issues in the repository
- **RunPod:** Consult RunPod documentation or support

Remember to update the checkpoint path and model name in the configuration files if using a different model!