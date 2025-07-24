# Talkative Autoencoder Web Interface

A secure, production-ready web interface for the Talkative Autoencoder (Consistency Lens) model with real-time visualization.

## 🚀 Quick Start

```bash
# Local development
make setup
make demo  # Runs both backend and frontend
```

Visit http://localhost:3000 for frontend, http://localhost:8000/docs for API docs.

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Deployment](#quick-deployment)
- [Security Features](#security-features)
- [Development](#development)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  GitHub Pages   │  HTTPS  │    RunPod GPU    │         │     Model       │
│   (Frontend)    │ ──────> │    (Backend)     │ ──────> │   Checkpoint    │
│     FREE        │   WSS   │   $0.40-4/hour   │         │  (Volume Mount) │
└─────────────────┘         └──────────────────┘         └─────────────────┘
```

### Why This Architecture?

- **Cost-Effective**: Free frontend hosting, pay-per-use GPU backend
- **Secure**: HTTPS/WSS encryption, API keys, rate limiting
- **Scalable**: Independent scaling of frontend/backend
- **Simple**: No complex infrastructure to manage

## 🚀 Quick Deployment

### Prerequisites

- RunPod account with credits
- GitHub account with Pages enabled  
- Model checkpoint file (~28GB for 14B model)

### Step 1: Deploy Backend on RunPod

1. **Create RunPod Pod**:
   - GPU: RTX 3090 ($0.44/hr) or A100 ($1.14/hr)
   - Image: `runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel`
   - Disk: 50GB
   - Expose Port: 8000

2. **SSH into pod and run**:
   ```bash
   cd /workspace
   git clone https://github.com/kitft/talkative-probes.git
   cd talkative-probes/talkative_autoencoder/website
   
   # Automated secure setup
   make setup
   make secure-env
   cp backend/.env.secure backend/.env
   
   # Upload your model checkpoint to:
   # /workspace/checkpoints/qwen2_5_WCHAT_14b_frozen_nopostfix.pt
   
   # Start server
   make run
   ```

3. **Note your URL**: `https://YOUR-POD-ID-8000.proxy.runpod.net`

### Step 2: Deploy Frontend

#### Option A: Automated Deployment (Recommended)

1. **Initial Setup (one time)**:
   ```bash
   # In your github.io repository
   cd ~/kitft.github.io
   git submodule add https://github.com/kitft/talkative-probes.git talkative-autoencoder
   git commit -m "Add talkative-autoencoder as submodule"
   git push
   ```

2. **Set up GitHub Actions**:
   - Copy `github-pages-workflow.yml` to your `kitft.github.io` repo as `.github/workflows/update-talkative-autoencoder.yml`
   - In the talkative-probes repo settings, create a Personal Access Token (PAT) with `repo` scope
   - Add it as a secret named `PAGES_UPDATE_TOKEN` in the talkative-probes repo

3. **Deploy Frontend**:
   ```bash
   # Configure the frontend API URL
   cd /workspace/kitf/talkative-probes/talkative_autoencoder/website
   make deploy-frontend RUNPOD_URL=https://YOUR-POD-ID-8000.proxy.runpod.net
   
   # Commit and push - this triggers auto-update!
   git add frontend/app.js
   git commit -m "Update frontend API URL"
   git push
   ```

4. **Access**: `https://kitft.github.io/talkative-autoencoder/talkative_autoencoder/website/frontend/`

#### Option B: Manual Updates

```bash
# After updating frontend in talkative-probes
cd ~/kitft.github.io
git submodule update --remote talkative-autoencoder
git add talkative-autoencoder
git commit -m "Update talkative-autoencoder frontend"
git push
```

### Step 3: Access

Visit: `https://YOUR-USERNAME.github.io/talkative-autoencoder/`

## 🔒 Security Features

### Implemented Security

1. **API Key Authentication**: Required in production (auto-generated)
2. **Rate Limiting**: 60 requests/minute per IP
3. **CORS Protection**: Restricted to configured origins
4. **Host Header Validation**: Prevents host header attacks
5. **HTTPS/WSS**: Automatic via RunPod proxy
6. **Input Validation**: Max text length, Pydantic models
7. **No Direct Port Access**: Port 8000 only accessible through RunPod proxy

### Security Configuration

The `make secure-env` command generates:
- Random 32-byte API key
- Production environment settings
- Proper CORS origins
- Rate limiting configuration

## 💻 Development

### Environment Setup

```bash
# Install dependencies
make setup

# Run tests
make test

# Check security
make security-check

# Run locally
make run            # Backend only
make run-frontend   # Frontend only  
make demo          # Both
```

### Project Structure

```
website/
├── Makefile                # Deployment automation
├── README.md              # This file
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI + security
│   │   ├── inference.py   # Model management
│   │   ├── models.py      # Request/response models
│   │   ├── config.py      # Settings
│   │   └── websocket.py   # WebSocket manager
│   ├── tests/             # Test suite
│   └── pyproject.toml     # Dependencies (uv)
└── frontend/
    ├── index.html         # UI
    ├── app.js            # Client logic
    └── styles.css        # Styling
```

### Key Features

- **Real-time Updates**: WebSocket connection for live progress
- **Smart Batching**: Automatic batch size based on k_rollouts
- **Interactive Visualization**: Salience coloring, transposed views
- **Queue Management**: Handles concurrent requests gracefully

## 📡 API Reference

### Endpoints

| Method | Path | Description | Auth Required |
|--------|------|-------------|---------------|
| GET | `/` | API info | No |
| GET | `/health` | Health check | No |
| GET | `/metrics` | Prometheus metrics | No |
| POST | `/analyze` | Submit text analysis | Yes |
| GET | `/status/{id}` | Check request status | No |
| WS | `/ws` | WebSocket connection | Yes |

### Request Example

```bash
curl -X POST https://YOUR-API-URL/analyze \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "options": {
      "temperature": 0.1,
      "optimize_explanations_config": {
        "just_do_k_rollouts": 8,
        "batch_size_for_rollouts": 32
      }
    }
  }'
```

### WebSocket Protocol

```javascript
// Connect
const ws = new WebSocket('wss://YOUR-API-URL/ws');

// Send analysis request
ws.send(JSON.stringify({
  type: 'analyze',
  text: 'Your text here',
  options: {...}
}));

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Types: queued, processing, result, error
};
```

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Set to "production" for secure mode | development | Yes (prod) |
| `API_KEY` | Bearer token for authentication | None | Yes (prod) |
| `CHECKPOINT_PATH` | Model checkpoint location | /workspace/checkpoints/model.pt | Yes |
| `ALLOWED_ORIGINS` | CORS allowed origins | http://localhost:3000 | Yes |
| `LOAD_IN_8BIT` | Use 8-bit quantization | false | No |
| `MAX_TEXT_LENGTH` | Maximum input length | 1000 | No |
| `RATE_LIMIT_PER_MINUTE` | API rate limit | 60 | No |

### Cost Optimization

1. **GPU Selection**:
   - RTX 3090 (24GB): $0.44/hr - Good for 8-bit models
   - RTX 4090 (24GB): $0.74/hr - Faster inference
   - A100 (40GB): $1.14/hr - Full precision models
   - H100 (80GB): $3.50/hr - Maximum performance

2. **RunPod Serverless** (for low traffic):
   - Pay per second of GPU time
   - Auto-scales to zero
   - Cold start: 1-2 minutes

3. **Optimization Settings**:
   ```bash
   LOAD_IN_8BIT=true      # Reduces VRAM by 50%
   BATCH_SIZE=64          # Increase for better GPU utilization
   ```

## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Backend Offline" | Wait 1-2 min for model loading |
| "Unauthorized" | Check API_KEY in request headers |
| "Too Many Requests" | Rate limit hit, wait 1 minute |
| CORS Error | Verify ALLOWED_ORIGINS includes your domain |
| GPU OOM | Enable 8-bit quantization |

### Debug Commands

```bash
# Check backend logs
docker logs $(docker ps -q)

# Monitor GPU
nvidia-smi -l 1

# Test API key
curl -H "Authorization: Bearer YOUR_KEY" https://YOUR-API/health

# Check queue
curl https://YOUR-API/metrics | grep queue
```

### RunPod Tips

- Use persistent storage for model checkpoints
- Enable auto-stop after 30 min idle to save costs
- Monitor GPU utilization to right-size instance

## 🎯 Makefile Reference

| Command | Description |
|---------|-------------|
| `make help` | Show all commands |
| `make setup` | Install dependencies |
| `make test` | Run test suite |
| `make run` | Start backend |
| `make demo` | Run full stack locally |
| `make security-check` | Audit security |
| `make secure-env` | Generate production config |
| `make deploy-frontend RUNPOD_URL=xxx` | Configure frontend with API URL |
| `make clean` | Clean generated files |

## 🚀 GitHub Actions Setup

For automated frontend deployment:

1. **Create Personal Access Token**:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token with `repo` scope
   - Copy the token

2. **Add Secret to talkative-probes**:
   - Go to Settings → Secrets → Actions
   - Add new secret: `PAGES_UPDATE_TOKEN` with your PAT

3. **Copy Workflow to github.io repo**:
   ```bash
   cp website/github-pages-workflow.yml ~/kitft.github.io/.github/workflows/update-talkative-autoencoder.yml
   ```

4. **Push both workflows** and enjoy automatic updates!

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Run `make test` and `make security-check`
4. Submit pull request

## 📚 Additional Resources

- **Main Project**: https://github.com/kitft/talkative-probes
- **Model Details**: See main repository documentation
- **RunPod Docs**: https://docs.runpod.io
- **FastAPI**: https://fastapi.tiangolo.com

## 🔐 Security Disclosure

Found a security issue? Please email security concerns directly rather than creating public issues.

---

**Note**: This is a research project. Use appropriate security measures for production deployments.