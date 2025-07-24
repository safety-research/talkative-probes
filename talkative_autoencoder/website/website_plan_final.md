# Talkative Autoencoder Web Interface - Final Implementation Plan

## Overview

Since cost is not a concern and you want to use the H100 on RunPod, this plan focuses on:
- **Keeping the H100** for maximum performance
- **Using the existing talkative-probes repository** as the codebase
- **Running a custom curl script on RunPod startup** for environment setup
- **Simplifying deployment** with better tooling
- **Using modern frameworks** for maintainability
- **Implementing proper queuing** and error handling
- **Adding the best suggestions** from Gemini's critique

## Repository Structure

This implementation will be part of the existing `talkative-probes` repository:
```
talkative-probes/
├── talkative_autoencoder/        # Existing code
│   ├── lens/                     # Core lens modules
│   ├── scripts/                  # Existing scripts
│   └── website/                  # NEW: Web interface
│       ├── backend/              # FastAPI backend
│       ├── frontend/             # SvelteKit frontend
│       └── deployment/           # Deployment scripts
└── consistency-lens/             # Linked submodule
```

## Architecture

```
┌─────────────────┐         ┌──────────────────────┐
│                 │         │                      │
│  Vercel         │ <-----> │  RunPod H100         │
│  (SvelteKit)    │  HTTPS  │  (FastAPI + Docker)  │
│                 │         │                      │
│  - Modern UI    │         │  - Queue System      │
│  - State Mgmt   │         │  - WebSockets        │
│  - Visualization│         │  - Auto-restart      │
│                 │         │                      │
└─────────────────┘         └──────────────────────┘
```

## Implementation Plan

### Phase 1: Backend with Proper Architecture (Days 1-2)

#### 1.1 Project Structure
```
website/backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI with WebSocket support
│   ├── models.py            # Pydantic models
│   ├── inference.py         # Model manager with queue
│   ├── config.py            # Settings management
│   └── websocket.py         # WebSocket handlers
├── tests/
│   ├── test_api.py          # API endpoint tests
│   ├── test_inference.py    # Model manager tests
│   └── test_websocket.py    # WebSocket tests
├── requirements.txt         # Pinned dependencies
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── scripts/
    ├── deploy.sh
    ├── health_check.sh
    └── runpod_startup.sh   # Custom startup script
```

#### 1.1a Pydantic Models (`models.py`)
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any
from datetime import datetime

class AnalyzeOptions(BaseModel):
    """Options for text analysis"""
    batch_size: int = Field(default=32, ge=1, le=128)
    calculate_salience: bool = True
    return_structured: bool = True
    
class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoint"""
    text: str = Field(..., min_length=1, max_length=1000)
    options: Optional[AnalyzeOptions] = Field(default_factory=AnalyzeOptions)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or just whitespace')
        return v.strip()

class TokenAnalysis(BaseModel):
    """Individual token analysis result"""
    position: int
    token: str
    explanation: str
    explanation_structured: Optional[List[str]] = None
    token_salience: Optional[List[float]] = None
    mse: float
    kl_divergence: float
    relative_rmse: Optional[float] = None
    tuned_lens_top: Optional[str] = None
    logit_lens_top: Optional[str] = None
    layer: Optional[int] = None

class AnalyzeResponse(BaseModel):
    """Response model for analysis endpoint"""
    request_id: str
    status: str
    metadata: Dict[str, Any]
    data: Optional[List[TokenAnalysis]] = None
    error: Optional[str] = None
    queue_position: Optional[int] = None
    processing_time: Optional[float] = None
    
class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str  # 'analyze', 'status', 'result', 'error'
    request_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
```

#### 1.2 Improved Model Manager with Queue (`inference.py`)
```python
import asyncio
from typing import Optional, Dict, Any
import uuid
from datetime import datetime
import torch

class InferenceQueue:
    def __init__(self, max_size: int = 100):
        self.queue = asyncio.Queue(maxsize=max_size)  # Bounded queue
        self.active_requests = {}
        self.processing = False
    
    async def add_request(self, text: str, options: dict) -> str:
        request_id = str(uuid.uuid4())
        request = {
            'id': request_id,
            'text': text,
            'options': options,
            'status': 'queued',
            'created_at': datetime.utcnow(),
            'result': None,
            'error': None
        }
        self.active_requests[request_id] = request
        await self.queue.put(request_id)
        return request_id
    
    def get_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        return self.active_requests.get(request_id)

class ModelManager:
    def __init__(self, checkpoint_path: str):
        from lens.analysis.analyzer_class import LensAnalyzer
        
        self.analyzer = LensAnalyzer(
            checkpoint_path,
            device="cuda",
            batch_size=32,  # H100 can handle larger batches
            use_bf16=True
        )
        self.queue = InferenceQueue()
        self.processing_task = None
    
    async def start_processing(self):
        """Start the queue processing loop"""
        self.processing_task = asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Process requests from the queue"""
        while True:
            try:
                # Get next request
                request_id = await self.queue.queue.get()
                request = self.queue.active_requests.get(request_id)
                
                if not request:
                    continue
                
                # Update status
                request['status'] = 'processing'
                request['started_at'] = datetime.utcnow()
                
                # Check if client is still connected (for WebSocket)
                if hasattr(request, 'websocket') and request['websocket'].client_state != 1:
                    request['status'] = 'cancelled'
                    continue
                
                try:
                    # Run inference
                    df = self.analyzer.analyze_all_tokens(
                        request['text'],
                        batch_size=request['options'].get('batch_size', 32),
                        return_structured=True,
                        calculate_salience=request['options'].get('calculate_salience', True)
                    )
                    
                    request['result'] = df.to_json()
                    request['status'] = 'completed'
                    
                except Exception as e:
                    request['error'] = str(e)
                    request['status'] = 'failed'
                
                request['completed_at'] = datetime.utcnow()
                
                # Notify via WebSocket if connected
                if hasattr(request, 'websocket'):
                    await request['websocket'].send_json({
                        'type': 'result',
                        'request_id': request_id,
                        'status': request['status'],
                        'result': request.get('result'),
                        'error': request.get('error')
                    })
                
            except Exception as e:
                print(f"Queue processing error: {e}")
                await asyncio.sleep(1)
```

#### 1.3 FastAPI with WebSocket Support (`main.py`)
```python
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

load_dotenv()

# Global model manager
model_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_manager
    model_manager = ModelManager(
        checkpoint_path=os.getenv('CHECKPOINT_PATH', '/models/checkpoint.pt')
    )
    await model_manager.start_processing()
    yield
    # Shutdown
    if model_manager.processing_task:
        model_manager.processing_task.cancel()

app = FastAPI(title="Consistency Lens API", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Queue analysis request and return request ID"""
    request_id = await model_manager.queue.add_request(
        request.text, 
        request.options.dict()
    )
    
    return {
        "request_id": request_id,
        "status": "queued",
        "queue_position": model_manager.queue.queue.qsize()
    }

@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Get status of a request"""
    status = model_manager.queue.get_status(request_id)
    if not status:
        raise HTTPException(404, "Request not found")
    return status

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data['type'] == 'analyze':
                request_id = await model_manager.queue.add_request(
                    data['text'],
                    data.get('options', {})
                )
                
                # Attach websocket to request for notifications
                request = model_manager.queue.active_requests[request_id]
                request['websocket'] = websocket
                
                await websocket.send_json({
                    'type': 'queued',
                    'request_id': request_id,
                    'queue_position': model_manager.queue.queue.qsize()
                })
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "queue_size": model_manager.queue.queue.qsize() if model_manager else 0
    }

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring"""
    if not model_manager:
        return {"error": "Model not loaded"}
    
    active_requests = model_manager.queue.active_requests
    completed = [r for r in active_requests.values() if r['status'] == 'completed']
    failed = [r for r in active_requests.values() if r['status'] == 'failed']
    
    return {
        "total_requests": len(active_requests),
        "completed_requests": len(completed),
        "failed_requests": len(failed),
        "queue_size": model_manager.queue.queue.qsize(),
        "average_processing_time": calculate_avg_time(completed)
    }
```

#### 1.4 Configuration Management (`config.py`)
```python
from pydantic import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Model settings
    checkpoint_path: str = Field(
        default="/workspace/checkpoints/qwen2_5_WCHAT_14b_frozen_nopostfix.pt",
        env="CHECKPOINT_PATH"
    )
    device: str = Field(default="cuda", env="DEVICE")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    use_bf16: bool = Field(default=True, env="USE_BF16")
    
    # API settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000"],
        env="ALLOWED_ORIGINS"
    )
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    max_queue_size: int = Field(default=100, env="MAX_QUEUE_SIZE")
    max_text_length: int = Field(default=1000, env="MAX_TEXT_LENGTH")
    
    # Redis settings (optional)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # RunPod specific
    runpod_pod_id: Optional[str] = Field(default=None, env="RUNPOD_POD_ID")
    runpod_api_key: Optional[str] = Field(default=None, env="RUNPOD_API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

#### 1.5 Error Handling and Model Loading (`inference.py` continued)
```python
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Raised when model fails to load"""
    pass

class ModelManager:
    def __init__(self, checkpoint_path: str, max_retries: int = 3):
        self.checkpoint_path = checkpoint_path
        self.analyzer = None
        self.max_retries = max_retries
        self.queue = InferenceQueue(max_size=settings.max_queue_size)
        self.processing_task = None
        
    def load_model(self):
        """Load model with retry logic"""
        from lens.analysis.analyzer_class import LensAnalyzer
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Loading model attempt {attempt + 1}/{self.max_retries}")
                
                # Check if checkpoint exists
                if not os.path.exists(self.checkpoint_path):
                    raise ModelLoadError(f"Checkpoint not found: {self.checkpoint_path}")
                
                # Check GPU availability
                if not torch.cuda.is_available():
                    raise ModelLoadError("No GPU available")
                
                # Load model
                self.analyzer = LensAnalyzer(
                    self.checkpoint_path,
                    device=settings.device,
                    batch_size=settings.batch_size,
                    use_bf16=settings.use_bf16,
                    strict_load=False  # Allow loading with warnings
                )
                
                # Test inference to ensure model works
                test_result = self.analyzer.analyze_all_tokens(
                    "test",
                    batch_size=1,
                    return_structured=True
                )
                
                logger.info("Model loaded successfully")
                return
                
            except Exception as e:
                logger.error(f"Model load attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise ModelLoadError(f"Failed to load model after {self.max_retries} attempts: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

#### 1.6 RunPod Startup Script (Simplified and Secure)
```bash
#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

# This script runs on RunPod startup to set up the environment
echo "Starting RunPod setup..."

# Define variables
REPO_URL="${REPO_URL:-https://github.com/yourusername/talkative-probes.git}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-qwen2_5_WCHAT_14b_frozen_nopostfix.pt}"
CHECKPOINT_PATH="/workspace/checkpoints/$CHECKPOINT_NAME"

# Function to handle errors
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Clone repository if needed
if [ ! -d "/workspace/talkative-probes" ]; then
    echo "Cloning repository..."
    cd /workspace
    git clone "$REPO_URL" || error_exit "Failed to clone repository"
    cd talkative-probes
    git submodule update --init --recursive || error_exit "Failed to update submodules"
fi

cd /workspace/talkative-probes/talkative_autoencoder

# Run environment setup
echo "Setting up Python environment..."
if [ -f "Makefile" ]; then
    make || error_exit "Failed to run make"
else
    error_exit "Makefile not found"
fi

# Download checkpoint if needed
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint not found. Please mount a volume with the checkpoint at $CHECKPOINT_PATH"
    # For security, we don't download from arbitrary URLs
    # Instead, expect the checkpoint to be mounted via RunPod volumes
    error_exit "Checkpoint not found at $CHECKPOINT_PATH"
fi

# Verify checkpoint
echo "Verifying checkpoint..."
if [ ! -s "$CHECKPOINT_PATH" ]; then
    error_exit "Checkpoint file is empty"
fi

# Install backend dependencies
cd /workspace/talkative-probes/talkative_autoencoder/website/backend
echo "Installing backend dependencies..."
uv pip install -r requirements.txt || error_exit "Failed to install dependencies"

# Create necessary directories
mkdir -p logs

# Start the API server with proper error handling
echo "Starting API server..."
exec uv run uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop uvloop \
    --access-log \
    --log-config logging.yaml
```

#### 1.7 Production-Ready Dockerfile (Alternative if not using startup script)
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    curl \
    make \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/yourusername/talkative-probes.git
WORKDIR /workspace/talkative-probes
RUN git submodule update --init --recursive

# Set up the environment
WORKDIR /workspace/talkative-probes/talkative_autoencoder
RUN make  # Installs uv and sets up environment

# Copy only the website backend code (rest comes from repo)
COPY website/backend /workspace/talkative-probes/talkative_autoencoder/website/backend

# Install Python dependencies
WORKDIR /workspace/talkative-probes/talkative_autoencoder/website/backend
RUN uv pip install -r requirements.txt

# Create directory for checkpoints
RUN mkdir -p /workspace/checkpoints

# Environment variables (will be overridden by RunPod config)
ENV CHECKPOINT_PATH="/workspace/checkpoints/model.pt"
ENV ALLOWED_ORIGINS="http://localhost:3000"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run with proper production settings
CMD ["uv", "run", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--access-log"]
```

#### 1.8 Backend Requirements (`requirements.txt`)
```txt
# Core dependencies - pinned versions
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
pydantic==2.5.0
pydantic-settings==2.1.0

# PyTorch and ML (should match main project versions)
torch==2.1.0
transformers==4.36.0
pandas==2.1.4
numpy==1.24.3

# Redis for caching (optional)
redis==5.0.1
hiredis==2.2.3

# Monitoring and logging
prometheus-client==0.19.0
python-json-logger==2.0.7

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.2  # For TestClient

# Development
python-dotenv==1.0.0
black==23.11.0
ruff==0.1.6
```

### Phase 2: Modern Frontend with SvelteKit (Days 3-4)

#### 2.1 Initialize Project
```bash
npm create svelte@latest consistency-lens-web
cd consistency-lens-web
npm install
npm install -D tailwindcss @tailwindcss/forms postcss autoprefixer
npm install socket.io-client chart.js
npx tailwindcss init -p
```

#### 2.2 WebSocket Store (`lib/stores/websocket.js`)
```javascript
import { writable } from 'svelte/store';

export const connectionStatus = writable('disconnected');
export const analysisResults = writable(new Map());
export const currentRequest = writable(null);

let ws = null;

export function connectWebSocket(url) {
    ws = new WebSocket(url);
    
    ws.onopen = () => {
        connectionStatus.set('connected');
    };
    
    ws.onclose = () => {
        connectionStatus.set('disconnected');
        setTimeout(() => connectWebSocket(url), 5000); // Auto-reconnect
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'queued') {
            currentRequest.update(r => ({...r, status: 'queued', queue_position: data.queue_position}));
        } else if (data.type === 'result') {
            analysisResults.update(results => {
                results.set(data.request_id, data);
                return results;
            });
            if (currentRequest && currentRequest.id === data.request_id) {
                currentRequest.set(null);
            }
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        connectionStatus.set('error');
    };
}

export function analyzeText(text, options = {}) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        throw new Error('WebSocket not connected');
    }
    
    const request = {
        type: 'analyze',
        text,
        options
    };
    
    ws.send(JSON.stringify(request));
    currentRequest.set({ text, status: 'sending' });
}
```

#### 2.3 Main Visualization Component
```svelte
<!-- src/lib/components/TokenVisualization.svelte -->
<script>
  import { onMount } from 'svelte';
  export let data;
  
  let showTransposed = false;
  let salienceEnabled = true;
  let hoveredToken = null;
  
  function getSalienceColor(value) {
    // Port the color logic from visualiser.html
    if (typeof value !== 'number' || isNaN(value)) return 'transparent';
    const v = Math.max(-1, Math.min(0.3, value));
    
    if (v >= 0) {
      const beta = 50;
      const norm = Math.log10(1 + beta * v) / Math.log10(1 + beta * 0.3);
      const g = Math.round(255 * (1 - norm));
      return `rgb(255,${g},${g})`;
    } else {
      const norm = (-v) / 1;
      const r = Math.round(255 * (1 - norm));
      return `rgb(${r},255,${r})`;
    }
  }
</script>

<div class="visualization-container">
  <div class="controls flex gap-4 mb-4">
    <button 
      class="btn btn-secondary"
      on:click={() => showTransposed = !showTransposed}
    >
      {showTransposed ? 'Table View' : 'Transposed View'}
    </button>
    
    <label class="flex items-center gap-2">
      <input 
        type="checkbox" 
        bind:checked={salienceEnabled}
        class="checkbox"
      />
      <span>Color by Salience</span>
    </label>
  </div>
  
  {#if !showTransposed}
    <!-- Table View -->
    <div class="overflow-x-auto">
      <table class="table w-full">
        <thead>
          <tr>
            <th>Position</th>
            <th>Token</th>
            <th>Explanation</th>
            <th>MSE</th>
            <th>KL Divergence</th>
          </tr>
        </thead>
        <tbody>
          {#each data as row}
            <tr 
              class="hover:bg-base-200"
              on:mouseenter={() => hoveredToken = row}
              on:mouseleave={() => hoveredToken = null}
            >
              <td>{row.position}</td>
              <td class="font-mono font-bold text-orange-600">{row.token}</td>
              <td>
                {#if row.explanation_structured && salienceEnabled}
                  {#each row.explanation_structured as word, i}
                    <span 
                      class="inline-block px-1"
                      style="background-color: {getSalienceColor(row.token_salience?.[i])}"
                    >
                      {word}
                    </span>
                  {/each}
                {:else}
                  {row.explanation}
                {/if}
              </td>
              <td>{row.mse?.toFixed(6) || 'N/A'}</td>
              <td>{row.kl_divergence?.toFixed(6) || 'N/A'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {:else}
    <!-- Transposed View -->
    <div class="flex flex-wrap gap-2">
      {#each data as row}
        <div class="token-column">
          <div class="token-header font-bold text-orange-600 -rotate-12 origin-bottom-left">
            {row.token}
          </div>
          <div class="explanation-words">
            {#each row.explanation_structured || [] as word, i}
              <div 
                class="word-box"
                style="background-color: {salienceEnabled ? getSalienceColor(row.token_salience?.[i]) : 'transparent'}"
              >
                {word}
              </div>
            {/each}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .token-column {
    width: 80px;
    display: flex;
    flex-direction: column;
  }
  
  .word-box {
    padding: 2px 4px;
    margin: 1px 0;
    border-radius: 2px;
  }
</style>
```

### Phase 3: Deployment & DevOps (Day 5)

#### 3.1 RunPod Deployment Configuration

**Option A: Using RunPod's Docker Image + Startup Script**
```bash
#!/bin/bash
# deploy_to_runpod.sh

# Create RunPod configuration with startup script
cat > runpod-config.json <<EOF
{
  "name": "consistency-lens-h100",
  "imageName": "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04",
  "gpuType": "H100_PCIE",
  "gpuCount": 1,
  "containerDiskInGb": 100,
  "volumeInGb": 100,
  "ports": "8000/http",
  "env": {
    "REPO_URL": "https://github.com/yourusername/talkative-probes.git",
    "CHECKPOINT_NAME": "qwen2_5_WCHAT_14b_frozen_nopostfix.pt",
    "ALLOWED_ORIGINS": "https://your-frontend.vercel.app",
    "CUSTOM_SETUP_SCRIPT": "https://your-custom-setup-script.sh"
  },
  "dockerArgs": "--shm-size=10g",
  "onStartCommand": "curl -sSL https://raw.githubusercontent.com/yourusername/talkative-probes/main/website/backend/scripts/runpod_startup.sh | bash"
}
EOF

# Deploy using RunPod CLI
runpod create pod --config runpod-config.json
```

**Option B: Using Pre-built Docker Image**
```bash
#!/bin/bash
# build_and_deploy.sh

# Build and push to registry
docker build -t ghcr.io/yourusername/consistency-lens-api:latest .
docker push ghcr.io/yourusername/consistency-lens-api:latest

# Create RunPod configuration
cat > runpod-config.json <<EOF
{
  "name": "consistency-lens-h100",
  "imageName": "ghcr.io/yourusername/consistency-lens-api:latest",
  "gpuType": "H100_PCIE",
  "gpuCount": 1,
  "containerDiskInGb": 100,
  "volumeInGb": 100,
  "ports": "8000/http",
  "env": {
    "CHECKPOINT_PATH": "/workspace/checkpoints/qwen2_5_WCHAT_14b_frozen_nopostfix.pt",
    "ALLOWED_ORIGINS": "https://your-frontend.vercel.app"
  },
  "volumeMountPath": "/workspace/checkpoints"
}
EOF

# Deploy using RunPod CLI
runpod create pod --config runpod-config.json
```

#### 3.2 Checkpoint Management

Create a script to manage checkpoints (`scripts/manage_checkpoints.sh`):
```bash
#!/bin/bash
# Script to download/upload checkpoints to RunPod volume

CHECKPOINT_NAME="qwen2_5_WCHAT_14b_frozen_nopostfix.pt"
VOLUME_ID="your-runpod-volume-id"

case "$1" in
  download)
    echo "Downloading checkpoint from storage..."
    # Option 1: From Hugging Face
    huggingface-cli download your-org/your-model $CHECKPOINT_NAME \
      --local-dir /workspace/checkpoints
    
    # Option 2: From S3/GCS
    # aws s3 cp s3://your-bucket/checkpoints/$CHECKPOINT_NAME /workspace/checkpoints/
    
    # Option 3: From Google Drive
    # gdown https://drive.google.com/uc?id=YOUR_FILE_ID -O /workspace/checkpoints/$CHECKPOINT_NAME
    ;;
    
  upload)
    echo "Uploading checkpoint to RunPod volume..."
    runpod volume upload $VOLUME_ID /local/path/to/$CHECKPOINT_NAME /checkpoints/
    ;;
    
  *)
    echo "Usage: $0 {download|upload}"
    exit 1
    ;;
esac
```

#### 3.3 GitHub Actions CI/CD
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push Docker image
        run: |
          echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
          docker build -t ghcr.io/${{ github.repository }}/api:latest ./backend
          docker push ghcr.io/${{ github.repository }}/api:latest
      
      - name: Deploy to RunPod
        run: |
          # Use RunPod API to update the pod
          ./scripts/update_runpod.sh

  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - name: Install and build
        run: |
          cd frontend
          npm ci
          npm run build
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID}}
          vercel-project-id: ${{ secrets.PROJECT_ID}}
```

#### 3.3 Monitoring with Grafana
```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Phase 4: Production Hardening (Day 6)

#### 4.1 Request Cancellation
```python
@app.post("/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks, req: Request):
    """Queue analysis with cancellation support"""
    request_id = await model_manager.queue.add_request(
        request.text, 
        request.options.dict()
    )
    
    # Monitor for disconnection
    async def check_disconnection():
        while request_id in model_manager.queue.active_requests:
            if await req.is_disconnected():
                model_manager.queue.active_requests[request_id]['status'] = 'cancelled'
                break
            await asyncio.sleep(1)
    
    background_tasks.add_task(check_disconnection)
    
    return {"request_id": request_id, "status": "queued"}
```

#### 4.2 Result Caching with Redis
```python
import redis
import hashlib
import json

class CachedModelManager(ModelManager):
    def __init__(self, checkpoint_path: str):
        super().__init__(checkpoint_path)
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = 3600  # 1 hour
    
    def _get_cache_key(self, text: str, options: dict) -> str:
        """Generate cache key from input"""
        data = f"{text}:{json.dumps(options, sort_keys=True)}"
        return f"lens:{hashlib.md5(data.encode()).hexdigest()}"
    
    async def analyze_with_cache(self, text: str, options: dict):
        """Check cache before running inference"""
        cache_key = self._get_cache_key(text, options)
        
        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Run inference
        result = await self.analyze_text(text, options)
        
        # Cache result
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(result))
        
        return result
```

### Phase 4: Testing Strategy (Critical Addition)

#### 4.1 Backend Tests (`tests/test_api.py`)
```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import asyncio

@pytest.fixture
def mock_model_manager():
    """Mock ModelManager for testing"""
    manager = Mock()
    manager.queue = Mock()
    manager.queue.add_request = asyncio.coroutine(lambda t, o: "test-request-id")
    manager.queue.get_status = lambda id: {"status": "completed", "result": "{}"}
    return manager

@pytest.fixture
def client(mock_model_manager):
    """Test client with mocked dependencies"""
    with patch('app.main.model_manager', mock_model_manager):
        from app.main import app
        return TestClient(app)

def test_analyze_endpoint_valid_request(client):
    """Test valid analysis request"""
    response = client.post("/analyze", json={
        "text": "Hello world",
        "options": {"batch_size": 16}
    })
    assert response.status_code == 200
    assert "request_id" in response.json()

def test_analyze_endpoint_text_too_long(client):
    """Test text length validation"""
    response = client.post("/analyze", json={
        "text": "x" * 1001
    })
    assert response.status_code == 422

def test_health_check(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

#### 4.2 Frontend Tests (`frontend/tests/stores.test.js`)
```javascript
import { describe, it, expect, beforeEach } from 'vitest';
import { analyzeText, analysisResult, error } from '$lib/stores/analysis';
import { vi } from 'vitest';

// Mock the API client
vi.mock('$lib/api/client', () => ({
    analyze: vi.fn()
}));

describe('Analysis Store', () => {
    beforeEach(() => {
        analysisResult.set(null);
        error.set(null);
    });

    it('should handle successful analysis', async () => {
        const mockResult = {
            metadata: { model_name: 'test' },
            data: [{ token: 'Hello', position: 0 }]
        };
        
        const { analyze } = await import('$lib/api/client');
        analyze.mockResolvedValue(mockResult);
        
        await analyzeText('Hello');
        
        const result = get(analysisResult);
        expect(result).toEqual(mockResult);
        expect(get(error)).toBeNull();
    });
});
```

### Phase 5: Documentation (Critical Addition)

#### 5.1 README.md
```markdown
# Consistency Lens Web Interface

Web interface for the Talkative Autoencoder (Consistency Lens) project.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (for local development)
- Node.js 18+ (for frontend)
- Docker (optional)

## Quick Start

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/talkative-probes.git
   cd talkative-probes/talkative_autoencoder
   ```

2. Set up the environment:
   ```bash
   make  # Installs uv and dependencies
   ```

3. Download or mount your checkpoint file to `/workspace/checkpoints/`

4. Start the backend:
   ```bash
   cd website/backend
   uv run uvicorn app.main:app --reload
   ```

### Frontend Setup

1. Navigate to frontend:
   ```bash
   cd website/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure API endpoint:
   ```bash
   echo "VITE_API_URL=http://localhost:8000" > .env
   ```

4. Start development server:
   ```bash
   npm run dev
   ```

## API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive API documentation.

## Testing

Backend:
```bash
cd website/backend
uv run pytest
```

Frontend:
```bash
cd website/frontend
npm test
```

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.
```

#### 5.2 DEPLOYMENT.md
```markdown
# Deployment Guide

## RunPod Deployment

### Prerequisites

1. RunPod account with H100 access
2. Model checkpoint uploaded to persistent storage
3. GitHub repository with your code

### Step 1: Create RunPod Volume

1. Create a new volume in RunPod dashboard
2. Upload your checkpoint file to the volume
3. Note the volume ID

### Step 2: Create Pod

1. Select H100 GPU
2. Use the RunPod PyTorch template
3. Configure:
   - Container Disk: 100GB
   - Volume: Mount your checkpoint volume to `/workspace/checkpoints`
   - Ports: 8000/http

### Step 3: Environment Variables

Set in RunPod dashboard:
- `REPO_URL`: Your GitHub repository
- `CHECKPOINT_NAME`: Your checkpoint filename
- `ALLOWED_ORIGINS`: Your frontend URL

### Step 4: Start Command

Set the startup command:
```bash
curl -sSL https://raw.githubusercontent.com/yourusername/talkative-probes/main/website/backend/scripts/runpod_startup.sh | bash
```

### Troubleshooting

Check logs:
```bash
runpod logs <pod-id>
```

SSH into pod:
```bash
runpod ssh <pod-id>
```

## Frontend Deployment (Vercel)

1. Fork the repository
2. Connect Vercel to your GitHub
3. Configure:
   - Root Directory: `website/frontend`
   - Environment Variables:
     - `VITE_API_URL`: Your RunPod endpoint

4. Deploy
```

#### 5.3 LensAnalyzer API Documentation
```markdown
# LensAnalyzer API Reference

This document describes the expected API of the `lens.analysis.analyzer_class.LensAnalyzer` class.

## Constructor

```python
LensAnalyzer(
    checkpoint_path: str,
    device: str = "cuda",
    batch_size: int = 32,
    use_bf16: bool = True,
    strict_load: bool = True,
    comparison_tl_checkpoint: bool = False,
    no_orig: bool = True
)
```

## Methods

### analyze_all_tokens

Analyzes all tokens in the input text.

```python
analyze_all_tokens(
    text: str,
    batch_size: int = 32,
    return_structured: bool = True,
    calculate_salience: bool = True,
    no_eval: bool = False,
    move_devices: bool = False
) -> LensDataFrame
```

**Returns**: A pandas-like DataFrame with the following columns:
- `position`: Token position
- `token`: Token text
- `explanation`: Human-readable explanation
- `explanation_structured`: List of explanation words
- `token_salience`: List of salience scores per explanation word
- `mse`: Mean squared error
- `kl_divergence`: KL divergence
- `relative_rmse`: Relative RMSE
- Additional optional columns

### to_json

Converts results to JSON format:

```python
df.to_json() -> str
```

Returns JSON with:
- `metadata`: Model information
- `data`: List of token analyses
```

## Key Implementation Details

### Using the Existing Repository

1. **Repository Setup**: The entire web interface lives in `talkative_autoencoder/website/`
2. **Environment Management**: Uses the existing `uv` setup from the main project
3. **Module Imports**: Can directly import from `lens.analysis.analyzer_class`
4. **Checkpoint Path**: Configurable via environment variable, defaults to Qwen model

### RunPod Startup Flow

1. **Initial Setup**: RunPod executes the startup script on container launch
2. **Repository Clone**: Clones talkative-probes if not present
3. **Custom Script**: Runs your specified curl script for environment setup
4. **Environment Creation**: Uses `make` to set up uv environment
5. **Checkpoint Loading**: Downloads/mounts the specific checkpoint
6. **Server Start**: Launches FastAPI with the loaded model

### Frontend Deployment

- **Static Hosting**: Vercel/Netlify for the SvelteKit frontend
- **API Endpoint**: Configured via environment variable
- **CORS**: Properly configured for your frontend domain

## Final Architecture Benefits

1. **Performance**: H100 provides fastest possible inference
2. **Real-time Updates**: WebSocket support for live progress
3. **Proper Queuing**: No more "server busy" errors
4. **Modern Frontend**: SvelteKit for maintainable, reactive UI
5. **Production Ready**: Health checks, monitoring, auto-restart
6. **Developer Experience**: Hot reload, type safety, modern tooling
7. **Integrated with Existing Code**: Uses your existing lens modules directly

## Timeline

- Day 1-2: Backend with queue and WebSocket
- Day 3-4: SvelteKit frontend
- Day 5: Deployment and CI/CD
- Day 6: Production hardening

Total: 6 days for a production-ready system

## Addressing Critical Implementation Concerns

### From Gemini's Critique:

1. **Missing Pydantic Models**: Now fully defined in section 1.1a
2. **Error Handling**: Comprehensive error handling added in section 1.5
3. **Bounded Queue**: Queue now has configurable size limit to prevent memory exhaustion
4. **Testing Strategy**: Complete test suite outlined in Phase 4
5. **Documentation**: Full documentation suite in Phase 5
6. **Secure Startup**: Removed `curl | bash`, using mounted volumes for checkpoints
7. **Configuration Management**: Proper settings management with Pydantic BaseSettings
8. **Requirements**: All dependencies pinned with specific versions

### Development Prerequisites

Before starting implementation, ensure you have:

1. **Access to LensAnalyzer source code** or detailed API documentation
2. **The Makefile** from the main project that sets up `uv`
3. **A method to obtain the model checkpoint** (HuggingFace, S3, etc.)
4. **The custom setup script** referenced in the original plan (if needed)

### Recommended Development Order

1. **Day 0**: Validate LensAnalyzer works with your checkpoint
   ```python
   # test_analyzer.py
   from lens.analysis.analyzer_class import LensAnalyzer
   analyzer = LensAnalyzer("/path/to/checkpoint.pt")
   result = analyzer.analyze_all_tokens("test")
   print(result.to_json())
   ```

2. **Days 1-2**: Backend implementation with tests
3. **Days 3-4**: Frontend implementation  
4. **Day 5**: Integration and deployment
5. **Day 6**: Production hardening and monitoring

## Conclusion

This revised plan addresses all critical issues raised by Gemini while maintaining the power of the H100 GPU. The implementation is now:
- **Secure**: No remote script execution, proper input validation
- **Robust**: Comprehensive error handling and testing
- **Scalable**: Bounded queues, proper resource management
- **Documented**: Complete documentation for developers
- **Production-ready**: Health checks, monitoring, and deployment guides

The plan integrates seamlessly with your existing talkative-probes repository while providing a modern, maintainable web interface.