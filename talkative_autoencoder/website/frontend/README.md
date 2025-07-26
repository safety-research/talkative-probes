# Talkative Autoencoder Frontend

A web interface for the Talkative Autoencoder that allows interactive analysis of text with various optimization parameters.

## Overview

This frontend provides two deployment options:
1. **Main Frontend** (`index.html`) - Connects to the inference server via WebSocket for real-time analysis
2. **Standalone Data Viewer** (`visualizer-standalone.html`) - Works offline for viewing previously generated analysis data

## Features

### Core Functionality
- **Real-time WebSocket connection** for live updates during analysis
- **Parameter controls** for fine-tuning the analysis:
  - Number of rollouts (k) with automatic batch size calculation
  - Temperature control (0.01-2.0)
  - Advanced settings for detailed configuration
- **Queue management** with position updates and status tracking

### Visualization Options
- **Table view** with sortable columns
- **Transposed view** for vertical text reading
- **Salience coloring** to highlight important tokens
- **Full text view** with interactive tooltips
- **Column visibility toggles** to customize display
- **Adjustable spacing** and column widths

### Advanced Settings
- Calculate token salience scores
- Tuned lens analysis
- Logit lens analysis
- Evaluation options (MSE only, no KL)
- Hard token generation
- Custom seed and batch sizes

## Current Implementation Status

✅ **All features now implemented**:
- WebSocket real-time updates
- All parameter controls from analyze_all_tokens
- Smart batch size calculation
- Table and transposed views with adjustable spacing
- Salience coloring with gradient visualization
- Column visibility toggles
- Full text rendering with interactive tooltips
- File upload and download capability
- Data sharing via Pantry/JSONBin
- Multiple transcript navigation
- Click-to-copy functionality
- Rich tooltips with salience details
- Settings persistence
- Side/bottom navigation buttons
- Enhanced metadata display
- Responsive design

## Local Development

### Prerequisites

1. **Ensure uv is available** (the project uses uv for Python package management):
   ```bash
   cd /workspace/kitf/talkative-probes/talkative_autoencoder
   source scripts/ensure_env.sh
   ```
   This script will:
   - Install uv if not present
   - Create a Python 3.11 virtual environment
   - Install all dependencies from pyproject.toml
   - Set up the UV_PROJECT_ROOT and UV_CACHE_DIR

### Running the Full Application (Frontend + Backend)

#### For Local Development (Recommended)

```bash
cd /workspace/kitf/talkative-probes/talkative_autoencoder/website
make demo  # Starts both backend and frontend servers
```

This runs everything locally - perfect for development and testing if you have a GPU.

#### For RunPod + Local Hybrid

When you need more GPU power (H100/A100):

1. **On your RunPod instance**:
   ```bash
   cd /workspace/kitf/talkative-probes/talkative_autoencoder
   source scripts/ensure_env.sh
   uv run python -m website.backend.app.main
   ```

2. **On your local machine**:
   ```bash
   # First, update the API URL in app.js:
   # const API_URL = 'https://[your-pod-id]-8000.proxy.runpod.net';
   
   cd /workspace/kitf/talkative-probes/talkative_autoencoder/website/frontend
   python -m http.server 8080
   ```

This setup gives you:
- Fast frontend iteration (local)
- Powerful GPU inference (RunPod)
- No need to deploy frontend changes

#### Manual Local Setup

If you prefer to run components separately:

1. **Start the inference backend**:
   ```bash
   cd /workspace/kitf/talkative-probes/talkative_autoencoder
   uv run uvicorn website.backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   The backend will start on http://localhost:8000
   - WebSocket endpoint: ws://localhost:8765
   - API docs: http://localhost:8000/docs

2. **Start the frontend server** (in a new terminal):
   ```bash
   cd /workspace/kitf/talkative-probes/talkative_autoencoder/website/frontend
   python -m http.server 8080
   ```
   Access the frontend at http://localhost:8080

### Running the Standalone Data Viewer

The standalone viewer doesn't need the backend server:

```bash
cd /workspace/kitf/talkative-probes/talkative_autoencoder/website/frontend
python -m http.server 8081
```
Then open http://localhost:8081/visualizer-standalone.html

## Usage Guide

1. **Basic Analysis**:
   - Enter text in the input field
   - Adjust k (rollouts) - higher = better but slower
   - Click "Analyze Text"
   - Monitor progress via WebSocket updates

2. **View Options**:
   - Toggle columns with the "Toggle Columns" button
   - Switch to transposed view for vertical reading
   - Enable salience coloring to see token importance
   - Adjust spacing with the slider

3. **Advanced Options**:
   - Click "Advanced Settings" to access all parameters
   - Enable salience calculation for token importance
   - Adjust temperature for explanation diversity
   - Set custom seeds for reproducibility

## Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| k (rollouts) | Explanation attempts per token | 8 | 1-64 |
| Batch size | Auto-calculated as 256÷k | 32 | 1-256 |
| Temperature | Randomness in generation | 0.1 | 0.01-2.0 |
| Calculate salience | Token importance scoring | true | - |
| Tuned lens | Use tuned lens analysis | false | - |
| Logit lens | Use logit lens analysis | false | - |

## Deployment

### Main Frontend Deployment

#### Automated GitHub Pages Deployment

The frontend is automatically deployed when you push changes:

1. **Configure API URL**:
   ```bash
   make deploy-frontend RUNPOD_URL=https://YOUR-POD-ID-8000.proxy.runpod.net
   ```

2. **Push changes**:
   ```bash
   git add frontend/app.js
   git commit -m "Update API URL"
   git push
   ```

3. **Access at**: https://kitft.github.io/talkative-autoencoder/

#### Manual Deployment

For other hosting services:
1. Update `API_URL` in `app.js`
2. Upload all frontend files
3. Ensure CORS is configured on backend

### Standalone Data Viewer Deployment

The standalone viewer is deployed at https://kitft.github.io/data-viewer/

**Structure on kitft.github.io**:
```
kitft.github.io/
├── data-viewer/
│   └── index.html  # Copy data-viewer-index.html here
└── talkative-autoencoder/  # Git submodule
    └── talkative_autoencoder/
        └── website/
            └── frontend/
                ├── visualizer-core.js
                ├── visualizer-standalone.js
                └── visualizer-styles.css
```

The data viewer references assets via relative paths from the talkative-autoencoder submodule.

## Architecture

### Modular Structure

```
frontend/
├── index.html                    # Main frontend with WebSocket connection
├── app.js                        # Main application logic & WebSocket handling
├── visualizer-core.js            # Shared visualization library
├── visualizer-styles.css         # Shared CSS styles
├── visualizer-standalone.html    # Standalone data viewer (no server needed)
├── visualizer-standalone.js      # Standalone viewer logic
├── data-viewer-index.html        # For deployment at kitft.github.io/data-viewer/
├── architecture-summary.md       # Detailed architecture documentation
└── README.md                     # This file
```

### Component Separation

1. **Shared Components** (`visualizer-core.js`, `visualizer-styles.css`):
   - Core visualization functions (table, transposed view, full text)
   - Salience coloring and tooltip generation
   - Column management and UI utilities
   - Used by both main frontend and standalone viewer

2. **Main Frontend** (`index.html`, `app.js`):
   - WebSocket connection to inference server
   - Real-time analysis with parameter controls
   - Imports visualizer-core as ES6 module

3. **Standalone Viewer** (`visualizer-standalone.html`, `visualizer-standalone.js`):
   - Works completely offline
   - File upload and data sharing capabilities
   - Uses visualizer-core as global object
   - Deployable to static hosting

## WebSocket Protocol

```javascript
// Connect
ws = new WebSocket('wss://api-url/ws')

// Send analysis request
ws.send(JSON.stringify({
  type: 'analyze',
  text: 'Your text',
  options: {
    temperature: 0.1,
    optimize_explanations_config: {
      best_of_k: 8
    }
  }
}))

// Receive updates
// Types: queued, processing, result, error
```

## Related Files

- **visualiser.html**: Standalone visualization tool with full features
- **backend/**: FastAPI server implementation
- **../README.md**: Complete project documentation