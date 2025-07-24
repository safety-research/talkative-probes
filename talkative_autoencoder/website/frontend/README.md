# Consistency Lens Frontend

A web interface for the Talkative Autoencoder (Consistency Lens) that allows interactive analysis of text with various optimization parameters.

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

✅ **Implemented**:
- WebSocket real-time updates
- All parameter controls from analyze_all_tokens
- Smart batch size calculation
- Table and transposed views
- Basic salience coloring
- Column visibility toggles
- Full text rendering
- Basic tooltips
- Responsive design

❌ **Not Yet Implemented** (from visualiser.html):
- File upload capability
- Data sharing (Pantry/JSONBin)
- Multiple transcript navigation
- Click-to-copy functionality
- Rich tooltips with salience details
- Settings persistence
- Side/bottom navigation buttons
- Enhanced metadata display

## Local Development

### Using Cursor

1. **Set up environment**:
   ```bash
   cd /workspace/kitf/talkative-probes
   source scripts/ensure_env.sh
   cd talkative_autoencoder/website
   ```

2. **Start the demo**:
   ```bash
   make demo  # Starts both backend and frontend
   ```

3. **Access in Cursor**:
   - When servers start, click "Open in Browser" in the popup
   - Or use Cmd/Ctrl+Shift+P → "Ports: Focus on Ports View"
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/docs

### Manual Setup

1. **Start backend**:
   ```bash
   cd backend
   uv run uvicorn app.main:app --reload
   ```

2. **Start frontend**:
   ```bash
   cd frontend
   python -m http.server 3000
   ```

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

### Automated GitHub Pages Deployment

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

3. **Access at**: https://kitft.github.io/talkative-lens/

### Manual Deployment

For other hosting services:
1. Update `API_URL` in `app.js`
2. Upload all frontend files
3. Ensure CORS is configured on backend

## Architecture

```
frontend/
├── index.html      # Main UI structure
├── app.js          # Application logic & WebSocket handling
├── styles.css      # Tailwind-based styling
└── README.md       # This file
```

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
      just_do_k_rollouts: 8
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