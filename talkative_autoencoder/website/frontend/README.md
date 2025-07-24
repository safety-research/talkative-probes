# Consistency Lens Frontend

A web interface for the Talkative Autoencoder (Consistency Lens) that allows interactive analysis of text with various optimization parameters.

## Features

- **Real-time WebSocket connection** for live updates during analysis
- **Parameter controls** for fine-tuning the analysis:
  - Number of rollouts (k) with automatic batch size calculation
  - Temperature control
  - Advanced settings for detailed configuration
- **Visualization options**:
  - Table view with sortable columns
  - Transposed view for better readability
  - Salience coloring to highlight important tokens
  - Full text view with tooltips
- **Based on the existing visualizer** with enhanced controls

## Local Development

1. **Start the backend** (see backend README)

2. **Open the frontend**:
   ```bash
   cd website/frontend
   # Open index.html in your browser, or use a local server:
   python -m http.server 8080
   ```

3. **Configure API endpoint**:
   Edit `app.js` and update the `API_URL` if your backend is not on localhost:8000

## Usage

1. Enter text in the input field
2. Adjust the number of rollouts (k) - higher values give better explanations but take longer
3. Click "Analyze Text" to process
4. View results in different formats:
   - Toggle between table and transposed view
   - Enable salience coloring to see token importance
   - Hover over tokens for detailed explanations

## Key Parameters

- **Number of Rollouts (k)**: How many explanation attempts to generate per token
- **Batch Size**: Auto-calculated as 256÷k for optimal GPU utilization
- **Temperature**: Controls randomness in explanation generation
- **Calculate Salience**: Measures importance of each decoded token

## Deployment

For production deployment:
1. Update `API_URL` in `app.js` to point to your deployed backend
2. Host files on any static web server (Vercel, Netlify, GitHub Pages, etc.)