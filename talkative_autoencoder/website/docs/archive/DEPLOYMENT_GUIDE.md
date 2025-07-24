# Talkative Autoencoder - Complete Deployment Guide

## 🎯 Deployment Strategy Overview

### Architecture
```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  GitHub Pages   │  HTTPS  │    RunPod GPU    │         │     Model       │
│   (Frontend)    │ ──────> │    (Backend)     │ ──────> │   Checkpoint    │
│     FREE        │   WSS   │   $3-4/hour      │         │  (Volume Mount) │
└─────────────────┘         └──────────────────┘         └─────────────────┘
```

### Why This Architecture?

✅ **Cost-Effective**: Free frontend hosting, pay-per-use GPU backend  
✅ **Scalable**: Independent scaling of frontend/backend  
✅ **Secure**: CORS protection, optional API keys, HTTPS  
✅ **Simple**: No complex infrastructure to manage  

## 🚀 Quick Deployment

### Option A: RunPod Serverless (Recommended for Demos)

**Pros**: Pay-per-second billing, auto-scaling to zero  
**Cons**: 1-2 minute cold starts

1. **Create Serverless Endpoint on RunPod**:
   - Select GPU: A100 40GB (cheaper than H100, sufficient for 14B model)
   - Container: `runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel`
   - Max Workers: 1-3 (based on expected load)

2. **Deploy with this handler**:
   ```python
   # handler.py for RunPod Serverless
   import runpod
   from app.main import app
   
   def handler(event):
       # Your serverless logic here
       return runpod.serverless.return_json(result)
   
   runpod.serverless.start({"handler": handler})
   ```

### Option B: RunPod Pod (For Production)

**Pros**: Always-on, instant responses  
**Cons**: Continuous billing

Follow the detailed steps in the main documentation.

## 💡 Cost Optimization Tips

### 1. GPU Selection Guide

| GPU | VRAM | Cost/hr | Use Case |
|-----|------|---------|----------|
| RTX 3090 | 24GB | ~$0.40 | Testing, 8-bit models |
| RTX 4090 | 24GB | ~$0.60 | Better performance |
| A100 40GB | 40GB | ~$1.50 | Production, full precision |
| H100 80GB | 80GB | ~$3.50 | Maximum performance |

### 2. Model Optimization

```bash
# Enable 8-bit quantization (reduces VRAM by ~50%)
echo "LOAD_IN_8BIT=true" >> .env

# Result: 14B model fits in ~16GB VRAM instead of 32GB
```

### 3. Auto-Shutdown Script

```bash
# Add to RunPod startup script
cat > /workspace/auto_shutdown.sh << 'EOF'
#!/bin/bash
IDLE_MINUTES=30
LAST_REQUEST_FILE="/tmp/last_request"

while true; do
    if [ -f "$LAST_REQUEST_FILE" ]; then
        IDLE_TIME=$(($(date +%s) - $(stat -c %Y "$LAST_REQUEST_FILE")))
        if [ $IDLE_TIME -gt $((IDLE_MINUTES * 60)) ]; then
            echo "Shutting down after $IDLE_MINUTES minutes of inactivity"
            shutdown now
        fi
    fi
    sleep 60
done
EOF

chmod +x /workspace/auto_shutdown.sh
nohup /workspace/auto_shutdown.sh &
```

## 🔧 Enhanced Frontend Features

### 1. Backend Status Indicator

Add to `frontend/app.js`:
```javascript
// Check backend health on page load
async function checkBackendStatus() {
    const statusElement = document.getElementById('backend-status');
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            statusElement.innerHTML = '🟢 Backend Online';
            statusElement.className = 'status-online';
        } else {
            throw new Error('Backend offline');
        }
    } catch (error) {
        statusElement.innerHTML = '🔴 Backend Offline';
        statusElement.className = 'status-offline';
        
        // Show helpful message for cold starts
        if (API_URL.includes('runpod')) {
            statusElement.innerHTML += '<br><small>The model may be starting up. Please wait 1-2 minutes.</small>';
        }
    }
}

// Check on load and periodically
checkBackendStatus();
setInterval(checkBackendStatus, 30000); // Every 30 seconds
```

### 2. Improved Loading Messages

```javascript
// In WebSocket message handler
if (data.type === 'status' && data.message === 'Loading model...') {
    showStatus('🚀 Waking up the AI model... This may take 1-2 minutes on first use.', 'info');
}
```

## 📊 Monitoring & Analytics

### 1. Simple Analytics

Add to frontend:
```html
<!-- Plausible Analytics (privacy-friendly) -->
<script defer data-domain="kitft.github.io" src="https://plausible.io/js/script.js"></script>
```

### 2. Backend Metrics

The `/metrics` endpoint provides Prometheus-compatible metrics:
- Request count
- Queue length  
- Average processing time
- GPU utilization

### 3. Cost Tracking

```python
# Add to backend for cost awareness
import time

class CostTracker:
    def __init__(self, gpu_cost_per_hour=1.50):
        self.start_time = time.time()
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.request_count = 0
    
    def track_request(self):
        self.request_count += 1
    
    def get_stats(self):
        runtime_hours = (time.time() - self.start_time) / 3600
        total_cost = runtime_hours * self.gpu_cost_per_hour
        cost_per_request = total_cost / max(1, self.request_count)
        
        return {
            "runtime_hours": runtime_hours,
            "total_cost": f"${total_cost:.2f}",
            "requests": self.request_count,
            "cost_per_request": f"${cost_per_request:.4f}"
        }
```

## 🔒 Production Security Checklist

- [ ] Set strong `API_KEY` environment variable
- [ ] Restrict `ALLOWED_ORIGINS` to your domain only
- [ ] Enable HTTPS on RunPod (automatic with proxy)
- [ ] Set up rate limiting (already configured at 60 req/min)
- [ ] Monitor `/metrics` for abuse patterns
- [ ] Regular security updates (`uv sync --upgrade`)

## 🚦 User Access Flow

1. **User visits**: `https://kitft.github.io/talkative-autoencoder/`
2. **Frontend loads** from GitHub Pages (instant, cached)
3. **Status check** shows backend availability
4. **User enters text** and adjusts parameters
5. **WebSocket connection** established to RunPod
6. **Real-time updates** show progress
7. **Results displayed** with interactive visualization

## 🆘 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Backend Offline" | Wait 1-2 min for cold start or check RunPod dashboard |
| CORS errors | Verify `ALLOWED_ORIGINS` includes your domain |
| Slow responses | Check GPU utilization, consider batch size |
| WebSocket fails | Ensure RunPod proxy supports WebSocket |

### Debug Commands

```bash
# Check backend logs
docker logs $(docker ps -q)

# Monitor GPU usage
nvidia-smi -l 1

# Test backend locally
curl http://localhost:8000/health

# Check queue status
curl http://localhost:8000/metrics | grep queue
```

## 📈 Scaling Considerations

### When to Scale

- **>10 concurrent users**: Consider multiple RunPod workers
- **>100 daily users**: Add Redis caching for common queries
- **>1000 daily users**: Consider dedicated GPU cluster

### Scaling Options

1. **Horizontal**: Add more RunPod serverless workers
2. **Vertical**: Upgrade to larger GPU (A100 → H100)
3. **Caching**: Implement Redis for repeated queries
4. **CDN**: Use Cloudflare for frontend assets

## 🎉 Launch Checklist

- [ ] Model checkpoint uploaded to RunPod volume
- [ ] Backend deployed and `/health` returns 200
- [ ] Frontend `API_URL` updated with RunPod endpoint
- [ ] CORS origins configured correctly
- [ ] Frontend deployed to GitHub Pages
- [ ] Test end-to-end flow with sample text
- [ ] Monitor first few user interactions
- [ ] Set up cost alerts on RunPod

---

**Remember**: Start with RunPod Serverless for testing, then upgrade to dedicated Pod based on usage patterns and user feedback!