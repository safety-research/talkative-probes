# 🚀 Quick Deployment Guide

Deploy the Talkative Autoencoder web interface in 10 minutes!

## Prerequisites

- RunPod account with credits
- GitHub account with Pages enabled
- Model checkpoint file

## 🎯 3-Step Deployment

### Step 1: Deploy Backend on RunPod

1. **Create RunPod Pod**:
   ```
   GPU: RTX 3090 or A100
   Image: runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel
   Disk: 50GB
   Expose Port: 8000
   ```

2. **SSH into pod and run**:
   ```bash
   cd /workspace
   git clone https://github.com/kitft/talkative-probes.git
   cd talkative-probes/talkative_autoencoder/website
   
   # One command setup!
   make setup
   make secure-env
   
   # Edit the generated secure config
   nano backend/.env.secure
   # Update CHECKPOINT_PATH with your model location
   # Save and copy to .env
   cp backend/.env.secure backend/.env
   
   # Start the server
   make run
   ```

3. **Note your URL**: `https://YOUR-POD-ID-8000.proxy.runpod.net`

### Step 2: Deploy Frontend on GitHub Pages

On your local machine:

```bash
# Clone and prepare frontend
git clone https://github.com/kitft/talkative-probes.git
cd talkative-probes/talkative_autoencoder/website

# Deploy with your RunPod URL
make deploy-frontend RUNPOD_URL=https://YOUR-POD-ID-8000.proxy.runpod.net

# Copy to your GitHub Pages repo
cp -r frontend/* ~/YOUR-USERNAME.github.io/talkative-autoencoder/
cd ~/YOUR-USERNAME.github.io
git add .
git commit -m "Deploy Talkative Autoencoder"
git push
```

### Step 3: Access Your App

Visit: `https://YOUR-USERNAME.github.io/talkative-autoencoder/`

## 🔒 Security Features

The Makefile automatically:
- ✅ Generates secure API keys
- ✅ Restricts CORS to your domain
- ✅ Configures rate limiting
- ✅ Uses HTTPS via RunPod proxy
- ✅ Validates security settings

## 🛠️ Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make setup` | Install dependencies |
| `make test` | Run tests |
| `make run` | Start backend locally |
| `make security-check` | Audit security settings |
| `make secure-env` | Generate secure config |
| `make deploy-backend` | Backend deployment guide |
| `make deploy-frontend RUNPOD_URL=xxx` | Deploy frontend |

## 💡 Cost Optimization

### RunPod Serverless (Recommended for demos)
Instead of a dedicated pod, use RunPod Serverless:
- Pay only when processing requests
- Auto-scales to zero
- ~$0.001-0.01 per request

### GPU Selection
- **RTX 3090**: $0.44/hr - Good for 8-bit models
- **RTX 4090**: $0.74/hr - Better performance
- **A100 40GB**: $1.14/hr - Production ready

## 🚨 Troubleshooting

### "Backend Offline"
- Wait 1-2 minutes for model to load
- Check RunPod logs: `docker logs $(docker ps -q)`

### CORS Errors
- Verify `ALLOWED_ORIGINS` includes your GitHub Pages URL
- Restart backend after changing .env

### Port Security
- Port 8000 is **only** accessible through RunPod's authenticated proxy
- Direct access requires RunPod login
- All traffic is HTTPS encrypted

## 📞 Support

- **Issues**: https://github.com/kitft/talkative-probes/issues
- **RunPod Help**: https://docs.runpod.io
- **Model Questions**: See main repository docs