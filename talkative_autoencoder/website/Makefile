# Talkative Autoencoder Web Interface - Deployment Makefile
# Usage: make [target]

SHELL := /bin/zsh

.PHONY: help setup test run deploy-backend deploy-frontend deploy clean security-check

# Configuration
BACKEND_DIR = backend
FRONTEND_DIR = frontend
API_PORT = 8000
RUNPOD_URL ?= # Set via environment or command line: make deploy RUNPOD_URL=xxx

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

# Default target - show help
help:
	@echo "$(GREEN)Talkative Autoencoder Web Interface$(NC)"
	@echo ""
	@echo "$(YELLOW)Local Development:$(NC)"
	@echo "  make setup          - Install dependencies and prepare environment"
	@echo "  make test           - Run backend tests"
	@echo "  make run            - Run backend server locally"
	@echo "  make run-frontend   - Serve frontend locally"
	@echo ""
	@echo "$(YELLOW)Deployment:$(NC)"
	@echo "  make deploy-backend - Deploy backend to RunPod (interactive)"
	@echo "  make deploy-frontend RUNPOD_URL=xxx - Deploy frontend with API URL"
	@echo ""
	@echo "$(YELLOW)Security:$(NC)"
	@echo "  make security-check - Run security audit"
	@echo "  make secure-env     - Generate secure environment configuration"
	@echo ""
	@echo "$(YELLOW)Utilities:$(NC)"
	@echo "  make clean          - Clean up generated files"
	@echo "  make check-deps     - Check if all dependencies are installed"

# Setup environment using ensure_env.sh
setup:
	@echo "$(YELLOW)Setting up environment using ensure_env.sh...$(NC)"
	@source ../scripts/ensure_env.sh
	@echo "$(GREEN)✓ Environment ready$(NC)"
	@echo ""
	@echo "$(YELLOW)Creating .env file from example...$(NC)"
	@if [ ! -f $(BACKEND_DIR)/.env ]; then \
		cp $(BACKEND_DIR)/.env.example $(BACKEND_DIR)/.env 2>/dev/null || \
		echo "# Auto-generated environment file\nCHECKPOINT_PATH=/workspace/checkpoints/model.pt\nALLOWED_ORIGINS=http://localhost:3000\nDEVICE=cuda\nHOST=0.0.0.0\nPORT=$(API_PORT)" > $(BACKEND_DIR)/.env; \
		echo "$(GREEN)✓ Created .env file - please update with your settings$(NC)"; \
	else \
		echo "$(GREEN)✓ .env file already exists$(NC)"; \
	fi

# Run tests
test:
	@echo "$(YELLOW)Running backend tests...$(NC)"
	@source ../scripts/ensure_env.sh && uv run pytest backend/tests/ -v
	@echo "$(GREEN)✓ All tests passed$(NC)"

# Run backend locally
run:
	@echo "$(YELLOW)Starting backend server on http://localhost:$(API_PORT)$(NC)"
	@echo "$(YELLOW)API docs available at http://localhost:$(API_PORT)/docs$(NC)"
	@source ../scripts/ensure_env.sh && cd .. && uv run uvicorn website.backend.app.main:app --host 0.0.0.0 --port $(API_PORT) --reload || true

# Run frontend locally
run-frontend:
	@echo "$(YELLOW)Starting frontend server on http://localhost:3000$(NC)"
	@cd $(FRONTEND_DIR) && python3 -m http.server 3000 || true

# Security check
security-check: check-deps
	@echo "$(YELLOW)Running security audit...$(NC)"
	@echo ""
	@echo "$(YELLOW)Checking for hardcoded secrets...$(NC)"
	@! grep -r "api_key\|password\|secret" $(BACKEND_DIR)/app --include="*.py" | grep -v "Field\|None\|example" || echo "$(GREEN)✓ No hardcoded secrets found$(NC)"
	@echo ""
	@echo "$(YELLOW)Checking CORS configuration...$(NC)"
	@grep -q "allow_methods=\[\"GET\", \"POST\"\]" $(BACKEND_DIR)/app/main.py && echo "$(GREEN)✓ CORS methods restricted$(NC)" || echo "$(RED)✗ CORS methods not properly restricted$(NC)"
	@echo ""
	@echo "$(YELLOW)Checking API key authentication...$(NC)"
	@grep -q "api_key" $(BACKEND_DIR)/app/config.py && echo "$(GREEN)✓ API key support configured$(NC)" || echo "$(YELLOW)⚠ API key authentication not configured$(NC)"
	@echo ""
	@echo "$(YELLOW)Checking rate limiting...$(NC)"
	@grep -q "rate_limit" $(BACKEND_DIR)/app/config.py && echo "$(GREEN)✓ Rate limiting configured$(NC)" || echo "$(YELLOW)⚠ Rate limiting not found$(NC)"

# Generate secure environment configuration
secure-env:
	@echo "$(YELLOW)Generating secure environment configuration...$(NC)"
	@echo "# Secure environment configuration" > $(BACKEND_DIR)/.env.secure
	@echo "# Generated on $$(date)" >> $(BACKEND_DIR)/.env.secure
	@echo "" >> $(BACKEND_DIR)/.env.secure
	@echo "# Model configuration" >> $(BACKEND_DIR)/.env.secure
	@echo "CHECKPOINT_PATH=/workspace/checkpoints/qwen2_5_WCHAT_14b_frozen_nopostfix.pt" >> $(BACKEND_DIR)/.env.secure
	@echo "DEVICE=cuda" >> $(BACKEND_DIR)/.env.secure
	@echo "LOAD_IN_8BIT=true  # Reduce memory usage" >> $(BACKEND_DIR)/.env.secure
	@echo "" >> $(BACKEND_DIR)/.env.secure
	@echo "# Security configuration" >> $(BACKEND_DIR)/.env.secure
	@echo "ENVIRONMENT=production" >> $(BACKEND_DIR)/.env.secure
	@echo "API_KEY=$$(openssl rand -base64 32)" >> $(BACKEND_DIR)/.env.secure
	@echo "ALLOWED_ORIGINS=https://kitft.github.io" >> $(BACKEND_DIR)/.env.secure
	@echo "" >> $(BACKEND_DIR)/.env.secure
	@echo "# Network configuration" >> $(BACKEND_DIR)/.env.secure
	@echo "HOST=0.0.0.0  # RunPod requires 0.0.0.0" >> $(BACKEND_DIR)/.env.secure
	@echo "PORT=$(API_PORT)" >> $(BACKEND_DIR)/.env.secure
	@echo "" >> $(BACKEND_DIR)/.env.secure
	@echo "# Performance configuration" >> $(BACKEND_DIR)/.env.secure
	@echo "MAX_QUEUE_SIZE=50" >> $(BACKEND_DIR)/.env.secure
	@echo "MAX_TEXT_LENGTH=1000" >> $(BACKEND_DIR)/.env.secure
	@echo "RATE_LIMIT_PER_MINUTE=60" >> $(BACKEND_DIR)/.env.secure
	@echo ""
	@echo "$(GREEN)✓ Secure configuration generated in $(BACKEND_DIR)/.env.secure$(NC)"
	@echo "$(YELLOW)  Copy this to your RunPod deployment as .env$(NC)"

# Deploy backend to RunPod
deploy-backend: security-check
	@echo "$(GREEN)=== RunPod Backend Deployment ===$(NC)"
	@echo ""
	@echo "$(YELLOW)Prerequisites:$(NC)"
	@echo "  1. RunPod account with credits"
	@echo "  2. Model checkpoint uploaded to RunPod storage"
	@echo ""
	@echo "$(YELLOW)Step 1: Create RunPod Pod$(NC)"
	@echo "  - GPU: A100 40GB or RTX 3090 (for cost savings)"
	@echo "  - Container: runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel"
	@echo "  - Disk: 50GB+"
	@echo "  - Expose Port: $(API_PORT)"
	@echo ""
	@echo "$(YELLOW)Step 2: SSH into your pod and run:$(NC)"
	@echo "  cd /workspace"
	@echo "  git clone https://github.com/kitft/talkative-probes.git"
	@echo "  cd talkative-probes/talkative_autoencoder/website"
	@echo "  make setup"
	@echo "  make secure-env"
	@echo "  # Edit backend/.env with your settings"
	@echo "  make run"
	@echo ""
	@echo "$(YELLOW)Step 3: Note your RunPod URL:$(NC)"
	@echo "  https://YOUR-POD-ID-$(API_PORT).proxy.runpod.net"
	@echo ""
	@echo "$(YELLOW)Security Note:$(NC)"
	@echo "  - RunPod proxy provides HTTPS automatically"
	@echo "  - Port $(API_PORT) is only accessible through their proxy"
	@echo "  - Direct pod access requires RunPod authentication"

# Deploy frontend to GitHub Pages
deploy-frontend:
	@if [ -z "$(RUNPOD_URL)" ]; then \
		echo "$(RED)Error: RUNPOD_URL not set$(NC)"; \
		echo "Usage: make deploy-frontend RUNPOD_URL=https://your-pod-id-$(API_PORT).proxy.runpod.net"; \
		exit 1; \
	fi
	@echo "$(GREEN)=== Frontend Deployment ===$(NC)"
	@echo ""
	@echo "$(YELLOW)Updating frontend with RunPod URL...$(NC)"
	@cp $(FRONTEND_DIR)/app.js $(FRONTEND_DIR)/app.js.backup
	@sed -i.bak "s|'https://your-api-endpoint.com'|'$(RUNPOD_URL)'|g" $(FRONTEND_DIR)/app.js || \
		sed -i '' "s|'https://your-api-endpoint.com'|'$(RUNPOD_URL)'|g" $(FRONTEND_DIR)/app.js
	@echo "$(GREEN)✓ Updated API URL to: $(RUNPOD_URL)$(NC)"
	@echo ""
	@echo "$(YELLOW)Frontend has been configured!$(NC)"
	@echo ""
	@echo "$(YELLOW)To deploy via Git submodule:$(NC)"
	@echo "  1. Commit these changes to the talkative-probes repo:"
	@echo "     git add $(FRONTEND_DIR)/app.js"
	@echo "     git commit -m 'Update frontend API URL'"
	@echo "     git push"
	@echo ""
	@echo "  2. Update the submodule in your github.io repo:"
	@echo "     cd ~/kitft.github.io"
	@echo "     git submodule update --remote talkative-autoencoder"
	@echo "     git add talkative-autoencoder"
	@echo "     git commit -m 'Update talkative-autoencoder frontend'"
	@echo "     git push"
	@echo ""
	@echo "$(YELLOW)Frontend will be available at:$(NC)"
	@echo "  https://kitft.github.io/talkative-autoencoder/talkative_autoencoder/website/frontend/"
	@echo ""
	@echo "$(YELLOW)Important: Don't forget to update ALLOWED_ORIGINS in backend .env!$(NC)"

# Clean up
clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@rm -f $(FRONTEND_DIR)/app.js.backup 2>/dev/null || true
	@rm -f $(FRONTEND_DIR)/app.js.bak 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Quick deployment check
deployment-check:
	@echo "$(YELLOW)Deployment Checklist:$(NC)"
	@echo ""
	@echo "Backend:"
	@[ -f $(BACKEND_DIR)/.env ] && echo "  $(GREEN)✓ .env file exists$(NC)" || echo "  $(RED)✗ .env file missing$(NC)"
	@[ -f $(BACKEND_DIR)/uv.lock ] && echo "  $(GREEN)✓ Dependencies locked$(NC)" || echo "  $(RED)✗ Dependencies not locked$(NC)"
	@echo ""
	@echo "Frontend:"
	@grep -q "window.location.origin" $(FRONTEND_DIR)/app.js && echo "  $(GREEN)✓ API URL configurable$(NC)" || echo "  $(RED)✗ API URL not configurable$(NC)"
	@echo ""
	@echo "Security:"
	@grep -q "allow_methods=\[\"GET\", \"POST\"\]" $(BACKEND_DIR)/app/main.py && echo "  $(GREEN)✓ CORS restricted$(NC)" || echo "  $(RED)✗ CORS not restricted$(NC)"

# One-command local demo
demo:
	@echo "$(GREEN)Starting local demo...$(NC)"
	@echo ""
	@echo "Backend will run on http://localhost:$(API_PORT)"
	@echo "Frontend will run on http://localhost:3000"
	@echo ""
	@echo "$(YELLOW)Press Ctrl+C to stop both servers$(NC)"
	@echo ""
	@bash -c '\
	CLEANED_UP=0; \
	cleanup() { \
		if [ $$CLEANED_UP -eq 0 ]; then \
			CLEANED_UP=1; \
			echo ""; \
			echo "$(RED)Stopping servers...$(NC)"; \
			pkill -f "uvicorn website.backend.app.main:app" 2>/dev/null; \
			pkill -f "python3 -m http.server 3000" 2>/dev/null; \
			kill $$(jobs -p) 2>/dev/null; \
			exit 0; \
		fi; \
	}; \
	trap cleanup INT TERM EXIT; \
	$(MAKE) -s run & \
	BACKEND_PID=$$!; \
	sleep 5; \
	$(MAKE) -s run-frontend & \
	FRONTEND_PID=$$!; \
	wait $$BACKEND_PID $$FRONTEND_PID'