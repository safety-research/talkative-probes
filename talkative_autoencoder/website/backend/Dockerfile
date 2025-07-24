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
RUN git clone https://github.com/kitft/talkative-probes.git
WORKDIR /workspace/talkative-probes
RUN git submodule update --init --recursive

# Set up the environment
WORKDIR /workspace/talkative-probes/talkative_autoencoder
RUN make  # Installs uv and sets up environment

# Copy only the website backend code (rest comes from repo)
COPY . /workspace/talkative-probes/talkative_autoencoder/website/backend

# Install Python dependencies
WORKDIR /workspace/talkative-probes/talkative_autoencoder/website/backend
RUN uv sync --frozen  # Use frozen for reproducible builds

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