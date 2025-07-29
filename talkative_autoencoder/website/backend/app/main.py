import os

from dotenv import load_dotenv

# Load .env files FIRST before any other imports that might use HF_TOKEN
# Load .env file from backend directory
dotenv_location = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_location)

# Also load parent .env file (talkative_autoencoder) for HF_TOKEN
parent_dotenv_location = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".env"
)
load_dotenv(parent_dotenv_location, override=False)  # Don't override already set values

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

import torch
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .config import load_settings
from .model_manager import ModelManager, ModelLoadError
from .inference_service import InferenceService
from .models import AnalyzeRequest
from .websocket import manager

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

logger.info(f"Loaded .env from {dotenv_location}")
logger.info(f"Loaded parent .env from {parent_dotenv_location}")


# GPU Stats Monitor Class
class GPUStatsMonitor:
    def __init__(self, update_interval: float = 0.5):
        self.update_interval = update_interval
        self.stats = {
            "available": torch.cuda.is_available(),
            "utilization": 0,
            "memory_used": 0,
            "memory_total": 0,
            "memory_percent": 0,
            "peak_utilization": 0,
            "last_update": None,
            "is_computing": False,
        }
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self._last_reset_time = time.time()

    def start(self):
        """Start the monitoring thread"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("GPU stats monitor started")

    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("GPU stats monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop running in separate thread"""
        pynvml_available = False
        try:
            import pynvml

            pynvml.nvmlInit()
            pynvml_available = True
            logger.info("pynvml initialized successfully for GPU monitoring")
        except Exception as e:
            logger.warning(f"pynvml not available for GPU monitoring: {e}")

        while self.running:
            try:
                stats = {"available": torch.cuda.is_available(), "last_update": datetime.utcnow().isoformat()}

                if torch.cuda.is_available():
                    # Always get memory stats from PyTorch
                    memory_used = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    stats["memory_used"] = round(memory_used, 2)
                    stats["memory_total"] = round(memory_total, 2)
                    stats["memory_percent"] = round((memory_used / memory_total) * 100, 1)

                    # Try to get GPU utilization from pynvml
                    if pynvml_available:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            stats["utilization"] = utilization.gpu

                            # Track peak utilization
                            with self.lock:
                                if utilization.gpu > self.stats.get("peak_utilization", 0):
                                    self.stats["peak_utilization"] = utilization.gpu

                        except Exception as e:
                            logger.debug(f"Failed to get GPU utilization: {e}")
                            stats["utilization"] = 0
                    else:
                        stats["utilization"] = 0

                    # Detect if we're computing based on memory usage changes
                    with self.lock:
                        prev_memory = self.stats.get("memory_used", 0)
                        stats["is_computing"] = abs(memory_used - prev_memory) > 0.1  # Changed by >100MB

                # Update shared stats
                with self.lock:
                    self.stats.update(stats)

                    # Reset peak utilization every 30 seconds if not computing
                    if time.time() - self._last_reset_time > 30 and not stats.get("is_computing", False):
                        self.stats["peak_utilization"] = stats.get("utilization", 0)
                        self._last_reset_time = time.time()

            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")

            time.sleep(self.update_interval)

    def get_stats(self):
        """Get current GPU stats (thread-safe)"""
        with self.lock:
            return self.stats.copy()


# Global instances
model_manager: ModelManager | None = None
inference_service: InferenceService | None = None
gpu_monitor = GPUStatsMonitor()
queue_broadcast_task = None

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
settings = load_settings()


async def broadcast_queue_status_periodically():
    """Broadcast queue status to all connected clients every 2 seconds"""
    while True:
        try:
            if inference_service and manager.active_connections:
                stats = inference_service.queue.get_queue_stats()
                
                # Get list of queued request IDs for position tracking
                queued_ids = list(inference_service.queue.queue.queue) if hasattr(inference_service.queue.queue, 'queue') else []
                
                message = {
                    "type": "queue_update",
                    "queue_size": stats["queue_size"],
                    "queued_requests": stats["queued_requests"],
                    "processing_requests": stats["processing_requests"],
                    "total_active": stats["total_active"],
                    "queued_ids": queued_ids
                }
                
                await manager.broadcast(message)
        except Exception as e:
            logger.error(f"Error broadcasting queue status: {e}")
        
        await asyncio.sleep(2)  # Update every 2 seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_manager, inference_service

    # Start GPU monitoring
    gpu_monitor.start()

    try:
        # Initialize model manager
        model_manager = ModelManager(settings)
        model_manager.websocket_manager = manager
        
        # Initialize inference service
        inference_service = InferenceService(model_manager, settings, manager)

        # Load default model on startup if lazy loading is disabled
        if not settings.lazy_load_model:
            logger.info("Loading default model on startup (lazy_load_model=false)")
            # Always use the model manager's default model from registry
            await model_manager.initialize_default_model()
        else:
            logger.info("Model will be loaded on first request (lazy_load_model=true)")

        await inference_service.start_processing()
        
        # Start periodic queue status broadcaster
        global queue_broadcast_task
        queue_broadcast_task = asyncio.create_task(broadcast_queue_status_periodically())
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Continue startup even if model loading fails

    yield

    # Shutdown
    logger.info("Shutting down application")

    # Stop GPU monitoring
    gpu_monitor.stop()
    
    # Stop queue broadcaster
    if queue_broadcast_task:
        queue_broadcast_task.cancel()
        try:
            await queue_broadcast_task
        except asyncio.CancelledError:
            pass

    if inference_service:
        await inference_service.stop_processing()


app = FastAPI(
    title="Consistency Lens API",
    description="API for Talkative Autoencoder (Consistency Lens)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
)

# Add rate limiter state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security: Trusted host middleware (prevents host header attacks)
if os.getenv("ENABLE_TRUSTED_HOST", "true").lower() == "true":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.proxy.runpod.net", "localhost", "127.0.0.1", "*"],  # Added wildcard for debugging
    )

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Added OPTIONS for preflight
    allow_headers=["*"],  # Allow all headers for WebSocket compatibility
)

# API Key Security
security = HTTPBearer(auto_error=False)


async def verify_api_key(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    """Verify API key if configured"""
    # Check if we're in production (not localhost)
    is_production = os.getenv("ENVIRONMENT", "development") == "production"

    if not settings.api_key:
        if is_production:
            logger.error("API key not configured in production environment!")
            raise HTTPException(status_code=500, detail="Server configuration error: API key required")
        else:
            logger.warning("API key not configured - running without authentication")
            return True

    if not credentials or credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


@app.get("/")
async def root():
    """Root endpoint"""
    return {"name": "Consistency Lens API", "version": "1.0.0", "status": "running"}


@app.post("/analyze", response_model=dict)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def analyze(
    request: AnalyzeRequest, background_tasks: BackgroundTasks, req: Request, authorized: bool = Depends(verify_api_key)
):
    """Queue analysis request and return request ID"""
    if not inference_service:
        raise HTTPException(500, "Inference service not initialized")

    # Validate text length
    if len(request.text) > settings.max_text_length:
        raise HTTPException(400, f"Text too long. Maximum length is {settings.max_text_length} characters")

    # Lazy load model on first request
    if model_manager and not model_manager.is_model_loaded():
        try:
            # Load default model
            await model_manager.initialize_default_model()
        except Exception as e:
            raise HTTPException(500, f"Failed to load model: {str(e)}") from e

    request_id = await inference_service.queue.add_request(request.text, request.options.dict())

    # Monitor for disconnection
    async def check_disconnection():
        while request_id in inference_service.queue.active_requests:
            if await req.is_disconnected():
                inference_service.queue.active_requests[request_id]["status"] = "cancelled"
                break
            await asyncio.sleep(1)

    background_tasks.add_task(check_disconnection)

    # Get actual position in queue
    position = inference_service.queue.get_position_in_queue(request_id)
    stats = inference_service.queue.get_queue_stats()

    return {"request_id": request_id, "status": "queued", "queue_position": position, "queue_size": stats["queue_size"]}


@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Get status of a request"""
    if not inference_service:
        raise HTTPException(500, "Inference service not initialized")

    status = inference_service.queue.get_status(request_id)
    if not status:
        raise HTTPException(404, "Request not found")

    # Convert datetime objects to strings for JSON serialization
    response = {
        "request_id": request_id,
        "status": status["status"],
        "created_at": status["created_at"].isoformat() if isinstance(status.get("created_at"), datetime) else None,
        "started_at": status.get("started_at").isoformat() if isinstance(status.get("started_at"), datetime) else None,
        "completed_at": status.get("completed_at").isoformat()
        if isinstance(status.get("completed_at"), datetime)
        else None,
        "processing_time": status.get("processing_time"),
        "result": status.get("result"),
        "error": status.get("error"),
    }

    return response


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    # Log headers for debugging
    logger.info(f"WebSocket connection attempt from: {websocket.client}")
    logger.info(f"WebSocket headers: {websocket.headers}")

    await manager.connect(websocket)

    # Send model info from model_manager
    if model_manager:
        model_info = model_manager.get_current_model_info()
        # Transform the response to match expected format
        if model_info.get("status") == "no_model_loaded":
            # Model not loaded yet
            await websocket.send_json({
                "type": "model_info",
                "loaded": False,
                "model_name": "No model loaded",
                "auto_batch_size_max": settings.auto_batch_size_max,
            })
        else:
            # Model is loaded
            await websocket.send_json({
                "type": "model_info",
                "loaded": True,
                "model_id": model_info.get("model_id"),
                "display_name": model_info.get("display_name", "Unknown"),
                "model_name": model_info.get("display_name", "Unknown"),  # For backwards compatibility
                "description": model_info.get("description"),
                "batch_size": model_info.get("batch_size"),
                "auto_batch_size_max": model_info.get("auto_batch_size_max", settings.auto_batch_size_max),
                "layer": model_info.get("layer"),
                "checkpoint_path": model_info.get("checkpoint_path"),
                "checkpoint_filename": model_info.get("checkpoint_filename"),
                "cached_models": model_info.get("cached_models", []),
                "is_switching": model_info.get("is_switching", False),
            })

    # Send initial queue status
    if inference_service:
        stats = inference_service.queue.get_queue_stats()
        await websocket.send_json(
            {
                "type": "queue_update",
                "queue_size": stats["queue_size"],
                "queued_requests": stats["queued_requests"],
                "processing_requests": stats["processing_requests"],
                "total_active": stats["total_active"],
            }
        )

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "analyze":
                if not inference_service:
                    await websocket.send_json({"type": "error", "error": "Inference service not initialized"})
                    continue

            elif data["type"] == "generate":
                if not inference_service:
                    await websocket.send_json({"type": "generation_error", "error": "Inference service not initialized"})
                    continue

                # Lazy load model if needed through model_manager
                if model_manager and not model_manager.is_model_loaded():
                    await websocket.send_json(
                        {
                            "type": "status",
                            "message": "Loading model checkpoint... This may take a minute on first load.",
                        }
                    )
                    try:
                        await model_manager.ensure_model_loaded()
                        await websocket.send_json({"type": "status", "message": "Model loaded successfully!"})
                        # Send updated model info
                        model_info = model_manager.get_current_model_info()
                        await websocket.send_json(
                            {
                                "type": "model_info",
                                "loaded": True,
                                "model_id": model_info.get("model_id"),
                                "model_name": model_info.get("display_name", "Unknown"),
                                "description": model_info.get("description"),
                                "batch_size": model_info.get("batch_size"),
                                "auto_batch_size_max": model_info.get("auto_batch_size_max", settings.auto_batch_size_max),
                            }
                        )
                    except ModelLoadError as e:
                        await websocket.send_json(
                            {"type": "generation_error", "error": f"Failed to load model: {str(e)}"}
                        )
                        continue

                # Add generation request to queue through inference_service
                text = data.get("text", "")
                options = data.get("options", {})

                # Add websocket to request for real-time updates
                options["websocket"] = websocket

                # Add to queue with type 'generate'
                request_id = await inference_service.queue.add_request(text, options, request_type="generate")

                # Get current queue position
                position = inference_service.queue.get_position_in_queue(request_id)
                stats = inference_service.queue.get_queue_stats()

                await websocket.send_json(
                    {
                        "type": "queued",
                        "request_id": request_id,
                        "queue_position": position,
                        "queue_size": stats["queue_size"],
                        "context": "generation",
                    }
                )

                continue

                # This code was orphaned - it should be part of the analyze block
                # Moving it inside the analyze condition

            if data["type"] == "analyze":
                # Lazy load model if needed through model_manager
                if model_manager and not model_manager.is_model_loaded():
                    await websocket.send_json(
                        {
                            "type": "status",
                            "message": "Loading model checkpoint... This may take a minute on first load.",
                        }
                    )
                    try:
                        await model_manager.ensure_model_loaded()
                        await websocket.send_json({"type": "status", "message": "Model loaded successfully!"})
                        # Send updated model info
                        model_info = model_manager.get_current_model_info()
                        await websocket.send_json(
                            {
                                "type": "model_info",
                                "loaded": True,
                                "model_id": model_info.get("model_id"),
                                "model_name": model_info.get("display_name", "Unknown"),
                                "description": model_info.get("description"),
                                "batch_size": model_info.get("batch_size"),
                                "auto_batch_size_max": model_info.get("auto_batch_size_max", settings.auto_batch_size_max),
                            }
                        )
                    except ModelLoadError as e:
                        await websocket.send_json({"type": "error", "error": f"Failed to load model: {str(e)}"})
                        continue

                # Get options and add websocket
                options = data.get("options", {})
                options["websocket"] = websocket
                
                request_id = await inference_service.queue.add_request(data["text"], options)

                # Get actual position in queue
                position = inference_service.queue.get_position_in_queue(request_id)
                stats = inference_service.queue.get_queue_stats()

                await websocket.send_json(
                    {
                        "type": "queued",
                        "request_id": request_id,
                        "queue_position": position,
                        "queue_size": stats["queue_size"],
                        "context": "analysis",
                    }
                )

            elif data["type"] == "status":
                request_id = data.get("request_id")
                if request_id and inference_service:
                    status = inference_service.queue.get_status(request_id)
                    if status:
                        await websocket.send_json(
                            {
                                "type": "status_update",
                                "request_id": request_id,
                                "status": status["status"],
                                "error": status.get("error"),
                            }
                        )

            elif data["type"] == "list_models":
                # List available models
                if not model_manager:
                    await websocket.send_json({"type": "error", "error": "Model manager not initialized"})
                    continue
                    
                from .model_registry import list_available_models
                models = list_available_models()
                current_info = model_manager.get_current_model_info()
                
                # Get queue stats to show if there are active requests
                queue_stats = inference_service.queue.get_queue_stats() if inference_service else {"queue_size": 0, "processing_requests": 0}
                
                await websocket.send_json({
                    "type": "models_list",
                    "models": models,
                    "current_model": current_info.get("model_id"),
                    "is_switching": current_info.get("is_switching", False),
                    "queue_stats": queue_stats
                })
                
            elif data["type"] == "switch_model":
                # Switch to a different model
                if not model_manager:
                    await websocket.send_json({"type": "error", "error": "Model manager not initialized"})
                    continue
                    
                model_id = data.get("model_id")
                if not model_id:
                    await websocket.send_json({"type": "error", "error": "No model_id provided"})
                    continue
                    
                # Send immediate acknowledgment
                await websocket.send_json({
                    "type": "model_switch_acknowledged",
                    "model_id": model_id,
                    "message": "Model switch initiated. This affects all users."
                })
                
                try:
                    # Perform the switch
                    result = await model_manager.switch_model(model_id)
                    
                    # Send success message
                    model_info = model_manager.get_current_model_info()
                    await websocket.send_json({
                        "type": "model_switch_complete",
                        "model_id": model_id,
                        "message": result["message"],
                        "model_info": model_info,
                        "generation_config": model_info.get("generation_config", {})
                    })
                    
                except Exception as e:
                    logger.error(f"Model switch failed: {e}")
                    await websocket.send_json({
                        "type": "model_switch_error",
                        "model_id": model_id,
                        "error": str(e)
                    })
                    
            elif data["type"] == "get_model_info":
                # Get current model info
                if not model_manager:
                    await websocket.send_json({"type": "error", "error": "Model manager not initialized"})
                    continue
                    
                model_info = model_manager.get_current_model_info()
                await websocket.send_json({
                    "type": "model_info_update",
                    **model_info
                })
                
            elif data["type"] == "reload_models":
                # Reload model registry from JSON
                # Check for API key authentication
                api_key = data.get("api_key")
                if settings.api_key and api_key != settings.api_key:
                    await websocket.send_json({
                        "type": "error", 
                        "error": "Authentication required. Please provide a valid API key."
                    })
                    logger.warning(f"Unauthorized reload_models attempt from {websocket.client}")
                    continue
                    
                try:
                    from .model_registry import model_registry, list_available_models
                    model_registry.reload_models()
                    
                    # Send updated model list
                    models = list_available_models()
                    current_info = model_manager.get_current_model_info()
                    
                    await websocket.send_json({
                        "type": "models_reloaded",
                        "models": models,
                        "current_model": current_info.get("model_id"),
                        "message": "Model registry reloaded successfully"
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Failed to reload models: {str(e)}"
                    })
                
            elif data["type"] == "interrupt":
                request_id = data.get("request_id")
                context = data.get("context", "analysis")

                if request_id and inference_service:
                    # Mark the request as cancelled
                    request = inference_service.queue.active_requests.get(request_id)
                    if request and request["status"] == "processing":
                        request["status"] = "cancelled"
                        request["error"] = "Interrupted by user"

                        logger.info(f"Interrupt requested for {context} request: {request_id}")

                        # Send confirmation
                        await websocket.send_json({"type": "interrupted", "request_id": request_id, "context": context})
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "error": f"Cannot interrupt request {request_id} - not currently processing",
                            }
                        )

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "model_loaded": model_manager is not None and model_manager.is_model_loaded(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "queue_size": inference_service.queue.queue.qsize() if inference_service else 0,
    }

    # If GPU is not available, mark as unhealthy
    if not health_status["gpu_available"]:
        health_status["status"] = "degraded"
        health_status["warning"] = "No GPU available"

    return health_status


@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring"""
    if not inference_service:
        return {"error": "Inference service not initialized"}

    active_requests = inference_service.queue.active_requests
    completed = [r for r in active_requests.values() if r["status"] == "completed"]
    failed = [r for r in active_requests.values() if r["status"] == "failed"]

    # Calculate average processing time
    processing_times = [r.get("processing_time", 0) for r in completed if "processing_time" in r]
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

    return {
        "total_requests": len(active_requests),
        "completed_requests": len(completed),
        "failed_requests": len(failed),
        "queue_size": inference_service.queue.queue.qsize(),
        "average_processing_time": avg_processing_time,
        "active_websocket_connections": len(manager.active_connections),
    }


@app.post("/reset")
async def reset_model():
    """Reset the model and clear memory"""
    global model_manager, inference_service

    if inference_service:
        # Stop inference processing
        await inference_service.stop_processing()
        
        # Clear the queue
        inference_service.queue.active_requests.clear()

    if model_manager:
        # Clear the model from memory through model_manager
        await model_manager.clear_current_model()

        # Force garbage collection
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model reset and memory cleared")

        # Reload model if lazy loading is disabled
        if not settings.lazy_load_model:
            logger.info("Reloading model after reset (lazy_load_model=false)")
            try:
                await model_manager.initialize_default_model()
                logger.info("Model reloaded successfully")
                # Restart inference processing
                if inference_service:
                    await inference_service.start_processing()
                return {"status": "reset", "message": "Model cleared and reloaded."}
            except ModelLoadError as e:
                logger.error(f"Failed to reload model after reset: {e}")
                raise HTTPException(500, f"Model cleared, but failed to reload: {e}") from e

    return {"status": "reset", "message": "Model cleared from memory"}


@app.get("/api/models")
async def list_models():
    """List available models"""
    from .model_registry import list_available_models
    
    models = list_available_models()
    current_info = model_manager.get_current_model_info() if model_manager else {"status": "no_model_manager"}
    
    return {
        "models": models,
        "current_model": current_info.get("model_id") if "model_id" in current_info else None,
        "is_switching": current_info.get("is_switching", False),
        "model_status": current_info
    }

@app.post("/api/models/switch")
async def switch_model(model_id: str, authorized: bool = Depends(verify_api_key)):
    """Switch to a different model (requires API key)"""
    if not model_manager:
        raise HTTPException(500, "Model manager not initialized")
        
    try:
        result = await model_manager.switch_model(model_id)
        return {
            "status": "success",
            **result,
            "model_info": model_manager.get_current_model_info()
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        raise HTTPException(500, f"Model switch failed: {str(e)}")

@app.get("/api/models/cache")
async def get_model_cache_status():
    """Get model cache status"""
    if not model_manager:
        return {"error": "Model manager not initialized"}
    
    cache_status = await model_manager.get_cache_status()
    memory_usage = model_manager.get_memory_usage()
    
    return {
        "cache": cache_status,
        "memory": memory_usage,
    }

@app.post("/api/models/reload")
async def reload_models(authorized: bool = Depends(verify_api_key)):
    """Reload model registry from JSON file (requires API key)"""
    try:
        from .model_registry import model_registry
        model_registry.reload_models()
        
        # Get updated list
        models = model_registry.list_available_models()
        
        return {
            "status": "success",
            "message": "Model registry reloaded successfully",
            "models_count": len(models),
            "models": models
        }
    except Exception as e:
        logger.error(f"Failed to reload model registry: {e}")
        raise HTTPException(500, f"Failed to reload models: {str(e)}")

@app.get("/api/gpu_stats")
async def get_gpu_stats():
    """Get GPU utilization and memory stats from the background monitor"""
    # Get cached stats from the monitor
    stats = gpu_monitor.get_stats()

    # Remove internal fields that frontend doesn't need
    stats.pop("is_computing", None)
    stats.pop("last_update", None)

    # Ensure all expected fields are present for backward compatibility
    return {
        "available": stats.get("available", False),
        "utilization": stats.get("utilization", 0),
        "memory_used": stats.get("memory_used", 0),
        "memory_total": stats.get("memory_total", 0),
        "memory_percent": stats.get("memory_percent", 0),
        "peak_utilization": stats.get("peak_utilization", 0),
    }
