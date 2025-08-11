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
from .model_manager import UnifiedModelManager
from .inference_service import InferenceService
from .models import AnalyzeRequest
from .websocket import manager
from . import unified_api


# Custom logging filter to suppress GPU stats access logs
class SuppressGPUStatsFilter(logging.Filter):
    def filter(self, record):
        # Suppress uvicorn access logs for /api/gpu_stats endpoint
        if hasattr(record, 'args') and record.args:
            # Check if this is a uvicorn access log
            if len(record.args) >= 3 and '/api/gpu_stats' in str(record.args[1]):
                return False
        # Also check the message directly
        if record.getMessage and '/api/gpu_stats' in record.getMessage():
            return False
        return True


# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Add filter to uvicorn access logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(SuppressGPUStatsFilter())

logger = logging.getLogger(__name__)

logger.info(f"Loaded .env from {dotenv_location}")
logger.info(f"Loaded parent .env from {parent_dotenv_location}")


# GPU Stats Monitor Class
class GPUStatsMonitor:
    def __init__(self, devices=None, update_interval: float = 0.5):
        self.update_interval = update_interval
        self.devices = devices or ["cuda:0"]
        
        # Initialize stats for each device
        self.stats = {
            "available": torch.cuda.is_available(),
            "last_update": None,
            "devices": {}
        }
        
        # Initialize per-device stats
        for device in self.devices:
            device_num = int(device.split(':')[1]) if ':' in device else 0
            self.stats["devices"][device] = {
                "utilization": 0,
                "memory_used": 0,
                "memory_total": 0,
                "memory_percent": 0,
                "peak_utilization": 0,
                "is_computing": False,
                "device_num": device_num
            }
        
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self._last_reset_times = {device: time.time() for device in self.devices}

    def start(self):
        """Start the monitoring thread"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"GPU stats monitor started for devices: {self.devices}")

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
                stats = {
                    "available": torch.cuda.is_available(), 
                    "last_update": datetime.utcnow().isoformat(),
                    "devices": {}
                }

                if torch.cuda.is_available():
                    # Monitor each device
                    for device in self.devices:
                        device_num = int(device.split(':')[1]) if ':' in device else 0
                        device_stats = {"device_num": device_num}
                        
                        try:
                            # Get memory stats from PyTorch
                            with torch.cuda.device(device_num):
                                memory_used = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                                memory_total = torch.cuda.get_device_properties(device_num).total_memory / 1024**3
                                device_stats["memory_used"] = round(memory_used, 2)
                                device_stats["memory_total"] = round(memory_total, 2)
                                device_stats["memory_percent"] = round((memory_used / memory_total) * 100, 1)

                            # Try to get GPU utilization from pynvml
                            if pynvml_available:
                                try:
                                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_num)
                                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                    device_stats["utilization"] = utilization.gpu

                                    # Track peak utilization
                                    with self.lock:
                                        prev_peak = self.stats["devices"].get(device, {}).get("peak_utilization", 0)
                                        if utilization.gpu > prev_peak:
                                            device_stats["peak_utilization"] = utilization.gpu
                                        else:
                                            device_stats["peak_utilization"] = prev_peak

                                except Exception as e:
                                    logger.debug(f"Failed to get GPU utilization for device {device}: {e}")
                                    device_stats["utilization"] = 0
                                    device_stats["peak_utilization"] = self.stats["devices"].get(device, {}).get("peak_utilization", 0)
                            else:
                                device_stats["utilization"] = 0
                                device_stats["peak_utilization"] = self.stats["devices"].get(device, {}).get("peak_utilization", 0)

                            # Detect if we're computing based on memory usage changes
                            with self.lock:
                                prev_memory = self.stats["devices"].get(device, {}).get("memory_used", 0)
                                device_stats["is_computing"] = abs(memory_used - prev_memory) > 0.1  # Changed by >100MB
                                
                        except Exception as e:
                            logger.error(f"Error monitoring device {device}: {e}")
                            # Keep previous stats if available
                            if device in self.stats["devices"]:
                                device_stats = self.stats["devices"][device].copy()
                            
                        stats["devices"][device] = device_stats

                # Update shared stats
                with self.lock:
                    self.stats.update(stats)

                    # Reset peak utilization every 30 seconds if not computing
                    current_time = time.time()
                    for device in self.devices:
                        if (current_time - self._last_reset_times.get(device, 0) > 30 and 
                            not stats["devices"].get(device, {}).get("is_computing", False)):
                            if device in self.stats["devices"]:
                                self.stats["devices"][device]["peak_utilization"] = stats["devices"].get(device, {}).get("utilization", 0)
                            self._last_reset_times[device] = current_time

            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")

            time.sleep(self.update_interval)

    def get_stats(self):
        """Get current GPU stats (thread-safe)"""
        with self.lock:
            return self.stats.copy()


# Load settings first
settings = load_settings()

# Global instances
model_manager: UnifiedModelManager | None = None
inference_service: InferenceService | None = None
gpu_monitor = GPUStatsMonitor(devices=settings.devices)  # Pass configured devices
queue_broadcast_task = None
cleanup_task = None

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


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


async def cleanup_old_requests_periodically():
    """Periodically clear out old, completed requests to prevent memory leaks."""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        if inference_service:
            try:
                inference_service.queue.clear_completed_requests(older_than_seconds=3600)
                logger.info("Cleared old completed requests from queue")
            except Exception as e:
                logger.error(f"Error during periodic request cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_manager, inference_service

    # Start GPU monitoring
    gpu_monitor.start()

    try:
        # Initialize unified model manager
        model_manager = UnifiedModelManager(settings)
        await model_manager.start()
        
        # Initialize inference service with unified model manager
        inference_service = InferenceService(model_manager, settings, manager)
        
        # Set the model manager in api module
        unified_api.model_manager = model_manager

        # Models will be loaded on-demand in the unified system
        if not settings.lazy_load_model:
            logger.info("Non-lazy mode: models will be loaded on first request")
        else:
            logger.info("Lazy mode: models will be loaded on-demand")

        await inference_service.start_processing()
        
        # Start periodic queue status broadcaster
        global queue_broadcast_task, cleanup_task
        queue_broadcast_task = asyncio.create_task(broadcast_queue_status_periodically())
        
        # Start periodic cleanup task
        cleanup_task = asyncio.create_task(cleanup_old_requests_periodically())
        
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
    
    # Stop cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    if inference_service:
        await inference_service.stop_processing()
    
    if model_manager:
        await model_manager.stop()


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

# Include unified API routes
app.include_router(unified_api.router)

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
            if not hasattr(verify_api_key, "_warn_count"):
                verify_api_key._warn_count = 0
            if verify_api_key._warn_count < 3:
                logger.warning("API key not configured - running without authentication")
                verify_api_key._warn_count += 1
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

    # In the new architecture, models are loaded on-demand
    # No need to check or preload models

    # Enrich options with origin for file logging split
    options = request.options.dict()
    try:
        options["origin"] = req.headers.get("origin")
    except Exception:
        pass
    request_id = await inference_service.queue.add_request(request.text, options)

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

    # In the unified architecture, we just send a connection established message
    if model_manager:
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to unified inference server",
            "num_devices": model_manager.num_devices
        })
        
        # Send current system state
        system_state = model_manager.get_system_state()
        await websocket.send_json({
            "type": "system_state",
            "data": system_state
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

                # Add generation request to queue through inference_service
                text = data.get("text", "")
                options = data.get("options", {})
                # Include origin if available in websocket headers
                try:
                    if hasattr(websocket, "headers") and websocket.headers is not None:
                        origin = websocket.headers.get("origin") if hasattr(websocket.headers, "get") else None
                        if origin:
                            options["origin"] = origin
                except Exception:
                    pass

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
                # Get options and add websocket
                options = data.get("options", {})
                options["websocket"] = websocket
                # Include origin if available in websocket headers
                try:
                    if hasattr(websocket, "headers") and websocket.headers is not None:
                        origin = websocket.headers.get("origin") if hasattr(websocket.headers, "get") else None
                        if origin:
                            options["origin"] = origin
                except Exception:
                    pass
                
                # Include model_id if provided
                if "model_id" in data:
                    options["model_id"] = data["model_id"]
                
                # Include client_request_id if provided
                if "client_request_id" in data:
                    options["client_request_id"] = data["client_request_id"]
                
                request_id = await inference_service.queue.add_request(data["text"], options)

                # Get actual position in queue
                position = inference_service.queue.get_position_in_queue(request_id)
                stats = inference_service.queue.get_queue_stats()

                response = {
                    "type": "queued",
                    "request_id": request_id,
                    "queue_position": position,
                    "queue_size": stats["queue_size"],
                    "context": "analysis",
                }
                # Include client_request_id if provided
                if "client_request_id" in options:
                    response["client_request_id"] = options["client_request_id"]
                
                await websocket.send_json(response)

            elif data["type"] == "generate_and_analyze":
                # Chained request: generate (n=1) then analyze using click-time settings
                gen = data.get("generation", {})
                ana = data.get("analysis", {})
                client_request_id = data.get("client_request_id")

                # Prepare generation options (force single continuation)
                gen_options = gen.get("options", {})
                gen_text = gen.get("text", "")
                # Include websocket and origin
                gen_options["websocket"] = websocket
                try:
                    if hasattr(websocket, "headers") and websocket.headers is not None:
                        origin = websocket.headers.get("origin") if hasattr(websocket.headers, "get") else None
                        if origin:
                            gen_options["origin"] = origin
                except Exception:
                    pass
                # Force n=1 but keep other settings
                gen_options["num_completions"] = 1
                # If chat formatting is requested elsewhere, honor it like regular generation

                # Determine model for both requests from payload or selected model
                if "model_id" in gen:
                    gen_options["model_id"] = gen["model_id"]
                if "model_id" in ana:
                    # analysis model explicit wins; otherwise will use gen model via payload
                    pass

                # Prepare analysis options and text
                ana_options = ana.get("options", {})
                ana_text = ana.get("text", "")
                ana_options["websocket"] = websocket
                try:
                    if hasattr(websocket, "headers") and websocket.headers is not None:
                        origin = websocket.headers.get("origin") if hasattr(websocket.headers, "get") else None
                        if origin:
                            ana_options["origin"] = origin
                except Exception:
                    pass
                # If analysis should use the generated completion directly, set flag
                # When set, the analysis processor will take the prior completion text
                if ana.get("use_generated_completion", True):
                    ana_options["use_prior_generated_text"] = True

                # Propagate client_request_id so completion events can be matched on frontend
                if client_request_id:
                    ana_options["client_request_id"] = client_request_id

                # Model selection: if not explicitly set for analysis, mirror generation model
                if "model_id" not in ana_options and "model_id" in gen_options:
                    ana_options["model_id"] = gen_options["model_id"]

                # Queue both requests back-to-back
                gen_id, ana_id = await inference_service.queue.add_chained_requests(
                    gen_text,
                    gen_options,
                    "generate",
                    ana_text,
                    ana_options,
                    "analyze",
                    depends=True,
                )

                # Report both queued positions
                gen_pos = inference_service.queue.get_position_in_queue(gen_id)
                ana_pos = inference_service.queue.get_position_in_queue(ana_id)
                stats = inference_service.queue.get_queue_stats()

                gen_response = {
                    "type": "queued",
                    "request_id": gen_id,
                    "queue_position": gen_pos,
                    "queue_size": stats["queue_size"],
                    "context": "generation",
                }
                if client_request_id:
                    gen_response["client_request_id"] = client_request_id + "_gen"
                await websocket.send_json(gen_response)
                ana_response = {
                    "type": "queued",
                    "request_id": ana_id,
                    "queue_position": ana_pos,
                    "queue_size": stats["queue_size"],
                    "context": "analysis",
                }
                if client_request_id:
                    ana_response["client_request_id"] = client_request_id
                await websocket.send_json(
                    ana_response
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
                # List all models with unified format
                if not model_manager:
                    await websocket.send_json({"type": "error", "error": "Model manager not initialized"})
                    continue
                    
                # Determine if this is a public request
                origin_header = websocket.headers.get('origin') or ""
                host_header = websocket.headers.get('host') or ""
                public_only = ("kitft.com" in origin_header) or ("kitft.com" in host_header)
                
                from .model_registry import get_all_models
                all_models = get_all_models(public_only=public_only)
                
                # Enhance with location information
                models_with_locations = []
                for model_data in all_models:
                    model_id = model_data["id"]
                    location_info = model_manager.get_model_location(model_id)
                    
                    model_info = {
                        **model_data,
                        "locations": location_info["locations"] if location_info else []
                    }
                    models_with_locations.append(model_info)
                
                # Get system state
                system_state = model_manager.get_system_state()
                
                await websocket.send_json({
                    "type": "models_list",
                    "models": models_with_locations,
                    "system_state": system_state
                })
                
                    
            elif data["type"] == "load_model":
                # Load a model in the unified system
                if not model_manager:
                    await websocket.send_json({"type": "error", "error": "Model manager not initialized"})
                    continue
                    
                model_id = data.get("model_id")
                device_id = data.get("device_id")  # Optional
                
                if not model_id:
                    await websocket.send_json({"type": "error", "error": "No model_id provided"})
                    continue
                
                try:
                    # Check if model is publicly available
                    origin_header = websocket.headers.get('origin') or ""
                    host_header = websocket.headers.get('host') or ""
                    public_only = ("kitft.com" in origin_header) or ("kitft.com" in host_header)
                    
                    if public_only:
                        from .model_registry import get_model_info
                        model_info = get_model_info(model_id)
                        if model_info and (model_info.get("backend_only", False) or model_info.get("visible") is False):
                            await websocket.send_json({
                                "type": "model_load_error",
                                "model_id": model_id,
                                "error": "This model is not available publicly"
                            })
                            continue
                    
                    # Load the model
                    result = await model_manager.load_model(model_id, device_id)
                    
                    # Get model info
                    from .model_registry import get_model_info
                    model_info = get_model_info(model_id)
                    
                    await websocket.send_json({
                        "type": "model_loaded",
                        "model_id": model_id,
                        "device_id": result["device_id"],
                        "group_id": result.get("group_id"),
                        "model_info": model_info,
                        "generation_config": model_info.get("generation_config", {})
                    })
                    
                except Exception as e:
                    logger.error(f"Model load error: {e}")
                    await websocket.send_json({
                        "type": "model_load_error",
                        "model_id": model_id,
                        "error": str(e)
                    })
                    
            elif data["type"] == "load_group":
                # Load a group in the unified system
                if not model_manager:
                    await websocket.send_json({"type": "error", "error": "Model manager not initialized"})
                    continue
                    
                group_id = data.get("group_id")
                device_id = data.get("device_id")  # Optional
                
                if not group_id:
                    await websocket.send_json({"type": "error", "error": "No group_id provided"})
                    continue
                    
                try:
                    result = await model_manager.load_group(group_id, device_id)
                    await websocket.send_json({
                        "type": "group_loaded",
                        "group_id": group_id,
                        "device_id": result["device_id"],
                        "loaded_models": result["loaded_models"],
                        "message": f"Successfully loaded group {group_id} on device {result['device_id']}"
                    })
                except Exception as e:
                    logger.error(f"Group load failed: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Failed to load group: {str(e)}"
                    })
                    
            elif data["type"] == "clear_device":
                # Clear a device in the unified system
                if not model_manager:
                    await websocket.send_json({"type": "error", "error": "Model manager not initialized"})
                    continue
                    
                device_id = data.get("device_id")
                if device_id is None:
                    await websocket.send_json({"type": "error", "error": "No device_id provided"})
                    continue
                    
                try:
                    result = await model_manager.clear_device(device_id)
                    
                    await websocket.send_json({
                        "type": "device_cleared",
                        "device_id": device_id,
                        "message": f"Successfully cleared all models from device {device_id}"
                    })
                    
                    # Broadcast to all clients
                    await manager.broadcast({
                        "type": "device_cleared",
                        "device_id": device_id
                    })
                    
                except Exception as e:
                    logger.error(f"Device clear failed: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Failed to clear device: {str(e)}"
                    })
                    
            elif data["type"] == "get_model_info":
                # Get model info for a specific model
                model_id = data.get("model_id")
                
                if model_manager and model_id:
                    from .model_registry import get_model_info
                    model_info = get_model_info(model_id)
                    
                    if model_info:
                        # Check if model is loaded
                        location_info = model_manager.get_model_location(model_id)
                        
                        await websocket.send_json({
                            "type": "model_info",
                            "model_id": model_id,
                            "display_name": model_info.get("name"),
                            "layer": model_info.get("layer"),
                            "auto_batch_size_max": model_info.get("auto_batch_size_max"),
                            "generation_config": model_info.get("generation_config", {}),
                            "loaded": location_info is not None,
                            "locations": location_info["locations"] if location_info else []
                        })
                    else:
                        await websocket.send_json({"type": "error", "error": f"Unknown model: {model_id}"})
                else:
                    await websocket.send_json({"type": "error", "error": "Model manager not initialized"})
                
                
            elif data["type"] == "cancel_request":
                # Cancel a queued or processing request
                request_id = data.get("request_id")
                
                if request_id and inference_service:
                    request = inference_service.queue.active_requests.get(request_id)
                    if request:
                        current_status = request["status"]
                        if current_status in ["queued", "processing", "waiting_for_group_switch"]:
                            # Mark as cancelled
                            request["status"] = "cancelled"
                            request["error"] = "Cancelled by user"
                            request["completed_at"] = datetime.utcnow()
                            
                            logger.info(f"Cancelled {current_status} request: {request_id} (type: {request.get('type')})")
                            
                            # Send confirmation
                            await websocket.send_json({
                                "type": "request_cancelled",
                                "request_id": request_id,
                                "message": "Request cancelled successfully"
                            })
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "error": f"Cannot cancel request {request_id} - status is {current_status}"
                            })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Request {request_id} not found"
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
                        response = {"type": "interrupted", "request_id": request_id, "context": context}
                        
                        # Include client_request_id if available in the request
                        if "options" in request and "client_request_id" in request["options"]:
                            response["client_request_id"] = request["options"]["client_request_id"]
                            
                        await websocket.send_json(response)
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
        # Cancel any pending/processing requests associated with this websocket
        if inference_service:
            cancelled_count = 0
            for request_id, request in list(inference_service.queue.active_requests.items()):
                # Check if this request is associated with the disconnecting websocket
                if (request.get("options", {}).get("websocket") == websocket and 
                    request.get("status") in ["queued", "processing", "waiting_for_group_switch"]):
                    request["status"] = "cancelled"
                    request["error"] = "Client disconnected"
                    request["completed_at"] = datetime.utcnow()
                    cancelled_count += 1
                    logger.info(f"Cancelled request {request_id} due to websocket disconnection")
            
            if cancelled_count > 0:
                logger.info(f"Cancelled {cancelled_count} requests due to websocket disconnection")
        
        manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "model_loaded": model_manager is not None and any(state.loaded_models for state in model_manager.device_states.values()),
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

    try:
        if inference_service:
            # Stop inference processing
            await inference_service.stop_processing()
            
            # Clear the queue by re-initializing it
            inference_service.queue.queue = asyncio.Queue()
            inference_service.queue.active_requests.clear()
            logger.info("Cleared inference queue")

        # Use unified model manager
        manager_to_use = model_manager

        if manager_to_use:
            # Clear all models from memory through manager
            if hasattr(manager_to_use, 'clear_all_models'):
                await manager_to_use.clear_all_models()
            elif hasattr(manager_to_use, 'clear_current_model'):
                await manager_to_use.clear_current_model()
            else:
                # For grouped model manager, unload current group
                if hasattr(manager_to_use, 'current_group_id') and manager_to_use.current_group_id:
                    await manager_to_use._unload_group(manager_to_use.current_group_id)
                    manager_to_use.current_group_id = None

            # Force garbage collection multiple times
            import gc
            for _ in range(3):
                gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Wait a bit for memory to be fully released
                await asyncio.sleep(0.5)

            logger.info("Model reset and memory cleared")

            # Only reload model if explicitly not in lazy mode and we have enough memory
            if not settings.lazy_load_model:
                # Check available GPU memory first
                if torch.cuda.is_available():
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    free_memory_gb = free_memory / (1024 ** 3)
                    logger.info(f"Available GPU memory after reset: {free_memory_gb:.2f} GB")
                    
                    # Only reload if we have enough memory (at least 20GB free)
                    if free_memory_gb < 20:
                        logger.warning(f"Not enough GPU memory to reload model ({free_memory_gb:.2f} GB free)")
                        # Restart inference processing anyway
                        if inference_service:
                            await inference_service.start_processing()
                        return {"status": "reset", "message": "Model cleared. Low memory - model will be loaded on next request."}
                
                logger.info("Reloading model after reset (lazy_load_model=false)")
                try:
                    if hasattr(manager_to_use, 'ensure_group_loaded'):
                        await manager_to_use.ensure_group_loaded()
                    else:
                        await manager_to_use.initialize_default_model()
                    logger.info("Model reloaded successfully")
                    # Restart inference processing
                    if inference_service:
                        await inference_service.start_processing()
                    return {"status": "reset", "message": "Model cleared and reloaded."}
                except Exception as e:
                    logger.error(f"Failed to reload model after reset: {e}")
                    # Don't raise - just continue without model
                    if inference_service:
                        await inference_service.start_processing()
                    return {"status": "reset", "message": f"Model cleared. Failed to reload: {str(e)}"}
            else:
                # Restart inference processing
                if inference_service:
                    await inference_service.start_processing()

        return {"status": "reset", "message": "Model cleared from memory"}
        
    except Exception as e:
        logger.error(f"Error during reset: {e}")
        # Try to restart inference processing even if reset failed
        if inference_service:
            try:
                await inference_service.start_processing()
            except:
                pass
        raise HTTPException(500, f"Reset failed: {str(e)}") from e






@app.get("/api/gpu_stats")
async def get_gpu_stats():
    """Get GPU utilization and memory stats from the background monitor"""
    # Get cached stats from the monitor
    stats = gpu_monitor.get_stats()

    # Format response with multi-GPU support
    response = {
        "available": stats.get("available", False),
        "devices": {}
    }
    
    # Add per-device stats
    for device, device_stats in stats.get("devices", {}).items():
        # Remove internal fields
        cleaned_stats = {k: v for k, v in device_stats.items() if k not in ["is_computing", "device_num"]}
        response["devices"][device] = cleaned_stats
    
    # For backward compatibility, also include stats for first device at top level
    if settings.devices:
        first_device = settings.devices[0]
        first_stats = stats.get("devices", {}).get(first_device, {})
        response.update({
            "utilization": first_stats.get("utilization", 0),
            "memory_used": first_stats.get("memory_used", 0),
            "memory_total": first_stats.get("memory_total", 0),
            "memory_percent": first_stats.get("memory_percent", 0),
            "peak_utilization": first_stats.get("peak_utilization", 0),
        })
    
    return response
