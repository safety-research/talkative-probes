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
from .model_manager_grouped import GroupedModelManager
from .inference_service import InferenceService
from .models import AnalyzeRequest
from .websocket import manager
from . import api_grouped


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
grouped_model_manager: GroupedModelManager | None = None  # New grouped manager
inference_service: InferenceService | None = None
gpu_monitor = GPUStatsMonitor()
queue_broadcast_task = None
cleanup_task = None

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
    global grouped_model_manager, inference_service

    # Start GPU monitoring
    gpu_monitor.start()

    try:
        # Initialize grouped model manager
        grouped_model_manager = GroupedModelManager(settings)
        grouped_model_manager.websocket_manager = manager
        
        # Initialize inference service with grouped model manager
        # (can switch to use grouped_model_manager as primary)
        inference_service = InferenceService(grouped_model_manager, settings, manager)
        
        # Set the instances in slim_api module after they're created
        from .api import slim_api
        slim_api.inference_service = inference_service
        slim_api.grouped_model_manager = grouped_model_manager
        slim_api.settings = settings

        # Optionally preload groups on startup
        if not settings.lazy_load_model and grouped_model_manager:
            # Check if we should preload all groups from the config with env overrides
            from .config import Settings
            config_with_overrides = Settings.get_model_config_with_overrides(
                {"settings": grouped_model_manager.config_settings}
            )
            preload_all = config_with_overrides.get('preload_groups', False)
            default_group = config_with_overrides.get('default_group', 'gemma3-27b-it')
            
            if preload_all:
                logger.info("Preloading all groups on startup")
                try:
                    await grouped_model_manager.preload_all_groups(default_group)
                except Exception as e:
                    logger.warning(f"Failed to preload all groups: {e}")
            else:
                logger.info("Preloading default group on startup")
                try:
                    # Always load the default group first
                    if default_group in grouped_model_manager.model_groups:
                        await grouped_model_manager._switch_to_group(default_group)
                    else:
                        await grouped_model_manager.ensure_group_loaded()
                except Exception as e:
                    logger.warning(f"Failed to preload default group: {e}")
        else:
            logger.info("Groups will be loaded on-demand as needed")

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

# Include grouped model API routes
api_grouped.grouped_model_manager = grouped_model_manager
app.include_router(api_grouped.router)

# Include slim API routes
from .api import slim_api
app.include_router(slim_api.router)

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

    # In the new architecture, we don't have a current model on connection
    # Models are specified per-request
    if grouped_model_manager:
        # Just send a connection established message
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to inference server",
            # Note: auto_batch_size_max is now model-specific, see model info
        })
        
        # If a group switch is in progress, notify the new client
        if grouped_model_manager.is_switching_group:
            # Find which group we're switching to
            target_group_id = None
            if inference_service:
                for req_id, req in inference_service.queue.active_requests.items():
                    if req.get("type") == "group_switch" and req.get("status") == "processing":
                        target_group_id = req.get("target_group_id")
                        break
            
            await websocket.send_json({
                "type": "group_switch_status",
                "status": "starting",
                "group_id": target_group_id or "unknown",
                "timestamp": datetime.now().isoformat(),
                "message": "A group switch is in progress. All requests are queued."
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

                
            elif data["type"] == "list_model_groups":
                # List model groups (new grouped format)
                if not grouped_model_manager:
                    await websocket.send_json({"type": "error", "error": "Grouped model manager not initialized"})
                    continue
                
                # Check if this is a refresh request
                if data.get("refresh", False):
                    grouped_model_manager.reload_config()
                    
                # Determine if this is a public request (e.g., from kitft.com)
                origin_header = websocket.headers.get('origin') or ""
                host_header = websocket.headers.get('host') or ""
                public_only = ("kitft.com" in origin_header) or ("kitft.com" in host_header)
                
                groups = grouped_model_manager.get_model_list(public_only=public_only)
                group_info = grouped_model_manager.get_current_group_info()
                memory_info = grouped_model_manager.get_memory_usage()
                
                # Check if there's a queued group switch
                queued_switch_info = None
                if inference_service:
                    for req_id, req in inference_service.queue.active_requests.items():
                        if req.get("type") == "group_switch" and req.get("status") == "queued":
                            queue_position = inference_service.queue.get_position_in_queue(req_id)
                            active_count = len([r for r in inference_service.queue.active_requests.values() 
                                              if r["status"] == "processing"])
                            queued_switch_info = {
                                "request_id": req_id,
                                "target_group_id": req.get("target_group_id"),
                                "model_id": req.get("model_id"),
                                "queue_position": queue_position,
                                "active_requests": active_count,
                                "queued_ahead": queue_position - 1 if queue_position > 0 else 0
                            }
                            break
                
                response_data = {
                    "type": "model_groups_list",
                    "groups": groups,
                    "current_group": group_info["current_group_id"],
                    "is_switching": group_info["is_switching"],
                    "model_status": {
                        "cache_info": {
                            "groups_loaded": group_info["groups_loaded"],
                            "models_cached": list(grouped_model_manager.lens_cache.keys()),
                            "base_locations": group_info["base_locations"]
                        },
                        "memory": memory_info
                    }
                }
                
                # Add queued switch info if present
                if queued_switch_info:
                    response_data["queued_switch"] = queued_switch_info
                
                await websocket.send_json(response_data)
                
                    
            elif data["type"] == "switch_model_grouped":
                # Handle group switch request
                if not grouped_model_manager:
                    await websocket.send_json({"type": "error", "error": "Grouped model manager not initialized"})
                    continue
                    
                model_id = data.get("model_id")
                if not model_id:
                    await websocket.send_json({"type": "error", "error": "No model_id provided"})
                    continue
                
                # Check if this requires a group switch
                try:
                    # Block backend-only models for public requests
                    origin_header = websocket.headers.get('origin') or ""
                    host_header = websocket.headers.get('host') or ""
                    public_only = ("kitft.com" in origin_header) or ("kitft.com" in host_header)
                    if public_only:
                        target_group_for_check = grouped_model_manager.model_to_group.get(model_id)
                        if target_group_for_check:
                            group_cfg = grouped_model_manager.model_groups.get(target_group_for_check)
                            if group_cfg:
                                model_cfg = next((m for m in group_cfg.models if m.get("id") == model_id), None)
                                if model_cfg and (model_cfg.get("backend_only", False) or model_cfg.get("visible") is False):
                                    await websocket.send_json({
                                        "type": "model_switch_error",
                                        "model_id": model_id,
                                        "error": "This model is not available publicly"
                                    })
                                    continue

                    target_group_id = grouped_model_manager.model_to_group.get(model_id)
                    if not target_group_id:
                        await websocket.send_json({
                            "type": "model_switch_error",
                            "model_id": model_id,
                            "error": f"Unknown model ID: {model_id}"
                        })
                        continue
                    
                    # Check if we need to switch groups
                    if target_group_id != grouped_model_manager.current_group_id:
                        # Queue the group switch instead of doing it immediately
                        logger.info(f"Queueing group switch: {grouped_model_manager.current_group_id} -> {target_group_id}")
                        
                        # Add to the queue
                        request_id = await inference_service.queue.add_group_switch_request(
                            target_group_id, model_id, websocket
                        )
                        
                        # Get actual queue position
                        queue_position = inference_service.queue.get_position_in_queue(request_id)
                        active_requests = len([r for r in inference_service.queue.active_requests.values() 
                                             if r["status"] == "processing"])
                        queued_ahead = queue_position - 1 if queue_position > 0 else 0
                        
                        await websocket.send_json({
                            "type": "group_switch_queued",
                            "request_id": request_id,
                            "model_id": model_id,
                            "target_group_id": target_group_id,
                            "queue_position": queue_position,
                            "active_requests": active_requests,
                            "queued_ahead": queued_ahead,
                            "message": f"Group switch queued at position {queue_position}. Will start after {active_requests + queued_ahead} request(s) complete."
                        })
                    else:
                        # Same group - just return model info immediately
                        model_info = grouped_model_manager.get_model_info(model_id)
                        await websocket.send_json({
                            "type": "model_switch_complete",
                            "model_id": model_id,
                            "message": "Model selected (within same group)",
                            "model_info": model_info,
                            "generation_config": model_info.get("generation_config", {})
                        })
                    
                except Exception as e:
                    logger.error(f"Model switch error: {e}")
                    await websocket.send_json({
                        "type": "model_switch_error",
                        "model_id": model_id,
                        "error": str(e)
                    })
                    
            elif data["type"] == "preload_group":
                # Preload all models in a group
                if not grouped_model_manager:
                    await websocket.send_json({"type": "error", "error": "Grouped model manager not initialized"})
                    continue
                    
                group_id = data.get("group_id")
                if not group_id:
                    await websocket.send_json({"type": "error", "error": "No group_id provided"})
                    continue
                    
                try:
                    await grouped_model_manager.preload_group(group_id)
                    await websocket.send_json({
                        "type": "group_preload_complete",
                        "group_id": group_id,
                        "message": f"Successfully preloaded all models in group {group_id}"
                    })
                except Exception as e:
                    logger.error(f"Group preload failed: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Failed to preload group: {str(e)}"
                    })
                    
            elif data["type"] == "unload_group":
                # Unload a group from memory
                if not grouped_model_manager:
                    await websocket.send_json({"type": "error", "error": "Grouped model manager not initialized"})
                    continue
                    
                group_id = data.get("group_id")
                if not group_id:
                    await websocket.send_json({"type": "error", "error": "No group_id provided"})
                    continue
                    
                try:
                    # Check if this is the current group
                    if grouped_model_manager.current_group_id == group_id:
                        await websocket.send_json({
                            "type": "group_unload_error",
                            "group_id": group_id,
                            "error": "Cannot unload the currently active group"
                        })
                        continue
                    
                    # Unload the group
                    result = await grouped_model_manager.unload_group(group_id)
                    
                    # Get group name for better message
                    group_config = grouped_model_manager.model_groups.get(group_id)
                    group_name = group_config.group_name if group_config else group_id
                    
                    await websocket.send_json({
                        "type": "group_unload_complete",
                        "group_id": group_id,
                        "group_name": group_name,
                        "message": f"Successfully unloaded {group_name} from memory"
                    })
                    
                    # Broadcast to all clients that the group was unloaded
                    await manager.broadcast({
                        "type": "group_unload_complete",
                        "group_id": group_id,
                        "group_name": group_name
                    })
                    
                except Exception as e:
                    logger.error(f"Group unload failed: {e}")
                    await websocket.send_json({
                        "type": "group_unload_error",
                        "group_id": group_id,
                        "error": str(e)
                    })
                    
            elif data["type"] == "get_model_info":
                # Get model info for a specific model
                model_id = data.get("model_id")
                
                if grouped_model_manager and model_id:
                    # Get info for specific model from grouped manager
                    model_info = grouped_model_manager.get_model_info(model_id)
                    await websocket.send_json({
                        "type": "model_info",
                        "model_id": model_info.get("model_id"),
                        "display_name": model_info.get("display_name"),
                        "layer": model_info.get("layer"),
                        "auto_batch_size_max": model_info.get("auto_batch_size_max"),
                        "generation_config": model_info.get("generation_config", {}),
                        "loaded": model_info.get("is_loaded", False)
                    })
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
        "model_loaded": grouped_model_manager is not None and grouped_model_manager.is_model_loaded(),
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
    global grouped_model_manager, inference_service

    try:
        if inference_service:
            # Stop inference processing
            await inference_service.stop_processing()
            
            # Clear the queue by re-initializing it
            inference_service.queue.queue = asyncio.Queue()
            inference_service.queue.active_requests.clear()
            logger.info("Cleared inference queue")

        # Use grouped model manager
        manager_to_use = grouped_model_manager

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
