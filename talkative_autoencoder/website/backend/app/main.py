import os
from dotenv import load_dotenv

# Load .env files FIRST before any other imports that might use HF_TOKEN
# Load .env file from backend directory
dotenv_location = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_location)

# Also load parent .env file (talkative_autoencoder) for HF_TOKEN
parent_dotenv_location = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env')
load_dotenv(parent_dotenv_location, override=False)  # Don't override already set values

from fastapi import FastAPI, WebSocket, HTTPException, Request, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import torch
import logging
import asyncio
from typing import Optional
import threading
import time
from datetime import datetime

from .models import AnalyzeRequest, AnalyzeResponse, WebSocketMessage
from .config import load_settings
from .inference import ModelManager, ModelLoadError
from .websocket import manager

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
            "is_computing": False
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
                stats = {
                    "available": torch.cuda.is_available(),
                    "last_update": datetime.utcnow().isoformat()
                }
                
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
model_manager: Optional[ModelManager] = None
gpu_monitor = GPUStatsMonitor()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
settings = load_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_manager
    
    # Start GPU monitoring
    gpu_monitor.start()
    
    try:
        model_manager = ModelManager(
            checkpoint_path=settings.checkpoint_path
        )
        
        # Load model on startup if lazy loading is disabled
        if not settings.lazy_load_model:
            logger.info("Loading model on startup (lazy_load_model=false)")
            await model_manager.load_model()
        else:
            logger.info("Model will be loaded on first request (lazy_load_model=true)")
        
        await model_manager.start_processing()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Continue startup even if model loading fails
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Stop GPU monitoring
    gpu_monitor.stop()
    
    if model_manager and model_manager.processing_task:
        model_manager.processing_task.cancel()
        try:
            await model_manager.processing_task
        except asyncio.CancelledError:
            pass

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
        allowed_hosts=["*.proxy.runpod.net", "localhost", "127.0.0.1", "*"]  # Added wildcard for debugging
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

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify API key if configured"""
    # Check if we're in production (not localhost)
    is_production = os.getenv("ENVIRONMENT", "development") == "production"
    
    if not settings.api_key:
        if is_production:
            logger.error("API key not configured in production environment!")
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: API key required"
            )
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
    return {
        "name": "Consistency Lens API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/analyze", response_model=dict)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def analyze(
    request: AnalyzeRequest, 
    background_tasks: BackgroundTasks, 
    req: Request,
    authorized: bool = Depends(verify_api_key)
):
    """Queue analysis request and return request ID"""
    if not model_manager:
        raise HTTPException(500, "Model manager not initialized")
    
    # Validate text length
    if len(request.text) > settings.max_text_length:
        raise HTTPException(
            400, 
            f"Text too long. Maximum length is {settings.max_text_length} characters"
        )
    
    # Lazy load model on first request with lock to prevent race conditions
    async with model_manager._load_lock:
        if model_manager.analyzer is None:
            try:
                await model_manager.load_model()
            except ModelLoadError as e:
                raise HTTPException(500, f"Failed to load model: {str(e)}")
    
    request_id = await model_manager.queue.add_request(
        request.text, 
        request.options.dict()
    )
    
    # Monitor for disconnection
    async def check_disconnection():
        while request_id in model_manager.queue.active_requests:
            if await req.is_disconnected():
                model_manager.queue.active_requests[request_id]['status'] = 'cancelled'
                break
            await asyncio.sleep(1)
    
    background_tasks.add_task(check_disconnection)
    
    queue_position = model_manager.queue.queue.qsize()
    
    return {
        "request_id": request_id,
        "status": "queued",
        "queue_position": queue_position
    }

@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Get status of a request"""
    if not model_manager:
        raise HTTPException(500, "Model manager not initialized")
    
    status = model_manager.queue.get_status(request_id)
    if not status:
        raise HTTPException(404, "Request not found")
    
    # Convert datetime objects to strings for JSON serialization
    response = {
        "request_id": request_id,
        "status": status['status'],
        "created_at": status['created_at'].isoformat() if isinstance(status.get('created_at'), datetime) else None,
        "started_at": status.get('started_at').isoformat() if isinstance(status.get('started_at'), datetime) else None,
        "completed_at": status.get('completed_at').isoformat() if isinstance(status.get('completed_at'), datetime) else None,
        "processing_time": status.get('processing_time'),
        "result": status.get('result'),
        "error": status.get('error')
    }
    
    return response

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    # Log headers for debugging
    logger.info(f"WebSocket connection attempt from: {websocket.client}")
    logger.info(f"WebSocket headers: {websocket.headers}")
    
    await manager.connect(websocket)
    
    # Send model info - either loaded or pending
    if model_manager:
        if model_manager.analyzer:
            # Model is loaded
            model_info = {
                'type': 'model_info',
                'loaded': True,
                'checkpoint_path': model_manager.checkpoint_path,
                'model_name': model_manager.model_name or 'Unknown',
                'auto_batch_size_max': settings.auto_batch_size_max
            }
        else:
            # Model not loaded yet, but we know what will be loaded
            checkpoint_name = os.path.basename(model_manager.checkpoint_path)
            model_info = {
                'type': 'model_info',
                'loaded': False,
                'checkpoint_path': model_manager.checkpoint_path,
                'model_name': checkpoint_name,  # Show checkpoint name as preview
                'auto_batch_size_max': settings.auto_batch_size_max
            }
        await websocket.send_json(model_info)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data['type'] == 'analyze':
                if not model_manager:
                    await websocket.send_json({
                        'type': 'error',
                        'error': 'Model manager not initialized'
                    })
                    continue
                
            elif data['type'] == 'generate':
                if not model_manager:
                    await websocket.send_json({
                        'type': 'generation_error',
                        'error': 'Model manager not initialized'
                    })
                    continue
                
                # Add generation request to queue
                text = data.get('text', '')
                options = data.get('options', {})
                
                # Add websocket to request for real-time updates
                options['websocket'] = websocket
                
                # Add to queue with type 'generate'
                request_id = await model_manager.queue.add_request(text, options, request_type='generate')
                
                # Get current queue position
                queue_position = model_manager.queue.queue.qsize()
                
                await websocket.send_json({
                    'type': 'queued',
                    'request_id': request_id,
                    'queue_position': queue_position,
                    'context': 'generation'
                })
                
                # Store request in context with websocket
                request = model_manager.queue.active_requests[request_id]
                request['websocket'] = websocket
                
                continue
                
                # This code was orphaned - it should be part of the analyze block
                # Moving it inside the analyze condition
                
            if data['type'] == 'analyze':
                # Model loading is already checked at the top
                # Lazy load model if needed with lock to prevent race conditions
                async with model_manager._load_lock:
                    if model_manager.analyzer is None:
                        await websocket.send_json({
                            'type': 'status',
                            'message': 'Loading model checkpoint... This may take a minute on first load.'
                        })
                        try:
                            await model_manager.load_model()
                            await websocket.send_json({
                                'type': 'status',
                                'message': 'Model loaded successfully!'
                            })
                            # Send model info with the extracted name
                            await websocket.send_json({
                                'type': 'model_info',
                                'loaded': True,
                                'checkpoint_path': model_manager.checkpoint_path,
                                'model_name': model_manager.model_name or 'Unknown',
                                'auto_batch_size_max': settings.auto_batch_size_max
                            })
                        except ModelLoadError as e:
                            await websocket.send_json({
                                'type': 'error',
                                'error': f'Failed to load model: {str(e)}'
                            })
                            continue
                
                request_id = await model_manager.queue.add_request(
                    data['text'],
                    data.get('options', {})
                )
                
                # Attach websocket to request for notifications
                request = model_manager.queue.active_requests[request_id]
                request['websocket'] = websocket
                
                await websocket.send_json({
                    'type': 'queued',
                    'request_id': request_id,
                    'queue_position': model_manager.queue.queue.qsize(),
                    'context': 'analysis'
                })
            
            elif data['type'] == 'status':
                request_id = data.get('request_id')
                if request_id and model_manager:
                    status = model_manager.queue.get_status(request_id)
                    if status:
                        await websocket.send_json({
                            'type': 'status_update',
                            'request_id': request_id,
                            'status': status['status'],
                            'error': status.get('error')
                        })
            
            elif data['type'] == 'interrupt':
                request_id = data.get('request_id')
                context = data.get('context', 'analysis')
                
                if request_id and model_manager:
                    # Mark the request as cancelled
                    request = model_manager.queue.active_requests.get(request_id)
                    if request and request['status'] == 'processing':
                        request['status'] = 'cancelled'
                        request['error'] = 'Interrupted by user'
                        
                        logger.info(f"Interrupt requested for {context} request: {request_id}")
                        
                        # Send confirmation
                        await websocket.send_json({
                            'type': 'interrupted',
                            'request_id': request_id,
                            'context': context
                        })
                    else:
                        await websocket.send_json({
                            'type': 'error',
                            'error': f'Cannot interrupt request {request_id} - not currently processing'
                        })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "model_loaded": model_manager is not None and model_manager.analyzer is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "queue_size": model_manager.queue.queue.qsize() if model_manager else 0
    }
    
    # If GPU is not available, mark as unhealthy
    if not health_status["gpu_available"]:
        health_status["status"] = "degraded"
        health_status["warning"] = "No GPU available"
    
    return health_status

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring"""
    if not model_manager:
        return {"error": "Model not loaded"}
    
    active_requests = model_manager.queue.active_requests
    completed = [r for r in active_requests.values() if r['status'] == 'completed']
    failed = [r for r in active_requests.values() if r['status'] == 'failed']
    
    # Calculate average processing time
    processing_times = [r.get('processing_time', 0) for r in completed if 'processing_time' in r]
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    return {
        "total_requests": len(active_requests),
        "completed_requests": len(completed),
        "failed_requests": len(failed),
        "queue_size": model_manager.queue.queue.qsize(),
        "average_processing_time": avg_processing_time,
        "active_websocket_connections": len(manager.active_connections)
    }

@app.post("/reset")
async def reset_model():
    """Reset the model and clear memory"""
    global model_manager
    
    if model_manager:
        # Cancel processing task
        if model_manager.processing_task:
            model_manager.processing_task.cancel()
            try:
                await model_manager.processing_task
            except asyncio.CancelledError:
                pass
        
        # Clear the model from memory
        if model_manager.analyzer:
            model_manager.analyzer = None
        if model_manager.chat_tokenizer:
            model_manager.chat_tokenizer = None
        
        # Clear the queue
        model_manager.queue.active_requests.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model reset and memory cleared")
    
    return {"status": "reset", "message": "Model cleared from memory"}

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
        "peak_utilization": stats.get("peak_utilization", 0)
    }
