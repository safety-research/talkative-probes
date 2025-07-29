"""Model manager for handling multiple LensAnalyzers with GPU memory management"""

import asyncio
import torch
import gc
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import sys
import os

class ModelLoadError(Exception):
    """Raised when model fails to load"""
    pass

# Add parent directory to path to import lens module
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)

from lens.analysis.analyzer_class import LensAnalyzer
from .model_registry import ModelConfig, get_model_config
from .config import Settings

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages multiple LensAnalyzer models with GPU memory management"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.current_model_id: Optional[str] = None
        self.current_analyzer: Optional[LensAnalyzer] = None
        self.current_config: Optional[ModelConfig] = None
        self.loading_lock = asyncio.Lock()
        self.is_switching = False
        self.switch_start_time: Optional[datetime] = None
        self.queued_requests: List[asyncio.Future] = []  # Queue for requests during model switch
        self.websocket_manager = None  # Will be set by the application
        self.default_model_id = "qwen2.5-14b-wildchat"  # Can be overridden
        
    async def initialize_default_model(self, model_id: Optional[str] = None):
        """Initialize with a default model"""
        if model_id is None:
            model_id = self.default_model_id
        await self.switch_model(model_id)
        
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if not self.current_model_id or not self.current_config:
            return {"status": "no_model_loaded"}
            
        return {
            "model_id": self.current_model_id,
            "display_name": self.current_config.display_name,
            "description": self.current_config.description,
            "batch_size": self.current_config.batch_size,
            "auto_batch_size_max": self.current_config.auto_batch_size_max,
            "is_switching": self.is_switching,
            "switch_start_time": self.switch_start_time.isoformat() if self.switch_start_time else None,
        }
    
    def get_current_batch_size(self) -> int:
        """Get the batch size for the current model"""
        if self.current_config:
            return self.current_config.batch_size
        return self.settings.batch_size  # Fallback to default
        
    async def get_analyzer(self) -> Optional[LensAnalyzer]:
        """Get the current analyzer instance, waiting if model is switching"""
        if self.is_switching:
            # Create a future to wait for the switch to complete
            future = asyncio.Future()
            self.queued_requests.append(future)
            await future  # Wait until the switch is complete
            
        return self.current_analyzer
    
    def get_analyzer_sync(self) -> Optional[LensAnalyzer]:
        """Get the current analyzer instance synchronously (for compatibility)"""
        if self.is_switching:
            raise RuntimeError("Model is currently being switched. Please wait.")
        return self.current_analyzer
        
    async def switch_model(self, model_id: str) -> Dict[str, Any]:
        """Switch to a different model"""
        async with self.loading_lock:
            if self.current_model_id == model_id and self.current_analyzer is not None:
                return {
                    "status": "already_loaded",
                    "model_id": model_id,
                    "message": f"Model {model_id} is already loaded"
                }
                
            # Validate model ID
            config = get_model_config(model_id)
            if not config:
                raise ValueError(f"Unknown model ID: {model_id}")
                
            try:
                self.is_switching = True
                self.switch_start_time = datetime.now()
                
                # Notify all connected clients that switching has started
                await self._broadcast_switch_status("starting", model_id)
                
                # Clear current model from GPU memory
                if self.current_analyzer is not None:
                    logger.info(f"Unloading current model: {self.current_model_id}")
                    await self._unload_current_model()
                    
                # Load new model
                logger.info(f"Loading new model: {model_id}")
                await self._load_model(config)
                
                self.current_model_id = model_id
                self.is_switching = False
                self.switch_start_time = None
                
                # Resume any queued requests
                for future in self.queued_requests:
                    if not future.done():
                        future.set_result(None)
                self.queued_requests.clear()
                
                # Notify all connected clients that switching is complete
                await self._broadcast_switch_status("completed", model_id)
                
                return {
                    "status": "success",
                    "model_id": model_id,
                    "message": f"Successfully switched to {config.display_name}"
                }
                
            except Exception as e:
                self.is_switching = False
                self.switch_start_time = None
                
                # Cancel any queued requests
                for future in self.queued_requests:
                    if not future.done():
                        future.set_exception(RuntimeError(f"Model switch failed: {str(e)}"))
                self.queued_requests.clear()
                
                await self._broadcast_switch_status("failed", model_id, str(e))
                logger.error(f"Failed to switch to model {model_id}: {e}")
                raise
                
    async def _unload_current_model(self):
        """Unload current model and free GPU memory"""
        if self.current_analyzer is None:
            return
            
        # Delete the analyzer
        del self.current_analyzer
        self.current_analyzer = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Wait a bit for memory to be freed
        await asyncio.sleep(0.5)
        
    async def _load_model(self, config: ModelConfig):
        """Load a new model based on configuration"""
        # Create analyzer with model-specific settings in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        def create_analyzer():
            return LensAnalyzer(
                config.checkpoint_path,
                device=self.settings.device,
                batch_size=config.batch_size,
                use_bf16=config.use_bf16,
                strict_load=config.strict_load,
                comparison_tl_checkpoint=config.comparison_tl_checkpoint,
                do_not_load_weights=self.settings.do_not_load_weights,
                make_xl=config.make_xl,
                t_text=config.t_text,
                no_orig=config.no_orig,
                different_activations_orig=config.different_activations_model,
                initialise_on_cpu=self.settings.initialise_on_cpu,
            )
        
        # Run the blocking initialization in a thread
        self.current_analyzer = await loop.run_in_executor(None, create_analyzer)
        
        # Store current model config for reference
        self.current_config = config
        
    async def _broadcast_switch_status(self, status: str, model_id: str, error: Optional[str] = None):
        """Broadcast model switch status to all connected clients"""
        message = {
            "type": "model_switch_status",
            "status": status,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        if error:
            message["error"] = error
            
        # Broadcast via WebSocket manager if available
        if self.websocket_manager:
            await self.websocket_manager.broadcast(message)
        else:
            # Fallback to logging if no WebSocket manager
            logger.info(f"Model switch status: {message}")
            
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self.current_analyzer is not None
        
    async def ensure_model_loaded(self):
        """Ensure a model is loaded, loading default if necessary"""
        if not self.is_model_loaded():
            await self.initialize_default_model()
            
    async def clear_current_model(self):
        """Clear the current model from memory"""
        await self._unload_current_model()
        self.current_model_id = None
        self.current_config = None
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
            
        return {
            "gpu_available": True,
            "allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_memory_gb": torch.cuda.memory_reserved() / 1e9,
            "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        }