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
        
        # Model caching for fast switching
        self.model_cache: Dict[str, LensAnalyzer] = {}  # Stores model instances
        self.model_locations: Dict[str, str] = {}  # Tracks 'cuda', 'cpu', or 'unloaded'
        self.cache_order: List[str] = []  # Track usage order for potential LRU eviction
        self.max_cpu_models = getattr(settings, 'max_cpu_cached_models', 3)  # Maximum models to keep in CPU memory
        
    async def initialize_default_model(self, model_id: Optional[str] = None):
        """Initialize with a default model"""
        if model_id is None:
            model_id = self.default_model_id
        await self.switch_model(model_id)
        
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if not self.current_model_id or not self.current_config:
            return {"status": "no_model_loaded"}
            
        info = {
            "model_id": self.current_model_id,
            "display_name": self.current_config.display_name,
            "description": self.current_config.description,
            "batch_size": self.current_config.batch_size,
            "auto_batch_size_max": self.current_config.auto_batch_size_max,
            "layer": self.current_config.layer,
            "model_family": "Qwen" if "qwen" in self.current_model_id else "Gemma",
            "checkpoint_path": self.current_config.checkpoint_path,
            "checkpoint_filename": os.path.basename(self.current_config.checkpoint_path),
            "is_switching": self.is_switching,
            "switch_start_time": self.switch_start_time.isoformat() if self.switch_start_time else None,
            "cached_models": list(self.model_cache.keys()),
            "cache_info": {
                "total_cached": len(self.model_cache),
                "on_gpu": len([m for m, loc in self.model_locations.items() if loc == 'cuda']),
                "on_cpu": len([m for m, loc in self.model_locations.items() if loc == 'cpu']),
            }
        }
        return info
    
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
        """Move current model from GPU to CPU instead of deleting"""
        if self.current_analyzer is None or self.current_model_id is None:
            return
            
        logger.info(f"Moving model {self.current_model_id} from GPU to CPU")
        
        # Move model to CPU
        self.current_analyzer.to('cpu')
        self.model_locations[self.current_model_id] = 'cpu'
        
        # Clear reference but keep in cache
        self.current_analyzer = None
        
        # Clear CUDA cache to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # No delay needed - torch.cuda.synchronize() already ensures completion
        
    async def _load_model(self, config: ModelConfig):
        """Load a model, either from cache (CPU) or disk"""
        model_id = config.name
        
        # Check if model is already in cache
        if model_id in self.model_cache and self.model_locations.get(model_id) == 'cpu':
            logger.info(f"Moving cached model {model_id} from CPU to GPU")
            
            # Get cached model and move to GPU
            self.current_analyzer = self.model_cache[model_id]
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.current_analyzer.to(self.settings.device))
            
            self.model_locations[model_id] = 'cuda'
            
            # Update cache order for LRU
            if model_id in self.cache_order:
                self.cache_order.remove(model_id)
            self.cache_order.append(model_id)
            
        else:
            logger.info(f"Loading new model {model_id} from disk to GPU")
            
            # Check if we need to evict a model from CPU cache
            await self._manage_cache_size()
            
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
            
            # Add to cache
            self.model_cache[model_id] = self.current_analyzer
            self.model_locations[model_id] = 'cuda'
            self.cache_order.append(model_id)
        
        # Store current model config for reference
        self.current_config = config
        
        # Update settings to reflect the loaded model's configuration
        # This ensures compatibility with code that reads from settings
        self.settings.batch_size = config.batch_size
        self.settings.auto_batch_size_max = config.auto_batch_size_max
        self.settings.model_name = config.model_name
        self.settings.use_bf16 = config.use_bf16
        self.settings.no_orig = config.no_orig
        self.settings.comparison_tl_checkpoint = config.comparison_tl_checkpoint
        self.settings.different_activations_model = config.different_activations_model
        
    async def _manage_cache_size(self):
        """Manage CPU cache size by evicting least recently used models"""
        cpu_models = [m for m, loc in self.model_locations.items() if loc == 'cpu']
        
        if len(cpu_models) >= self.max_cpu_models:
            # Find the least recently used model
            lru_model = None
            for model_id in self.cache_order:
                if self.model_locations.get(model_id) == 'cpu':
                    lru_model = model_id
                    break
                    
            if lru_model:
                logger.info(f"Evicting model {lru_model} from CPU cache to free memory")
                # Delete the model
                del self.model_cache[lru_model]
                del self.model_locations[lru_model]
                self.cache_order.remove(lru_model)
                
                # Force garbage collection
                gc.collect()
        
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
        
    async def get_cache_status(self) -> Dict[str, Any]:
        """Get the status of the model cache (thread-safe)"""
        # Create a snapshot under lock to ensure consistency
        async with self.loading_lock:
            cached_models = list(self.model_cache.keys())
            locations = self.model_locations.copy()
            order = self.cache_order.copy()
            
        return {
            "cached_models": cached_models,
            "model_locations": locations,
            "cache_order": order,
            "max_cpu_models": self.max_cpu_models,
            "cpu_models_count": len([m for m, loc in locations.items() if loc == 'cpu']),
        }
            
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