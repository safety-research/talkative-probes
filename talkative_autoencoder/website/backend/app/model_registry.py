"""Model registry for managing multiple LensAnalyzer configurations"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import logging
import os
import json
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    """Configuration for a specific model"""
    name: str
    display_name: str
    checkpoint_path: str
    batch_size: int = Field(default=32)
    auto_batch_size_max: int = Field(default=256)
    model_name: str  # HuggingFace model name
    use_bf16: bool = Field(default=True)
    comparison_tl_checkpoint: str | bool = Field(default=False)  # Default False like .env
    tuned_lens_dir: Optional[str] = Field(default=None)
    description: str = Field(default="")
    # Model-specific settings
    make_xl: bool = Field(default=False)
    t_text: Optional[str] = Field(default=None)
    strict_load: bool = Field(default=False)
    no_orig: bool = Field(default=True)  # Default True like .env
    no_kl: bool = Field(default=True)  # From .env NO_KL=true
    different_activations_model: Optional[str] = Field(default=None)
    do_not_load_weights: bool = Field(default=False)
    initialise_on_cpu: bool = Field(default=False)
    
    # Layer configuration
    layer: int = Field(default=30)  # Single layer to analyze
    
    # Memory requirements (e.g., "45GB")
    estimated_gpu_memory: str = Field(default="20GB")
    
    # Additional model-specific parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('checkpoint_path')
    def validate_checkpoint_path(cls, v):
        """Validate checkpoint path to prevent path traversal attacks"""
        if not v:
            return v
            
        # Convert to Path object for proper validation
        path = Path(v)
        
        # Check for path traversal attempts
        if '..' in path.parts:
            raise ValueError("Path traversal detected in checkpoint_path")
            
        # Ensure absolute path
        if not path.is_absolute():
            raise ValueError("Checkpoint path must be absolute")
            
        # Check if path exists (but allow non-existent files for flexibility)
        # This is a warning, not an error
        if not path.exists():
            logger.warning(f"Checkpoint path does not exist: {v}")
            
        return str(path)  # Return as string
        
    @validator('tuned_lens_dir')
    def validate_tuned_lens_dir(cls, v):
        """Validate tuned lens directory path"""
        if not v:
            return v
            
        path = Path(v)
        if '..' in path.parts:
            raise ValueError("Path traversal detected in tuned_lens_dir")
            
        if not path.is_absolute():
            raise ValueError("Tuned lens directory must be absolute path")
            
        return str(path)
        
    @validator('comparison_tl_checkpoint')
    def validate_comparison_tl_checkpoint(cls, v):
        """Validate comparison checkpoint path if it's a string path"""
        # Skip validation if it's a boolean
        if isinstance(v, bool) or not v:
            return v
            
        # If it's a string path, validate it
        if isinstance(v, str):
            path = Path(v)
            if '..' in path.parts:
                raise ValueError("Path traversal detected in comparison_tl_checkpoint")
                
            if not path.is_absolute():
                raise ValueError("Comparison checkpoint path must be absolute")
                
            return str(path)
            
        return v

class ModelRegistry:
    """Registry for managing model configurations from JSON file"""
    
    def __init__(self, models_json_path: Optional[str] = None):
        if models_json_path is None:
            # Default to models.json in the same directory
            models_json_path = Path(__file__).parent / "models.json"
        
        self.models_json_path = Path(models_json_path)
        self._models: Dict[str, ModelConfig] = {}
        self._lock = threading.RLock()  # Use RLock to allow recursive locking
        
        # Load models with lock
        with self._lock:
            self._load_models()
    
    def _load_models(self):
        """Load models from JSON file (internal method, caller must hold lock)"""
        try:
            if not self.models_json_path.exists():
                logger.warning(f"Models file not found: {self.models_json_path}")
                return
            
            with open(self.models_json_path, 'r') as f:
                data = json.load(f)
            
            new_models = {}
            for model_id, model_data in data.get("models", {}).items():
                try:
                    # Add the name field from the key
                    model_data["name"] = model_id
                    # Create ModelConfig instance
                    config = ModelConfig(**model_data)
                    new_models[model_id] = config
                except Exception as e:
                    logger.error(f"Failed to load model {model_id}: {e}")
            
            # Only update if loading was successful
            self._models = new_models
            logger.info(f"Loaded {len(self._models)} models from {self.models_json_path}")
            
        except Exception as e:
            logger.error(f"Failed to load models from {self.models_json_path}: {e}")
    
    def reload_models(self):
        """Reload models from JSON file"""
        logger.info("Reloading model registry...")
        with self._lock:
            self._load_models()
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        with self._lock:
            return self._models.get(model_id)
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their info"""
        with self._lock:
            # Create a copy to avoid issues if models change during iteration
            models_snapshot = dict(self._models)
        
        # Process outside the lock to avoid holding it too long
        return {
            model_id: {
                "display_name": config.display_name,
                "description": config.description,
                "estimated_gpu_memory": config.estimated_gpu_memory,
                "batch_size": config.batch_size,
                "layer": config.layer,
                "model_family": "Qwen" if "qwen" in model_id else "Gemma",
                "checkpoint_path": config.checkpoint_path,
                "checkpoint_filename": os.path.basename(config.checkpoint_path),
            }
            for model_id, config in models_snapshot.items()
        }
    
    def validate_model_id(self, model_id: str) -> bool:
        """Check if a model ID is valid"""
        with self._lock:
            return model_id in self._models

# Create global instance
model_registry = ModelRegistry()

# Export convenience functions for backward compatibility
def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model"""
    return model_registry.get_model_config(model_id)

def list_available_models() -> Dict[str, Dict[str, Any]]:
    """List all available models with their info"""
    return model_registry.list_available_models()

def validate_model_id(model_id: str) -> bool:
    """Check if a model ID is valid"""
    return model_registry.validate_model_id(model_id)