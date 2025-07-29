"""Model registry for managing multiple LensAnalyzer configurations"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

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
    comparison_tl_checkpoint: str | bool = Field(default=True)
    tuned_lens_dir: Optional[str] = Field(default=None)
    description: str = Field(default="")
    # Model-specific settings
    make_xl: bool = Field(default=False)
    t_text: Optional[str] = Field(default=None)
    strict_load: bool = Field(default=False)
    no_orig: bool = Field(default=False)
    different_activations_model: Optional[str] = Field(default=None)
    
    # Memory requirements (in GB)
    estimated_gpu_memory: float = Field(default=20.0)
    
    # Additional model-specific parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict)

# Define available models
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "qwen2.5-14b": ModelConfig(
        name="qwen2.5-14b",
        display_name="Qwen 2.5 14B",
        checkpoint_path="/checkpoint/qwen2.5-14b/final",  # Update with actual path
        model_name="Qwen/Qwen2.5-14B-Instruct",
        batch_size=32,
        auto_batch_size_max=256,
        description="Qwen 2.5 14B model with talkative autoencoder",
        estimated_gpu_memory=30.0,
    ),
    "gemma-7b": ModelConfig(
        name="gemma-7b",
        display_name="Gemma 7B",
        checkpoint_path="/checkpoint/gemma-7b/final",  # Update with actual path
        model_name="google/gemma-7b-it",
        batch_size=64,  # Smaller model can handle larger batches
        auto_batch_size_max=512,
        description="Gemma 7B model with talkative autoencoder",
        estimated_gpu_memory=15.0,
    ),
    # Add more models as needed
}

def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model"""
    return MODEL_REGISTRY.get(model_id)

def list_available_models() -> Dict[str, Dict[str, Any]]:
    """List all available models with their info"""
    return {
        model_id: {
            "display_name": config.display_name,
            "description": config.description,
            "estimated_gpu_memory": config.estimated_gpu_memory,
            "batch_size": config.batch_size,
        }
        for model_id, config in MODEL_REGISTRY.items()
    }

def validate_model_id(model_id: str) -> bool:
    """Check if a model ID is valid"""
    return model_id in MODEL_REGISTRY