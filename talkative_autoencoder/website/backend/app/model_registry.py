"""Model registry for managing multiple LensAnalyzer configurations"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import os

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
    
    # Memory requirements (in GB)
    estimated_gpu_memory: float = Field(default=20.0)
    
    # Additional model-specific parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict)

# Define available models
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "qwen2.5-14b-wildchat": ModelConfig(
        name="qwen2.5-14b-wildchat",
        display_name="Qwen 2.5 14B (WildChat)",
        checkpoint_path="/workspace/kitf/talkative-probes/talkative_autoencoder/outputs/checkpoints/qwen2_5_WCHAT_14b_frozen_nopostfix_QQ1IAW1S_Qwen2.5-14B-Instruct_L36_e36_frozen_lr1e-3_t8_4ep_resume_0729_073454_NO_ENC_PROJ8_higherELR_PT_TO_CHAT_OTF_dist4_slurm6708",
        model_name="Qwen/Qwen2.5-14B-Instruct",
        different_activations_model="Qwen/Qwen2.5-14B-Instruct",
        batch_size=32,
        auto_batch_size_max=1024,
        layer=36,  # From L36 in checkpoint path
        description="Qwen 2.5 14B trained on WildChat dataset",
        estimated_gpu_memory=30.0,
        use_bf16=True,
        no_orig=True,
        no_kl=True,
        comparison_tl_checkpoint=False,
    ),
    "gemma2-9b-wildchat": ModelConfig(
        name="gemma2-9b-wildchat",
        display_name="Gemma 2 9B (WildChat)",
        checkpoint_path="/workspace/kitf/talkative-probes/talkative_autoencoder/outputs/checkpoints/gemma2_WILDCHAT_9b_frozen_nopostfix_GG29AW1S_gemma-2-9b-it_L30_e30_frozen_lr1e-4_t8_2ep_resume_0723_124516_NO_ENC_PROJ8_CHAT_DIR_OTF_dist2_slurm6696",
        model_name="google/gemma-2-9b-it",
        different_activations_model="google/gemma-2-9b-it",
        batch_size=48,  # Smaller model can handle larger batches
        auto_batch_size_max=2048,
        layer=30,  # From L30 in checkpoint path
        description="Gemma 2 9B trained on WildChat dataset",
        estimated_gpu_memory=20.0,
        use_bf16=True,
        no_orig=True,
        no_kl=True,
        comparison_tl_checkpoint=False,
    ),
    "gemma3-27b-chat": ModelConfig(
        name="gemma3-27b-chat",
        display_name="Gemma 3 27B (Chat)",
        checkpoint_path="/workspace/kitf/talkative-probes/talkative_autoencoder/outputs/checkpoints/gemma3_CHAT_27b_frozen_nopostfix_GG32PAW1S_gemma-3-27b-it_L45_e45_frozen_lr3e-4_t8_4ep_resume_0724_191127_frozenenc_add_patch5_suffix1p0enc_NO_PROJ_IT_E_D_OTF_dist8",
        model_name="google/gemma-3-27b-it", 
        different_activations_model="google/gemma-3-27b-it",
        batch_size=16,  # Large model needs smaller batches
        auto_batch_size_max=512,  # From .env AUTO_BATCH_SIZE_MAX=512
        layer=45,  # From L45 in checkpoint path
        description="Gemma 3 27B trained on chat data",
        estimated_gpu_memory=50.0,
        use_bf16=True,
        no_orig=True,
        no_kl=True,
        comparison_tl_checkpoint=False,
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
            "layer": config.layer,
            "model_family": "Qwen" if "qwen" in model_id else "Gemma",
            "checkpoint_path": config.checkpoint_path,
            "checkpoint_filename": os.path.basename(config.checkpoint_path),
        }
        for model_id, config in MODEL_REGISTRY.items()
    }

def validate_model_id(model_id: str) -> bool:
    """Check if a model ID is valid"""
    return model_id in MODEL_REGISTRY