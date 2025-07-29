from pydantic import BaseModel, Field
from typing import List, Optional, Union
import os
import logging
logger = logging.getLogger(__name__)

class Settings(BaseModel):
    """Application settings with environment variable support"""
    
    # Model settings
    checkpoint_path: str = Field(default="/None")
    device: str = Field(default="cuda")
    batch_size: int = Field(default=32)
    use_bf16: bool = Field(default=True)
    
    # API settings
    allowed_origins: List[str] = Field(default=["http://localhost:3000"])
    api_key: Optional[str] = Field(default=None)
    max_queue_size: int = Field(default=100)
    max_text_length: int = Field(default=1000)
    
    # Redis settings (optional)
    redis_url: Optional[str] = Field(default=None)
    cache_ttl: int = Field(default=3600)
    
    # RunPod specific
    runpod_pod_id: Optional[str] = Field(default=None)
    runpod_api_key: Optional[str] = Field(default=None)
    
    # Additional settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    request_timeout: int = Field(default=300)
    load_in_8bit: bool = Field(default=False)
    model_name: str = Field(default="Qwen/Qwen2.5-14B-Instruct")
    rate_limit_per_minute: int = Field(default=60)
    auto_batch_size_max: int = Field(default=256)  # Maximum value for auto-calculated batch size
    max_cpu_cached_models: int = Field(default=3)  # Maximum models to keep in CPU memory
    
    # Tuned lens settings
    tuned_lens_dir: Optional[str] = Field(default=None)
    
    # LensAnalyzer settings
    comparison_tl_checkpoint: Union[str, bool] = Field(default=False)  # Changed default to False to match .env
    do_not_load_weights: bool = Field(default=False)
    make_xl: bool = Field(default=False)
    t_text: Optional[str] = Field(default=None)
    strict_load: bool = Field(default=False)
    no_orig: bool = Field(default=True)  # Changed default to True to match .env
    no_kl: bool = Field(default=True)  # Added from .env NO_KL=true
    different_activations_model: Optional[str] = Field(default=None)
    initialise_on_cpu: bool = Field(default=False)
    lazy_load_model: bool = Field(default=False)  # Whether to load model lazily on first request

def load_settings():
    """Load settings with environment variable support"""
    settings_dict = {}
    
    # Map environment variables to settings fields
    env_mapping = {
        "CHECKPOINT_PATH": "checkpoint_path",
        "DEVICE": "device",
        "BATCH_SIZE": "batch_size",
        "USE_BF16": "use_bf16",
        "ALLOWED_ORIGINS": "allowed_origins",
        "API_KEY": "api_key",
        "MAX_QUEUE_SIZE": "max_queue_size",
        "MAX_TEXT_LENGTH": "max_text_length",
        "REDIS_URL": "redis_url",
        "CACHE_TTL": "cache_ttl",
        "RUNPOD_POD_ID": "runpod_pod_id",
        "RUNPOD_API_KEY": "runpod_api_key",
        "HOST": "host",
        "PORT": "port",
        "REQUEST_TIMEOUT": "request_timeout",
        "LOAD_IN_8BIT": "load_in_8bit",
        "MODEL_NAME": "model_name",
        "RATE_LIMIT_PER_MINUTE": "rate_limit_per_minute",
        "AUTO_BATCH_SIZE_MAX": "auto_batch_size_max",
        "MAX_CPU_CACHED_MODELS": "max_cpu_cached_models",
        "TUNED_LENS_DIR": "tuned_lens_dir",
        "COMPARISON_TL_CHECKPOINT": "comparison_tl_checkpoint",
        "DO_NOT_LOAD_WEIGHTS": "do_not_load_weights",
        "MAKE_XL": "make_xl",
        "T_TEXT": "t_text",
        "STRICT_LOAD": "strict_load",
        "NO_ORIG": "no_orig",
        "NO_KL": "no_kl",
        "DIFFERENT_ACTIVATIONS_MODEL": "different_activations_model",
        "INITIALISE_ON_CPU": "initialise_on_cpu",
        "LAZY_LOAD_MODEL": "lazy_load_model"
    }
    
    # Load from environment
    for env_key, field_name in env_mapping.items():
        env_value = os.getenv(env_key)
        if env_value is not None:
            # Handle different types
            if field_name in ["batch_size", "port", "max_queue_size", "max_text_length", 
                              "cache_ttl", "request_timeout", "rate_limit_per_minute", "auto_batch_size_max", "max_cpu_cached_models"]:
                settings_dict[field_name] = int(env_value)
            elif field_name in ["use_bf16", "load_in_8bit", "do_not_load_weights", "make_xl", "strict_load", "no_orig", "no_kl", "initialise_on_cpu", "lazy_load_model"]:
                settings_dict[field_name] = env_value.lower() in ["true", "1", "yes"]
            elif field_name == "allowed_origins":
                settings_dict[field_name] = env_value.split(",")
            elif field_name == "comparison_tl_checkpoint":
                # Handle both boolean and string values
                if env_value.lower() in ["true", "false"]:
                    settings_dict[field_name] = env_value.lower() == "true"
                else:
                    settings_dict[field_name] = env_value
            else:
                settings_dict[field_name] = env_value
            logger.info(f"Loaded {field_name} from environment: {env_value}")
    
    return Settings(**settings_dict)

