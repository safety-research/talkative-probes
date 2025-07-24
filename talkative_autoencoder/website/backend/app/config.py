from pydantic import BaseModel, Field
from typing import List, Optional
import os

class Settings(BaseModel):
    """Application settings with environment variable support"""
    
    # Model settings
    checkpoint_path: str = Field(default="/workspace/checkpoints/qwen2_5_WCHAT_14b_frozen_nopostfix.pt")
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
        "RATE_LIMIT_PER_MINUTE": "rate_limit_per_minute"
    }
    
    # Load from environment
    for env_key, field_name in env_mapping.items():
        env_value = os.getenv(env_key)
        if env_value is not None:
            # Handle different types
            if field_name in ["batch_size", "port", "max_queue_size", "max_text_length", 
                              "cache_ttl", "request_timeout", "rate_limit_per_minute"]:
                settings_dict[field_name] = int(env_value)
            elif field_name in ["use_bf16", "load_in_8bit"]:
                settings_dict[field_name] = env_value.lower() in ["true", "1", "yes"]
            elif field_name == "allowed_origins":
                settings_dict[field_name] = env_value.split(",")
            else:
                settings_dict[field_name] = env_value
    
    return Settings(**settings_dict)

# Global settings instance
settings = load_settings()