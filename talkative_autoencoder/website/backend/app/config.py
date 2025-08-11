from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Settings(BaseModel):
    """Application settings with environment variable support"""
    
    # Infrastructure settings (from env vars)
    device: str = Field(default="cuda")  # Kept for backward compatibility
    devices: List[str] = Field(default=["cuda:0"])  # List of available GPU devices
    num_workers_per_gpu: int = Field(default=1)  # Number of workers per GPU
    allowed_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:3001"])
    api_key: Optional[str] = Field(default=None)
    max_queue_size: int = Field(default=100)
    max_text_length: int = Field(default=20000)
    lazy_load_model: bool = Field(default=False)
    
    # Redis settings (optional)
    redis_url: Optional[str] = Field(default=None)
    cache_ttl: int = Field(default=3600)
    
    # RunPod specific
    runpod_pod_id: Optional[str] = Field(default=None)
    runpod_api_key: Optional[str] = Field(default=None)
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    request_timeout: int = Field(default=300)
    rate_limit_per_minute: int = Field(default=60)
    
    # Model management settings (defaults that can be overridden)
    max_cpu_cached_models: int = Field(default=5)
    default_group: Optional[str] = Field(default=None)  # Default model group to load
    
    # Tuned lens settings
    tuned_lens_dir: Optional[str] = Field(default=None)
    
    # Legacy settings (kept for backward compatibility)
    checkpoint_path: Optional[str] = Field(default=None)
    batch_size: Optional[int] = Field(default=None)
    use_bf16: Optional[bool] = Field(default=None)
    no_orig: Optional[bool] = Field(default=None)
    no_kl: Optional[bool] = Field(default=None)
    different_activations_model: Optional[str] = Field(default=None)
    auto_batch_size_max: Optional[int] = Field(default=None)
    
    # Unused legacy settings (kept to avoid breaking changes)
    comparison_tl_checkpoint: Union[str, bool] = Field(default=False)
    do_not_load_weights: bool = Field(default=False)
    make_xl: bool = Field(default=False)
    t_text: Optional[str] = Field(default=None)
    strict_load: bool = Field(default=False)
    initialise_on_cpu: bool = Field(default=False)
    load_in_8bit: bool = Field(default=False)
    model_name: str = Field(default="Qwen/Qwen2.5-14B-Instruct")
    
    @classmethod
    def get_model_config_with_overrides(cls, model_groups_config: Dict[str, Any], 
                                      model_id: str = None, group_id: str = None) -> Dict[str, Any]:
        """
        Get configuration for a specific model/group with environment variable overrides.
        
        Priority order:
        1. Model-specific env var (MODEL_<model_id>_<setting>)
        2. Group-specific env var (MODEL_GROUP_<group_id>_<setting>)
        3. Global env var (GLOBAL_<setting>)
        4. model_groups.json value
        5. Default value
        """
        # Start with settings from model_groups.json
        config = model_groups_config.get("settings", {}).copy()
        
        # Settings that can be overridden
        overridable_settings = [
            "use_bf16", "no_orig", "no_kl", "batch_size", "auto_batch_size_max",
            "comparison_tl_checkpoint", "preload_groups", "default_group", "max_cpu_models"
        ]
        
        # Apply overrides in priority order
        for setting in overridable_settings:
            # Check model-specific override
            if model_id:
                env_key = f"MODEL_{model_id}_{setting.upper()}"
                env_value = os.getenv(env_key)
                if env_value is not None:
                    parsed_value = cls._parse_env_value(setting, env_value)
                    if parsed_value is not None:
                        config[setting] = parsed_value
                        logger.info(f"Applied model-specific override: {env_key}={env_value}")
                        continue
            
            # Check group-specific override
            if group_id:
                env_key = f"MODEL_GROUP_{group_id}_{setting.upper()}"
                env_value = os.getenv(env_key)
                if env_value is not None:
                    parsed_value = cls._parse_env_value(setting, env_value)
                    if parsed_value is not None:
                        config[setting] = parsed_value
                        logger.info(f"Applied group-specific override: {env_key}={env_value}")
                        continue
            
            # Check global override
            env_key = f"GLOBAL_{setting.upper()}"
            env_value = os.getenv(env_key)
            if env_value is not None:
                parsed_value = cls._parse_env_value(setting, env_value)
                if parsed_value is not None:
                    config[setting] = parsed_value
                    logger.info(f"Applied global override: {env_key}={env_value}")
        
        return config
    
    @staticmethod
    def _parse_env_value(setting_name: str, env_value: str) -> Any:
        """Parse environment variable value based on setting type"""
        # Boolean settings
        if setting_name in ["use_bf16", "no_orig", "no_kl", "preload_groups"]:
            return env_value.lower() in ["true", "1", "yes"]
        
        # Integer settings
        if setting_name in ["batch_size", "auto_batch_size_max", "max_cpu_models"]:
            try:
                return int(env_value)
            except ValueError:
                logger.warning(f"Invalid integer value for {setting_name}: {env_value}, ignoring override")
                return None  # Return None to indicate parsing failed
        
        # String settings
        return env_value

def load_settings():
    """Load settings with environment variable support"""
    settings_dict = {}
    
    # Map environment variables to settings fields (infrastructure only)
    env_mapping = {
        "DEVICE": "device",
        "DEVICES": "devices",  # Comma-separated list of devices
        "NUM_WORKERS_PER_GPU": "num_workers_per_gpu",
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
        "RATE_LIMIT_PER_MINUTE": "rate_limit_per_minute",
        "MAX_CPU_CACHED_MODELS": "max_cpu_cached_models",
        "DEFAULT_GROUP": "default_group",
        "TUNED_LENS_DIR": "tuned_lens_dir",
        "LAZY_LOAD_MODEL": "lazy_load_model",
        
        # Legacy settings (for backward compatibility)
        "CHECKPOINT_PATH": "checkpoint_path",
        "BATCH_SIZE": "batch_size",
        "USE_BF16": "use_bf16",
        "NO_ORIG": "no_orig",
        "NO_KL": "no_kl",
        "DIFFERENT_ACTIVATIONS_MODEL": "different_activations_model",
        "AUTO_BATCH_SIZE_MAX": "auto_batch_size_max",
        "COMPARISON_TL_CHECKPOINT": "comparison_tl_checkpoint",
        "DO_NOT_LOAD_WEIGHTS": "do_not_load_weights",
        "MAKE_XL": "make_xl",
        "T_TEXT": "t_text",
        "STRICT_LOAD": "strict_load",
        "INITIALISE_ON_CPU": "initialise_on_cpu",
        "LOAD_IN_8BIT": "load_in_8bit",
        "MODEL_NAME": "model_name"
    }
    
    # Load from environment
    for env_key, field_name in env_mapping.items():
        env_value = os.getenv(env_key)
        if env_value is not None:
            # Handle different types
            if field_name in ["batch_size", "port", "max_queue_size", "max_text_length", 
                              "cache_ttl", "request_timeout", "rate_limit_per_minute", 
                              "auto_batch_size_max", "max_cpu_cached_models", "num_workers_per_gpu"]:
                settings_dict[field_name] = int(env_value)
            elif field_name in ["use_bf16", "load_in_8bit", "do_not_load_weights", 
                                "make_xl", "strict_load", "no_orig", "no_kl", 
                                "initialise_on_cpu", "lazy_load_model"]:
                settings_dict[field_name] = env_value.lower() in ["true", "1", "yes"]
            elif field_name in ["allowed_origins", "devices"]:
                settings_dict[field_name] = env_value.split(",")
            elif field_name == "comparison_tl_checkpoint":
                # Handle both boolean and string values
                if env_value.lower() in ["true", "false"]:
                    settings_dict[field_name] = env_value.lower() == "true"
                else:
                    settings_dict[field_name] = env_value
            else:
                settings_dict[field_name] = env_value
            
            # Only log infrastructure settings, not legacy ones
            if env_key not in ["CHECKPOINT_PATH", "BATCH_SIZE", "USE_BF16", "NO_ORIG", 
                               "NO_KL", "DIFFERENT_ACTIVATIONS_MODEL", "AUTO_BATCH_SIZE_MAX",
                               "COMPARISON_TL_CHECKPOINT", "DO_NOT_LOAD_WEIGHTS", "MAKE_XL",
                               "T_TEXT", "STRICT_LOAD", "INITIALISE_ON_CPU", "LOAD_IN_8BIT",
                               "MODEL_NAME"]:
                logger.info(f"Loaded {field_name} from environment: {env_value}")
    
    # Handle backward compatibility for single device
    if "devices" not in settings_dict and "device" in settings_dict:
        # Convert single device to list
        device = settings_dict["device"]
        if device == "cuda":
            settings_dict["devices"] = ["cuda:0"]
        elif device.startswith("cuda:"):
            settings_dict["devices"] = [device]
        else:
            settings_dict["devices"] = [device]
        logger.info(f"Converted single device '{device}' to devices list: {settings_dict['devices']}")
    
    return Settings(**settings_dict)