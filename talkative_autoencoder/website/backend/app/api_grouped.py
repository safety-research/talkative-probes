"""API endpoints for grouped model management"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging

from .model_manager_grouped import GroupedModelManager

logger = logging.getLogger(__name__)

# Create router for grouped model endpoints
router = APIRouter(prefix="/api/v2", tags=["grouped_models"])

# Request models
class SwitchModelRequest(BaseModel):
    """Request body for switching models"""
    group_id: Optional[str] = None  # Optional - can be inferred from model_id
    model_id: str

class PreloadGroupRequest(BaseModel):
    """Request body for preloading a group"""
    group_id: str

# This will be injected by the main app
grouped_model_manager: Optional[GroupedModelManager] = None

def get_model_manager() -> GroupedModelManager:
    """Dependency to get the model manager"""
    if not grouped_model_manager:
        raise HTTPException(503, "Model manager not initialized")
    return grouped_model_manager

@router.get("/models")
async def list_model_groups(manager: GroupedModelManager = Depends(get_model_manager)):
    """List all model groups and their models"""
    groups = manager.get_model_list()
    current_info = manager.get_current_model_info()
    
    return {
        "groups": groups,
        "current_model": current_info.get("model_id") if "model_id" in current_info else None,
        "current_group": current_info.get("group_id") if "group_id" in current_info else None,
        "is_switching": current_info.get("is_switching", False),
        "model_status": current_info
    }

@router.post("/models/switch")
async def switch_model(
    request: SwitchModelRequest,
    manager: GroupedModelManager = Depends(get_model_manager)
):
    """Switch to a different model, potentially in a different group"""
    try:
        # If group_id not provided, it will be inferred from model_id
        result = await manager.switch_model(request.model_id)
        
        return {
            "status": "success",
            **result,
            "model_info": manager.get_current_model_info()
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        raise HTTPException(500, f"Model switch failed: {str(e)}")

@router.post("/groups/{group_id}/preload")
async def preload_group(
    group_id: str,
    manager: GroupedModelManager = Depends(get_model_manager)
):
    """Preload all models in a group for fast switching"""
    try:
        await manager.preload_group(group_id)
        return {
            "status": "success",
            "message": f"Preloaded all models in group {group_id}",
            "group_id": group_id
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Group preload failed: {e}")
        raise HTTPException(500, f"Group preload failed: {str(e)}")

@router.get("/models/memory")
async def get_memory_status(manager: GroupedModelManager = Depends(get_model_manager)):
    """Get detailed memory status including group locations"""
    memory_usage = manager.get_memory_usage()
    current_info = manager.get_current_model_info()
    
    return {
        "memory": memory_usage,
        "cache_info": current_info.get("cache_info", {}),
        "current_model": current_info.get("model_id"),
        "current_group": current_info.get("group_id"),
    }

@router.get("/models/{model_id}/info")
async def get_model_info(
    model_id: str,
    manager: GroupedModelManager = Depends(get_model_manager)
):
    """Get information about a specific model"""
    # Find the model in groups
    group_id = manager.model_to_group.get(model_id)
    if not group_id:
        raise HTTPException(404, f"Model {model_id} not found")
    
    group_config = manager.model_groups[group_id]
    model_config = next((m for m in group_config.models if m["id"] == model_id), None)
    
    if not model_config:
        raise HTTPException(404, f"Model {model_id} not found in group {group_id}")
    
    return {
        "model_id": model_id,
        "group_id": group_id,
        "group_name": group_config.group_name,
        "model_name": model_config["name"],
        "description": model_config.get("description", ""),
        "layer": model_config.get("layer", 30),
        "batch_size": model_config.get("batch_size", 32),
        "base_model": group_config.base_model_path,
        "is_loaded": model_id in manager.lens_cache,
        "is_current": model_id == manager.current_model_id
    }

# Backward compatibility endpoints
@router.get("/models/legacy")
async def list_models_legacy(manager: GroupedModelManager = Depends(get_model_manager)):
    """List models in legacy flat format for backward compatibility"""
    groups = manager.get_model_list()
    
    # Flatten groups into a single list
    models = {}
    for group in groups:
        for model in group["models"]:
            models[model["id"]] = {
                "display_name": f"{group['group_name']} - {model['name']}",
                "description": model["description"],
                "estimated_gpu_memory": manager.config_settings.get("estimated_gpu_memory", {}).get(group["group_id"], "20GB"),
                "batch_size": model.get("batch_size", 32),
                "layer": model["layer"],
                "model_family": "Gemma" if "gemma" in model["id"] else "Qwen",
                "checkpoint_path": f"Group: {group['group_id']}",
                "checkpoint_filename": model["name"],
            }
    
    current_info = manager.get_current_model_info()
    
    return {
        "models": models,
        "current_model": current_info.get("model_id") if "model_id" in current_info else None,
        "is_switching": current_info.get("is_switching", False),
        "model_status": current_info
    }

def setup_grouped_model_manager(app, settings):
    """Setup function to be called from main.py"""
    global grouped_model_manager
    
    # Create the grouped model manager
    grouped_model_manager = GroupedModelManager(settings)
    
    # Include the router
    app.include_router(router)
    
    return grouped_model_manager