"""
Unified API for Model Management
================================

This module provides a clean API that naturally handles both single and
multi-GPU deployments without separate code paths.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging

from .model_manager import UnifiedModelManager
from .legacy.model_registry import list_available_models as get_all_models_dict, get_model_config as get_model_info_obj

logger = logging.getLogger(__name__)

# Helper functions to adapt legacy registry
def get_all_models(public_only: bool = False) -> List[Dict[str, Any]]:
    """Get all models as a list"""
    models_dict = get_all_models_dict()
    models_list = []
    
    for model_id, model_data in models_dict.items():
        # Filter out non-public models if requested
        if public_only and (model_data.get("backend_only", False) or model_data.get("visible") is False):
            continue
            
        model_info = {
            "id": model_id,
            "name": model_data.get("name", model_id),
            "description": model_data.get("description", ""),
            "dataset": model_data.get("dataset", "unknown"),
            "architecture": model_data.get("architecture", "unknown"),
            "memory_estimate": model_data.get("memory_estimate", 5.0)
        }
        models_list.append(model_info)
    
    return models_list

def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model info as a dict"""
    config = get_model_info_obj(model_id)
    if not config:
        return None
    
    return {
        "id": model_id,
        "name": config.name,
        "description": getattr(config, 'description', ''),
        "dataset": config.dataset,
        "architecture": config.architecture,
        "memory_estimate": getattr(config, 'memory_estimate', 5.0),
        "layer": config.layer,
        "auto_batch_size_max": getattr(config, 'auto_batch_size_max', 16),
        "generation_config": getattr(config, 'generation_config', {})
    }

# Create router
router = APIRouter(prefix="/api/v2", tags=["models"])

# Request/Response models

class LoadModelRequest(BaseModel):
    """Request to load a model"""
    model_id: str
    device_id: Optional[int] = Field(None, description="Specific device to use (optional)")

class LoadGroupRequest(BaseModel):
    """Request to load a model group"""
    group_id: str
    device_id: Optional[int] = Field(None, description="Specific device to use (optional)")

class ProcessPromptRequest(BaseModel):
    """Request to process a prompt"""
    model_id: str
    prompt: str
    num_predictions: int = Field(3, ge=1, le=10)
    device_id: Optional[int] = Field(None, description="Specific device to use (optional)")

class ModelLocation(BaseModel):
    """Location of a loaded model"""
    device_id: int
    group_id: Optional[str]

class ModelInfo(BaseModel):
    """Information about a model"""
    id: str
    name: str
    description: Optional[str]
    dataset: str
    architecture: str
    memory_estimate: float
    locations: List[ModelLocation] = Field(default_factory=list)

class GroupInfo(BaseModel):
    """Information about a model group"""
    id: str
    name: str
    models: List[str]
    memory_estimate_gb: float

class DeviceInfo(BaseModel):
    """Information about a GPU device"""
    device_id: int
    current_group: Optional[Dict[str, str]]
    loaded_models: List[str]
    is_switching: bool
    memory: Dict[str, float]

class SystemState(BaseModel):
    """Complete system state"""
    num_devices: int
    devices: List[DeviceInfo]
    groups: List[GroupInfo]

# Dependency injection
model_manager: Optional[UnifiedModelManager] = None

def get_model_manager() -> UnifiedModelManager:
    """Dependency to get the model manager"""
    if not model_manager:
        raise HTTPException(503, "Model manager not initialized")
    return model_manager

# Endpoints

@router.get("/system/state", response_model=SystemState)
async def get_system_state(manager: UnifiedModelManager = Depends(get_model_manager)):
    """Get the complete system state including all devices and loaded models"""
    return manager.get_system_state()

@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    request: Request,
    manager: UnifiedModelManager = Depends(get_model_manager)
):
    """List all available models with their current locations"""
    # Check origin for public/private filtering
    origin_header = request.headers.get('origin', '')
    host_header = request.headers.get('host', '')
    public_only = ("kitft.com" in origin_header) or ("kitft.com" in host_header)
    
    all_models = get_all_models(public_only=public_only)
    
    # Enhance with location information
    models_info = []
    for model_data in all_models:
        model_id = model_data["id"]
        
        # Get location info
        location_info = manager.get_model_location(model_id)
        locations = []
        if location_info:
            locations = [
                ModelLocation(**loc) for loc in location_info
            ]
        
        model_info = ModelInfo(
            id=model_id,
            name=model_data["name"],
            description=model_data.get("description"),
            dataset=model_data["dataset"],
            architecture=model_data["architecture"],
            memory_estimate=model_data.get("memory_estimate", 5.0),
            locations=locations
        )
        models_info.append(model_info)
    
    return models_info

@router.get("/groups", response_model=List[GroupInfo])
async def list_groups(manager: UnifiedModelManager = Depends(get_model_manager)):
    """List all model groups"""
    return [
        GroupInfo(
            id=group.id,
            name=group.name,
            models=[m.id for m in group.models],
            memory_estimate_gb=group.memory_estimate
        )
        for group in manager.groups.values()
    ]

@router.post("/models/load")
async def load_model(
    request: LoadModelRequest,
    manager: UnifiedModelManager = Depends(get_model_manager)
):
    """
    Load a model on the best available device or a specific device.
    
    For single-GPU systems, device_id can be omitted.
    For multi-GPU systems, the system will automatically select the best device
    unless a specific device_id is provided.
    """
    try:
        result = await manager.load_model(
            model_id=request.model_id,
            device_id=request.device_id
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(500, f"Failed to load model: {str(e)}")

@router.post("/groups/load")
async def load_group(
    request: LoadGroupRequest,
    manager: UnifiedModelManager = Depends(get_model_manager)
):
    """Load an entire group of models on a device"""
    try:
        result = await manager.load_group(
            group_id=request.group_id,
            device_id=request.device_id
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Failed to load group: {e}")
        raise HTTPException(500, f"Failed to load group: {str(e)}")

@router.post("/models/process")
async def process_prompt(
    request: ProcessPromptRequest,
    manager: UnifiedModelManager = Depends(get_model_manager)
):
    """
    Process a prompt with a specific model.
    
    The model will be automatically loaded if not already available.
    For multi-GPU systems, the best device will be selected unless specified.
    """
    try:
        result = await manager.process_prompt(
            model_id=request.model_id,
            prompt=request.prompt,
            num_predictions=request.num_predictions,
            device_id=request.device_id
        )
        return result
    except Exception as e:
        logger.error(f"Failed to process prompt: {e}")
        raise HTTPException(500, f"Failed to process prompt: {str(e)}")

@router.delete("/devices/{device_id}/clear")
async def clear_device(
    device_id: int,
    manager: UnifiedModelManager = Depends(get_model_manager)
):
    """Clear all models from a specific device"""
    if device_id >= manager.num_devices:
        raise HTTPException(404, f"Device {device_id} not found")
    
    try:
        # Convert int to string format for device_id
        device_str = f"cuda:{device_id}"
        result = await manager.clear_device(device_str)
        return result
    except Exception as e:
        logger.error(f"Failed to clear device: {e}")
        raise HTTPException(500, f"Failed to clear device: {str(e)}")

@router.get("/devices/{device_id}/memory")
async def get_device_memory(
    device_id: int,
    manager: UnifiedModelManager = Depends(get_model_manager)
):
    """Get memory usage for a specific device"""
    if device_id >= manager.num_devices:
        raise HTTPException(404, f"Device {device_id} not found")
    
    # Convert int to string format
    device_str = f"cuda:{device_id}"
    device_state = manager.device_states[device_str]
    
    import torch
    torch.cuda.set_device(device_id)
    free, total = torch.cuda.mem_get_info(device_id)
    
    return {
        "device_id": device_id,
        "total_gb": total / (1024**3),
        "free_gb": free / (1024**3),
        "allocated_gb": device_state.memory_allocated,
        "reserved_gb": device_state.memory_reserved,
        "models_loaded": len(device_state.loaded_models)
    }

# WebSocket endpoint for real-time updates

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    manager: UnifiedModelManager = Depends(get_model_manager)
):
    """WebSocket endpoint for real-time system state updates"""
    await websocket.accept()
    
    # Add to manager's connections
    await manager.add_websocket(websocket)
    
    try:
        # Keep connection alive and handle any incoming messages
        while True:
            # We could handle commands here if needed
            data = await websocket.receive_text()
            
            # For now, just echo back
            await websocket.send_json({
                "type": "echo",
                "data": data
            })
            
    except WebSocketDisconnect:
        await manager.remove_websocket(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.remove_websocket(websocket)

# Backward compatibility endpoints
# These map old single-GPU API calls to the new unified system

@router.get("/current", deprecated=True)
async def get_current_model_compat(manager: UnifiedModelManager = Depends(get_model_manager)):
    """
    DEPRECATED: Get the 'current' model (for backward compatibility).
    
    Returns the first model loaded on the first device.
    """
    if manager.num_devices == 0:
        return {"status": "no_devices"}
    
    device_state = manager.device_states[0]
    if not device_state.loaded_models:
        return {"status": "no_model_loaded"}
    
    # Return info about the first loaded model
    model_id = list(device_state.loaded_models.keys())[0]
    model_info = get_model_info(model_id)
    
    return {
        "model_id": model_id,
        "device_id": 0,
        **model_info
    }

@router.post("/switch", deprecated=True)
async def switch_model_compat(
    request: LoadModelRequest,
    manager: UnifiedModelManager = Depends(get_model_manager)
):
    """
    DEPRECATED: Switch to a different model (for backward compatibility).
    
    Maps to loading a model on device 0.
    """
    return await load_model(
        LoadModelRequest(model_id=request.model_id, device_id=0),
        manager
    )

# Setup function

def setup_model_manager(app, settings):
    """Setup function to be called from main.py"""
    # This function is not used anymore - model_manager is set directly from main.py
    # Include the router
    app.include_router(router)