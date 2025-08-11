"""
Unified Model Manager for Multi-GPU Support
==========================================

This module provides a unified model manager that treats single-GPU deployments
as a special case of multi-GPU, eliminating the need for separate code paths.

Architecture:
- Models are always part of groups (single-model groups for standalone models)
- Devices always have a current group (which may contain one or many models)
- All operations are inherently device-aware
"""

import asyncio
import json
import logging
import time
import sys
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple

import torch

# Optional import for WebSocket support
try:
    from fastapi import WebSocket
except ImportError:
    WebSocket = None

# Add parent directory to path to import lens module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lens.analysis.analyzer_class import LensAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model"""
    id: str
    memory_estimate: float
    dataset: str
    architecture: str
    
    @classmethod
    def from_registry(cls, model_id: str) -> "ModelConfig":
        """Create ModelConfig from registry information"""
        # Import here to avoid circular import
        from .legacy.model_registry import get_model_config
        
        config = get_model_config(model_id)
        if not config:
            raise ValueError(f"Unknown model: {model_id}")
        
        return cls(
            id=model_id,
            memory_estimate=getattr(config, 'memory_estimate', 5.0),
            dataset=config.dataset,
            architecture=config.architecture
        )


@dataclass
class GroupConfig:
    """Configuration for a model group"""
    id: str
    name: str
    models: List[ModelConfig]
    memory_estimate: float = 0.0
    
    def __post_init__(self):
        """Calculate total memory estimate for the group"""
        self.memory_estimate = sum(model.memory_estimate for model in self.models)
    
    @classmethod
    def from_single_model(cls, model_id: str) -> "GroupConfig":
        """Create a single-model group"""
        model_config = ModelConfig.from_registry(model_id)
        return cls(
            id=f"single_{model_id}",
            name=f"Single: {model_id}",
            models=[model_config]
        )


@dataclass
class DeviceState:
    """State of a single GPU device"""
    device_id: int
    current_group: Optional[GroupConfig] = None
    loaded_models: Dict[str, LensAnalyzer] = field(default_factory=dict)
    is_switching: bool = False
    switch_start_time: Optional[datetime] = None
    memory_allocated: float = 0.0
    memory_reserved: float = 0.0
    
    @property
    def available_models(self) -> Set[str]:
        """Get IDs of models currently loaded on this device"""
        return set(self.loaded_models.keys())
    
    def has_model(self, model_id: str) -> bool:
        """Check if a specific model is loaded"""
        return model_id in self.loaded_models
    
    def get_analyzer(self, model_id: str) -> Optional[LensAnalyzer]:
        """Get the lens analyzer for a model"""
        return self.loaded_models.get(model_id)


class UnifiedModelManager:
    """
    Unified model manager that handles both single and multi-GPU deployments
    with a consistent interface.
    """
    
    def __init__(self, settings):
        self.settings = settings
        
        # Device management - use string keys to match GroupedModelManager
        self.num_devices = torch.cuda.device_count()
        self.devices = [f"cuda:{i}" for i in range(self.num_devices)]
        self.device_states: Dict[str, DeviceState] = {
            f"cuda:{i}": DeviceState(device_id=i) for i in range(self.num_devices)
        }
        
        # Model and group configurations
        self.groups: Dict[str, GroupConfig] = {}
        self.model_to_groups: Dict[str, Set[str]] = {}  # model_id -> group_ids
        self.model_to_group: Dict[str, str] = {}  # model_id -> primary group_id for compatibility
        
        # Load group configurations
        self._load_group_configs()
        
        # WebSocket connections for updates
        self.websocket_connections: Set[WebSocket] = set()
        
        # Background tasks
        self.cleanup_task = None
        
        logger.info(f"Initialized UnifiedModelManager with {self.num_devices} devices")
    
    def _load_group_configs(self):
        """Load model group configurations"""
        groups_file = Path(__file__).parent / "model_groups.json"
        
        if groups_file.exists():
            with open(groups_file) as f:
                groups_data = json.load(f)
            
            for group_data in groups_data.get("groups", []):
                models = [
                    ModelConfig.from_registry(m["id"]) 
                    for m in group_data["models"]
                ]
                
                group = GroupConfig(
                    id=group_data["id"],
                    name=group_data["name"],
                    models=models
                )
                
                self.groups[group.id] = group
                
                # Update model-to-groups mapping
                for model in models:
                    if model.id not in self.model_to_groups:
                        self.model_to_groups[model.id] = set()
                    self.model_to_groups[model.id].add(group.id)
                    
                    # Set primary group for compatibility
                    if model.id not in self.model_to_group:
                        self.model_to_group[model.id] = group.id
        
        # Create single-model groups for any models not in a group
        from .legacy.model_registry import list_available_models
        all_models_dict = list_available_models()
        
        for model_id, model_info in all_models_dict.items():
            if model_id not in self.model_to_groups:
                group = GroupConfig.from_single_model(model_id)
                self.groups[group.id] = group
                self.model_to_groups[model_id] = {group.id}
                self.model_to_group[model_id] = group.id
        
        logger.info(f"Loaded {len(self.groups)} model groups")
    
    async def start(self):
        """Start the model manager"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Model manager started")
    
    async def stop(self):
        """Stop the model manager"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all models
        for device_id in self.device_states:
            await self.clear_device(device_id)
        
        logger.info("Model manager stopped")
    
    async def _cleanup_loop(self):
        """Background task to clean up unused models"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._cleanup_unused_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_unused_models(self):
        """Clean up models that haven't been used recently"""
        # This is a placeholder - implement based on usage tracking
        pass
    
    # Core API Methods
    
    async def load_model(
        self, 
        model_id: str, 
        device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load a model on a specific device or find the best device.
        
        This is the primary method for model loading that works for both
        single and multi-GPU setups.
        """
        if model_id not in self.model_to_groups:
            raise ValueError(f"Unknown model: {model_id}")
        
        # Find the best device if not specified
        if device_id is None:
            device_id = self.find_best_device_for_model(model_id)
        else:
            # Convert integer device_id to string format
            if isinstance(device_id, int):
                device_id = f"cuda:{device_id}"
        
        device_state = self.device_states[device_id]
        
        # Check if model is already loaded
        if device_state.has_model(model_id):
            return {
                "status": "already_loaded",
                "model_id": model_id,
                "device_id": device_id
            }
        
        # Find a group containing this model
        group_id = self._find_best_group_for_model(model_id, device_id)
        
        # Load the group if needed
        if not device_state.current_group or device_state.current_group.id != group_id:
            await self._switch_device_to_group(device_id, group_id)
        
        # Load the specific model
        await self._load_model_on_device(model_id, device_id)
        
        # Broadcast update
        await self._broadcast_state_update()
        
        return {
            "status": "loaded",
            "model_id": model_id,
            "device_id": device_id,
            "group_id": group_id
        }
    
    async def process_prompt(
        self,
        model_id: str,
        prompt: str,
        num_predictions: int = 3,
        device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a prompt with a specific model.
        
        Automatically handles model loading and device selection.
        """
        # Ensure model is loaded
        load_result = await self.load_model(model_id, device_id)
        device_id = load_result["device_id"]
        
        # Get the lens analyzer
        device_state = self.device_states[device_id]
        analyzer = device_state.get_analyzer(model_id)
        
        if not analyzer:
            raise RuntimeError(f"Model {model_id} failed to load on device {device_id}")
        
        # Process the prompt using the analyzer
        try:
            # Convert prompt to the format expected by LensAnalyzer
            result = await asyncio.to_thread(
                analyzer.analyze_text,
                prompt,
                k=num_predictions
            )
            
            return {
                "model_id": model_id,
                "device_id": device_id,
                "prompt": prompt,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Analysis error on device {device_id}: {e}")
            raise
    
    async def load_group(
        self,
        group_id: str,
        device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Load an entire group of models on a device"""
        if group_id not in self.groups:
            raise ValueError(f"Unknown group: {group_id}")
        
        group = self.groups[group_id]
        
        # Find the best device if not specified
        if device_id is None:
            device_id = self._find_best_device_for_group(group)
        
        # Switch device to this group
        await self._switch_device_to_group(device_id, group_id)
        
        # Load all models in the group
        loaded_models = []
        for model_config in group.models:
            try:
                await self._load_model_on_device(model_config.id, device_id)
                loaded_models.append(model_config.id)
            except Exception as e:
                logger.error(f"Failed to load {model_config.id}: {e}")
        
        await self._broadcast_state_update()
        
        return {
            "status": "loaded",
            "group_id": group_id,
            "device_id": device_id,
            "loaded_models": loaded_models
        }
    
    async def clear_device(self, device_id: str) -> Dict[str, Any]:
        """Clear all models from a device"""
        # Handle both string and int inputs
        if isinstance(device_id, int):
            device_id = f"cuda:{device_id}"
        
        device_state = self.device_states[device_id]
        
        # Clear all loaded models
        for model_id in list(device_state.loaded_models.keys()):
            try:
                del device_state.loaded_models[model_id]
            except Exception as e:
                logger.error(f"Error clearing {model_id}: {e}")
        
        # Clear any shared base model
        device_state.loaded_models.clear()
        
        device_state.current_group = None
        device_state.memory_allocated = 0.0
        device_state.memory_reserved = 0.0
        
        # Force garbage collection
        torch.cuda.empty_cache()
        
        await self._broadcast_state_update()
        
        return {
            "status": "cleared",
            "device_id": device_id
        }
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get the complete system state"""
        devices = []
        
        for device_id, state in self.device_states.items():
            device_num = int(device_id.split(':')[1]) if ':' in device_id else 0
            device_info = {
                "device_id": device_num,  # Return as integer for API compatibility
                "device": device_id,  # Full string format
                "current_group": {
                    "id": state.current_group.id,
                    "name": state.current_group.name
                } if state.current_group else None,
                "loaded_models": list(state.available_models),
                "is_switching": state.is_switching,
                "memory": {
                    "allocated_gb": state.memory_allocated,
                    "reserved_gb": state.memory_reserved
                }
            }
            devices.append(device_info)
        
        return {
            "num_devices": self.num_devices,
            "devices": devices,
            "groups": [
                {
                    "id": group.id,
                    "name": group.name,
                    "models": [m.id for m in group.models],
                    "memory_estimate_gb": group.memory_estimate
                }
                for group in self.groups.values()
            ]
        }
    
    def get_model_location(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Find which device(s) have a model loaded"""
        locations = []
        
        for device_id, state in self.device_states.items():
            if state.has_model(model_id):
                locations.append({
                    "device_id": device_id,
                    "group_id": state.current_group.id if state.current_group else None
                })
        
        if not locations:
            return None
        
        return {
            "model_id": model_id,
            "locations": locations
        }
    
    # Internal methods
    
    def find_best_device_for_model(self, model_id: str, preferred_device: Optional[str] = None) -> str:
        """Find the best device to load a model"""
        # If preferred device is specified and available, check it first
        if preferred_device and preferred_device in self.device_states:
            state = self.device_states[preferred_device]
            if state.has_model(model_id):
                return preferred_device
        
        # First, check if model is already loaded somewhere
        for device_id, state in self.device_states.items():
            if state.has_model(model_id):
                return device_id
        
        # Find device with a compatible group loaded
        for group_id in self.model_to_groups[model_id]:
            for device_id, state in self.device_states.items():
                if state.current_group and state.current_group.id == group_id:
                    return device_id
        
        # Find device with most free memory
        best_device = self.devices[0] if self.devices else "cuda:0"
        max_free_memory = 0.0
        
        for device_id in self.devices:
            free_memory = self._get_device_free_memory(device_id)
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_device = device_id
        
        return best_device
    
    def _find_best_device_for_group(self, group: GroupConfig) -> int:
        """Find the best device to load a group"""
        # Check if group is already loaded
        for device_id, state in self.device_states.items():
            if state.current_group and state.current_group.id == group.id:
                return device_id
        
        # Find device with enough free memory
        required_memory = group.memory_estimate
        
        for device_id in range(self.num_devices):
            free_memory = self._get_device_free_memory(device_id)
            if free_memory >= required_memory * 1.2:  # 20% buffer
                return device_id
        
        # Fall back to device with most free memory
        return max(
            range(self.num_devices),
            key=lambda d: self._get_device_free_memory(d)
        )
    
    def _find_best_group_for_model(self, model_id: str, device_id: int) -> str:
        """Find the best group to use for loading a model"""
        device_state = self.device_states[device_id]
        
        # If device already has a group with this model, use it
        if device_state.current_group:
            for model in device_state.current_group.models:
                if model.id == model_id:
                    return device_state.current_group.id
        
        # Find the smallest group containing this model
        candidate_groups = [
            self.groups[gid] for gid in self.model_to_groups[model_id]
        ]
        
        # Sort by memory requirement (prefer smaller groups)
        candidate_groups.sort(key=lambda g: g.memory_estimate)
        
        # Check if we can fit any group
        free_memory = self._get_device_free_memory(device_id)
        for group in candidate_groups:
            if group.memory_estimate <= free_memory * 0.8:  # Leave 20% buffer
                return group.id
        
        # Fall back to smallest group
        return candidate_groups[0].id
    
    def _get_device_free_memory(self, device_id: str) -> float:
        """Get free memory on a device in GB"""
        device_num = int(device_id.split(':')[1]) if ':' in device_id else 0
        torch.cuda.set_device(device_num)
        
        # Get memory info
        free_memory = torch.cuda.mem_get_info(device_num)[0] / (1024**3)
        
        # Account for current group that will be unloaded
        device_state = self.device_states[device_id]
        if device_state.current_group:
            free_memory += device_state.memory_allocated
        
        return free_memory
    
    async def _switch_device_to_group(self, device_id: str, group_id: str):
        """Switch a device to a different group"""
        device_state = self.device_states[device_id]
        group = self.groups[group_id]
        
        # Mark as switching
        device_state.is_switching = True
        device_state.switch_start_time = datetime.utcnow()
        
        try:
            # Clear current models if switching groups
            if device_state.current_group and device_state.current_group.id != group_id:
                await self.clear_device(device_id)
            
            # Set new group
            device_state.current_group = group
            
            logger.info(f"Device {device_id} switched to group {group_id}")
            
        finally:
            device_state.is_switching = False
            device_state.switch_start_time = None
    
    async def _load_model_on_device(self, model_id: str, device_id: str):
        """Load a specific model on a device"""
        device_state = self.device_states[device_id]
        
        if model_id in device_state.loaded_models:
            return  # Already loaded
        
        logger.info(f"Loading model {model_id} on device {device_id}")
        
        # Create lens analyzer with proper configuration
        analyzer = LensAnalyzer(
            model_id,
            device=device_id,
            # Pass other required parameters from settings
            dtype=torch.float16 if self.settings.use_half_precision else torch.float32
        )
        
        # Store the analyzer
        device_state.loaded_models[model_id] = analyzer
        
        # Update memory tracking
        device_num = int(device_id.split(':')[1]) if ':' in device_id else 0
        torch.cuda.set_device(device_num)
        device_state.memory_allocated = torch.cuda.memory_allocated(device_num) / (1024**3)
        device_state.memory_reserved = torch.cuda.memory_reserved(device_num) / (1024**3)
        
        logger.info(f"Model {model_id} loaded successfully on device {device_id}")
    
    # WebSocket support
    
    async def add_websocket(self, websocket: WebSocket):
        """Add a WebSocket connection for state updates"""
        self.websocket_connections.add(websocket)
        
        # Send initial state
        await websocket.send_json({
            "type": "state_update",
            "data": self.get_system_state()
        })
    
    async def remove_websocket(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.websocket_connections.discard(websocket)
    
    async def _broadcast_state_update(self):
        """Broadcast state update to all connected clients"""
        if not self.websocket_connections:
            return
        
        state = self.get_system_state()
        message = {
            "type": "state_update",
            "data": state
        }
        
        # Send to all connections
        disconnected = set()
        for ws in self.websocket_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.add(ws)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected
    
    # Additional methods for compatibility with InferenceService
    
    async def get_default_model_id(self) -> str:
        """Get the default model ID"""
        # Return the first available model
        if self.model_to_group:
            return list(self.model_to_group.keys())[0]
        raise ValueError("No models available")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        from .legacy.model_registry import get_model_config
        config = get_model_config(model_id)
        if not config:
            return {"id": model_id, "error": "Unknown model"}
        
        info = {
            "id": model_id,
            "model_id": model_id,  # For compatibility
            "name": config.name,
            "display_name": config.name,  # For compatibility
            "description": getattr(config, 'description', ''),
            "dataset": config.dataset,
            "architecture": config.architecture,
            "memory_estimate": getattr(config, 'memory_estimate', 5.0),
            "layer": config.layer,
            "auto_batch_size_max": getattr(config, 'auto_batch_size_max', 16),
            "generation_config": getattr(config, 'generation_config', {})
        }
        
        # Add location information
        location_info = self.get_model_location(model_id)
        if location_info:
            info["locations"] = location_info["locations"]
            info["is_loaded"] = True
        else:
            info["locations"] = []
            info["is_loaded"] = False
        
        return info
    
    async def get_analyzer_for_model(
        self, 
        model_id: str, 
        device: Optional[str] = None
    ) -> Tuple[LensAnalyzer, str]:
        """Get analyzer for a model, loading if necessary. Returns (analyzer, device)."""
        # Ensure model is loaded
        result = await self.load_model(model_id, device)
        device_id = result["device_id"]
        
        # Get the analyzer
        device_state = self.device_states[device_id]
        analyzer = device_state.get_analyzer(model_id)
        
        if not analyzer:
            raise RuntimeError(f"Failed to get analyzer for model {model_id}")
        
        return analyzer, device_id
    
    def get_chat_tokenizer_for_model(self, model_id: str):
        """Get chat tokenizer for a model"""
        # This would need to be implemented based on the model's tokenizer
        # For now, return None as this seems to be optional in the inference service
        return None
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage across all devices"""
        memory_info = {}
        
        for device_id in self.devices:
            device_num = int(device_id.split(':')[1]) if ':' in device_id else 0
            torch.cuda.set_device(device_num)
            
            free, total = torch.cuda.mem_get_info(device_num)
            allocated = torch.cuda.memory_allocated(device_num)
            reserved = torch.cuda.memory_reserved(device_num)
            
            memory_info[device_id] = {
                "total_gb": total / (1024**3),
                "free_gb": free / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "used_percent": ((total - free) / total) * 100
            }
        
        return memory_info
    
    def is_model_loaded(self) -> bool:
        """Check if any model is loaded on any device"""
        for state in self.device_states.values():
            if state.loaded_models:
                return True
        return False