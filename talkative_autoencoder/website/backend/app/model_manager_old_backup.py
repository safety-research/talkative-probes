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
        # For now, create a minimal config from the model_id
        # This matches the structure in model_groups.json
        return cls(
            id=model_id,
            memory_estimate=5.0,  # Default estimate
            dataset="unknown",
            architecture="unknown"
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
    shared_base_model: Optional[Any] = None  # For compatibility with grouped model manager
    
    @property
    def current_group_id(self) -> Optional[str]:
        """Get current group ID for compatibility"""
        return self.current_group.id if self.current_group else None
    
    @property
    def lens_cache(self) -> Dict[str, LensAnalyzer]:
        """Alias for loaded_models for backward compatibility"""
        return self.loaded_models
    
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
        if hasattr(settings, 'devices') and settings.devices:
            # Use configured devices
            self.devices = settings.devices
            self.num_devices = len(self.devices)
        else:
            # Fall back to auto-detection
            self.num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
            if self.num_devices > 0 and torch.cuda.is_available():
                self.devices = [f"cuda:{i}" for i in range(self.num_devices)]
            else:
                # CPU fallback
                self.devices = ["cpu"]
                self.num_devices = 1
        
        self.device_states: Dict[str, DeviceState] = {}
        self.device_locks: Dict[str, asyncio.Lock] = {}  # Add device-level locks
        for device in self.devices:
            if device.startswith("cuda:"):
                device_id = int(device.split(":")[1])
            else:
                device_id = 0
            self.device_states[device] = DeviceState(device_id=device_id)
            self.device_locks[device] = asyncio.Lock()  # Create lock for each device
        
        # Model and group configurations
        self.groups: Dict[str, GroupConfig] = {}
        self.model_to_groups: Dict[str, Set[str]] = {}  # model_id -> group_ids
        self.model_to_group: Dict[str, str] = {}  # model_id -> primary group_id for compatibility
        
        # CPU cache management
        self.cpu_cached_groups: Dict[str, Dict[str, Any]] = {}  # group_id -> {"base_model": ..., "models": {...}}
        self.group_usage_order: List[str] = []  # LRU tracking for CPU cache
        self.max_cpu_groups = getattr(settings, 'max_cpu_cached_models', 2)
        self.cpu_cache_lock = asyncio.Lock()  # Lock for CPU cache modifications
        
        # Track shared base models across devices (path -> model)
        self.shared_base_models: Dict[str, Any] = {}
        self.shared_orig_models: Dict[str, Any] = {}  # For orig models used by analyzers
        
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
            
            # The JSON uses "model_groups" not "groups"
            for group_data in groups_data.get("model_groups", []):
                models = []
                for model_data in group_data["models"]:
                    # Create ModelConfig with data from JSON
                    model = ModelConfig(
                        id=model_data["id"],
                        memory_estimate=model_data.get("batch_size", 32) * 0.1,  # Rough estimate
                        dataset="unknown",
                        architecture="unknown"
                    )
                    models.append(model)
                
                group = GroupConfig(
                    id=group_data["group_id"],  # Note: JSON uses "group_id" not "id"
                    name=group_data["group_name"],  # Note: JSON uses "group_name" not "name"
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
        device_id: Optional[str] = None
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
    
    async def clear_device(self, device_id: str, move_to_cpu: bool = False) -> Dict[str, Any]:
        """Clear all models from a device
        
        Args:
            device_id: Device to clear
            move_to_cpu: If True, move models to CPU cache instead of deleting
        """
        # Handle both string and int inputs
        if isinstance(device_id, int):
            device_id = f"cuda:{device_id}"
        
        device_state = self.device_states[device_id]
        
        # If we have a current group and should move to CPU, do that first
        if move_to_cpu and device_state.current_group:
            await self._move_group_to_cpu(device_state.current_group.id, device_id)
        else:
            # Clear all loaded models - explicitly delete analyzer objects
            for model_id in list(device_state.loaded_models.keys()):
                try:
                    analyzer = device_state.loaded_models[model_id]
                    del analyzer  # Explicitly delete the analyzer object
                    del device_state.loaded_models[model_id]
                except Exception as e:
                    logger.error(f"Error clearing {model_id}: {e}")
            
            # Clear any shared base model
            device_state.loaded_models.clear()
            device_state.shared_base_model = None  # Explicitly clear shared base model
            
            device_state.current_group = None
            device_state.memory_allocated = 0.0
            device_state.memory_reserved = 0.0
            
            # Force garbage collection and clear CUDA cache
            import gc
            gc.collect()
            if torch.cuda.is_available():
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
            ],
            "cpu_cached_groups": list(self.cpu_cached_groups.keys())
        }
    
    def get_model_location(self, model_id: str) -> Optional[List[Dict[str, Any]]]:
        """Find which device(s) have a model loaded"""
        locations = []
        
        # Check devices
        for device_id, state in self.device_states.items():
            if state.has_model(model_id):
                locations.append({
                    "device_id": device_id,
                    "group_id": state.current_group.id if state.current_group else None
                })
        
        # Check CPU cache
        for group_id, group_cache in self.cpu_cached_groups.items():
            # Ensure group_cache is a dict with models
            if isinstance(group_cache, dict) and "models" in group_cache:
                if model_id in group_cache["models"]:
                    locations.append({
                        "device_id": "cpu",
                        "group_id": group_id
                    })
                    break
        
        return locations if locations else None
    
    # Internal methods
    
    async def consider_duplicating_group(
        self, 
        group_id: str, 
        device_load_stats: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """
        Consider duplicating a group to another GPU for load balancing.
        For now, returns None as UnifiedModelManager doesn't fully support
        group duplication yet. This is a stub for compatibility.
        
        Returns the device ID if duplication occurred, None otherwise.
        """
        # TODO: Implement group duplication for UnifiedModelManager
        # For now, just return None to avoid errors
        logger.debug(f"Group duplication requested for {group_id} but not implemented in UnifiedModelManager")
        return None
    
    def find_all_devices_for_model(self, model_id: str) -> List[str]:
        """
        Find all devices that have a model or its group loaded.
        
        Args:
            model_id: The model to find devices for
            
        Returns:
            List of device IDs that have the model or its group loaded
        """
        devices = []
        
        # Check if model_to_groups exists and has this model
        if not hasattr(self, 'model_to_groups') or model_id not in self.model_to_groups:
            # Fallback: just check if model is directly loaded
            for device_id, state in self.device_states.items():
                if state.has_model(model_id):
                    devices.append(device_id)
            return devices
        
        # Find all devices with the model's group loaded
        for group_id in self.model_to_groups[model_id]:
            for device_id, state in self.device_states.items():
                if state.current_group and state.current_group.id == group_id and device_id not in devices:
                    devices.append(device_id)
        
        # Also check if model is directly loaded (might be loaded outside of a group)
        for device_id, state in self.device_states.items():
            if state.has_model(model_id) and device_id not in devices:
                devices.append(device_id)
        
        return devices
    
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
    
    def _find_best_device_for_group(self, group: GroupConfig) -> str:
        """Find the best device to load a group"""
        # Check if group is already loaded
        for device_id, state in self.device_states.items():
            if state.current_group and state.current_group.id == group.id:
                return device_id
        
        # Find device with enough free memory
        required_memory = group.memory_estimate
        
        best_device = None
        max_free_memory = 0.0
        
        for device_id in self.devices:
            free_memory = self._get_device_free_memory(device_id)
            if free_memory >= required_memory * 1.2 and free_memory > max_free_memory:
                best_device = device_id
                max_free_memory = free_memory
        
        # Fall back to device with most free memory if none have enough
        if best_device is None:
            best_device = max(self.devices, key=lambda d: self._get_device_free_memory(d))
        
        return best_device
    
    def _find_best_group_for_model(self, model_id: str, device_id: str) -> str:
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
        if device_id == "cpu":
            # For CPU, return a large number to indicate plenty of memory
            return 100.0
        
        if not torch.cuda.is_available():
            return 0.0
            
        device_num = int(device_id.split(':')[1]) if ':' in device_id else 0
        
        try:
            torch.cuda.set_device(device_num)
            # Get memory info
            free_memory = torch.cuda.mem_get_info(device_num)[0] / (1024**3)
            
            # Account for current group that will be unloaded
            device_state = self.device_states[device_id]
            if device_state.current_group:
                free_memory += device_state.memory_allocated
            
            return free_memory
        except Exception as e:
            logger.warning(f"Could not get memory info for {device_id}: {e}")
            return 10.0  # Default fallback
    
    async def _switch_device_to_group(self, device_id: str, group_id: str):
        """Switch a device to a different group"""
        # Use device lock to prevent concurrent switches
        async with self.device_locks[device_id]:
            device_state = self.device_states[device_id]
            group = self.groups[group_id]
            
            # Check if already on this group
            if device_state.current_group and device_state.current_group.id == group_id:
                logger.info(f"Device {device_id} already on group {group_id}, skipping switch")
                return
            
            # Mark as switching
            device_state.is_switching = True
            device_state.switch_start_time = datetime.utcnow()
            
            try:
                # Move current group to CPU if exists
                if device_state.current_group:
                    logger.info(f"Moving group {device_state.current_group.id} from {device_id} to CPU cache")
                    await self._move_group_to_cpu(device_state.current_group.id, device_id)
                
                # Load target group from CPU cache or fresh
                if group_id in self.cpu_cached_groups:
                    logger.info(f"Loading group {group_id} from CPU cache to {device_id}")
                    await self._load_group_from_cpu(group_id, device_id)
                    # Now that loading is complete, set the current group
                    device_state.current_group = group
                    logger.info(f"Device {device_id} fully loaded group {group_id}")
                else:
                    # Just set the group - models will be loaded on demand
                    device_state.current_group = group
                    logger.info(f"Device {device_id} switched to group {group_id}")
                
            finally:
                device_state.is_switching = False
                device_state.switch_start_time = None
    
    async def _move_group_to_cpu(self, group_id: str, device_id: str):
        """Move a group from a specific device to CPU cache"""
        device_state = self.device_states[device_id]
        
        if group_id not in self.groups:
            return
        
        # Use lock for CPU cache modifications
        async with self.cpu_cache_lock:
            # Manage CPU cache size
            await self._manage_cpu_cache()
            
            # Move models to CPU
            # First move the shared base model to CPU if it exists
            if device_state.shared_base_model is not None and hasattr(device_state.shared_base_model, 'to'):
                device_state.shared_base_model.to('cpu')
                logger.debug("Moved shared base model to CPU")
            
            group_cache = {
                "base_model": device_state.shared_base_model,
                "models": {}
            }
            
            # Move each model to CPU
            for model_id, analyzer in list(device_state.loaded_models.items()):
                try:
                    # Actually move the model tensors to CPU
                    analyzer.to('cpu')
                    group_cache["models"][model_id] = analyzer
                    logger.debug(f"Moved {model_id} to CPU")
                except Exception as e:
                    logger.error(f"Error moving {model_id} to CPU: {e}")
            
            # Store in CPU cache
            self.cpu_cached_groups[group_id] = group_cache
            
            # Update LRU tracking
            if group_id in self.group_usage_order:
                self.group_usage_order.remove(group_id)
            self.group_usage_order.append(group_id)
        
        # Clear device state
        device_state.shared_base_model = None
        device_state.loaded_models.clear()
        device_state.current_group = None
        device_state.memory_allocated = 0.0
        device_state.memory_reserved = 0.0
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if device_id.startswith("cuda:"):
                torch.cuda.synchronize()
    
    async def _load_group_from_cpu(self, group_id: str, device_id: str):
        """Load a group from CPU cache to a specific device"""
        device_state = self.device_states[device_id]
        
        # Use lock to safely access CPU cache
        async with self.cpu_cache_lock:
            if group_id not in self.cpu_cached_groups:
                raise ValueError(f"Group {group_id} not in CPU cache")
            
            group_cache = self.cpu_cached_groups[group_id]
            
            # CRITICAL: Remove from CPU cache since we're moving the models in-place
            # If we don't remove it, the cache will hold references to models that are now on GPU!
            del self.cpu_cached_groups[group_id]
            # Also remove from LRU tracking
            if group_id in self.group_usage_order:
                self.group_usage_order.remove(group_id)
            logger.info(f"Removed group {group_id} from CPU cache before moving to {device_id}")
        
        # Move base model to device first
        if group_cache["base_model"] is not None:
            # Move the shared base model to the target device
            if hasattr(group_cache["base_model"], 'to'):
                group_cache["base_model"].to(device_id)
            device_state.shared_base_model = group_cache["base_model"]
            logger.info(f"Moved shared base model to {device_id}")
        
        # Move models to device
        for model_id, analyzer in group_cache["models"].items():
            # Move the analyzer and all its components to device
            # The analyzer.to() method now properly moves shared_base_model
            analyzer.to(device_id)
            
            # Ensure the orig_model is also moved if it exists
            if hasattr(analyzer, 'orig_model') and analyzer.orig_model is not None:
                if hasattr(analyzer.orig_model, 'model'):
                    analyzer.orig_model.model.to(device_id)
                    logger.debug(f"Moved orig_model for {model_id} to {device_id}")
            
            device_state.loaded_models[model_id] = analyzer
        
        # Don't set current_group here - let the caller (_switch_device_to_group) do it
        # after confirming all models are loaded
        
        # Update memory stats for CUDA devices
        if device_id.startswith("cuda:") and torch.cuda.is_available():
            device_num = int(device_id.split(':')[1])
            torch.cuda.set_device(device_num)
            device_state.memory_allocated = torch.cuda.memory_allocated(device_num) / (1024**3)
            device_state.memory_reserved = torch.cuda.memory_reserved(device_num) / (1024**3)
        
        logger.info(f"Loaded group {group_id} from CPU cache to {device_id}")
    
    async def _manage_cpu_cache(self):
        """Manage CPU cache by evicting least recently used groups"""
        if len(self.cpu_cached_groups) >= self.max_cpu_groups:
            # Find LRU group in CPU cache
            lru_group = None
            for group_id in self.group_usage_order:
                if group_id in self.cpu_cached_groups:
                    # Check if this group is currently on any GPU
                    on_gpu = any(
                        state.current_group and state.current_group.id == group_id 
                        for state in self.device_states.values()
                    )
                    if not on_gpu:
                        lru_group = group_id
                        break
            
            if lru_group:
                logger.info(f"Evicting group {lru_group} from CPU cache")
                del self.cpu_cached_groups[lru_group]
                self.group_usage_order.remove(lru_group)
                import gc
                gc.collect()
    
    async def _load_model_on_device(self, model_id: str, device_id: str):
        """Load a specific model on a device"""
        device_state = self.device_states[device_id]
        
        if model_id in device_state.loaded_models:
            return  # Already loaded
        
        logger.info(f"Loading model {model_id} on device {device_id}")
        
        # Get model data from JSON
        groups_file = Path(__file__).parent / "model_groups.json"
        model_json_data = {}
        base_model_path = None
        groups_data = {}
        
        if groups_file.exists():
            with open(groups_file) as f:
                groups_data = json.load(f)
            for group_data in groups_data.get("model_groups", []):
                for model_data in group_data["models"]:
                    if model_data["id"] == model_id:
                        model_json_data = model_data
                        base_model_path = group_data.get("base_model_path")
                        break
                if model_json_data:
                    break
        
        if not model_json_data:
            raise ValueError(f"Model {model_id} not found in configuration")
        
        # Create lens analyzer with proper configuration
        try:
            # Import the grouped model manager's approach to creating analyzers
            from lens.analysis.analyzer_class import LensAnalyzer
            
            # Get settings from JSON
            settings = groups_data.get("settings", {})
            use_bf16 = settings.get("use_bf16", True)
            batch_size = model_json_data.get("batch_size", 32)
            no_orig = settings.get("no_orig", True)
            comparison_tl = settings.get("comparison_tl_checkpoint", False)
            
            # Get different_activations_orig if specified - reuse if already loaded
            different_activations_orig = None
            orig_path = model_json_data.get("different_activations_orig_path")
            if orig_path:
                # Check if we already have this orig model loaded
                if orig_path in self.shared_orig_models:
                    different_activations_orig = self.shared_orig_models[orig_path]
                    logger.info(f"Reusing shared orig model: {orig_path}")
                else:
                    # Pass the path and let LensAnalyzer load it
                    different_activations_orig = orig_path
            
            # Check if we can reuse a shared base model
            shared_base_to_use = None
            if device_state.shared_base_model is not None:
                # Already have a base model for this device/group
                shared_base_to_use = device_state.shared_base_model
                logger.info(f"Reusing shared base model on device {device_id}")
            elif base_model_path and base_model_path in self.shared_base_models:
                # We have this base model loaded elsewhere, reuse it
                shared_base_to_use = self.shared_base_models[base_model_path]
                logger.info(f"Reusing shared base model from cache: {base_model_path}")
            
            # Create analyzer with correct arguments matching LensAnalyzer.__init__
            analyzer = LensAnalyzer(
                checkpoint_path=model_json_data.get("lens_checkpoint_path"),
                device=device_id,
                batch_size=batch_size,
                use_bf16=use_bf16,
                strict_load=False,
                no_orig=no_orig,
                comparison_tl_checkpoint=comparison_tl,
                shared_base_model=shared_base_to_use,
                different_activations_orig=different_activations_orig
            )
            
            # Store the analyzer
            device_state.loaded_models[model_id] = analyzer
            
            # Extract and store shared base model if this is the first model in the group
            if hasattr(analyzer, 'shared_base_model') and device_state.shared_base_model is None:
                device_state.shared_base_model = analyzer.shared_base_model
                if base_model_path:
                    self.shared_base_models[base_model_path] = analyzer.shared_base_model
                logger.info(f"Extracted shared base model for group on device {device_id}")
            
            # Store orig model if available and not already cached
            if hasattr(analyzer, 'orig_model') and analyzer.orig_model is not None:
                orig_path = model_json_data.get("different_activations_orig_path")
                if orig_path and orig_path not in self.shared_orig_models:
                    self.shared_orig_models[orig_path] = analyzer.orig_model
                    logger.info(f"Cached shared orig model: {orig_path}")
            
            # Update memory tracking for CUDA devices
            if device_id.startswith("cuda:") and torch.cuda.is_available():
                device_num = int(device_id.split(':')[1])
                torch.cuda.set_device(device_num)
                device_state.memory_allocated = torch.cuda.memory_allocated(device_num) / (1024**3)
                device_state.memory_reserved = torch.cuda.memory_reserved(device_num) / (1024**3)
            
            logger.info(f"Model {model_id} loaded successfully on device {device_id}")
            logger.info(f"Device {device_id} now has models: {list(device_state.loaded_models.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id} on device {device_id}: {e}")
            raise
    
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
        # Find model in groups
        model_config = None
        group_info = None
        
        for group_id in self.model_to_groups.get(model_id, []):
            group = self.groups[group_id]
            for model in group.models:
                if model.id == model_id:
                    model_config = model
                    group_info = group
                    break
            if model_config:
                break
        
        if not model_config:
            return {"id": model_id, "error": "Unknown model"}
        
        # Get model data from model_groups.json structure
        groups_file = Path(__file__).parent / "model_groups.json"
        model_json_data = {}
        group_json_data = {}
        if groups_file.exists():
            with open(groups_file) as f:
                groups_data = json.load(f)
            for group_data in groups_data.get("model_groups", []):
                for model_data in group_data["models"]:
                    if model_data["id"] == model_id:
                        model_json_data = model_data
                        group_json_data = group_data
                        break
        
        info = {
            "id": model_id,
            "model_id": model_id,  # For compatibility
            "name": model_json_data.get("name", model_id),
            "display_name": model_json_data.get("name", model_id),  # For compatibility
            "description": model_json_data.get("description", ""),
            "dataset": model_config.dataset,
            "architecture": model_config.architecture,
            "memory_estimate": model_config.memory_estimate,
            "layer": model_json_data.get("layer", 0),
            "auto_batch_size_max": model_json_data.get("auto_batch_size_max", 16),
            "generation_config": {},
            # Add fields expected by inference service
            "base_model": group_json_data.get("base_model_path", "Unknown"),
            # Use different_activations_orig_path if available, otherwise use base_model_path
            "donor_model": model_json_data.get("different_activations_orig_path") or group_json_data.get("base_model_path", "Unknown"),
            "shared_base_model": group_json_data.get("base_model_path", "Unknown"),
            # Add checkpoint path for metadata display
            "checkpoint": model_json_data.get("lens_checkpoint_path", "Unknown"),
            "checkpoint_path": model_json_data.get("lens_checkpoint_path", "Unknown"),  # Also add with _path suffix for compatibility
            "encoder_decoder_model": model_json_data.get("name", model_id)  # For metadata display
        }
        
        # Add location information
        location_info = self.get_model_location(model_id)
        if location_info:
            info["locations"] = location_info
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
        # Check if we have the model loaded
        for device_state in self.device_states.values():
            if model_id in device_state.loaded_models:
                analyzer = device_state.loaded_models[model_id]
                if hasattr(analyzer, 'tokenizer'):
                    return analyzer.tokenizer
        
        # If model not loaded yet, we could load just the tokenizer
        # but for now return None
        return None
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage across all devices"""
        memory_info = {}
        
        for device_id in self.devices:
            if device_id == "cpu":
                # For CPU, return placeholder values
                memory_info[device_id] = {
                    "total_gb": 100.0,
                    "free_gb": 80.0,
                    "allocated_gb": 20.0,
                    "reserved_gb": 20.0,
                    "used_percent": 20.0
                }
                continue
                
            if not torch.cuda.is_available():
                continue
                
            device_num = int(device_id.split(':')[1]) if ':' in device_id else 0
            
            try:
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
            except Exception as e:
                logger.warning(f"Could not get memory usage for {device_id}: {e}")
                # Add placeholder values
                memory_info[device_id] = {
                    "total_gb": 0.0,
                    "free_gb": 0.0,
                    "allocated_gb": 0.0,
                    "reserved_gb": 0.0,
                    "used_percent": 0.0
                }
        
        return memory_info
    
    def is_model_loaded(self) -> bool:
        """Check if any model is loaded on any device"""
        for state in self.device_states.values():
            if state.loaded_models:
                return True
        return False