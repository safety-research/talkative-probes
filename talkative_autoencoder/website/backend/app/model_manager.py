"""Enhanced Model Manager with support for grouped models sharing base models across multiple GPUs"""

import asyncio
import torch
import gc
import json
import logging
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime
from pathlib import Path
import sys
import os
from dataclasses import dataclass, field

# Add parent directory to path to import lens module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lens.analysis.analyzer_class import LensAnalyzer
from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class DeviceState:
    """Tracks the state of a single GPU device"""
    device_id: str
    current_group_id: Optional[str] = None
    is_switching: bool = False
    switch_start_time: Optional[datetime] = None
    lens_cache: Dict[str, LensAnalyzer] = field(default_factory=dict)
    shared_base_model: Optional[Any] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def is_available_for_group(self, group_id: str) -> bool:
        """Check if this device is available for a specific group"""
        return not self.is_switching and (self.current_group_id == group_id or self.current_group_id is None)
    
    def has_group_loaded(self, group_id: str) -> bool:
        """Check if this device has a specific group loaded"""
        return self.current_group_id == group_id and self.shared_base_model is not None


class ModelGroupConfig:
    """Configuration for a model group"""
    def __init__(self, group_data: Dict[str, Any]):
        self.group_id = group_data["group_id"]
        self.group_name = group_data["group_name"]
        self.base_model_path = group_data["base_model_path"]
        self.description = group_data.get("description", "")
        # Visibility flags from config
        self.visible = group_data.get("visible", True)
        self.backend_only = group_data.get("backend_only", False)
        self.models = group_data["models"]
        
        # Memory estimate based on model name (rough estimates)
        self.memory_estimate = self._estimate_memory_requirement()
    
    def _estimate_memory_requirement(self) -> float:
        """Estimate memory requirement in GB based on model name"""
        model_name = self.base_model_path.lower()
        
        # Rough estimates based on common model sizes
        if "70b" in model_name or "llama-2-70b" in model_name:
            return 140.0  # 70B models need ~140GB
        elif "30b" in model_name or "34b" in model_name:
            return 70.0   # 30-34B models need ~70GB
        elif "13b" in model_name or "llama-2-13b" in model_name:
            return 26.0   # 13B models need ~26GB
        elif "9b" in model_name or "gemma-2-9b" in model_name:
            return 20.0   # 9B models need ~20GB (includes encoder/decoder)
        elif "7b" in model_name or "llama-2-7b" in model_name:
            return 15.0   # 7B models need ~15GB
        elif "3b" in model_name:
            return 8.0    # 3B models need ~8GB
        elif "2b" in model_name or "gemma-2b" in model_name:
            return 6.0    # 2B models need ~6GB
        else:
            # Default conservative estimate
            return 15.0


class UnifiedModelManager:
    """
    Unified model manager that handles both single and multi-GPU deployments.
    Manages models grouped by shared base models for efficient switching across multiple GPUs.
    
    This manager supports:
    - Multiple GPU devices with independent model groups
    - Efficient group switching on each device
    - Smart request routing to minimize group switches
    - Per-device caching and state management
    - Thread-safe operations across devices
    """

    def __init__(self, settings: Settings, groups_config_path: Optional[str] = None):
        self.settings = settings
        self.websocket_manager = None
        
        # Initialize devices from settings
        self.devices = settings.devices if settings.devices else ["cuda:0"]
        logger.info(f"Initializing model manager with devices: {self.devices}")

        # Initialize data structures first (before loading config)
        self.model_groups = {}
        self.model_to_group = {}  # model_id -> group_id
        self.config_settings = {}  # Will be populated by _load_groups_config
        
        # Load groups configuration
        if groups_config_path is None:
            # model_groups.json is colocated with backend .env (one directory above app/)
            groups_config_path = Path(__file__).parent.parent / "model_groups.json"
        self.groups_config_path = Path(groups_config_path)
        self._load_groups_config()

        # Per-device state tracking
        self.device_states: Dict[str, DeviceState] = {
            device: DeviceState(device_id=device) for device in self.devices
        }
        
        # Global state tracking (other tracking beyond what was initialized above)
        self.model_locks: Dict[str, asyncio.Lock] = {}  # model_id -> lock for loading
        self.global_cache_lock = asyncio.Lock()  # Lock for cross-device operations
        
        # Shared model caches (across all devices)
        self.shared_orig_models: Dict[str, Any] = {}  # model_path -> orig model wrapper
        self.model_last_used: Dict[str, datetime] = {}  # model_id -> last used time
        
        # CPU cache management
        self.cpu_cached_groups: Dict[str, Dict[str, Any]] = {}  # group_id -> {"base_model": ..., "models": {...}}
        self.group_usage_order: List[str] = []  # LRU tracking for CPU cache
        
        # Memory management settings
        from .config import Settings
        config_with_overrides = Settings.get_model_config_with_overrides({"settings": self.config_settings})
        self.max_cpu_groups = config_with_overrides.get('max_cpu_models', 
                                                        getattr(settings, 'max_cpu_cached_models', 2))
        
        # Chat tokenizers (shared across devices)
        self.chat_tokenizers: Dict[str, Any] = {}  # model_path -> tokenizer

    def _load_groups_config(self):
        """Load model groups configuration from JSON"""
        try:
            with open(self.groups_config_path, 'r') as f:
                data = json.load(f)

            # Clear existing data
            self.model_groups.clear()
            self.model_to_group.clear()

            for group_data in data.get("model_groups", []):
                group_config = ModelGroupConfig(group_data)
                self.model_groups[group_config.group_id] = group_config
                # Load ALL models, including backend_only ones
                for model in group_config.models:
                    self.model_to_group[model["id"]] = group_config.group_id
                    # Debug log for backend-only models
                    if model.get("backend_only"):
                        logger.debug(f"Loaded backend-only model: {model['id']}")

            self.config_settings = data.get("settings", {})
            logger.info(f"Loaded {len(self.model_groups)} model groups with {len(self.model_to_group)} total models")
        except Exception as e:
            logger.error(f"Failed to load model groups config: {e}")
            raise

    def reload_config(self):
        """Reload the model groups configuration from JSON file"""
        logger.info("Reloading model groups configuration")
        self._load_groups_config()
    
    @property
    def is_switching_group(self) -> bool:
        """Check if any device is currently switching groups"""
        return any(state.is_switching for state in self.device_states.values())
    
    @property
    def current_group_id(self) -> Optional[str]:
        """
        Get the current group ID for backward compatibility.
        Returns the group ID from the first device, or None if no groups loaded.
        """
        if self.devices:
            first_device_state = self.device_states.get(self.devices[0])
            if first_device_state:
                return first_device_state.current_group_id
        return None
    
    @property
    def lens_cache(self) -> Dict[str, Any]:
        """
        Get all lens analyzers across all devices for backward compatibility.
        Returns a merged dictionary of all device caches.
        """
        merged_cache = {}
        for device_state in self.device_states.values():
            merged_cache.update(device_state.lens_cache)
        return merged_cache

    def get_model_list(self, public_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get structured list of model groups for frontend.
        
        Args:
            public_only: When True, exclude groups/models flagged as backend_only or not visible.
            
        Returns:
            List of group dictionaries with model information and device status
        """
        groups = []
        for group_id, group_config in self.model_groups.items():
            # Apply group-level visibility filtering
            if not getattr(group_config, "visible", True):
                continue
            if public_only and getattr(group_config, "backend_only", False):
                continue
                
            # Collect device information for this group
            device_info = []
            for device_id, device_state in self.device_states.items():
                device_info.append({
                    "device": device_id,
                    "has_group": device_state.has_group_loaded(group_id),
                    "is_current": device_state.current_group_id == group_id,
                    "is_switching": device_state.is_switching
                })
            
            # Check if group is in CPU cache
            is_cpu_cached = group_id in self.cpu_cached_groups
            
            group_data = {
                "group_id": group_config.group_id,
                "group_name": group_config.group_name,
                "description": group_config.description,
                "base_model": group_config.base_model_path,
                "device_info": device_info,
                "is_cpu_cached": is_cpu_cached,
                "models": []
            }
            
            # Add model information
            for model in group_config.models:
                # Skip models that are not visible
                if model.get("visible") is False:
                    continue
                # Skip backend-only models for public contexts
                if public_only and model.get("backend_only", False):
                    continue
                    
                # Extract checkpoint folder name for display
                checkpoint_path = model.get("lens_checkpoint_path", "")
                checkpoint_filename = "Unknown"
                checkpoint_full = ""
                if checkpoint_path:
                    path_parts = checkpoint_path.split('/')
                    checkpoint_folder = path_parts[-1] if path_parts else ""
                    checkpoint_full = checkpoint_folder
                    if len(checkpoint_folder) > 40:
                        checkpoint_filename = f"{checkpoint_folder[:20]}...{checkpoint_folder[-15:]}"
                    else:
                        checkpoint_filename = checkpoint_folder
                
                # Check which devices have this model loaded
                loaded_on_devices = []
                for device_id, device_state in self.device_states.items():
                    if model["id"] in device_state.lens_cache:
                        loaded_on_devices.append(device_id)
                
                model_info = {
                    "id": model["id"],
                    "name": model["name"],
                    "description": model.get("description", ""),
                    "layer": model.get("layer", 30),
                    "checkpoint_filename": checkpoint_filename,
                    "checkpoint_full": checkpoint_full,
                    "loaded_on_devices": loaded_on_devices,
                    "last_used": self.model_last_used.get(model["id"], "").isoformat() if model["id"] in self.model_last_used else None
                }
                group_data["models"].append(model_info)
            groups.append(group_data)
        return groups

    def get_device_status(self) -> Dict[str, Any]:
        """Get status information for all devices"""
        device_status = {}
        for device_id, device_state in self.device_states.items():
            device_status[device_id] = {
                "current_group": device_state.current_group_id,
                "is_switching": device_state.is_switching,
                "models_loaded": list(device_state.lens_cache.keys()),
                "switch_start_time": device_state.switch_start_time.isoformat() if device_state.switch_start_time else None
            }
        return {
            "devices": device_status,
            "cpu_cached_groups": list(self.cpu_cached_groups.keys()),
            "total_models_loaded": sum(len(state.lens_cache) for state in self.device_states.values())
        }
    
    def get_current_group_info(self) -> Dict[str, Any]:
        """Get information about current groups across all devices"""
        # Find if any device is switching
        is_switching = False
        switch_start_time = None
        for device_state in self.device_states.values():
            if device_state.is_switching:
                is_switching = True
                if device_state.switch_start_time:
                    switch_start_time = device_state.switch_start_time
                    break
        
        # Get groups loaded on any device
        groups_loaded = set()
        for device_state in self.device_states.values():
            if device_state.current_group_id:
                groups_loaded.add(device_state.current_group_id)
        
        # Add CPU cached groups
        groups_loaded.update(self.cpu_cached_groups.keys())
        
        # Build device location map
        base_locations = {}
        for device_id, device_state in self.device_states.items():
            if device_state.current_group_id:
                base_locations[device_state.current_group_id] = device_id
        for group_id in self.cpu_cached_groups:
            if group_id not in base_locations:
                base_locations[group_id] = 'cpu'
        
        return {
            "current_group_id": self.current_group_id,  # For backward compatibility
            "is_switching": is_switching,
            "switch_start_time": switch_start_time.isoformat() if switch_start_time else None,
            "groups_loaded": list(groups_loaded),
            "base_locations": base_locations,
            "device_states": self.get_device_status()  # Include full device info
        }

    def find_all_devices_for_model(self, model_id: str) -> List[str]:
        """
        Find all devices that have a model's group loaded.
        
        Args:
            model_id: The model to find devices for
            
        Returns:
            List of device IDs that have the model's group loaded
        """
        target_group_id = self.model_to_group.get(model_id)
        if not target_group_id:
            return []
        
        devices = []
        for device_id, device_state in self.device_states.items():
            if device_state.has_group_loaded(target_group_id) and not device_state.is_switching:
                devices.append(device_id)
        
        return devices
    
    def find_best_device_for_model(self, model_id: str, preferred_device: Optional[str] = None) -> str:
        """
        Find the best device to run a model on.
        
        Priority order:
        1. Preferred device if specified and available
        2. Device with the model's group already loaded
        3. Device with the model already in cache
        4. Device with no group loaded (empty)
        5. Device with fewest models loaded
        
        Args:
            model_id: The model to find a device for
            preferred_device: Optional preferred device to use
            
        Returns:
            The best device ID to use
        """
        target_group_id = self.model_to_group.get(model_id)
        if not target_group_id:
            logger.error(f"Model ID {model_id} not found in model_to_group mapping")
            logger.error(f"Available models: {list(self.model_to_group.keys())[:10]}...")  # Show first 10
            logger.error(f"Total models in mapping: {len(self.model_to_group)}")
            raise ValueError(f"Unknown model ID: {model_id}")
        
        # Check preferred device first
        if preferred_device and preferred_device in self.device_states:
            device_state = self.device_states[preferred_device]
            if device_state.is_available_for_group(target_group_id):
                return preferred_device
        
        # Find device with group already loaded
        for device_id, device_state in self.device_states.items():
            if device_state.has_group_loaded(target_group_id) and not device_state.is_switching:
                logger.debug(f"Found device {device_id} with group {target_group_id} already loaded")
                return device_id
        
        # Find device with model already cached
        for device_id, device_state in self.device_states.items():
            if model_id in device_state.lens_cache and not device_state.is_switching:
                logger.debug(f"Found device {device_id} with model {model_id} already cached")
                return device_id
        
        # Find empty device
        for device_id, device_state in self.device_states.items():
            if device_state.current_group_id is None and not device_state.is_switching:
                logger.debug(f"Found empty device {device_id}")
                return device_id
        
        # Find least loaded device
        best_device = None
        min_models = float('inf')
        for device_id, device_state in self.device_states.items():
            if not device_state.is_switching:
                model_count = len(device_state.lens_cache)
                if model_count < min_models:
                    min_models = model_count
                    best_device = device_id
        
        if best_device:
            logger.debug(f"Using least loaded device {best_device} with {min_models} models")
            return best_device
        
        # Fallback to first device if all are switching
        return self.devices[0]
    
    async def load_model(self, model_id: str, device_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model on a specific device (API compatibility method).
        Optimized to handle within-group switches efficiently.
        Broadcasts status updates via WebSocket.
        
        Args:
            model_id: The model to load
            device_id: Optional specific device to use
            
        Returns:
            Dict with status information
        """
        try:
            target_group_id = self.model_to_group.get(model_id)
            if not target_group_id:
                raise ValueError(f"Unknown model ID: {model_id}")
            
            # Fast path: Check if model's group is already loaded somewhere
            for dev_id, dev_state in self.device_states.items():
                if dev_state.current_group_id == target_group_id and model_id in dev_state.lens_cache:
                    # Model is already loaded, this is just a selection change
                    logger.info(f"Model {model_id} already loaded on {dev_id} (fast within-group switch)")
                    
                    # Still broadcast the state update for UI consistency
                    if self.websocket_manager:
                        await self._broadcast_system_state_update()
                    
                    return {
                        "status": "success",
                        "model_id": model_id,
                        "device": dev_id,
                        "device_id": dev_id,
                        "group_id": target_group_id,
                        "message": f"Model {model_id} selected (already loaded on {dev_id})",
                        "fast_switch": True  # Indicate this was a fast switch
                    }
            
            # Slow path: Need to actually load the model/group
            analyzer, device = await self.get_analyzer_for_model(model_id, device_id)
            
            # Broadcast the update to all connected clients
            if self.websocket_manager:
                await self._broadcast_model_loaded(model_id, device)
            
            return {
                "status": "success",
                "model_id": model_id,
                "device": device,
                "device_id": device,  # Add device_id for compatibility
                "group_id": target_group_id,
                "message": f"Model {model_id} loaded on {device}"
            }
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    async def _broadcast_model_loaded(self, model_id: str, device: str):
        """Broadcast model loaded status to all connected clients"""
        # Use centralized broadcast method
        await self._broadcast_system_state_update()
    
    async def _broadcast_system_state_update(self):
        """
        Centralized method to broadcast complete system state to all connected clients.
        This ensures consistency across all state changes.
        """
        if not self.websocket_manager:
            return
        
        try:
            # Get complete system state
            system_state = self.get_system_state()
            
            # Build model groups list (similar to what main.py does)
            model_groups = []
            for group_id, group_config in self.model_groups.items():
                if not group_config.visible and not group_config.backend_only:
                    continue
                    
                group_data = {
                    "group_id": group_id,
                    "group_name": group_config.group_name,
                    "base_model": group_config.base_model_path,
                    "models": []
                }
                
                # Add model information
                for model in group_config.models:
                    model_id = model["id"]
                    # Check if model is loaded on any device
                    is_loaded = any(
                        model_id in device_state.lens_cache 
                        for device_state in self.device_states.values()
                    )
                    
                    model_data = {
                        "id": model_id,
                        "name": model.get("name", model_id),
                        "layer": model.get("layer", 0),
                        "is_loaded": is_loaded,
                        "is_current": False  # Will be set based on current selections
                    }
                    group_data["models"].append(model_data)
                
                model_groups.append(group_data)
            
            # Build cache info
            cache_info = {
                "groups_loaded": [],
                "models_cached": [],
                "base_locations": {}
            }
            
            # Add loaded groups and models from each device
            for device_state in self.device_states.values():
                if device_state.current_group_id:
                    cache_info["groups_loaded"].append(device_state.current_group_id)
                    cache_info["base_locations"][device_state.current_group_id] = device_state.device_id
                    
                    # Add cached models
                    cache_info["models_cached"].extend(list(device_state.lens_cache.keys()))
            
            # Add CPU cached groups
            for group_id in self.cpu_cached_groups:
                if group_id not in cache_info["groups_loaded"]:
                    cache_info["groups_loaded"].append(group_id)
                cache_info["base_locations"][group_id] = "cpu"
                
                # Add CPU cached models
                group_cache = self.cpu_cached_groups.get(group_id, {})
                if isinstance(group_cache, dict) and "models" in group_cache:
                    cache_info["models_cached"].extend(group_cache["models"].keys())
            
            # Get current model (from first device with loaded models)
            current_model = None
            current_group = None
            for device_state in self.device_states.values():
                if device_state.lens_cache:
                    current_model = list(device_state.lens_cache.keys())[0]
                    current_group = device_state.current_group_id
                    break
            
            # Construct the message
            message = {
                "type": "model_groups_list",
                "groups": model_groups,
                "current_group": current_group,
                "current_model": current_model,
                "is_switching": self.is_switching_group,
                "system_state": system_state,
                "model_status": {
                    "cache_info": cache_info
                }
            }
            
            # Broadcast to all connected clients
            await self.websocket_manager.broadcast(message)
            logger.info(f"Broadcast system state update to all clients")
            
        except Exception as e:
            logger.error(f"Failed to broadcast system state update: {e}")

    async def get_analyzer_for_model(self, model_id: str, device: Optional[str] = None) -> Tuple[LensAnalyzer, str]:
        """
        Get analyzer for a specific model, loading if necessary.
        
        Args:
            model_id: The model to get analyzer for
            device: Optional specific device to use
            
        Returns:
            Tuple of (analyzer, device_id) where analyzer is ready to use on device_id
        """
        # Check if model exists
        if model_id not in self.model_to_group:
            raise ValueError(f"Unknown model ID: {model_id}")
        
        target_group_id = self.model_to_group[model_id]
        
        # Find best device if not specified
        if device is None:
            device = self.find_best_device_for_model(model_id)
        elif device not in self.device_states:
            raise ValueError(f"Unknown device: {device}")
        
        device_state = self.device_states[device]
        
        # Update last used time
        self.model_last_used[model_id] = datetime.now()
        
        # Fast path: already loaded on this device
        if model_id in device_state.lens_cache:
            logger.debug(f"Model {model_id} already loaded on device {device}")
            return device_state.lens_cache[model_id], device
        
        # Need to ensure group is loaded on device
        if device_state.current_group_id != target_group_id:
            await self._switch_device_to_group(device, target_group_id)
        
        # Load the model on this device
        await self._load_model_on_device(model_id, device)
        
        return device_state.lens_cache[model_id], device

    async def _switch_device_to_group(self, device: str, target_group_id: str):
        """
        Switch a specific device to a different group.
        
        This operation:
        1. Moves current group to CPU cache if exists
        2. Loads target group from CPU cache or fresh
        3. Updates device state
        
        Args:
            device: Device to switch
            target_group_id: Group to switch to
        """
        device_state = self.device_states[device]
        
        async with device_state.lock:
            # Re-check after acquiring lock (another request may have loaded it)
            if device_state.current_group_id == target_group_id:
                logger.info(f"Group {target_group_id} already loaded on {device} (caught race condition)")
                return  # Already on target group
            
            try:
                device_state.is_switching = True
                device_state.switch_start_time = datetime.now()
                
                # Broadcast switch starting
                logger.info(f"About to broadcast switch starting, websocket_manager: {self.websocket_manager}")
                if self.websocket_manager:
                    logger.info(f"Active connections: {len(self.websocket_manager.active_connections)}")
                # Include the source group if we're switching between groups
                source_group = device_state.current_group_id if device_state.current_group_id else None
                await self._broadcast_group_switch_status("starting", target_group_id, device, source_group_id=source_group)
                
                # Memory check: Only needed when no group is currently loaded
                # When switching groups, we'll unload the current one first, so no check needed
                target_group = self.model_groups.get(target_group_id)
                if target_group:
                    memory_needed = target_group.memory_estimate
                    if not device_state.current_group_id:
                        # First load - check if we have enough free memory
                        if not self._check_memory_available(device, memory_needed):
                            raise RuntimeError(f"Insufficient memory on {device} for group {target_group_id} (needs {memory_needed:.1f}GB)")
                    else:
                        # Switching groups - current will be unloaded first, so no check needed
                        logger.info(f"Switching from {device_state.current_group_id} to {target_group_id} on {device} (estimated {memory_needed:.1f}GB)")
                
                # Move current group to CPU if exists
                if device_state.current_group_id:
                    logger.info(f"Moving group {device_state.current_group_id} from {device} to CPU cache")
                    # Send periodic keepalive updates during the long operation
                    # Pass target_group_id so progress messages show "Switching from X to Y"
                    await self._move_group_to_cpu_with_keepalive(device_state.current_group_id, device, target_group_id)
                
                # Load target group on this device
                if target_group_id in self.cpu_cached_groups:
                    logger.info(f"Loading group {target_group_id} from CPU cache to {device}")
                    # Pass source_group so progress messages show full switch context
                    await self._load_group_from_cpu_with_keepalive(target_group_id, device, source_group)
                else:
                    logger.info(f"Loading new group {target_group_id} on {device}")
                    group_config = self.model_groups[target_group_id]
                    first_model = group_config.models[0]
                    await self._load_group_fresh(target_group_id, first_model["id"], first_model, device)
                
                # Update device state
                device_state.current_group_id = target_group_id
                
                # Update LRU order
                if target_group_id in self.group_usage_order:
                    self.group_usage_order.remove(target_group_id)
                self.group_usage_order.append(target_group_id)
                
                # Broadcast completion
                await self._broadcast_group_switch_status("completed", target_group_id, device)
                
            except Exception as e:
                logger.error(f"Group switch failed on {device}: {e}")
                
                # Try to restore device to a safe state
                try:
                    # Clear the device if switch failed midway
                    if device_state.current_group_id != target_group_id:
                        logger.info(f"Clearing device {device} after failed switch")
                        device_state.lens_cache.clear()
                        device_state.shared_base_model = None
                        device_state.current_group_id = None
                        
                        # Clear GPU memory
                        if torch.cuda.is_available() and device.startswith("cuda:"):
                            device_num = int(device.split(':')[1])
                            with torch.cuda.device(device_num):
                                torch.cuda.empty_cache()
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup after failed switch: {cleanup_error}")
                
                await self._broadcast_group_switch_status("failed", target_group_id, device, str(e))
                raise
                
            finally:
                device_state.is_switching = False
                device_state.switch_start_time = None

    async def _move_group_to_cpu_with_keepalive(self, group_id: str, device: str, target_group_id: Optional[str] = None):
        """
        Move a group to CPU with periodic keepalive updates to prevent WebSocket timeout.
        """
        # Create a keepalive task
        keepalive_event = asyncio.Event()
        
        async def send_keepalive():
            """Send periodic status updates to keep WebSocket alive"""
            while not keepalive_event.is_set():
                try:
                    # Send progress update immediately, then every 2 seconds
                    if not keepalive_event.is_set():
                        await self._broadcast_group_switch_status(
                            "progress", 
                            target_group_id or group_id,  # Use target group for the main group_id field
                            device, 
                            f"Moving group to CPU...",
                            source_group_id=group_id  # Pass the group being moved as source
                        )
                    await asyncio.sleep(2)  # More frequent updates
                except Exception as e:
                    logger.warning(f"Error sending keepalive: {e}")
        
        # Start keepalive task
        keepalive_task = asyncio.create_task(send_keepalive())
        
        try:
            # Do the actual move
            await self._move_group_to_cpu(group_id, device)
        finally:
            # Stop keepalive
            keepalive_event.set()
            await keepalive_task
    
    async def _move_group_to_cpu(self, group_id: str, device: str):
        """Move a group from a specific device to CPU cache"""
        device_state = self.device_states[device]
        
        if group_id not in self.model_groups:
            return
        
        # Manage CPU cache size
        await self._manage_cpu_cache()
        
        # Move models to CPU
        group_cache = {
            "base_model": device_state.shared_base_model,
            "models": {}
        }
        
        loop = asyncio.get_event_loop()
        group_config = self.model_groups[group_id]
        
        # Move each model to CPU in a thread pool to avoid blocking the event loop
        for model in group_config.models:
            if model["id"] in device_state.lens_cache:
                analyzer = device_state.lens_cache[model["id"]]
                # Run the blocking .to('cpu') operation in a thread pool
                await loop.run_in_executor(None, analyzer.to, 'cpu')
                group_cache["models"][model["id"]] = analyzer
        
        # Store in CPU cache
        self.cpu_cached_groups[group_id] = group_cache
        
        # Clear device state
        device_state.shared_base_model = None
        device_state.lens_cache.clear()
        
        # Clear GPU memory in thread pool to avoid blocking
        if torch.cuda.is_available():
            def clear_gpu_memory():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            await loop.run_in_executor(None, clear_gpu_memory)

    async def _load_group_from_cpu_with_keepalive(self, group_id: str, device: str, source_group_id: Optional[str] = None):
        """
        Load a group from CPU with periodic keepalive updates to prevent WebSocket timeout.
        """
        # Create a keepalive task
        keepalive_event = asyncio.Event()
        
        async def send_keepalive():
            """Send periodic status updates to keep WebSocket alive"""
            while not keepalive_event.is_set():
                try:
                    # Send progress update immediately, then every 2 seconds
                    if not keepalive_event.is_set():
                        await self._broadcast_group_switch_status(
                            "progress", 
                            group_id, 
                            device, 
                            f"Loading group from CPU...",
                            source_group_id=source_group_id  # Pass the source if switching between groups
                        )
                    await asyncio.sleep(2)  # More frequent updates
                except Exception as e:
                    logger.warning(f"Error sending keepalive: {e}")
        
        # Start keepalive task
        keepalive_task = asyncio.create_task(send_keepalive())
        
        try:
            # Do the actual load
            await self._load_group_from_cpu(group_id, device)
        finally:
            # Stop keepalive
            keepalive_event.set()
            await keepalive_task
    
    async def _load_group_from_cpu(self, group_id: str, device: str):
        """Load a group from CPU cache to a specific device"""
        device_state = self.device_states[device]
        
        if group_id not in self.cpu_cached_groups:
            raise ValueError(f"Group {group_id} not in CPU cache")
        
        group_cache = self.cpu_cached_groups[group_id]
        loop = asyncio.get_event_loop()
        
        # Move base model to device
        device_state.shared_base_model = group_cache["base_model"]
        
        # Move models to device - properly run blocking operations in thread pool
        for model_id, analyzer in group_cache["models"].items():
            # Use a proper function call instead of lambda to avoid scope issues
            await loop.run_in_executor(None, analyzer.to, device)
            device_state.lens_cache[model_id] = analyzer
        
        logger.info(f"Loaded group {group_id} from CPU cache to {device}")

    async def _load_group_fresh(self, group_id: str, target_model_id: str, target_model_config: Dict[str, Any], device: str):
        """Load a group fresh on a specific device"""
        device_state = self.device_states[device]
        group_config = self.model_groups[group_id]
        loop = asyncio.get_event_loop()
        
        def load_base():
            # Handle donor model for the target model
            orig_model_path = target_model_config.get("different_activations_orig_path")
            logger.info(f"Loading model {target_model_id} on {device} with orig_model_path: {orig_model_path}")
            
            # Check if we already have this donor model cached
            orig_model = None
            if orig_model_path and orig_model_path in self.shared_orig_models:
                logger.info(f"Using cached donor model: {orig_model_path}")
                orig_model = self.shared_orig_models[orig_model_path]
            else:
                # Pass the path and let LensAnalyzer load it
                orig_model = orig_model_path
            
            # Get configuration with environment variable overrides
            from .config import Settings
            config_with_overrides = Settings.get_model_config_with_overrides(
                {"settings": self.config_settings},
                model_id=target_model_id.replace('-', '_'),
                group_id=group_id.replace('-', '_')
            )
            
            # If we have a different donor model, we can't use no_orig
            no_orig = config_with_overrides.get("no_orig", True) and not orig_model_path
            
            # Use model-specific batch_size if provided
            batch_size = target_model_config.get("batch_size")
            if batch_size is None:
                batch_size = config_with_overrides.get("batch_size", 32)
            
            # Create analyzer on the specific device
            with torch.cuda.device(device):
                analyzer = LensAnalyzer(
                    target_model_config["lens_checkpoint_path"],
                    device=device,
                    batch_size=batch_size,
                    use_bf16=config_with_overrides.get("use_bf16", True),
                    strict_load=False,
                    comparison_tl_checkpoint=config_with_overrides.get("comparison_tl_checkpoint", False),
                    do_not_load_weights=False,
                    make_xl=False,
                    no_orig=no_orig,
                    different_activations_orig=orig_model,
                    initialise_on_cpu=False,
                )
            return analyzer
        
        analyzer = await loop.run_in_executor(None, load_base)
        
        # Extract shared base model
        if hasattr(analyzer, 'shared_base_model'):
            device_state.shared_base_model = analyzer.shared_base_model
        else:
            logger.error(f"No shared_base_model found in analyzer for group {group_id}")
            raise RuntimeError("Failed to extract shared base model")
        
        # Store analyzer
        device_state.lens_cache[target_model_id] = analyzer
        
        # Cache the donor model if it was loaded
        orig_model_path = target_model_config.get("different_activations_orig_path")
        if (orig_model_path and 
            hasattr(analyzer, 'orig_model') and 
            analyzer.orig_model is not None and
            orig_model_path not in self.shared_orig_models):
            logger.info(f"Caching donor model: {orig_model_path}")
            self.shared_orig_models[orig_model_path] = analyzer.orig_model

    async def _load_model_on_device(self, model_id: str, device: str):
        """Load a specific model on a device (group must already be loaded)"""
        device_state = self.device_states[device]
        
        # Create lock for this model if needed
        async with self.global_cache_lock:
            if model_id not in self.model_locks:
                self.model_locks[model_id] = asyncio.Lock()
        
        async with self.model_locks[model_id]:
            # Double-check after acquiring lock
            if model_id in device_state.lens_cache:
                return
            
            # Get model configuration
            group_id = self.model_to_group[model_id]
            group_config = self.model_groups[group_id]
            model_config = next((m for m in group_config.models if m["id"] == model_id), None)
            
            if not model_config:
                raise ValueError(f"Model {model_id} not found in group {group_id}")
            
            # Create analyzer using shared base
            logger.info(f"Loading model {model_id} on device {device}")
            loop = asyncio.get_event_loop()
            
            def create_analyzer():
                # Get donor model
                orig_model_path = model_config.get("different_activations_orig_path")
                orig_model = None
                if orig_model_path:
                    if orig_model_path in self.shared_orig_models:
                        logger.info(f"Using cached donor model: {orig_model_path}")
                        orig_model = self.shared_orig_models[orig_model_path]
                    else:
                        logger.info(f"Donor model not cached, will load: {orig_model_path}")
                        orig_model = orig_model_path
                
                # Get configuration with overrides
                from .config import Settings
                config_with_overrides = Settings.get_model_config_with_overrides(
                    {"settings": self.config_settings},
                    model_id=model_id.replace('-', '_'),
                    group_id=group_id.replace('-', '_')
                )
                
                no_orig = config_with_overrides.get("no_orig", True) and not orig_model_path
                batch_size = model_config.get("batch_size")
                if batch_size is None:
                    batch_size = config_with_overrides.get("batch_size", 32)
                
                # Create analyzer with shared base on specific device
                with torch.cuda.device(device):
                    return LensAnalyzer(
                        model_config["lens_checkpoint_path"],
                        device=device,
                        batch_size=batch_size,
                        use_bf16=config_with_overrides.get("use_bf16", True),
                        strict_load=False,
                        comparison_tl_checkpoint=config_with_overrides.get("comparison_tl_checkpoint", False),
                        do_not_load_weights=False,
                        make_xl=False,
                        no_orig=no_orig,
                        shared_base_model=device_state.shared_base_model,
                        different_activations_orig=orig_model,
                        initialise_on_cpu=False,
                    )
            
            analyzer = await loop.run_in_executor(None, create_analyzer)
            
            # Store in device cache
            device_state.lens_cache[model_id] = analyzer
            
            # Cache donor model if loaded
            orig_model_path = model_config.get("different_activations_orig_path")
            if (orig_model_path and 
                hasattr(analyzer, 'orig_model') and 
                analyzer.orig_model is not None and
                orig_model_path not in self.shared_orig_models):
                logger.info(f"Caching newly loaded donor model: {orig_model_path}")
                self.shared_orig_models[orig_model_path] = analyzer.orig_model

    async def _manage_cpu_cache(self):
        """Manage CPU cache by evicting least recently used groups"""
        if len(self.cpu_cached_groups) >= self.max_cpu_groups:
            # Find LRU group in CPU cache
            lru_group = None
            for group_id in self.group_usage_order:
                if group_id in self.cpu_cached_groups:
                    # Check if this group is currently on any GPU
                    on_gpu = any(state.current_group_id == group_id for state in self.device_states.values())
                    if not on_gpu:
                        lru_group = group_id
                        break
            
            if lru_group:
                logger.info(f"Evicting group {lru_group} from CPU cache")
                del self.cpu_cached_groups[lru_group]
                self.group_usage_order.remove(lru_group)
                gc.collect()

    async def _broadcast_group_switch_status(self, status: str, group_id: str, device: str, error: Optional[str] = None, source_group_id: Optional[str] = None):
        """Broadcast group switch status to all connected clients"""
        message = {
            "type": "group_switch_status",
            "status": status,
            "group_id": group_id,
            "device": device,
            "timestamp": datetime.now().isoformat(),
        }
        if error:
            message["error"] = error
        if source_group_id:
            message["source_group_id"] = source_group_id
        
        logger.info(f"Broadcasting group switch status: {status} for group {group_id} on device {device}")
        
        if self.websocket_manager:
            await self.websocket_manager.broadcast(message)

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        group_id = self.model_to_group.get(model_id)
        if not group_id:
            return {"status": "invalid_model", "error": f"Unknown model ID: {model_id}"}

        group_config = self.model_groups[group_id]
        model_config = next((m for m in group_config.models if m["id"] == model_id), None)
        if not model_config:
            return {"status": "model_not_found", "error": f"Model {model_id} not found in group {group_id}"}

        # Get checkpoint info
        checkpoint_path = model_config.get("lens_checkpoint_path", "")
        if checkpoint_path:
            path_parts = checkpoint_path.split('/')
            checkpoint_folder = path_parts[-1] if path_parts else ""
            if len(checkpoint_folder) > 40:
                checkpoint_filename = f"{checkpoint_folder[:20]}...{checkpoint_folder[-15:]}"
            else:
                checkpoint_filename = checkpoint_folder
        else:
            checkpoint_filename = "Unknown"
        
        # Get donor model info
        donor_model_path = model_config.get("different_activations_orig_path")
        donor_model_name = "Same as base model"
        if donor_model_path:
            donor_model_name = donor_model_path.split('/')[-1] if '/' in donor_model_path else donor_model_path
        
        # Get device information
        loaded_on_devices = []
        for device_id, device_state in self.device_states.items():
            if model_id in device_state.lens_cache:
                loaded_on_devices.append(device_id)
        
        # Get configuration with overrides
        from .config import Settings
        config_with_overrides = Settings.get_model_config_with_overrides(
            {"settings": self.config_settings},
            model_id=model_id.replace('-', '_'),
            group_id=group_id.replace('-', '_')
        )
        
        batch_size = model_config.get("batch_size")
        if batch_size is None:
            batch_size = config_with_overrides.get("batch_size", 32)
        
        auto_batch_size_max = model_config.get("auto_batch_size_max")
        if auto_batch_size_max is None:
            auto_batch_size_max = config_with_overrides.get("auto_batch_size_max", 512)
        
        return {
            "model_id": model_id,
            "group_id": group_id,
            "group_name": group_config.group_name,
            "display_name": model_config["name"],
            "description": model_config.get("description", ""),
            "layer": model_config.get("layer", 30),
            "batch_size": batch_size,
            "auto_batch_size_max": auto_batch_size_max,
            "base_model": group_config.base_model_path,
            "donor_model": donor_model_name,
            "donor_model_path": donor_model_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint_filename": checkpoint_filename,
            "loaded_on_devices": loaded_on_devices,
            "last_used": self.model_last_used.get(model_id, "").isoformat() if model_id in self.model_last_used else None,
            "device_status": self.get_device_status()
        }

    async def preload_model(self, model_id: str, device: Optional[str] = None) -> Dict[str, Any]:
        """Preload a model into memory"""
        try:
            analyzer, device_used = await self.get_analyzer_for_model(model_id, device)
            model_info = self.get_model_info(model_id)
            return {
                "status": "success",
                "model_id": model_id,
                "device": device_used,
                "message": f"Model {model_info['display_name']} loaded successfully on {device_used}"
            }
        except Exception as e:
            logger.error(f"Failed to preload model {model_id}: {e}")
            return {
                "status": "failed",
                "model_id": model_id,
                "error": str(e)
            }

    async def clear_all_models(self):
        """Clear all models from all devices"""
        logger.info("Clearing all models from memory")
        
        # Clear each device
        for device_id, device_state in self.device_states.items():
            device_state.lens_cache.clear()
            device_state.shared_base_model = None
            device_state.current_group_id = None
        
        # Clear CPU cache
        self.cpu_cached_groups.clear()
        
        # Clear shared caches
        self.shared_orig_models.clear()
        self.model_last_used.clear()
        self.group_usage_order.clear()
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
        
        # Clear CUDA cache on all devices
        if torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        logger.info("All models cleared from memory")

    async def ensure_group_loaded(self):
        """Ensure at least one group is loaded on the first device"""
        if self.model_groups and not any(state.current_group_id for state in self.device_states.values()):
            # Load first group on first device
            first_group_id = next(iter(self.model_groups.keys()))
            await self._switch_device_to_group(self.devices[0], first_group_id)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current GPU memory usage for all devices"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        device_memory = {}
        for device in self.devices:
            device_num = int(device.split(':')[1]) if ':' in device else 0
            with torch.cuda.device(device_num):
                device_state = self.device_states[device]
                device_memory[device] = {
                    "allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
                    "reserved_memory_gb": torch.cuda.memory_reserved() / 1e9,
                    "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "current_group": device_state.current_group_id,
                    "models_loaded": len(device_state.lens_cache)
                }
        
        return {
            "gpu_available": True,
            "devices": device_memory,
            "cpu_cached_groups": list(self.cpu_cached_groups.keys()),
            "total_models_in_memory": sum(len(state.lens_cache) for state in self.device_states.values())
        }

    def get_chat_tokenizer_for_model(self, model_id: str):
        """Get the chat tokenizer for a specific model if available"""
        if model_id not in self.model_to_group:
            return None
        
        group_id = self.model_to_group[model_id]
        group_config = self.model_groups[group_id]
        model_config = next((m for m in group_config.models if m["id"] == model_id), None)
        
        if model_config:
            tokenizer_path = (model_config.get("different_activations_orig_path") or
                              group_config.base_model_path)
            if 'gemma' in tokenizer_path and '-pt' in tokenizer_path:
                tokenizer_path = tokenizer_path.replace('-pt', '-it')
            elif 'google/gemma-2-9b' in tokenizer_path:
                tokenizer_path = tokenizer_path.replace('google/gemma-2-9b', 'google/gemma-2-9b-it')
            if tokenizer_path not in self.chat_tokenizers:
                try:
                    from transformers import AutoTokenizer
                    logger.info(f"Loading chat tokenizer from: {tokenizer_path}")
                    self.chat_tokenizers[tokenizer_path] = AutoTokenizer.from_pretrained(tokenizer_path)
                except Exception as e:
                    logger.warning(f"Failed to load chat tokenizer: {e}")
                    return None
            return self.chat_tokenizers.get(tokenizer_path)
        return None

    async def get_default_model_id(self) -> Optional[str]:
        """Get the ID of a default model to use"""
        # Check if any device has a group loaded
        for device_state in self.device_states.values():
            if device_state.current_group_id and device_state.current_group_id in self.model_groups:
                group = self.model_groups[device_state.current_group_id]
                if group.models:
                    return group.models[0]["id"]
        
        # Otherwise return first model from first group
        if self.model_groups:
            first_group = next(iter(self.model_groups.values()))
            if first_group.models:
                return first_group.models[0]["id"]
        return None
    
    def _check_memory_available(self, device: str, required_gb: float) -> bool:
        """
        Check if a device has enough free memory for a model/group.
        
        Args:
            device: Device ID (e.g., "cuda:0")
            required_gb: Required memory in GB
            
        Returns:
            True if enough memory is available, False otherwise
        """
        if not torch.cuda.is_available() or required_gb is None:
            return True  # Assume enough if no CUDA or no estimate
        
        try:
            device_num = int(device.split(':')[1])
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_num)
            free_gb = free_bytes / 1e9
            
            # Add a buffer of 1GB for safety
            required_with_buffer = required_gb + 1.0
            
            logger.info(f"Device {device} has {free_gb:.2f}GB free, requires {required_with_buffer:.2f}GB (including 1GB buffer)")
            return free_gb > required_with_buffer
        except Exception as e:
            logger.error(f"Failed to check memory on {device}: {e}")
            return False  # Be conservative on error
    
    async def consider_duplicating_group(
        self, 
        group_id: str, 
        device_load_stats: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """
        Consider duplicating a group to another GPU for load balancing.
        
        Returns the device ID if duplication occurred, None otherwise.
        """
        # Check if group exists
        if group_id not in self.model_groups:
            return None
        
        # Find devices that currently have this group
        devices_with_group = [
            device for device, state in self.device_states.items()
            if state.current_group_id == group_id and not state.is_switching
        ]
        
        if not devices_with_group:
            return None
        
        # Check load on devices with the group
        max_load = 0.0
        busiest_device = None
        for device in devices_with_group:
            if device in device_load_stats:
                stats = device_load_stats[device]
                # Consider a device busy if queue depth > 3 or utilization > 0.7
                load_score = stats["queue_depth"] * 0.5 + stats["utilization"] * 10
                if load_score > max_load:
                    max_load = load_score
                    busiest_device = device
        
        # Only consider duplication if load is high
        if max_load < 5.0:  # Threshold for considering duplication
            return None
        
        # Find an idle GPU that doesn't have this group
        best_idle_device = None
        min_load = float('inf')
        
        for device, state in self.device_states.items():
            # Skip devices that already have this group or are switching
            if state.current_group_id == group_id or state.is_switching:
                continue
            
            # Check if device has enough memory
            group = self.model_groups[group_id]
            memory_needed = group.memory_estimate
            if not self._check_memory_available(device, memory_needed):
                continue
            
            # Check device load
            if device in device_load_stats:
                stats = device_load_stats[device]
                load_score = stats["queue_depth"] * 0.5 + stats["utilization"] * 10
                if load_score < min_load and load_score < 2.0:  # Threshold for idle
                    min_load = load_score
                    best_idle_device = device
        
        # If we found a good idle device, duplicate the group
        if best_idle_device:
            logger.info(f"Duplicating group {group_id} from {busiest_device} (load={max_load:.2f}) to {best_idle_device} (load={min_load:.2f})")
            await self._switch_device_to_group(best_idle_device, group_id)
            return best_idle_device
        
        return None
    
    async def _unload_group(self, group_id: str):
        """
        Unload a group from the first device for backward compatibility.
        This method is kept for compatibility with the old API.
        """
        if self.devices:
            first_device = self.devices[0]
            device_state = self.device_states[first_device]
            if device_state.current_group_id == group_id:
                await self._move_group_to_cpu(group_id, first_device)
                device_state.current_group_id = None
    
    async def _switch_to_group(self, target_group_id: str):
        """
        Switch to a group on the first device for backward compatibility.
        This method is kept for compatibility with the old API.
        """
        if self.devices:
            first_device = self.devices[0]
            await self._switch_device_to_group(first_device, target_group_id)
    
    async def start(self):
        """Start the model manager (compatibility method)"""
        logger.info("Model manager started")
        # The new manager doesn't need a cleanup loop as it manages resources differently
    
    async def stop(self):
        """Stop the model manager (compatibility method)"""
        logger.info("Model manager stopped")
        # Clean up resources if needed
        await self.clear_all_models()
    
    def set_websocket_manager(self, manager):
        """Sets the WebSocket manager for broadcasting updates."""
        self.websocket_manager = manager
        logger.info("WebSocket manager set in UnifiedModelManager")

    async def load_group(self, group_id: str, device_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model group onto a specific or best-available device.
        This ensures the group's base model is loaded and ready.
        """
        if group_id not in self.model_groups:
            raise ValueError(f"Unknown group ID: {group_id}")

        # Find best device if not specified
        if device_id is None:
            # Find a device that is either empty or already has the group
            available_devices = [
                d for d, s in self.device_states.items()
                if s.current_group_id == group_id or s.current_group_id is None
            ]
            if not available_devices:
                # If no ideal device, pick the first device
                device_id = self.devices[0] if self.devices else "cuda:0"
            else:
                device_id = available_devices[0]
        
        elif device_id not in self.device_states:
            raise ValueError(f"Unknown device: {device_id}")

        # Switch the device to the target group
        await self._switch_device_to_group(device_id, group_id)

        # Pre-load first model in the group to ensure everything is ready
        group_config = self.model_groups[group_id]
        if group_config.models:
            first_model_id = group_config.models[0]["id"]
            if first_model_id not in self.device_states[device_id].lens_cache:
                await self._load_model_on_device(first_model_id, device_id)

        # Broadcast the state update
        await self._broadcast_system_state_update()
        
        return {
            "group_id": group_id,
            "device_id": device_id,
            "message": f"Group {group_id} is ready on {device_id}"
        }

    async def unload_group(self, group_id: str):
        """Unload a group from all devices it might be loaded on."""
        logger.info(f"Unloading group {group_id} from all devices.")
        for device_id, state in self.device_states.items():
            if state.current_group_id == group_id:
                await self._move_group_to_cpu(group_id, device_id)
                state.current_group_id = None
        
        # Broadcast the state update
        await self._broadcast_system_state_update()
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get a comprehensive state of the system for the frontend"""
        devices_state = []
        for device_id, state in self.device_states.items():
            current_group_info = None
            if state.current_group_id:
                group_config = self.model_groups.get(state.current_group_id)
                if group_config:
                    current_group_info = {
                        "id": group_config.group_id,
                        "name": group_config.group_name,
                    }
            
            devices_state.append({
                "device": device_id,
                "current_group": current_group_info,
                "is_switching": state.is_switching,
                "loaded_models": list(state.lens_cache.keys())
            })
        
        groups_info = []
        for group_id, group_config in self.model_groups.items():
            groups_info.append({
                "id": group_id,
                "name": group_config.group_name,
                "base_model": group_config.base_model_path,
                "memory_estimate": group_config.memory_estimate
            })
        
        return {
            "devices": devices_state,
            "groups": groups_info,
            "cpu_cached_groups": list(self.cpu_cached_groups.keys())
        }
    
    def get_model_location(self, model_id: str) -> Optional[List[Dict[str, str]]]:
        """Get the location(s) of a model (GPU or CPU)"""
        locations = []
        
        # Check GPU devices
        for device_id, state in self.device_states.items():
            if model_id in state.lens_cache:
                locations.append({"device_id": device_id, "status": "loaded"})
        
        # Check CPU cache
        group_id = self.model_to_group.get(model_id)
        if group_id and group_id in self.cpu_cached_groups:
            group_cache = self.cpu_cached_groups[group_id]
            if "models" in group_cache and model_id in group_cache["models"]:
                locations.append({"device_id": "cpu", "status": "cached"})
        
        return locations if locations else None
    
    async def clear_device(self, device_id: str) -> Dict[str, Any]:
        """Clear all models from a specific device"""
        if device_id not in self.device_states:
            raise ValueError(f"Unknown device: {device_id}")
        
        device_state = self.device_states[device_id]
        
        async with device_state.lock:
            if device_state.current_group_id:
                logger.info(f"Clearing group {device_state.current_group_id} from device {device_id}")
                
                # Move to CPU cache if not already there
                if device_state.current_group_id not in self.cpu_cached_groups:
                    await self._move_group_to_cpu(device_state.current_group_id, device_id)
                else:
                    # Just clear from device
                    device_state.shared_base_model = None
                    device_state.lens_cache.clear()
                
                device_state.current_group_id = None
                
                # Clear GPU memory
                if torch.cuda.is_available() and device_id.startswith("cuda:"):
                    device_num = int(device_id.split(':')[1])
                    with torch.cuda.device(device_num):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                logger.info(f"Device {device_id} cleared")
            else:
                logger.info(f"Device {device_id} is already clear")
        
        # Broadcast the state update (after lock is released)
        await self._broadcast_system_state_update()
        
        return {"status": "success", "device_id": device_id}