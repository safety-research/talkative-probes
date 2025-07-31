"""Enhanced Model Manager with support for grouped models sharing base models"""

import asyncio
import torch
import gc
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path to import lens module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lens.analysis.analyzer_class import LensAnalyzer
from .config import Settings

logger = logging.getLogger(__name__)

class ModelGroupConfig:
    """Configuration for a model group"""
    def __init__(self, group_data: Dict[str, Any]):
        self.group_id = group_data["group_id"]
        self.group_name = group_data["group_name"]
        self.base_model_path = group_data["base_model_path"]
        self.description = group_data.get("description", "")
        self.models = group_data["models"]

class GroupedModelManager:
    """Manages models grouped by shared base models for efficient switching"""

    def __init__(self, settings: Settings, groups_config_path: Optional[str] = None):
        self.settings = settings
        self.websocket_manager = None

        # Load groups configuration
        if groups_config_path is None:
            groups_config_path = Path(__file__).parent / "model_groups.json"
        self.groups_config_path = Path(groups_config_path)
        self._load_groups_config()

        # Current group state (global - only one group on GPU)
        self.current_group_id: Optional[str] = None
        self.is_switching_group = False
        self.group_switch_lock = asyncio.Lock()
        self.switch_start_time: Optional[datetime] = None
        self.queued_requests: List[asyncio.Future] = []

        # Thread-safe locks for model access
        self.model_locks: Dict[str, asyncio.Lock] = {}  # model_id -> lock for loading
        self.cache_lock = asyncio.Lock()  # Lock for accessing caches
        
        # Shared base model management
        self.shared_base_models: Dict[str, Any] = {}  # group_id -> base model
        self.shared_orig_models: Dict[str, Any] = {}  # model_path -> orig model wrapper
        self.lens_cache: Dict[str, LensAnalyzer] = {}  # model_id -> LensAnalyzer
        self.base_model_locations: Dict[str, str] = {}  # group_id -> 'cuda' or 'cpu'
        self.model_last_used: Dict[str, datetime] = {}  # model_id -> last used time

        # Memory management
        self.max_cpu_groups = getattr(settings, 'max_cpu_cached_models', 2)
        self.group_usage_order: List[str] = []  # LRU tracking

        # Chat tokenizers
        self.chat_tokenizers: Dict[str, Any] = {}  # model_path -> tokenizer

    def _load_groups_config(self):
        """Load model groups configuration from JSON"""
        try:
            with open(self.groups_config_path, 'r') as f:
                data = json.load(f)

            self.model_groups = {}
            self.model_to_group = {}

            for group_data in data.get("model_groups", []):
                group_config = ModelGroupConfig(group_data)
                self.model_groups[group_config.group_id] = group_config
                for model in group_config.models:
                    self.model_to_group[model["id"]] = group_config.group_id

            self.config_settings = data.get("settings", {})
            logger.info(f"Loaded {len(self.model_groups)} model groups with {len(self.model_to_group)} total models")
        except Exception as e:
            logger.error(f"Failed to load model groups config: {e}")
            raise

    def get_model_list(self) -> List[Dict[str, Any]]:
        """Get structured list of model groups for frontend"""
        # This method should not block on group switches - it's just reading state
        groups = []
        for group_id, group_config in self.model_groups.items():
            group_data = {
                "group_id": group_config.group_id,
                "group_name": group_config.group_name,
                "description": group_config.description,
                "base_model": group_config.base_model_path,
                "is_current": group_id == self.current_group_id,
                "is_loaded": group_id in self.shared_base_models,
                "location": self.base_model_locations.get(group_id, "not_loaded"),
                "models": []
            }
            for model in group_config.models:
                # Extract checkpoint folder name for display
                checkpoint_path = model.get("lens_checkpoint_path", "")
                checkpoint_filename = "Unknown"
                checkpoint_full = ""
                if checkpoint_path:
                    path_parts = checkpoint_path.split('/')
                    checkpoint_folder = path_parts[-1] if path_parts else ""
                    checkpoint_full = checkpoint_folder  # Store full name for tooltip
                    if len(checkpoint_folder) > 40:
                        checkpoint_filename = f"{checkpoint_folder[:20]}...{checkpoint_folder[-15:]}"
                    else:
                        checkpoint_filename = checkpoint_folder
                        
                model_info = {
                    "id": model["id"],
                    "name": model["name"],
                    "description": model.get("description", ""),
                    "layer": model.get("layer", 30),
                    "checkpoint_filename": checkpoint_filename,
                    "checkpoint_full": checkpoint_full,  # Full name for tooltip
                    "is_loaded": model["id"] in self.lens_cache,
                    "is_available": group_id == self.current_group_id,  # Can use without group switch
                    "last_used": self.model_last_used.get(model["id"], "").isoformat() if model["id"] in self.model_last_used else None
                }
                group_data["models"].append(model_info)
            groups.append(group_data)
        return groups

    def get_current_group_info(self) -> Dict[str, Any]:
        """Get information about the current group"""
        return {
            "current_group_id": self.current_group_id,
            "is_switching": self.is_switching_group,
            "switch_start_time": self.switch_start_time.isoformat() if self.switch_start_time else None,
            "groups_loaded": list(self.shared_base_models.keys()),
            "base_locations": self.base_model_locations,
        }
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        group_id = self.model_to_group.get(model_id)
        if not group_id:
            return {"status": "invalid_model", "error": f"Unknown model ID: {model_id}"}

        group_config = self.model_groups[group_id]
        model_config = next((m for m in group_config.models if m["id"] == model_id), None)
        if not model_config:
            return {"status": "model_not_found", "error": f"Model {model_id} not found in group {group_id}"}

        checkpoint_path = model_config.get("lens_checkpoint_path", "")
        # Extract folder name and checkpoint file from path
        if checkpoint_path:
            path_parts = checkpoint_path.split('/')
            # Get the checkpoint folder name (last part of path)
            checkpoint_folder = path_parts[-1] if path_parts else ""
            # For display, show shortened version if too long
            if len(checkpoint_folder) > 40:
                # Show first 20 and last 15 characters for long names
                checkpoint_filename = f"{checkpoint_folder[:20]}...{checkpoint_folder[-15:]}"
            else:
                checkpoint_filename = checkpoint_folder
        else:
            checkpoint_filename = "Unknown"
            
        # Get donor model info
        donor_model_path = model_config.get("different_activations_orig_path")
        donor_model_name = "Same as base model"
        if donor_model_path:
            # Extract a readable name from the path
            donor_model_name = donor_model_path.split('/')[-1] if '/' in donor_model_path else donor_model_path
        
        # Get generation config if model is loaded
        generation_config = {}
        if model_id in self.lens_cache:
            generation_config = self._get_generation_config_for_model(model_id)
        
        return {
            "model_id": model_id,
            "group_id": group_id,
            "group_name": group_config.group_name,
            "display_name": model_config["name"],
            "description": model_config.get("description", ""),
            "layer": model_config.get("layer", 30),
            "batch_size": model_config.get("batch_size", self.settings.batch_size),
            "auto_batch_size_max": model_config.get("auto_batch_size_max", self.settings.auto_batch_size_max),
            "base_model": group_config.base_model_path,
            "donor_model": donor_model_name,
            "donor_model_path": donor_model_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint_filename": checkpoint_filename,
            "is_loaded": model_id in self.lens_cache,
            "last_used": self.model_last_used.get(model_id, "").isoformat() if model_id in self.model_last_used else None,
            "cache_info": {
                "groups_loaded": list(self.shared_base_models.keys()),
                "models_cached": list(self.lens_cache.keys()),
                "donor_models_cached": list(self.shared_orig_models.keys()),
                "base_locations": self.base_model_locations,
            },
            "generation_config": generation_config,
        }

    def _get_generation_config_for_model(self, model_id: str) -> Dict[str, Any]:
        """Get generation config for a specific model"""
        if model_id not in self.lens_cache:
            logger.debug(f"Model {model_id} not loaded, no generation config available")
            return {}
            
        analyzer = self.lens_cache[model_id]
        try:
            # First check if the analyzer has orig_model.model
            if hasattr(analyzer, 'orig_model'):
                logger.debug(f"Analyzer for {model_id} has orig_model: {type(analyzer.orig_model)}")
                if hasattr(analyzer.orig_model, 'model'):
                    model = analyzer.orig_model.model
                    logger.debug(f"Found model object: {type(model)}")
                    if hasattr(model, 'generation_config'):
                        gen_config = model.generation_config
                        temperature = getattr(gen_config, 'temperature', 1.0)
                        top_p = getattr(gen_config, 'top_p', 1.0)
                        # Log the generation config for debugging
                        logger.info(f"Model {model_id} generation config: temperature={temperature}, top_p={top_p}")
                        return {
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": getattr(gen_config, 'top_k', None),
                            "do_sample": getattr(gen_config, 'do_sample', True),
                            "repetition_penalty": getattr(gen_config, 'repetition_penalty', 1.0),
                        }
                    else:
                        logger.debug("Model has no generation_config attribute")
                else:
                    logger.debug("orig_model has no model attribute")
            else:
                logger.debug("Analyzer has no orig_model attribute")
        except Exception as e:
            logger.warning(f"Failed to get generation config for {model_id}: {e}", exc_info=True)
        
        # Default values - just use standard defaults
        logger.info(f"Using default generation config for {model_id}")
        return {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "do_sample": True,
            "repetition_penalty": 1.0,
        }

    async def get_analyzer_for_model(self, model_id: str) -> LensAnalyzer:
        """Get analyzer for a specific model, loading if necessary"""
        # Check if model exists
        if model_id not in self.model_to_group:
            raise ValueError(f"Unknown model ID: {model_id}")
        
        target_group_id = self.model_to_group[model_id]
        
        # Check if we're in the wrong group
        if target_group_id != self.current_group_id:
            # This should not happen anymore - group switches go through the queue
            raise RuntimeError(f"Model {model_id} requires group {target_group_id} but current group is {self.current_group_id}. Group switch should have been queued.")
        
        # Wait if a group switch is in progress
        while self.is_switching_group:
            future = asyncio.Future()
            async with self.group_switch_lock:
                if self.is_switching_group:  # Double-check
                    self.queued_requests.append(future)
                else:
                    break
            try:
                await asyncio.wait_for(future, timeout=300)  # 5-minute timeout
            except asyncio.TimeoutError:
                raise RuntimeError("Group switch timeout")
            
        # Update last used time
        self.model_last_used[model_id] = datetime.now()
        
        # Fast path: already loaded
        async with self.cache_lock:
            if model_id in self.lens_cache:
                return self.lens_cache[model_id]
        
        # Need to load the model within current group - fix lock creation race
        async with self.cache_lock:
            if model_id not in self.model_locks:
                self.model_locks[model_id] = asyncio.Lock()
            model_lock = self.model_locks[model_id]
            
        async with model_lock:
            # Double-check after acquiring lock
            async with self.cache_lock:
                if model_id in self.lens_cache:
                    return self.lens_cache[model_id]
            
            # Load the model
            logger.info(f"Loading model {model_id} within group {self.current_group_id}")
            group_config = self.model_groups[self.current_group_id]
            model_config = next((m for m in group_config.models if m["id"] == model_id), None)
            
            if not model_config:
                raise ValueError(f"Model {model_id} not found in group {self.current_group_id}")
            
            # Create analyzer for this specific model (group is already on GPU)
            await self._create_model_analyzer(model_id, model_config, self.current_group_id)
            
            # Return the loaded analyzer
            return self.lens_cache[model_id]

    async def preload_model(self, model_id: str) -> Dict[str, Any]:
        """Preload a model into memory (useful for warming up)"""
        try:
            analyzer = await self.get_analyzer_for_model(model_id)
            model_info = self.get_model_info(model_id)
            return {
                "status": "success",
                "model_id": model_id,
                "message": f"Model {model_info['display_name']} loaded successfully"
            }
        except Exception as e:
            logger.error(f"Failed to preload model {model_id}: {e}")
            return {
                "status": "failed",
                "model_id": model_id,
                "error": str(e)
            }

    async def _create_model_analyzer(self, model_id: str, model_config: Dict[str, Any], group_id: str):
        """Create analyzer for a specific model within a group"""
        logger.info(f"Creating new lens analyzer for {model_id} with shared base")
        loop = asyncio.get_event_loop()
        shared_base = self.shared_base_models[group_id]
        
        # Handle donor model
        orig_model_path = model_config.get("different_activations_orig_path")
        logger.info(f"Model {model_id} requires donor model: {orig_model_path}")
        
        orig_model = None
        if orig_model_path:
            if orig_model_path in self.shared_orig_models:
                logger.info(f"Using cached donor model: {orig_model_path}")
                orig_model = self.shared_orig_models[orig_model_path]
            else:
                logger.info(f"Donor model not cached, will load: {orig_model_path}")
                orig_model = orig_model_path
        
        def create_analyzer():
            logger.info(f"Creating LensAnalyzer for {model_id}")
            logger.info(f"  - lens_checkpoint: {model_config['lens_checkpoint_path']}")
            logger.info(f"  - shared_base_model: {type(shared_base)}")
            logger.info(f"  - different_activations_orig: {orig_model_path} (type: {type(orig_model)})")
            
            # If we have a different donor model, we can't use no_orig
            no_orig = self.config_settings.get("no_orig", True) and not orig_model_path
            
            return LensAnalyzer(
                model_config["lens_checkpoint_path"],
                device=self.settings.device,
                batch_size=model_config.get("batch_size", self.settings.batch_size),
                use_bf16=self.config_settings.get("use_bf16", True),
                strict_load=False,
                comparison_tl_checkpoint=self.config_settings.get("comparison_tl_checkpoint", False),
                do_not_load_weights=False,
                make_xl=False,
                no_orig=no_orig,
                shared_base_model=shared_base,
                different_activations_orig=orig_model,
                initialise_on_cpu=False,
            )
        
        analyzer = await loop.run_in_executor(None, create_analyzer)
        
        # Store in cache
        async with self.cache_lock:
            self.lens_cache[model_id] = analyzer
        
        # Cache the donor model if it was loaded
        if (orig_model_path and 
            hasattr(analyzer, 'orig_model') and 
            analyzer.orig_model is not None and
            orig_model_path not in self.shared_orig_models):
            logger.info(f"Caching newly loaded donor model: {orig_model_path}")
            self.shared_orig_models[orig_model_path] = analyzer.orig_model

    async def _switch_to_group(self, target_group_id: str):
        """Switch to a different group (blocking operation)"""
        async with self.group_switch_lock:
            if self.current_group_id == target_group_id:
                return  # Already in target group
                
            try:
                self.is_switching_group = True
                self.switch_start_time = datetime.now()
                
                # Broadcast switch starting
                await self._broadcast_group_switch_status("starting", target_group_id)
                
                # Move current group to CPU if exists
                if self.current_group_id and self.current_group_id in self.shared_base_models:
                    logger.info(f"Moving group {self.current_group_id} to CPU")
                    await self._move_group_to_cpu(self.current_group_id)
                
                # Load or move target group to GPU
                if target_group_id in self.shared_base_models:
                    logger.info(f"Moving group {target_group_id} from CPU to GPU")
                    await self._move_group_to_gpu(target_group_id)
                else:
                    logger.info(f"Loading new group {target_group_id}")
                    # Load first model in group to establish base
                    group_config = self.model_groups[target_group_id]
                    first_model = group_config.models[0]
                    await self._load_group_base(target_group_id, first_model["id"], first_model)
                
                # Update current group
                self.current_group_id = target_group_id
                
                # Update LRU order
                if target_group_id in self.group_usage_order:
                    self.group_usage_order.remove(target_group_id)
                self.group_usage_order.append(target_group_id)
                
                # Broadcast completion
                await self._broadcast_group_switch_status("completed", target_group_id)
                
            except Exception as e:
                logger.error(f"Group switch failed: {e}")
                # Try to restore previous state
                self.current_group_id = self.current_group_id  # Keep old group
                await self._broadcast_group_switch_status("failed", target_group_id, str(e))
                raise
                
            finally:
                self.is_switching_group = False
                self.switch_start_time = None
                
                # Resume queued requests safely
                queued = self.queued_requests[:]  # Copy list
                self.queued_requests.clear()
                for future in queued:
                    if not future.done():
                        future.set_result(None)

    async def _load_group_base(self, group_id: str, target_model_id: str, target_model_config: Dict[str, Any]):
        """Load a group's base model using the specified model"""
        group_config = self.model_groups[group_id]
        await self._manage_cpu_cache()
        loop = asyncio.get_event_loop()
        def load_base():
            # Use the target model instead of always using the first model
            model_config = target_model_config
            model_id = target_model_id
            
            # Handle donor model for the target model
            orig_model_path = model_config.get("different_activations_orig_path")
            logger.info(f"Loading model {model_id} with orig_model_path: {orig_model_path}")
            
            # Check if we already have this donor model cached
            orig_model = None
            if orig_model_path and orig_model_path in self.shared_orig_models:
                logger.info(f"Using cached donor model: {orig_model_path}")
                orig_model = self.shared_orig_models[orig_model_path]
            else:
                # Pass the path and let LensAnalyzer load it
                orig_model = orig_model_path
                
            # If we have a different donor model, we can't use no_orig
            no_orig = self.config_settings.get("no_orig", True) and not orig_model_path
            
            analyzer = LensAnalyzer(
                model_config["lens_checkpoint_path"],
                device=self.settings.device,
                batch_size=model_config.get("batch_size", self.settings.batch_size),
                use_bf16=self.config_settings.get("use_bf16", True),
                strict_load=False,
                comparison_tl_checkpoint=self.config_settings.get("comparison_tl_checkpoint", False),
                do_not_load_weights=False,
                make_xl=False,
                no_orig=no_orig,
                different_activations_orig=orig_model,  # Pass the donor model
                initialise_on_cpu=False,
            )
            return analyzer
        analyzer = await loop.run_in_executor(None, load_base)
        
        # Extract shared base model
        if hasattr(analyzer, 'shared_base_model'):
            shared_base_model = analyzer.shared_base_model
        else:
            logger.error(f"No shared_base_model found in analyzer for group {group_id}")
            raise RuntimeError("Failed to extract shared base model")
        
        # Protect all writes to shared state with cache_lock
        async with self.cache_lock:
            self.shared_base_models[group_id] = shared_base_model
            self.base_model_locations[group_id] = 'cuda'
            self.lens_cache[target_model_id] = analyzer
            
            # Cache the donor model if it was loaded
            orig_model_path = target_model_config.get("different_activations_orig_path")
            if (orig_model_path and 
                hasattr(analyzer, 'orig_model') and 
                analyzer.orig_model is not None and
                orig_model_path not in self.shared_orig_models):
                logger.info(f"Caching donor model: {orig_model_path}")
                self.shared_orig_models[orig_model_path] = analyzer.orig_model

    async def _move_group_to_cpu(self, group_id: str):
        """Move a group's base model to CPU"""
        if group_id not in self.shared_base_models:
            return
        logger.info(f"Moving group {group_id} to CPU")
        group_config = self.model_groups[group_id]
        for model in group_config.models:
            if model["id"] in self.lens_cache:
                analyzer = self.lens_cache[model["id"]]
                analyzer.to('cpu')
        self.base_model_locations[group_id] = 'cpu'
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def _move_group_to_gpu(self, group_id: str):
        """Move a group's base model to GPU"""
        if group_id not in self.shared_base_models:
            return
        logger.info(f"Moving group {group_id} to GPU")
        loop = asyncio.get_event_loop()
        group_config = self.model_groups[group_id]
        for model in group_config.models:
            if model["id"] in self.lens_cache:
                analyzer = self.lens_cache[model["id"]]
                await loop.run_in_executor(None, lambda: analyzer.to(self.settings.device))
        self.base_model_locations[group_id] = 'cuda'

    async def _manage_cpu_cache(self):
        """Manage CPU cache by evicting least recently used groups"""
        cpu_groups = [g for g, loc in self.base_model_locations.items() if loc == 'cpu']
        if len(cpu_groups) >= self.max_cpu_groups:
            lru_group = None
            for group_id in self.group_usage_order:
                if self.base_model_locations.get(group_id) == 'cpu':
                    lru_group = group_id
                    break
            if lru_group:
                logger.info(f"Evicting group {lru_group} from CPU cache")
                group_config = self.model_groups[lru_group]
                for model in group_config.models:
                    if model["id"] in self.lens_cache:
                        del self.lens_cache[model["id"]]
                if lru_group in self.shared_base_models:
                    del self.shared_base_models[lru_group]
                del self.base_model_locations[lru_group]
                self.group_usage_order.remove(lru_group)
                gc.collect()

    async def _broadcast_group_switch_status(self, status: str, group_id: str, error: Optional[str] = None):
        """Broadcast group switch status to all connected clients"""
        message = {
            "type": "group_switch_status",
            "status": status,
            "group_id": group_id,
            "timestamp": datetime.now().isoformat(),
        }
        if error:
            message["error"] = error
        
        logger.info(f"Broadcasting group switch status: {status} for group {group_id}")
        
        if self.websocket_manager:
            await self.websocket_manager.broadcast(message)
            logger.info(f"Broadcast sent to {len(self.websocket_manager.active_connections)} connections")
        else:
            logger.warning(f"No websocket manager available for broadcast: {message}")

    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a specific model is loaded"""
        return model_id in self.lens_cache

    async def get_default_model_id(self) -> Optional[str]:
        """Get the ID of a default model to use"""
        # If we have a current group, return first model from it
        if self.current_group_id and self.current_group_id in self.model_groups:
            group = self.model_groups[self.current_group_id]
            if group.models:
                return group.models[0]["id"]
        
        # Otherwise return first model from first group
        if self.model_groups:
            first_group = next(iter(self.model_groups.values()))
            if first_group.models:
                return first_group.models[0]["id"]
        return None
    
    async def ensure_group_loaded(self):
        """Ensure at least one group is loaded"""
        if not self.current_group_id and self.model_groups:
            # Load first group
            first_group_id = next(iter(self.model_groups.keys()))
            await self._switch_to_group(first_group_id)

    async def preload_group(self, group_id: str):
        """Preload all models in a group"""
        if group_id not in self.model_groups:
            raise ValueError(f"Unknown group ID: {group_id}")
        group_config = self.model_groups[group_id]
        
        logger.info(f"Preloading all models in group {group_id}")
        # Load each model in the group
        for model_config in group_config.models:
            model_id = model_config["id"]
            if not self.is_model_loaded(model_id):
                await self.get_analyzer_for_model(model_id)
        
        logger.info(f"Preloaded {len(group_config.models)} models in group {group_id}")
    
    async def preload_all_groups(self, default_group: Optional[str] = None):
        """Preload all groups by cycling through them, ending with the default group"""
        logger.info("Starting preload of all model groups")
        
        # Get default group from settings if not provided
        if not default_group and hasattr(self, 'config_settings'):
            default_group = self.config_settings.get('default_group', 'gemma3-27b-it')
        elif not default_group:
            default_group = 'gemma3-27b-it'
        
        # List of all groups except the default
        all_groups = list(self.model_groups.keys())
        other_groups = [g for g in all_groups if g != default_group]
        
        # First, cycle through all non-default groups
        for group_id in other_groups:
            logger.info(f"Preloading group {group_id} to CPU by loading to GPU first")
            try:
                # Switch to the group (loads to GPU)
                await self._switch_to_group(group_id)
                # The group will be moved to CPU when we switch to the next one
            except Exception as e:
                logger.error(f"Failed to preload group {group_id}: {e}")
        
        # Finally, switch to the default group
        if default_group in self.model_groups:
            logger.info(f"Loading default group {default_group} to GPU")
            try:
                await self._switch_to_group(default_group)
            except Exception as e:
                logger.error(f"Failed to load default group {default_group}: {e}")
        
        logger.info("Completed preloading all groups")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        return {
            "gpu_available": True,
            "allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_memory_gb": torch.cuda.memory_reserved() / 1e9,
            "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            "groups_on_gpu": [g for g, loc in self.base_model_locations.items() if loc == 'cuda'],
            "groups_on_cpu": [g for g, loc in self.base_model_locations.items() if loc == 'cpu'],
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