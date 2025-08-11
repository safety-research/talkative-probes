"""Inference service with multi-GPU support that handles request processing using the GroupedModelManager"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List, Tuple
import sys
import json
from pathlib import Path
import traceback
from dataclasses import dataclass, field

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)

from .inference_queue import InferenceQueue
from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    """Tracks the state of a single worker"""
    worker_id: str
    device: str
    task: Optional[asyncio.Task] = None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    current_request: Optional[str] = None
    is_busy: bool = False


class RequestFileLogger:
    """Logs requests to files in the logs directory"""
    
    def __init__(self, log_dir: str = None):
        if log_dir is None:
            # Default to website/logs directory
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def log_request(self, request_type: str, request_id: str, text: str, options: Dict[str, Any] = None):
        """Log a request to file and stdout"""
        
        # Print to stdout (always)
        print(f"\n{'='*80}")
        print(f"[{request_type.upper()} REQUEST {request_id}]")
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        print(f"Text length: {len(str(text))} characters")
        print(f"Input text: {str(text)[:500]}{'...' if len(str(text)) > 500 else ''}")
        print(f"Options: {options}")
        print(f"{'='*80}\n")
        
        # Determine origin bucket
        origin = None
        if options:
            origin = options.get("origin")
            if not origin and isinstance(options.get("headers"), dict):
                origin = options["headers"].get("origin")
        is_kitft = False
        if isinstance(origin, str):
            is_kitft = "kitft.com" in origin

        # Select log file based on origin
        today = datetime.now().strftime("%Y-%m-%d")
        if is_kitft:
            log_file = self.log_dir / f"requests_kitft_{today}.jsonl"
        else:
            log_file = self.log_dir / f"requests_other_{today}.jsonl"

        # Create a copy of options without non-serializable objects
        clean_options = {}
        if options:
            for key, value in options.items():
                if key == "websocket" or hasattr(value, '__dict__'):
                    continue
                clean_options[key] = value
        
        # Log to file
        log_entry = {
            "request_id": request_id,
            "type": request_type,
            "timestamp": datetime.utcnow().isoformat(),
            "text_length": len(str(text)),
            "text_preview": str(text)[:1000],
            "options": clean_options,
            "pid": os.getpid()
        }
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write request log to file: {e}")


# Global request logger instance
request_logger = RequestFileLogger()


class InferenceService:
    """
    Service that handles inference requests using the GroupedModelManager with multi-GPU support.
    
    This service supports:
    - Multiple worker processes per GPU device
    - Smart request routing to optimal device/worker
    - Exclusive GPU assignment for requests
    - Efficient group switching coordination
    """
    
    def __init__(self, model_manager, settings: Settings, websocket_manager=None):
        self.model_manager = model_manager
        self.settings = settings
        self.websocket_manager = websocket_manager
        self.queue = InferenceQueue(websocket_manager)
        
        # Multi-worker configuration
        self.devices = settings.devices if settings.devices else ["cuda:0"]
        self.num_workers_per_gpu = settings.num_workers_per_gpu
        
        # Worker management
        self.workers: Dict[str, WorkerState] = {}
        self.dispatcher_task: Optional[asyncio.Task] = None
        
        # Initialize workers for each device
        worker_id = 0
        for device in self.devices:
            for i in range(self.num_workers_per_gpu):
                worker_name = f"{device}_worker_{i}"
                self.workers[worker_name] = WorkerState(
                    worker_id=worker_name,
                    device=device
                )
                worker_id += 1
        
        logger.info(f"Initialized {len(self.workers)} workers across {len(self.devices)} devices")
        
    async def start_processing(self):
        """Start the dispatcher and all workers"""
        # Start worker tasks
        for worker_name, worker_state in self.workers.items():
            if worker_state.task is None or worker_state.task.done():
                worker_state.task = asyncio.create_task(
                    self._worker_loop(worker_name)
                )
                logger.info(f"Started worker {worker_name}")
        
        # Start dispatcher
        if self.dispatcher_task is None or self.dispatcher_task.done():
            self.dispatcher_task = asyncio.create_task(self._dispatch_requests())
            logger.info("Started request dispatcher")
            
    async def stop_processing(self):
        """Stop all workers and the dispatcher"""
        # Cancel dispatcher
        if self.dispatcher_task:
            self.dispatcher_task.cancel()
            try:
                await self.dispatcher_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped request dispatcher")
        
        # Cancel all workers
        for worker_name, worker_state in self.workers.items():
            if worker_state.task:
                worker_state.task.cancel()
                try:
                    await worker_state.task
                except asyncio.CancelledError:
                    pass
                logger.info(f"Stopped worker {worker_name}")
    
    async def _dispatch_requests(self):
        """
        Central dispatcher that routes requests to appropriate workers.
        
        This implements smart routing to minimize group switches and balance load.
        """
        logger.info("Request dispatcher started")
        
        while True:
            try:
                # Get next request from main queue
                request_id, text, options, request_type = await self.queue.get_next_request()
                
                if request_id is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if request was cancelled while in queue
                request = self.queue.active_requests.get(request_id)
                if request and request["status"] == "cancelled":
                    logger.info(f"Skipping cancelled request {request_id}")
                    continue
                
                # Find best worker for this request
                worker_name = await self._find_best_worker_for_request(
                    request_id, text, options, request_type
                )
                
                # Route to selected worker
                worker = self.workers[worker_name]
                await worker.queue.put((request_id, text, options, request_type))
                logger.debug(f"Routed request {request_id} to worker {worker_name}")
                
            except Exception as e:
                logger.error(f"Dispatcher error: {e}")
                await asyncio.sleep(1)
    
    async def _find_best_worker_for_request(
        self, 
        request_id: str, 
        text: str, 
        options: Dict[str, Any], 
        request_type: str
    ) -> str:
        """
        Find the best worker to handle a request.
        
        Priority order:
        1. If exclusive GPU requested, use worker on that GPU
        2. For group switches, any worker on the target device
        3. For inference, worker on device with model's group loaded
        4. Least busy worker as fallback
        """
        # Check for exclusive GPU request
        exclusive_gpu = options.get("exclusive_gpu")
        if exclusive_gpu and exclusive_gpu in self.devices:
            # Find least busy worker on the exclusive GPU
            workers_on_gpu = [
                (name, worker) for name, worker in self.workers.items() 
                if worker.device == exclusive_gpu
            ]
            if workers_on_gpu:
                # Sort by queue size to find least busy
                workers_on_gpu.sort(key=lambda x: x[1].queue.qsize())
                selected_worker = workers_on_gpu[0][0]
                logger.info(f"Using exclusive GPU {exclusive_gpu} for request {request_id}")
                return selected_worker
        
        # For group switch requests
        if request_type == "group_switch":
            target_group_id = text  # For group switches, text contains the group ID
            # Find any worker on a device that needs this group
            # Prefer devices that don't have any group loaded yet
            best_worker = None
            min_queue_size = float('inf')
            
            for name, worker in self.workers.items():
                device_state = self.model_manager.device_states.get(worker.device)
                if device_state:
                    # Prefer empty devices or devices already on this group
                    if (device_state.current_group_id is None or 
                        device_state.current_group_id == target_group_id):
                        queue_size = worker.queue.qsize()
                        if queue_size < min_queue_size:
                            min_queue_size = queue_size
                            best_worker = name
            
            if best_worker:
                return best_worker
        
        # For regular inference requests
        model_id = options.get("model_id")
        if model_id:
            # Find device with the best conditions for this model
            best_device = self.model_manager.find_best_device_for_model(
                model_id, 
                preferred_device=exclusive_gpu
            )
            
            # Find least busy worker on that device
            workers_on_device = [
                (name, worker) for name, worker in self.workers.items() 
                if worker.device == best_device
            ]
            if workers_on_device:
                workers_on_device.sort(key=lambda x: x[1].queue.qsize())
                return workers_on_device[0][0]
        
        # Fallback: find globally least busy worker
        least_busy_worker = min(
            self.workers.items(),
            key=lambda x: x[1].queue.qsize()
        )
        return least_busy_worker[0]
    
    async def _worker_loop(self, worker_name: str):
        """
        Worker loop that processes requests from its queue.
        
        Each worker is bound to a specific GPU device.
        """
        worker = self.workers[worker_name]
        logger.info(f"Worker {worker_name} started on device {worker.device}")
        
        while True:
            try:
                # Get next request from worker's queue
                request_id, text, options, request_type = await worker.queue.get()
                
                # Mark worker as busy
                worker.is_busy = True
                worker.current_request = request_id
                
                try:
                    # Process the request on this worker's device
                    await self._handle_single_request(
                        request_id, text, options, request_type, worker.device
                    )
                finally:
                    # Mark worker as available
                    worker.is_busy = False
                    worker.current_request = None
                    
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_single_request(
        self, 
        request_id: str, 
        text: str, 
        options: Dict[str, Any], 
        request_type: str,
        device: str
    ):
        """Handle a single request on a specific device"""
        try:
            # Add device to options for model manager
            options["device"] = device
            
            # Process based on request type
            if request_type == "group_switch":
                await self._process_group_switch_request(request_id, text, options)
            else:
                # Regular requests need model/analyzer
                model_id = options.get("model_id")
                if not model_id:
                    model_id = await self.model_manager.get_default_model_id()
                    if not model_id:
                        request = self.queue.active_requests.get(request_id)
                        if request:
                            request["error"] = "No model specified and no default available"
                            request["status"] = "failed"
                        return
                
                # Handle dependencies
                request = self.queue.active_requests.get(request_id)
                if request and request.get("depends_on"):
                    await self._wait_for_dependency(request_id, request.get("depends_on"))
                
                # Check if we need a group switch
                target_group_id = self.model_manager.model_to_group.get(model_id)
                device_state = self.model_manager.device_states.get(device)
                
                if (target_group_id and device_state and 
                    target_group_id != device_state.current_group_id):
                    # Need to switch groups on this device
                    await self._handle_group_switch_for_request(
                        request_id, text, options, request_type, 
                        target_group_id, model_id, device
                    )
                    return
                
                # Get the analyzer for the specific model on this device
                try:
                    analyzer, used_device = await self.model_manager.get_analyzer_for_model(
                        model_id, device
                    )
                    # Update device in case model manager chose a different one
                    device = used_device
                except Exception as e:
                    request = self.queue.active_requests.get(request_id)
                    if request:
                        request["error"] = f"Failed to load model {model_id}: {str(e)}"
                        request["status"] = "failed"
                    return
                
                # Update request status
                request = self.queue.active_requests.get(request_id)
                if request and request["status"] != "cancelled":
                    if request["status"] == "waiting_for_group_switch":
                        request["status"] = "processing"
                    else:
                        request["status"] = "processing"
                        request["started_at"] = datetime.utcnow()
                    
                    # Add device info
                    request["device"] = device
                    
                    # Send status update
                    websocket = options.get("websocket")
                    if websocket:
                        await self.queue._send_websocket_update(
                            websocket,
                            {
                                "type": "processing",
                                "request_id": request_id,
                                "context": "generation" if request_type == "generate" else "analysis",
                                "device": device
                            }
                        )
                    
                    # Process based on request type
                    try:
                        if request_type == "generate":
                            await self._process_generation_request(
                                request_id, text, options, analyzer, device
                            )
                        elif request_type == "send_message":
                            await self._process_send_message_request(
                                request_id, text, options, analyzer, device
                            )
                        else:
                            await self._process_analysis_request(
                                request_id, text, options, analyzer, device
                            )
                    except Exception as e:
                        logger.error(f"Error processing request {request_id}: {e}")
                        request["error"] = str(e)
                        request["status"] = "failed"
                    
        except Exception as e:
            logger.error(f"Error handling request {request_id}: {e}")
            request = self.queue.active_requests.get(request_id)
            if request:
                request["error"] = str(e)
                request["status"] = "failed"
    
    async def _wait_for_dependency(self, request_id: str, depends_on: str):
        """Wait for a dependency request to complete"""
        for _ in range(50):  # up to ~5 seconds
            dep = self.queue.active_requests.get(depends_on)
            if dep and dep.get("status") in ["completed", "failed", "cancelled"]:
                break
            await asyncio.sleep(0.1)
    
    async def _handle_group_switch_for_request(
        self,
        request_id: str,
        text: str,
        options: Dict[str, Any],
        request_type: str,
        target_group_id: str,
        model_id: str,
        device: str
    ):
        """Handle group switching for a request"""
        # Check if a group switch is already queued
        switch_already_queued = any(
            req.get("type") == "group_switch" and 
            req.get("target_group_id") == target_group_id and
            req.get("device") == device and
            req.get("status") in ["queued", "processing"]
            for req in self.queue.active_requests.values()
        )
        
        if not switch_already_queued:
            logger.info(f"Request {request_id} needs group {target_group_id} on {device}, queuing group switch")
            
            # Queue the group switch
            websocket = options.get("websocket")
            switch_options = {"websocket": websocket, "model_id": model_id, "device": device}
            switch_request_id = await self.queue.add_group_switch_request(
                target_group_id, model_id, websocket
            )
            # Add device info to the switch request
            switch_request = self.queue.active_requests.get(switch_request_id)
            if switch_request:
                switch_request["device"] = device
            
            logger.info(f"Queued group switch {switch_request_id} for request {request_id} on {device}")
        else:
            logger.info(f"Group switch to {target_group_id} already queued on {device}, waiting...")
        
        # Mark this request as waiting
        request = self.queue.active_requests.get(request_id)
        if request:
            request["status"] = "waiting_for_group_switch"
            request["target_group"] = target_group_id
            request["device"] = device
            if not request.get("started_at"):
                request["started_at"] = datetime.utcnow()
        
        # Re-queue this request to the same worker
        worker_name = f"{device}_worker_0"  # Use first worker on device
        worker = self.workers.get(worker_name)
        if worker:
            await worker.queue.put((request_id, text, options, request_type))
    
    async def _process_group_switch_request(self, request_id: str, target_group_id: str, options: Dict[str, Any]):
        """Process a group switch request on a specific device"""
        request = self.queue.active_requests[request_id]
        websocket = options.get("websocket")
        model_id = options.get("model_id")
        device = options.get("device")
        
        try:
            logger.info(f"Processing group switch request {request_id} to group {target_group_id} on {device}")
            
            request["status"] = "processing"
            request["started_at"] = datetime.utcnow()
            
            # Notify user that switch is starting
            if websocket:
                await self.queue._send_websocket_update(
                    websocket,
                    {
                        "type": "group_switch_starting",
                        "request_id": request_id,
                        "target_group_id": target_group_id,
                        "model_id": model_id,
                        "device": device
                    }
                )
            
            # Perform the group switch on the specific device
            await self.model_manager._switch_device_to_group(device, target_group_id)
            
            request["status"] = "completed"
            request["completed_at"] = datetime.utcnow()
            request["processing_time"] = (request["completed_at"] - request["started_at"]).total_seconds()
            
            # Get model info
            model_info = self.model_manager.get_model_info(model_id) if model_id else {}
            
            # Notify completion
            if websocket:
                await self.queue._send_websocket_update(
                    websocket,
                    {
                        "type": "model_switch_complete",
                        "request_id": request_id,
                        "model_id": model_id,
                        "target_group_id": target_group_id,
                        "device": device,
                        "message": f"Switched to group {target_group_id} on {device}",
                        "model_info": model_info,
                        "generation_config": model_info.get("generation_config", {})
                    }
                )
                
        except Exception as e:
            logger.error(f"Group switch error for request {request_id}: {e}")
            request["error"] = str(e)
            request["status"] = "failed"
            request["completed_at"] = datetime.utcnow()
            
            if websocket:
                await self.queue._send_websocket_update(
                    websocket,
                    {
                        "type": "model_switch_error",
                        "request_id": request_id,
                        "model_id": model_id,
                        "device": device,
                        "error": str(e)
                    }
                )
    
    async def _process_analysis_request(
        self, 
        request_id: str, 
        text: str, 
        options: Dict[str, Any],
        analyzer: Any,
        device: str
    ):
        """Process an analysis request with the provided analyzer"""
        request = self.queue.active_requests[request_id]
        websocket = options.get("websocket")
        
        # Handle prior generated text substitution
        used_prior_generated_text = False
        if options.get("use_prior_generated_text") and request.get("depends_on"):
            prior = self.queue.active_requests.get(request["depends_on"])
            if prior and prior.get("status") == "completed":
                try:
                    comps = prior.get("result", {}).get("completions")
                    if comps and isinstance(comps, list) and len(comps) > 0:
                        text = comps[0]
                        used_prior_generated_text = True
                except Exception:
                    pass
        
        model_id = options.get("model_id")
        
        try:
            logger.info(f"Processing analysis request {request_id} on {device} with text length: {len(text)}")
            
            # Log the request
            request_logger.log_request("analysis", request_id, text, options)
            
            # Parse chat format if applicable
            text_to_analyze = text
            messages_list = None
            if options.get("use_chat_format", False) and not used_prior_generated_text:
                if isinstance(text, str):
                    text_to_analyze = await self._parse_chat_format(text, model_id)
                elif isinstance(text, list):
                    chat_tokenizer = self.model_manager.get_chat_tokenizer_for_model(model_id)
                    if chat_tokenizer and hasattr(chat_tokenizer, 'apply_chat_template'):
                        messages = text
                        messages_list = messages.copy()
                        if 'gemma' in model_id and messages and messages[0].get("role") == "system":
                            system_prompt = messages[0]["content"]
                            messages = messages[1:]
                            if messages:
                                messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]
                                logger.info(f"Merged system prompt into first message")
                        logger.info(f"Applying chat template to {len(messages)} messages")
                        text_to_analyze = chat_tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True if messages[-1].get("role") == "user" else False
                        )
                        logger.info(f"Chat template applied, output length: {len(text_to_analyze)}")
                    else:
                        raise ValueError(f"Chat format not supported for analysis due to lack of chat tokenizer for {model_id}")
            
            # Determine batch size
            batch_size = options.get("batch_size")
            if batch_size is None:
                optimize_config = options.get("optimize_explanations_config", {})
                k_rollouts = optimize_config.get("best_of_k", 8)
                model_info = self.model_manager.get_model_info(model_id)
                auto_batch_max = model_info.get("auto_batch_size_max", self.settings.auto_batch_size_max)
                batch_size = max(1, auto_batch_max // k_rollouts)
            
            # Create progress callback
            loop = asyncio.get_event_loop()
            
            def progress_callback(current: int, total: int, message: str = "") -> bool:
                if request["status"] == "cancelled":
                    logger.info(f"Analysis cancelled for request {request_id}")
                    return False
                
                if websocket:
                    progress = (current / total * 100) if total > 0 else 0
                    asyncio.run_coroutine_threadsafe(
                        self.queue._send_websocket_update(
                            websocket,
                            {
                                "type": "progress",
                                "request_id": request_id,
                                "current": current,
                                "total": total,
                                "progress": progress,
                                "message": message,
                                "device": device
                            }
                        ),
                        loop,
                    )
                return True
            
            # Run analysis in executor
            logger.info(f"Starting analysis in executor for request {request_id} on {device}")
            df = await loop.run_in_executor(
                None,
                lambda: analyzer.analyze_all_tokens(
                    text_to_analyze,
                    seed=options.get("seed", 42),
                    batch_size=batch_size,
                    no_eval=options.get("no_eval", False),
                    tuned_lens=options.get("tuned_lens", False) and self.settings.tuned_lens_dir is not None,
                    add_tokens=options.get("add_tokens"),
                    replace_left=options.get("replace_left"),
                    replace_right=options.get("replace_right"),
                    do_hard_tokens=options.get("do_hard_tokens", False),
                    return_structured=options.get("return_structured", True),
                    move_devices=options.get("move_devices", False),
                    logit_lens_analysis=options.get("logit_lens_analysis", False),
                    temperature=options.get("temperature", 1.0),
                    no_kl=options.get("no_kl", self.settings.no_kl if hasattr(self.settings, 'no_kl') else True),
                    calculate_token_salience=options.get("calculate_token_salience", True),
                    optimize_explanations_config=options.get("optimize_explanations_config"),
                    progress_callback=progress_callback if websocket else None,
                    messages_list=messages_list,
                    last_n_messages=options.get("last_n_messages"),
                ),
            )
            
            if df is None:
                raise ValueError("Analysis returned no data")
            
            # Convert result
            result_data = df.to_dict("records")
            
            # Get model info
            model_info = self.model_manager.get_model_info(model_id)
            
            # Get configuration
            optimize_config = options.get("optimize_explanations_config", {})
            best_of_k = optimize_config.get("best_of_k", 8)
            
            # Complete the request
            request["result"] = {
                "data": result_data,
                "metadata": {
                    "model_id": model_info.get("model_id"),
                    "encoder_decoder_model": model_info.get("display_name", "Unknown"),
                    "shared_base_model": model_info.get("base_model", model_info.get("donor_model", "Unknown")),
                    "donor_model": model_info.get("donor_model", "Unknown"),
                    "layer": model_info.get("layer", "Unknown"),
                    "batch_size": batch_size,
                    "best_of_k": best_of_k,
                    "temperature": options.get("temperature", 1.0),
                    "checkpoint_path": model_info.get("checkpoint_path", "Unknown"),
                    "device": device
                }
            }
            request["status"] = "completed"
            request["completed_at"] = datetime.utcnow()
            request["processing_time"] = (request["completed_at"] - request["started_at"]).total_seconds()
            
            # Send completion via WebSocket
            if websocket:
                response = {
                    "type": "completed",
                    "request_id": request_id,
                    "result": request["result"],
                    "context": "analysis",
                    "device": device
                }
                if "client_request_id" in request.get("options", {}):
                    response["client_request_id"] = request["options"]["client_request_id"]
                
                await self.queue._send_websocket_update(websocket, response)
                
        except Exception as e:
            logger.error(f"Analysis error for request {request_id}: {e}")
            logger.error(traceback.format_exc())
            request["error"] = str(e)
            request["status"] = "failed"
            
            if websocket:
                response = {
                    "type": "error",
                    "request_id": request_id,
                    "error": str(e),
                    "context": "analysis",
                    "device": device
                }
                if "client_request_id" in request.get("options", {}):
                    response["client_request_id"] = request["options"]["client_request_id"]
                
                await self.queue._send_websocket_update(websocket, response)
    
    async def _process_generation_request(
        self,
        request_id: str,
        text: str,
        options: Dict[str, Any],
        analyzer: Any,
        device: str
    ):
        """Process a generation request with the provided analyzer"""
        request = self.queue.active_requests[request_id]
        websocket = options.get("websocket")
        model_id = options.get("model_id")
        
        try:
            logger.info(f"Processing generation request {request_id} on {device}")
            
            # Log the request
            request_logger.log_request("generation", request_id, text, options)
            
            # Check if this is chat format
            is_chat = options.get("is_chat", False) or options.get("use_chat_format", False)
            
            # Parse the text if it's chat format
            if is_chat:
                prompt = self._parse_chat_input(text)
            else:
                prompt = text
            
            # Generation parameters
            num_completions = options.get("num_completions", 3)
            max_new_tokens = options.get("num_tokens") or options.get("max_new_tokens", 50)
            
            # Get model's default generation config
            model_info = self.model_manager.get_model_info(model_id)
            gen_config = model_info.get("generation_config", {})
            
            temperature = options.get("temperature", gen_config.get("temperature", 0.8))
            top_p = options.get("top_p", gen_config.get("top_p", 0.95))
            
            # Run generation in executor
            loop = asyncio.get_event_loop()
            
            # Build generation kwargs
            generation_kwargs = {
                "num_completions": num_completions,
                "num_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "return_full_text": True,
                "is_chat": is_chat,
                "use_cache": True,
            }
            
            # Add chat tokenizer if available
            if is_chat:
                chat_tokenizer = self.model_manager.get_chat_tokenizer_for_model(model_id)
                if chat_tokenizer:
                    generation_kwargs["chat_tokenizer"] = chat_tokenizer
                    logger.info(f"Using chat tokenizer for generation")
            
            completions = await loop.run_in_executor(
                None,
                lambda: analyzer.generate_continuation(prompt, **generation_kwargs)
            )
            
            logger.info(f"Generated {len(completions)} completions for request {request_id}")
            
            # Complete the request
            request["result"] = {
                "completions": completions,
                "metadata": {
                    "model_id": model_info.get("model_id"),
                    "model_name": model_info.get("display_name"),
                    "prompt": prompt,
                    "num_completions": num_completions,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "device": device
                }
            }
            request["status"] = "completed"
            request["completed_at"] = datetime.utcnow()
            request["processing_time"] = (request["completed_at"] - request["started_at"]).total_seconds()
            
            # Send completion via WebSocket
            if websocket:
                await self.queue._send_websocket_update(
                    websocket,
                    {
                        "type": "generation_complete",
                        "request_id": request_id,
                        "result": request["result"],
                        "device": device
                    }
                )
                
        except Exception as e:
            logger.error(f"Generation error for request {request_id}: {e}")
            request["error"] = str(e)
            request["status"] = "failed"
            
            if websocket:
                await self.queue._send_websocket_update(
                    websocket,
                    {
                        "type": "generation_error",
                        "request_id": request_id,
                        "error": str(e),
                        "device": device
                    }
                )
    
    async def _process_send_message_request(
        self,
        request_id: str,
        text: str,
        options: Dict[str, Any],
        analyzer: Any,
        device: str
    ):
        """Process a send_message request with the provided analyzer"""
        request = self.queue.active_requests[request_id]
        websocket = options.get("websocket")
        model_id = options.get("model_id")
        
        try:
            logger.info(f"Processing send_message request {request_id} on {device}")
            
            # Get messages from options
            messages = options.get("messages", [])
            if not messages:
                raise ValueError("No messages provided for send_message request")
            
            # Log the request
            request_logger.log_request("send_message", request_id, messages, options)
            
            # Update WebSocket status
            if websocket:
                await self.queue._send_websocket_update(
                    websocket,
                    {
                        "type": "processing",
                        "request_id": request_id,
                        "context": "send_message",
                        "device": device
                    }
                )
            
            # Extract generation parameters
            temperature = options.get("temperature", 1.0)
            max_tokens = options.get("max_tokens", 1024)
            top_p = options.get("top_p", 1.0)
            use_cache = options.get("use_cache", True)
            
            # Set up kwargs
            send_message_kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "use_cache": use_cache,
            }
            
            # Get chat tokenizer
            chat_tokenizer = self.model_manager.get_chat_tokenizer_for_model(model_id)
            if chat_tokenizer:
                send_message_kwargs["chat_tokenizer"] = chat_tokenizer
                logger.info(f"Using chat tokenizer for send_message")
            
            # Run send_message in executor
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                lambda: analyzer.send_message(messages, **send_message_kwargs)
            )
            
            logger.info(f"Generated response for request {request_id}")
            
            # Get model info
            model_info = self.model_manager.get_model_info(model_id)
            
            # Complete the request
            request["result"] = {
                "response": response_text,
                "metadata": {
                    "model_id": model_info.get("model_id"),
                    "model_name": model_info.get("display_name"),
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "device": device
                }
            }
            request["status"] = "completed"
            request["completed_at"] = datetime.utcnow()
            request["processing_time"] = (request["completed_at"] - request["started_at"]).total_seconds()
            
            # Send completion via WebSocket
            if websocket:
                await self.queue._send_websocket_update(
                    websocket,
                    {
                        "type": "send_message_complete",
                        "request_id": request_id,
                        "result": request["result"],
                        "device": device
                    }
                )
                
        except Exception as e:
            logger.error(f"Send message error for request {request_id}: {e}")
            request["error"] = str(e)
            request["status"] = "failed"
            request["completed_at"] = datetime.utcnow()
            
            if websocket:
                await self.queue._send_websocket_update(
                    websocket,
                    {
                        "type": "send_message_error",
                        "request_id": request_id,
                        "error": str(e),
                        "device": device
                    }
                )
    
    def _parse_chat_input(self, text: Any) -> Any:
        """Parse chat format input for generation"""
        if isinstance(text, (list, dict)):
            return text
        
        try:
            import ast
            import json
            
            # Check input size limit
            if len(text) > 1024 * 1024:
                raise ValueError("Input text too large (max 1MB)")
            
            text_stripped = text.strip()
            if text_stripped.startswith('[') or text_stripped.startswith('{'):
                try:
                    messages = json.loads(text)
                except json.JSONDecodeError:
                    try:
                        messages = ast.literal_eval(text)
                    except (ValueError, SyntaxError):
                        cleaned_text = text.replace("\n", "\\n").replace("\r", "\\r")
                        try:
                            messages = json.loads(cleaned_text)
                        except:
                            messages = [{"role": "user", "content": text}]
            else:
                messages = [{"role": "user", "content": text}]
            
            # Validate structure
            if isinstance(messages, list):
                for i, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        raise ValueError(f"Message {i} must be a dictionary")
                    if "role" not in msg or "content" not in msg:
                        raise ValueError(f"Message {i} must have 'role' and 'content' fields")
            
            return messages
            
        except Exception as e:
            logger.warning(f"Failed to parse chat format: {e}")
            return [{"role": "user", "content": text}]
    
    async def _parse_chat_format(self, text: str, model_id: str) -> str:
        """Parse chat format and apply chat template for analysis"""
        try:
            import json
            
            text_stripped = text.strip()
            messages = None
            
            if text_stripped.startswith('[') or text_stripped.startswith('{'):
                try:
                    messages = json.loads(text)
                    if not isinstance(messages, list):
                        messages = None
                except:
                    messages = None
            
            if messages is None:
                logger.info(f"Converting plain text to chat format for analysis")
                messages = [{"role": "user", "content": text}]
            
            # Get chat tokenizer
            chat_tokenizer = self.model_manager.get_chat_tokenizer_for_model(model_id)
            if chat_tokenizer and hasattr(chat_tokenizer, 'apply_chat_template'):
                if 'gemma' in model_id and messages[0]["role"] == "system":
                    system_prompt = messages[0]["content"]
                    messages = messages[1:]
                    messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]
                    logger.info(f"Added system prompt for Gemma as user prefix")
                
                logger.info(f"Applying chat template to {len(messages)} messages")
                formatted_text = chat_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                logger.info(f"Chat template applied, output length: {len(formatted_text)}")
                return formatted_text
            else:
                logger.warning("No chat tokenizer available, returning original text")
                return text
                
        except Exception as e:
            logger.warning(f"Failed to parse chat format: {e}", exc_info=True)
            return text
    
    def get_worker_status(self) -> Dict[str, Any]:
        """Get status of all workers"""
        worker_status = {}
        for name, worker in self.workers.items():
            worker_status[name] = {
                "device": worker.device,
                "is_busy": worker.is_busy,
                "current_request": worker.current_request,
                "queue_size": worker.queue.qsize()
            }
        
        return {
            "workers": worker_status,
            "total_workers": len(self.workers),
            "busy_workers": sum(1 for w in self.workers.values() if w.is_busy)
        }