"""Inference service that handles request processing using the ModelManager"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)

from .model_manager import ModelManager
from .inference_queue import InferenceQueue
from .config import Settings

logger = logging.getLogger(__name__)

class InferenceService:
    """Service that handles inference requests using the ModelManager"""
    
    def __init__(self, model_manager: ModelManager, settings: Settings, websocket_manager=None):
        self.model_manager = model_manager
        self.settings = settings
        self.websocket_manager = websocket_manager
        self.queue = InferenceQueue(websocket_manager)
        self.processing_task = None
        self.chat_tokenizer = None  # Will be loaded if needed
        
    async def start_processing(self):
        """Start processing the queue"""
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_queue())
            logger.info("Started queue processing")
            
    async def stop_processing(self):
        """Stop processing the queue"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped queue processing")
            
    async def _process_queue(self):
        """Process requests from the queue"""
        while True:
            try:
                # Get next request
                request_id, text, options, request_type = await self.queue.get_next_request()
                
                if request_id is None:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Get the analyzer, waiting if model is switching
                analyzer = await self.model_manager.get_analyzer()
                if analyzer is None:
                    # Model not loaded yet
                    request = self.queue.active_requests.get(request_id)
                    if request:
                        request["error"] = "No model loaded"
                        request["status"] = "failed"
                    continue
                    
                # Update request status
                request = self.queue.active_requests.get(request_id)
                if request and request["status"] != "cancelled":
                    request["status"] = "processing"
                    request["started_at"] = datetime.utcnow()
                    
                    # Send status update via WebSocket
                    websocket = options.get("websocket")
                    if websocket:
                        await self.queue._send_websocket_update(
                            websocket,
                            {
                                "type": "processing",
                                "request_id": request_id,
                                "context": "generation" if request_type == "generate" else "analysis"
                            }
                        )
                    
                    # Process based on request type
                    try:
                        if request_type == "generate":
                            await self._process_generation_request(request_id, text, options)
                        else:
                            await self._process_analysis_request(request_id, text, options)
                    except Exception as e:
                        logger.error(f"Error processing request {request_id}: {e}")
                        request["error"] = str(e)
                        request["status"] = "failed"
                        
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)
                
    async def _process_analysis_request(self, request_id: str, text: str, options: Dict[str, Any]):
        """Process an analysis request"""
        request = self.queue.active_requests[request_id]
        websocket = options.get("websocket")
        analyzer = await self.model_manager.get_analyzer()
        
        try:
            # Log the request
            logger.info(f"Processing analysis request {request_id} with text length: {len(text)}")
            
            # Parse chat format if applicable
            text_to_analyze = text
            if options.get("use_chat_format", False):
                text_to_analyze = await self._parse_chat_format(text)
                
            # Get batch size based on current model
            batch_size = options.get("batch_size")
            if batch_size is None:
                optimize_config = options.get("optimize_explanations_config", {})
                k_rollouts = optimize_config.get("best_of_k", 8)
                auto_batch_max = self.model_manager.get_current_model_info().get(
                    "auto_batch_size_max", self.settings.auto_batch_size_max
                )
                batch_size = max(1, auto_batch_max // k_rollouts)
                
            # Create progress callback
            loop = asyncio.get_event_loop()
            
            def progress_callback(progress: float, eta_seconds: int) -> bool:
                if request["status"] == "cancelled":
                    logger.info(f"Analysis cancelled for request {request_id}")
                    return False
                    
                if websocket:
                    asyncio.run_coroutine_threadsafe(
                        self.queue._send_websocket_update(
                            websocket,
                            {
                                "type": "progress",
                                "request_id": request_id,
                                "progress": progress,
                                "eta_seconds": eta_seconds,
                            }
                        ),
                        loop,
                    )
                return True
                
            # Run analysis in executor
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
                ),
            )
            
            # Convert result
            result_data = df.to_dict("records")
            
            # Get model info
            model_info = self.model_manager.get_current_model_info()
            
            # Complete the request
            request["result"] = {
                "data": result_data,
                "metadata": {
                    "model_id": model_info.get("model_id"),
                    "model_name": model_info.get("display_name"),
                    "batch_size": batch_size,
                    "checkpoint_path": self.model_manager.current_analyzer.checkpoint_path,
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
                        "type": "completed",
                        "request_id": request_id,
                        "result": request["result"],
                        "context": "analysis"
                    }
                )
                
        except Exception as e:
            logger.error(f"Analysis error for request {request_id}: {e}")
            request["error"] = str(e)
            request["status"] = "failed"
            
            if websocket:
                await self.queue._send_websocket_update(
                    websocket,
                    {
                        "type": "error",
                        "request_id": request_id,
                        "error": str(e),
                        "context": "analysis"
                    }
                )
                
    async def _process_generation_request(self, request_id: str, text: str, options: Dict[str, Any]):
        """Process a generation request"""
        request = self.queue.active_requests[request_id]
        websocket = options.get("websocket")
        analyzer = await self.model_manager.get_analyzer()
        
        try:
            logger.info(f"Processing generation request {request_id}")
            
            # Parse chat format if needed
            prompt = text
            if options.get("use_chat_format", False):
                prompt = await self._parse_chat_format(text)
                
            # Generation parameters
            num_completions = options.get("num_completions", 3)
            max_new_tokens = options.get("max_new_tokens", 50)
            temperature = options.get("temperature", 0.8)
            top_p = options.get("top_p", 0.95)
            
            # Run generation in executor
            loop = asyncio.get_event_loop()
            completions = await loop.run_in_executor(
                None,
                lambda: analyzer.generate_continuation(
                    prompt,
                    num_completions=num_completions,
                    num_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    return_full_text=True,
                )
            )
            
            # Get model info
            model_info = self.model_manager.get_current_model_info()
            
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
                    }
                )
                
    async def _parse_chat_format(self, text: str) -> str:
        """Parse chat format to extract the last user message"""
        # This is a simplified version - you may need to adapt based on your chat format
        # and tokenizer requirements
        try:
            # For now, just return the text as-is
            # In the full implementation, this would use the chat tokenizer
            return text
        except Exception as e:
            logger.warning(f"Failed to parse chat format: {e}")
            return text