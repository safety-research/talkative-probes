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
            
            # Log input text for visibility
            logger.info(f"[ANALYSIS REQUEST {request_id}] Text length: {len(text)} characters")
            logger.debug(f"Analysis request {request_id} input preview: {text[:500]}{'...' if len(text) > 500 else ''}")
            
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
            
            def progress_callback(current: int, total: int, message: str = "") -> bool:
                if request["status"] == "cancelled":
                    logger.info(f"Analysis cancelled for request {request_id}")
                    return False
                    
                if websocket:
                    # Calculate progress percentage
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
                            }
                        ),
                        loop,
                    )
                return True
                
            # Run analysis in executor
            logger.info(f"Starting analysis in executor for request {request_id}")
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
            logger.info(f"Analysis completed in executor for request {request_id}, df shape: {df.shape if df is not None else 'None'}")
            
            # Check if df is None
            if df is None:
                logger.error(f"Analysis returned None for request {request_id}")
                raise ValueError("Analysis returned no data")
                
            # Convert result
            try:
                logger.info(f"About to convert dataframe to dict for request {request_id}")
                result_data = df.to_dict("records")
                logger.info(f"Converted result to dict for request {request_id}, {len(result_data)} records")
            except Exception as e:
                logger.error(f"Error converting dataframe to dict for request {request_id}: {e}")
                logger.error(f"DataFrame info: columns={list(df.columns) if df is not None else 'None'}, dtypes={df.dtypes.to_dict() if df is not None else 'None'}")
                raise
            
            # Get model info
            model_info = self.model_manager.get_current_model_info()
            
            # Complete the request
            request["result"] = {
                "data": result_data,
                "metadata": {
                    "model_id": model_info.get("model_id"),
                    "model_name": model_info.get("display_name"),
                    "batch_size": batch_size,
                    "checkpoint_path": str(self.model_manager.current_analyzer.checkpoint_path),
                }
            }
            request["status"] = "completed"
            request["completed_at"] = datetime.utcnow()
            request["processing_time"] = (request["completed_at"] - request["started_at"]).total_seconds()
            
            logger.info(f"Request {request_id} marked as completed, processing time: {request['processing_time']}s")
            
            # Send completion via WebSocket
            if websocket:
                logger.info(f"Sending completion message via WebSocket for request {request_id}")
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
            
            # Log input text for visibility
            logger.info(f"[GENERATION REQUEST {request_id}] Text length: {len(str(text))} characters")
            logger.debug(f"Generation request {request_id} input preview: {str(text)[:500]}{'...' if len(str(text)) > 500 else ''}")
            
            # Check if this is chat format
            is_chat = options.get("is_chat", False) or options.get("use_chat_format", False)
            
            # Parse the text if it's chat format
            if is_chat:
                try:
                    import ast
                    import json
                    
                    # If it's already a list/dict, use it directly
                    if isinstance(text, (list, dict)):
                        messages = text
                    else:
                        # Check input size to prevent DoS (1MB limit)
                        if len(text) > 1024 * 1024:
                            raise ValueError("Input text too large (max 1MB)")
                        
                        # Try to parse as JSON string
                        # First attempt: direct parsing (handles most cases including escaped newlines)
                        try:
                            messages = json.loads(text)
                        except json.JSONDecodeError:
                            # Second attempt: Try to evaluate as Python literal if it looks like one
                            # This handles triple quotes and other Python syntax
                            try:
                                messages = ast.literal_eval(text)
                            except (ValueError, SyntaxError):
                                # Third attempt: handle literal newlines by escaping them
                                # This handles cases where users paste JSON with actual newlines
                                cleaned_text = text.replace("\n", "\\n").replace("\r", "\\r")
                                messages = json.loads(cleaned_text)
                        
                        # Validate structure: should be a list of dicts with 'role' and 'content'
                        if not isinstance(messages, list):
                            raise ValueError("Chat messages must be a list")
                        for i, msg in enumerate(messages):
                            if not isinstance(msg, dict):
                                raise ValueError(f"Message {i} must be a dictionary")
                            if "role" not in msg or "content" not in msg:
                                raise ValueError(f"Message {i} must have 'role' and 'content' fields")
                    
                    # Use messages directly - generate_continuation will handle chat formatting
                    prompt = messages
                except Exception as e:
                    # Catch any parsing or validation errors
                    logger.warning(f"Failed to parse chat format: {e}")
                    # Fall back to using as regular text
                    prompt = text
                    is_chat = False
            else:
                prompt = text
                
            # Generation parameters
            num_completions = options.get("num_completions", 3)
            # Handle both 'num_tokens' (frontend) and 'max_new_tokens' (backend) names
            max_new_tokens = options.get("num_tokens") or options.get("max_new_tokens", 50)
            temperature = options.get("temperature", 0.8)
            top_p = options.get("top_p", 0.95)
            
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
            }
            
            # Add chat tokenizer if available and needed
            if is_chat and self.model_manager.chat_tokenizer:
                generation_kwargs["chat_tokenizer"] = self.model_manager.chat_tokenizer
            
            completions = await loop.run_in_executor(
                None,
                lambda: analyzer.generate_continuation(prompt, **generation_kwargs)
            )
            
            logger.info(f"Generated {len(completions)} completions for request {request_id}")
            
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
                logger.info(f"Sending generation_complete for request {request_id}")
                await self.queue._send_websocket_update(
                    websocket,
                    {
                        "type": "generation_complete",
                        "request_id": request_id,
                        "result": request["result"],
                    }
                )
            else:
                logger.warning(f"No websocket available to send generation_complete for request {request_id}")
                
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