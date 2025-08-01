"""Inference service that handles request processing using the GroupedModelManager"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import sys
import json
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)

from .inference_queue import InferenceQueue
from .config import Settings

logger = logging.getLogger(__name__)

class RequestFileLogger:
    """Logs requests to files in the logs directory"""
    
    def __init__(self, log_dir: str = None):
        if log_dir is None:
            # Default to website/logs directory
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create date-based log file
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"requests_{today}.jsonl"
        
    def log_request(self, request_type: str, request_id: str, text: str, options: Dict[str, Any] = None):
        """Log a request to file and stdout"""
        
        # Print to stdout (always)
        print(f"\n{'='*80}")
        print(f"[{request_type.upper()} REQUEST {request_id}]")
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        print(f"Text length: {len(str(text))} characters")
        print(f"Input text: {str(text)[:500]}{'...' if len(str(text)) > 500 else ''}")
        print(f"{'='*80}\n")
        
        # Create a copy of options without non-serializable objects
        clean_options = {}
        if options:
            for key, value in options.items():
                # Skip WebSocket and other non-serializable objects
                if key == "websocket" or hasattr(value, '__dict__'):
                    continue
                clean_options[key] = value
        
        # Log to file
        log_entry = {
            "request_id": request_id,
            "type": request_type,
            "timestamp": datetime.utcnow().isoformat(),
            "text_length": len(str(text)),
            "text_preview": str(text)[:1000],  # Store more in file than shown in stdout
            "options": clean_options,
            "pid": os.getpid()
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write request log to file: {e}")

# Global request logger instance
request_logger = RequestFileLogger()

class InferenceService:
    """Service that handles inference requests using the GroupedModelManager"""
    
    def __init__(self, model_manager, settings: Settings, websocket_manager=None):
        self.model_manager = model_manager
        self.settings = settings
        self.websocket_manager = websocket_manager
        self.queue = InferenceQueue(websocket_manager)
        self.processing_task = None
        self.chat_tokenizer = None  # Will be loaded if needed
        self.active_processing_lock = asyncio.Lock()  # Ensure only one request processes at a time
        
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
                    
                # Check if request was cancelled while in queue
                request = self.queue.active_requests.get(request_id)
                if request and request["status"] == "cancelled":
                    logger.info(f"Skipping cancelled request {request_id}")
                    continue
                    
                # Process request sequentially to ensure group switches wait for active requests
                await self._handle_single_request(request_id, text, options, request_type)
                        
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)
                
    async def _handle_single_request(self, request_id: str, text: str, options: Dict[str, Any], request_type: str):
        """Handle a single request without blocking the queue"""
        try:
            # Process based on request type
            if request_type == "group_switch":
                # Group switches don't need model/analyzer loading
                await self._process_group_switch_request(request_id, text, options)
            else:
                # Regular requests need model/analyzer
                # Get model_id from options
                model_id = options.get("model_id")
                if not model_id:
                    # Try to get default model if none specified
                    model_id = await self.model_manager.get_default_model_id()
                    if not model_id:
                        request = self.queue.active_requests.get(request_id)
                        if request:
                            request["error"] = "No model specified and no default available"
                            request["status"] = "failed"
                        return
                
                # Check if we need to switch groups first
                target_group_id = self.model_manager.model_to_group.get(model_id)
                if target_group_id and target_group_id != self.model_manager.current_group_id:
                    # Check if a group switch is already queued for this group
                    switch_already_queued = any(
                        req.get("type") == "group_switch" and 
                        req.get("target_group_id") == target_group_id and
                        req.get("status") in ["queued", "processing"]
                        for req in self.queue.active_requests.values()
                    )
                    
                    if not switch_already_queued:
                        # Need to queue a group switch
                        logger.info(f"Request {request_id} needs group {target_group_id}, queuing group switch")
                        
                        # Queue the group switch
                        websocket = options.get("websocket")
                        switch_request_id = await self.queue.add_group_switch_request(
                            target_group_id, model_id, websocket
                        )
                        
                        logger.info(f"Queued group switch {switch_request_id} for request {request_id}")
                    else:
                        logger.info(f"Group switch to {target_group_id} already queued, waiting...")
                    
                    # Mark this request as waiting for group switch
                    request = self.queue.active_requests.get(request_id)
                    if request:
                        request["status"] = "waiting_for_group_switch"
                        request["target_group"] = target_group_id
                        # Set started_at if not already set
                        if not request.get("started_at"):
                            request["started_at"] = datetime.utcnow()
                    
                    # Re-queue this request to try again after the switch
                    await self.queue.queue.put((request_id, text, options, request_type))
                    return
                
                # Get the analyzer for the specific model
                try:
                    analyzer = await self.model_manager.get_analyzer_for_model(model_id)
                except Exception as e:
                    request = self.queue.active_requests.get(request_id)
                    if request:
                        request["error"] = f"Failed to load model {model_id}: {str(e)}"
                        request["status"] = "failed"
                    return
                    
                # Update request status
                request = self.queue.active_requests.get(request_id)
                if request and request["status"] != "cancelled":
                    # If this was waiting for group switch, update the started time
                    if request["status"] == "waiting_for_group_switch":
                        request["status"] = "processing"
                        # Don't update started_at since it was already set
                    else:
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
            logger.error(f"Error handling request {request_id}: {e}")
            request = self.queue.active_requests.get(request_id)
            if request:
                request["error"] = str(e)
                request["status"] = "failed"
                
    async def _process_group_switch_request(self, request_id: str, target_group_id: str, options: Dict[str, Any]):
        """Process a group switch request"""
        request = self.queue.active_requests[request_id]
        websocket = options.get("websocket")
        model_id = options.get("model_id")
        
        try:
            logger.info(f"Processing group switch request {request_id} to group {target_group_id}")
            
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
                        "model_id": model_id
                    }
                )
            
            # Perform the group switch through the model manager
            # This will trigger the _switch_to_group which handles all the heavy lifting
            await self.model_manager._switch_to_group(target_group_id)
            
            request["status"] = "completed"
            request["completed_at"] = datetime.utcnow()
            request["processing_time"] = (request["completed_at"] - request["started_at"]).total_seconds()
            
            # Get model info for the target model
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
                        "message": f"Switched to group {target_group_id}",
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
                        "error": str(e)
                    }
                )
                
    async def _process_analysis_request(self, request_id: str, text: str, options: Dict[str, Any]):
        """Process an analysis request"""
        request = self.queue.active_requests[request_id]
        websocket = options.get("websocket")
        
        # Get model_id and analyzer
        model_id = options.get("model_id")
        if not model_id:
            model_id = await self.model_manager.get_default_model_id()
            
        analyzer = await self.model_manager.get_analyzer_for_model(model_id)
        
        try:
            # Log the request
            logger.info(f"Processing analysis request {request_id} with text length: {len(text)}")
            
            # Log to stdout and file
            request_logger.log_request("analysis", request_id, text, options)
            
            # Parse chat format if applicable
            text_to_analyze = text
            if options.get("use_chat_format", False):
                text_to_analyze = await self._parse_chat_format(text, model_id)
                
            # Get batch size based on model
            batch_size = options.get("batch_size")
            if batch_size is None:
                optimize_config = options.get("optimize_explanations_config", {})
                k_rollouts = optimize_config.get("best_of_k", 8)
                model_info = self.model_manager.get_model_info(model_id)
                auto_batch_max = model_info.get(
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
            model_info = self.model_manager.get_model_info(model_id)
            
            # Get donor model info
            donor_model = model_info.get("donor_model", "Unknown")
            
            # Get best_of_k from options
            optimize_config = options.get("optimize_explanations_config", {})
            best_of_k = optimize_config.get("best_of_k", 8)
            
            # Complete the request
            request["result"] = {
                "data": result_data,
                "metadata": {
                    "model_id": model_info.get("model_id"),
                    "encoder_decoder_model": model_info.get("display_name", "Unknown"),
                    "shared_base_model": model_info.get("base_model", donor_model),
                    "donor_model": donor_model,
                    "layer": model_info.get("layer", "Unknown"),
                    "batch_size": batch_size,
                    "best_of_k": best_of_k,
                    "temperature": options.get("temperature", 1.0),
                    "checkpoint_path": model_info.get("checkpoint_path", "Unknown"),
                }
            }
            request["status"] = "completed"
            request["completed_at"] = datetime.utcnow()
            request["processing_time"] = (request["completed_at"] - request["started_at"]).total_seconds()
            
            logger.info(f"Request {request_id} marked as completed, processing time: {request['processing_time']}s")
            
            # Send completion via WebSocket
            if websocket:
                logger.info(f"Sending completion message via WebSocket for request {request_id}")
                response = {
                    "type": "completed",
                    "request_id": request_id,
                    "result": request["result"],
                    "context": "analysis"
                }
                # Include client_request_id if provided
                if "client_request_id" in request.get("options", {}):
                    response["client_request_id"] = request["options"]["client_request_id"]
                    
                await self.queue._send_websocket_update(websocket, response)
                
        except Exception as e:
            logger.error(f"Analysis error for request {request_id}: {e}")
            request["error"] = str(e)
            request["status"] = "failed"
            
            if websocket:
                response = {
                    "type": "error",
                    "request_id": request_id,
                    "error": str(e),
                    "context": "analysis"
                }
                # Include client_request_id if provided
                if "client_request_id" in request.get("options", {}):
                    response["client_request_id"] = request["options"]["client_request_id"]
                    
                await self.queue._send_websocket_update(websocket, response)
                
    async def _process_generation_request(self, request_id: str, text: str, options: Dict[str, Any]):
        """Process a generation request"""
        request = self.queue.active_requests[request_id]
        websocket = options.get("websocket")
        
        # Get model_id and analyzer
        model_id = options.get("model_id")
        if not model_id:
            model_id = await self.model_manager.get_default_model_id()
            
        analyzer = await self.model_manager.get_analyzer_for_model(model_id)
        
        try:
            logger.info(f"Processing generation request {request_id}")
            
            # Log to stdout and file
            request_logger.log_request("generation", request_id, text, options)
            
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
                        # First check if it looks like JSON (starts with [ or {)
                        text_stripped = text.strip()
                        if text_stripped.startswith('[') or text_stripped.startswith('{'):
                            # Try to parse as JSON
                            try:
                                messages = json.loads(text)
                            except json.JSONDecodeError:
                                # Try to evaluate as Python literal if it looks like one
                                try:
                                    messages = ast.literal_eval(text)
                                except (ValueError, SyntaxError):
                                    # Try handling literal newlines
                                    cleaned_text = text.replace("\n", "\\n").replace("\r", "\\r")
                                    try:
                                        messages = json.loads(cleaned_text)
                                    except:
                                        # If all JSON parsing fails, treat as plain text
                                        logger.info("JSON parsing failed, converting plain text to chat format")
                                        messages = [{"role": "user", "content": text}]
                        else:
                            # Plain text - convert to chat format
                            logger.info(f"Converting plain text to chat format: {text[:100]}...")
                            messages = [{"role": "user", "content": text}]
                        
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
                    logger.info(f"Using chat format with {len(messages)} messages")
                except Exception as e:
                    # Catch any parsing or validation errors
                    logger.warning(f"Failed to process chat format: {e}")
                    # Convert plain text to chat format as fallback
                    prompt = [{"role": "user", "content": text}]
                    logger.info("Falling back to single user message format")
            else:
                prompt = text
                
            # Generation parameters
            num_completions = options.get("num_completions", 3)
            # Handle both 'num_tokens' (frontend) and 'max_new_tokens' (backend) names
            max_new_tokens = options.get("num_tokens") or options.get("max_new_tokens", 50)
            
            # Get model's default generation config if not provided
            model_info = self.model_manager.get_model_info(model_id)
            gen_config = model_info.get("generation_config", {})
            
            temperature = options.get("temperature", gen_config.get("temperature", 0.8))
            top_p = options.get("top_p", gen_config.get("top_p", 0.95))
            
            # Run generation in executor
            loop = asyncio.get_event_loop()
            
            # Log prompt information
            logger.info(f"Generation request - is_chat: {is_chat}, prompt type: {type(prompt)}")
            if is_chat and isinstance(prompt, list):
                logger.info(f"Chat messages: {prompt}")
            
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
            
            # Add chat tokenizer if available and needed
            if is_chat:
                chat_tokenizer = self.model_manager.get_chat_tokenizer_for_model(model_id)
                if chat_tokenizer:
                    generation_kwargs["chat_tokenizer"] = chat_tokenizer
                    logger.info(f"Using chat tokenizer: {type(chat_tokenizer)}")
                else:
                    logger.warning("Chat format requested but no chat tokenizer available")
            
            completions = await loop.run_in_executor(
                None,
                lambda: analyzer.generate_continuation(prompt, **generation_kwargs)
            )
            
            logger.info(f"Generated {len(completions)} completions for request {request_id}")
            
            # Get model info
            model_info = self.model_manager.get_model_info(model_id)
            
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
                
    async def _parse_chat_format(self, text: str, model_id: str) -> str:
        """Parse chat format and apply chat template for analysis"""
        try:
            import json
            
            # First check if it's already JSON chat format
            text_stripped = text.strip()
            messages = None
            
            if text_stripped.startswith('[') or text_stripped.startswith('{'):
                # Try to parse as JSON
                try:
                    messages = json.loads(text)
                    if not isinstance(messages, list):
                        messages = None
                except:
                    messages = None
            
            # If not JSON or parsing failed, convert plain text to chat format
            if messages is None:
                logger.info(f"Converting plain text to chat format for analysis: {text[:100]}...")
                messages = [{"role": "user", "content": text}]
            
            # Get chat tokenizer for the model
            chat_tokenizer = self.model_manager.get_chat_tokenizer_for_model(model_id)
            if chat_tokenizer and hasattr(chat_tokenizer, 'apply_chat_template'):
                # Apply chat template to get the formatted text
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