import os
from dotenv import load_dotenv

# Load .env files FIRST before any other imports that might use HF_TOKEN
# Load .env file from backend directory
dotenv_location = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_location)

# Also load parent .env file (talkative_autoencoder) for HF_TOKEN
parent_dotenv_location = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(parent_dotenv_location, override=False)  # Don't override already set values

import asyncio
from typing import Optional, Dict, Any
import uuid
from datetime import datetime
import torch
import logging
import json
from contextlib import asynccontextmanager

from .config import load_settings
settings = load_settings()

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Raised when model fails to load"""
    pass

class InferenceQueue:
    def __init__(self, max_size: int = 100):
        self.queue = asyncio.Queue(maxsize=max_size)  # Bounded queue
        self.active_requests = {}
        self.processing = False
    
    async def add_request(self, text: str, options: dict, request_type: str = 'analyze') -> str:
        request_id = str(uuid.uuid4())
        request = {
            'id': request_id,
            'text': text,
            'options': options,
            'type': request_type,  # 'analyze' or 'generate'
            'status': 'queued',
            'created_at': datetime.utcnow(),
            'result': None,
            'error': None
        }
        self.active_requests[request_id] = request
        await self.queue.put(request_id)
        return request_id
    
    def get_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        return self.active_requests.get(request_id)

class ModelManager:
    def __init__(self, checkpoint_path: str, max_retries: int = 3):
        self.checkpoint_path = checkpoint_path
        self.analyzer = None
        self.chat_tokenizer = None  # For chat formatting
        self.model_name = None  # Store extracted model name
        self.max_retries = max_retries
        self.queue = InferenceQueue(max_size=settings.max_queue_size)
        self.processing_task = None
        self._load_lock = asyncio.Lock()  # Prevent concurrent model loading
        
    async def load_model(self):
        """Load model with retry logic"""
        # Import here to avoid circular imports and only when needed
        import sys
        sys.path.append('/workspace/kitf/talkative-probes/talkative_autoencoder')
        
        from lens.analysis.analyzer_class import LensAnalyzer
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Loading model attempt {attempt + 1}/{self.max_retries}")
                
                # Check if checkpoint exists
                if not os.path.exists(self.checkpoint_path):
                    raise ModelLoadError(f"Checkpoint not found: {self.checkpoint_path}")
                
                # Check GPU availability
                if not torch.cuda.is_available():
                    raise ModelLoadError("No GPU available")
                logger.info("full settings dict: ", dict(settings))
                # Load model
                self.analyzer = LensAnalyzer(
                    self.checkpoint_path,
                    device=settings.device,
                    batch_size=settings.batch_size,
                    use_bf16=settings.use_bf16,
                    strict_load=settings.strict_load,
                    comparison_tl_checkpoint=settings.comparison_tl_checkpoint,
                    do_not_load_weights=settings.do_not_load_weights,
                    make_xl=settings.make_xl,
                    t_text=settings.t_text,
                    no_orig=settings.no_orig,
                    different_activations_orig=settings.different_activations_model,
                    initialise_on_cpu=settings.initialise_on_cpu
                )
                
                # Skip test inference if no_orig is set
                if not settings.no_orig:
                    # Test inference to ensure model works
                    logger.info("Running test inference...")
                    try:
                        test_result = self.analyzer.analyze_all_tokens(
                            "test",
                            batch_size=8,
                            no_eval=True,  # Skip evaluation for test
                            tuned_lens=False,  # Skip tuned lens for test
                            logit_lens_analysis=False,
                            return_structured=True,
                            calculate_token_salience=False,  # Skip salience for test
                            optimize_explanations_config={
                                "just_do_k_rollouts": 1,  # Minimal rollouts for test
                                "use_batched": True
                            }
                        )
                        logger.info("Test inference successful")
                    except Exception as e:
                        logger.warning(f"Test inference failed (non-critical): {str(e)}")
                else:
                    logger.info("Skipping test inference (no_orig=True)")
                
                # Extract model name from checkpoint path
                checkpoint_name = os.path.basename(self.checkpoint_path)
                model_name_parts = checkpoint_name.split('_')
                if len(model_name_parts) > 4:
                    # Try to find where the timestamp starts
                    for i, part in enumerate(model_name_parts):
                        if part == 'resume' or part.startswith('0'):
                            self.model_name = '_'.join(model_name_parts[:i])
                            break
                    else:
                        # Fallback: just use first few parts
                        self.model_name = '_'.join(model_name_parts[:6])
                else:
                    self.model_name = checkpoint_name
                
                logger.info(f"Extracted model name: {self.model_name}")
                
                # Load chat tokenizer if we have a different_activations_model
                if settings.different_activations_model:
                    try:
                        from transformers import AutoTokenizer
                        logger.info(f"Loading chat tokenizer from: {settings.different_activations_model}")
                        self.chat_tokenizer = AutoTokenizer.from_pretrained(settings.different_activations_model)
                        logger.info("Chat tokenizer loaded successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load chat tokenizer: {e}")
                        self.chat_tokenizer = None
                
                logger.info("Model loaded successfully")
                return
                
            except Exception as e:
                logger.error(f"Model load attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise ModelLoadError(f"Failed to load model after {self.max_retries} attempts: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def start_processing(self):
        """Start the queue processing loop"""
        self.processing_task = asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Process requests from the queue"""
        while True:
            try:
                # Get next request
                request_id = await self.queue.queue.get()
                request = self.queue.active_requests.get(request_id)
                
                if not request:
                    continue
                
                # Update status
                request['status'] = 'processing'
                request['started_at'] = datetime.utcnow()
                
                # Check if client is still connected (for WebSocket)
                if hasattr(request, 'websocket') and request.get('websocket') and hasattr(request['websocket'], 'client_state') and request['websocket'].client_state != 1:
                    request['status'] = 'cancelled'
                    continue
                
                try:
                    # Check if model needs to be loaded
                    async with self._load_lock:
                        if self.analyzer is None:
                            # Send loading status
                            if 'websocket' in request and request['websocket']:
                                await request['websocket'].send_json({
                                    'type': 'status',
                                    'message': 'Loading model checkpoint... This may take a minute on first load.'
                                })
                            
                            # Load the model
                            await self.load_model()
                            
                            # Send success status
                            if 'websocket' in request and request['websocket']:
                                await request['websocket'].send_json({
                                    'type': 'status',
                                    'message': 'Model loaded successfully!'
                                })
                    
                    # Extract options
                    options = request['options']
                    
                    # Branch based on request type
                    if request.get('type') == 'generate':
                        # Handle generation request
                        await self._process_generation_request(request, request_id)
                    else:
                        # Handle analysis request (default)
                        await self._process_analysis_request(request, request_id)
                except Exception as e:
                    logger.error(f"Queue processing error: {e}")
                    await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_generation_request(self, request: dict, request_id: str):
        """Process a text generation request"""
        # Create a cancellation check function
        def is_cancelled():
            return request.get('status') == 'cancelled'
        
        try:
            text = request['text']
            options = request['options']
            
            # Check if cancelled before starting
            if is_cancelled():
                logger.info(f"Generation request {request_id} was cancelled before starting")
                return
            
            # Send processing status
            if 'websocket' in request and request['websocket']:
                await request['websocket'].send_json({
                    'type': 'processing',
                    'message': 'Generating text...',
                    'context': 'generation'
                })
            
            # Get the current event loop for run_in_executor
            loop = asyncio.get_event_loop()
            
            # Parse text as JSON if it's chat format
            if options.get('is_chat', False):
                try:
                    import json
                    messages = json.loads(text) if isinstance(text, str) else text
                    # Use messages directly with chat tokenizer
                    generation_kwargs = {
                        'num_tokens': options.get('num_tokens', 100),
                        'num_completions': options.get('num_completions', 1),
                        'temperature': options.get('temperature', 1.0),
                        'is_chat': True,
                        'return_full_text': options.get('return_full_text', True),
                        'use_cache': True
                    }
                    
                    # Add chat tokenizer if available
                    if self.chat_tokenizer:
                        generation_kwargs['chat_tokenizer'] = self.chat_tokenizer
                    
                    # For chat, pass cancellation check
                    generation_kwargs['cancellation_check'] = is_cancelled
                    
                    # Run generation in thread pool to avoid blocking event loop
                    continuations = await loop.run_in_executor(
                        None,  # Use default executor
                        lambda: self.analyzer.generate_continuation(
                            messages,
                            **generation_kwargs
                        )
                    )
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON format for chat messages")
            else:
                # Run generation in thread pool to avoid blocking event loop
                continuations = await loop.run_in_executor(
                    None,  # Use default executor
                    lambda: self.analyzer.generate_continuation(
                        text,
                        num_tokens=options.get('num_tokens', 100),
                        num_completions=options.get('num_completions', 1),
                        temperature=options.get('temperature', 1.0),
                        is_chat=False,
                        return_full_text=options.get('return_full_text', True),
                        cancellation_check=is_cancelled, 
                        use_cache=True
                    )
                )
            
            request['result'] = {
                'continuations': continuations
            }
            request['status'] = 'completed'
            
        except Exception as e:
            # Check if this was due to cancellation
            if is_cancelled():
                logger.info(f"Generation request {request_id} was cancelled")
                request['error'] = 'Cancelled by user'
                request['status'] = 'cancelled'
            else:
                logger.error(f"Generation error: {str(e)}")
                request['error'] = str(e)
                request['status'] = 'failed'
        
        request['completed_at'] = datetime.utcnow()
        
        # Calculate processing time
        if 'started_at' in request:
            processing_time = (request['completed_at'] - request['started_at']).total_seconds()
            request['processing_time'] = processing_time
        
        # Notify via WebSocket if connected
        if 'websocket' in request and request['websocket']:
            try:
                if request['status'] == 'cancelled':
                    await request['websocket'].send_json({
                        'type': 'interrupted',
                        'request_id': request_id,
                        'context': 'generation'
                    })
                else:
                    result_type = 'generation_result' if request['status'] == 'completed' else 'generation_error'
                    await request['websocket'].send_json({
                        'type': result_type,
                        'request_id': request_id,
                        'status': request['status'],
                        'result': request.get('result'),
                        'error': request.get('error'),
                        'processing_time': request.get('processing_time')
                    })
            except Exception as e:
                logger.error(f"WebSocket send error: {str(e)}")
    
    async def _process_analysis_request(self, request: dict, request_id: str):
        """Process a text analysis request"""
        # Create a cancellation check function
        def is_cancelled():
            return request.get('status') == 'cancelled'
        
        try:
            options = request['options']
            
            # Handle chat format if specified
            text_to_analyze = request['text']
            if options.get('is_chat', False):
                try:
                    import json
                    # If it's a string, parse it as JSON
                    if isinstance(text_to_analyze, str):
                        messages = json.loads(text_to_analyze)
                    else:
                        messages = text_to_analyze
                    
                    # Use the chat tokenizer if available, otherwise try model tokenizer
                    tokenizer = self.chat_tokenizer
                    if not tokenizer:
                        if hasattr(self.analyzer, 'orig_model') and hasattr(self.analyzer.orig_model, 'tokenizer'):
                            tokenizer = self.analyzer.orig_model.tokenizer
                        elif hasattr(self.analyzer, 'model') and hasattr(self.analyzer.model, 'tokenizer'):
                            tokenizer = self.analyzer.model.tokenizer
                    
                    if tokenizer and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                        try:
                            # Apply chat template
                            text_to_analyze = tokenizer.apply_chat_template(
                                messages, 
                                tokenize=False,
                                add_generation_prompt=False
                            )
                        except Exception as e:
                            logger.warning(f"Failed to apply chat template: {e}")
                            # Fallback to simple concatenation
                            text_parts = []
                            for msg in messages:
                                role = msg.get('role', 'user')
                                content = msg.get('content', '')
                                text_parts.append(f"{role}: {content}")
                            text_to_analyze = "\n".join(text_parts)
                    else:
                        # Fallback: concatenate messages
                        text_parts = []
                        for msg in messages:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            text_parts.append(f"{role}: {content}")
                        text_to_analyze = "\n".join(text_parts)
                except Exception as e:
                    logger.error(f"Failed to parse chat format: {e}")
                    # Fall back to using raw text
                    text_to_analyze = request['text']
            
            # Calculate batch size if not provided
            batch_size = options.get('batch_size')
            if batch_size is None:
                optimize_config = options.get('optimize_explanations_config', {})
                k_rollouts = optimize_config.get('just_do_k_rollouts', 8)
                batch_size = max(1, settings.auto_batch_size_max // k_rollouts)
            
            # Send processing status
            if 'websocket' in request and request['websocket']:
                await request['websocket'].send_json({
                    'type': 'processing',
                    'message': 'Starting text analysis...',
                    'context': 'analysis'
                })
            
            # Store WebSocket for use in callback
            websocket = request.get('websocket') if 'websocket' in request else None
            
            # Get the current event loop for thread-safe operations
            loop = asyncio.get_event_loop()
            
            # Track timing for ETA calculation
            import time
            start_time = time.time()
            batch_start_times = {}
            
            # Progress callback that will be called from sync code in another thread
            def progress_callback(current_batch, total_batches, message):
                # Check if cancelled
                if is_cancelled():
                    logger.info(f"Analysis cancelled at batch {current_batch}/{total_batches}")
                    # Return a special value to signal cancellation to analyzer
                    return False
                
                logger.info(f"Progress callback called: {current_batch}/{total_batches} - {message}")
                if websocket:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    
                    # Calculate ETA based on average time per batch
                    eta_seconds = None
                    if current_batch > 0:
                        avg_time_per_batch = elapsed_time / current_batch
                        remaining_batches = total_batches - current_batch
                        eta_seconds = avg_time_per_batch * remaining_batches
                    
                    # Schedule the async send in the main event loop from the worker thread
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({
                            'type': 'progress',
                            'current': current_batch,
                            'total': total_batches,
                            'message': message,
                            'percentage': (current_batch / total_batches * 100) if total_batches > 0 else 0,
                            'elapsed_seconds': elapsed_time,
                            'eta_seconds': eta_seconds
                        }),
                        loop
                    )
                return True  # Continue processing
            
            # Run inference in a thread pool to avoid blocking the event loop
            df = await loop.run_in_executor(
                None,  # Use default executor
                lambda: self.analyzer.analyze_all_tokens(
                    text_to_analyze,
                    seed=options.get('seed', 42),
                    batch_size=batch_size,
                    no_eval=options.get('no_eval', False),
                    tuned_lens=options.get('tuned_lens', False) and settings.tuned_lens_dir is not None,
                    add_tokens=options.get('add_tokens'),
                    replace_left=options.get('replace_left'),
                    replace_right=options.get('replace_right'),
                    do_hard_tokens=options.get('do_hard_tokens', False),
                    return_structured=options.get('return_structured', True),
                    move_devices=options.get('move_devices', False),
                    logit_lens_analysis=options.get('logit_lens_analysis', False),
                    temperature=options.get('temperature', 1.0),
                    no_kl=options.get('no_kl', False),
                    calculate_token_salience=options.get('calculate_token_salience', True),
                    optimize_explanations_config=options.get('optimize_explanations_config'),
                    progress_callback=progress_callback if websocket else None
                )
            )
            
            # Convert DataFrame to dict for JSON serialization
            result_data = df.to_dict('records')
            
            # Extract model name from checkpoint path
            checkpoint_name = os.path.basename(self.checkpoint_path)
            # Try to extract a cleaner name by removing timestamp and suffixes
            model_name_parts = checkpoint_name.split('_')
            if len(model_name_parts) > 4:
                # Try to find where the timestamp starts (usually after "resume" or a date pattern)
                for i, part in enumerate(model_name_parts):
                    if part == 'resume' or part.startswith('0'):
                        model_name = '_'.join(model_name_parts[:i])
                        break
                else:
                    # Fallback: just use first few parts
                    model_name = '_'.join(model_name_parts[:6])
            else:
                model_name = checkpoint_name
            
            request['result'] = {
                'metadata': {
                    'model_name': self.model_name or model_name,  # Use stored model_name if available
                    'device': settings.device,
                    'batch_size': request['options'].get('batch_size', 32)
                },
                'data': result_data
            }
            request['status'] = 'completed'
            
        except Exception as e:
            # Check if this was due to cancellation
            if is_cancelled():
                logger.info(f"Analysis request {request_id} was cancelled")
                request['error'] = 'Cancelled by user'
                request['status'] = 'cancelled'
            else:
                logger.error(f"Inference error: {str(e)}")
                request['error'] = str(e)
                request['status'] = 'failed'
        
        request['completed_at'] = datetime.utcnow()
        
        # Calculate processing time
        if 'started_at' in request:
            processing_time = (request['completed_at'] - request['started_at']).total_seconds()
            request['processing_time'] = processing_time
        
        # Notify via WebSocket if connected
        if 'websocket' in request and request['websocket']:
            try:
                if request['status'] == 'cancelled':
                    await request['websocket'].send_json({
                        'type': 'interrupted',
                        'request_id': request_id,
                        'context': 'analysis'
                    })
                else:
                    await request['websocket'].send_json({
                        'type': 'result',
                        'request_id': request_id,
                        'status': request['status'],
                        'result': request.get('result'),
                        'error': request.get('error'),
                        'processing_time': request.get('processing_time')
                    })
            except Exception as e:
                logger.error(f"WebSocket send error: {str(e)}")
