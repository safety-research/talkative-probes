import asyncio
from typing import Optional, Dict, Any
import uuid
from datetime import datetime
import torch
import logging
import os
import json
from contextlib import asynccontextmanager

from .config import settings

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Raised when model fails to load"""
    pass

class InferenceQueue:
    def __init__(self, max_size: int = 100):
        self.queue = asyncio.Queue(maxsize=max_size)  # Bounded queue
        self.active_requests = {}
        self.processing = False
    
    async def add_request(self, text: str, options: dict) -> str:
        request_id = str(uuid.uuid4())
        request = {
            'id': request_id,
            'text': text,
            'options': options,
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
        self.max_retries = max_retries
        self.queue = InferenceQueue(max_size=settings.max_queue_size)
        self.processing_task = None
        self._load_lock = asyncio.Lock()  # Prevent concurrent model loading
        
    async def load_model(self):
        """Load model with retry logic"""
        # Import here to avoid circular imports and only when needed
        import sys
        sys.path.append('/workspace/kitf/talkative-probes/talkative_autoencoder')
        sys.path.append('/workspace/kitf/talkative-probes/consistency-lens')
        
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
                
                # Load model
                self.analyzer = LensAnalyzer(
                    self.checkpoint_path,
                    device=settings.device,
                    batch_size=settings.batch_size,
                    use_bf16=settings.use_bf16,
                    strict_load=False  # Allow loading with warnings
                )
                
                # Test inference to ensure model works
                test_result = self.analyzer.analyze_all_tokens(
                    "test",
                    batch_size=1,
                    return_structured=True
                )
                
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
                    # Extract options
                    options = request['options']
                    
                    # Calculate batch size if not provided
                    batch_size = options.get('batch_size')
                    if batch_size is None:
                        optimize_config = options.get('optimize_explanations_config', {})
                        k_rollouts = optimize_config.get('just_do_k_rollouts', 8)
                        batch_size = max(1, 256 // k_rollouts)
                    
                    # Run inference with all parameters
                    df = self.analyzer.analyze_all_tokens(
                        request['text'],
                        seed=options.get('seed', 42),
                        batch_size=batch_size,
                        no_eval=options.get('no_eval', False),
                        tuned_lens=options.get('tuned_lens', True),
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
                        optimize_explanations_config=options.get('optimize_explanations_config')
                    )
                    
                    # Convert DataFrame to dict for JSON serialization
                    result_data = df.to_dict('records')
                    
                    request['result'] = {
                        'metadata': {
                            'model_name': 'qwen2_5_WCHAT_14b_frozen_nopostfix',
                            'device': settings.device,
                            'batch_size': request['options'].get('batch_size', 32)
                        },
                        'data': result_data
                    }
                    request['status'] = 'completed'
                    
                except Exception as e:
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
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)