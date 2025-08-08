"""
Slim API for generation and analysis with minimal options.
This runs in parallel with the main API but provides simplified endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import uuid
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/slim", tags=["slim"])
security = HTTPBearer(auto_error=False)

# These will be set by main.py
inference_service = None
grouped_model_manager = None
settings = None


class GenerationRequest(BaseModel):
    """Minimal generation request with chat-formatted input."""
    messages: List[Dict[str, str]] = Field(..., description="Chat-formatted messages")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    n_continuations: int = Field(1, ge=1, le=10, description="Number of continuations to generate")
    n_tokens: int = Field(100, ge=1, le=512, description="Number of tokens to generate")
    model_group: str = Field(..., description="Model group to use (e.g., 'gemma3-27b-it')")
    model_id: Optional[str] = Field(None, description="Specific model/layer ID to use (optional)")
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        for msg in v:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError("Role must be one of: system, user, assistant")
        return v


class AnalysisRequest(BaseModel):
    """Minimal analysis request with token salience option."""
    messages: List[Dict[str, str]] = Field(..., description="Chat-formatted messages to analyze")
    calculate_token_salience: bool = Field(True, description="Calculate token salience scores")
    best_of_k: int = Field(8, ge=1, le=32, description="Number of explanation rollouts")
    model_group: str = Field(..., description="Model group to use (e.g., 'gemma3-27b-it')")
    model_id: Optional[str] = Field(None, description="Specific model ID to use (e.g., 'gemma3-27b-it-layer20')")
    last_n_messages: Optional[int] = Field(None, description="Only analyze the last N messages (e.g., 2 for last user/assistant turn)")
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        for msg in v:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError("Role must be one of: system, user, assistant")
        return v


class SendMessageRequest(BaseModel):
    """Request for sending a message and getting a single response."""
    messages: List[Dict[str, str]] = Field(..., description="Chat-formatted messages")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(1024, ge=1, le=4096, description="Maximum tokens to generate")
    model_group: str = Field(..., description="Model group to use (e.g., 'gemma3-27b-it')")
    model_id: Optional[str] = Field(None, description="Specific model ID to use (optional)")
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        for msg in v:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError("Role must be one of: system, user, assistant")
        return v


class SlimResponse(BaseModel):
    """Response for both generation and analysis requests."""
    request_id: str
    status: str
    queue_position: Optional[int] = None
    message: str


def format_messages_to_text(messages: List[Dict[str, str]]) -> str:
    """Convert chat messages to a single text string for processing."""
    formatted_parts = []
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            formatted_parts.append(f"System: {content}")
        elif role == 'user':
            formatted_parts.append(f"User: {content}")
        elif role == 'assistant':
            formatted_parts.append(f"Assistant: {content}")
    return "\n".join(formatted_parts)


async def verify_api_key(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    """Verify API key if configured - mirrors main.py implementation"""
    # Check if we're in production (not localhost)
    is_production = os.getenv("ENVIRONMENT", "development") == "production"

    if not settings or not settings.api_key:
        if is_production:
            logger.error("API key not configured in production environment!")
            raise HTTPException(status_code=500, detail="Server configuration error: API key required")
        else:
            if not hasattr(verify_api_key, "_no_api_key_warn_count"):
                verify_api_key._no_api_key_warn_count = 0
            if verify_api_key._no_api_key_warn_count < 3:
                logger.warning("API key not configured - running without authentication")
                verify_api_key._no_api_key_warn_count += 1
            return True

    if settings.api_key and (not credentials or credentials.credentials != settings.api_key):
        logger.warning(f"Invalid or missing API key: {credentials.credentials if credentials else 'None'} != {settings.api_key}")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


@router.post("/generate", response_model=SlimResponse)
async def generate(
    request: GenerationRequest,
    authorized: bool = Depends(verify_api_key),
    req: Request = None,
):
    """
    Generate text continuations with minimal options.
    
    Required:
    - messages: Chat-formatted list of messages
    - model_group: Which model group to use
    
    Optional:
    - temperature: Sampling temperature (default: 1.0)
    - n_continuations: Number of continuations (default: 1)
    - n_tokens: Max tokens to generate (default: 100)
    """
    if not inference_service:
        raise HTTPException(500, "Inference service not initialized")
        
    try:
        # Convert messages to text
        #text = format_messages_to_text(request.messages)
        
        # Use provided model_id or find the first model in the requested group
        if request.model_id:
            model_id = request.model_id
        else:
            model_id = None
            if grouped_model_manager:
                origin_header = req.headers.get('origin', '') if req else ''
                host_header = req.headers.get('host', '') if req else ''
                public_only = ("kitft.com" in origin_header) or ("kitft.com" in host_header)
                groups = grouped_model_manager.get_model_list(public_only=public_only)
                for group in groups:
                    if group['group_id'] == request.model_group:
                        if group['models']:
                            model_id = group['models'][0]['id']
                        break
            
            if not model_id:
                raise HTTPException(400, f"Model group '{request.model_group}' not found")
        
        # Create generation options
        options = {
            "temperature": request.temperature,
            "max_new_tokens": request.n_tokens,
            "num_return_sequences": request.n_continuations,
            "do_sample": request.temperature > 0,
            "model_id": model_id
        }
        
        # Queue the request
        request_id = await inference_service.queue.add_request(
            text=request.messages,
            options={**options, **({"origin": req.headers.get("origin")} if req else {})},
            request_type="generate"
        )
        
        # Get queue position
        queue_size = inference_service.queue.queue.qsize()
        
        return SlimResponse(
            request_id=request_id,
            status="queued",
            queue_position=queue_size,
            message=f"Generation request queued at position {queue_size}"
        )
        
    except Exception as e:
        logger.error(f"Error in slim generate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=SlimResponse)
async def analyze(
    request: AnalysisRequest,
    authorized: bool = Depends(verify_api_key),
    req: Request = None,
):
    """
    Analyze text with minimal options.
    
    Required:
    - messages: Chat-formatted messages to analyze
    - model_group: Which model group to use
    
    Optional:
    - calculate_token_salience: Calculate salience scores (default: true)
    - best_of_k: Number of explanation rollouts (default: 8)
    - last_n_messages: Only analyze the last N messages (e.g., 2 for last user/assistant turn)
    """
    if not inference_service:
        raise HTTPException(500, "Inference service not initialized")
        
    try:
        # Convert messages to text
        #text = format_messages_to_text(request.messages)
        
        # Use provided model_id or find the first model in the requested group
        if request.model_id:
            model_id = request.model_id
        else:
            model_id = None
            if grouped_model_manager:
                origin_header = req.headers.get('origin', '') if req else ''
                host_header = req.headers.get('host', '') if req else ''
                public_only = ("kitft.com" in origin_header) or ("kitft.com" in host_header)
                groups = grouped_model_manager.get_model_list(public_only=public_only)
                for group in groups:
                    if group['group_id'] == request.model_group:
                        if group['models']:
                            model_id = group['models'][0]['id']
                        break
            
            if not model_id:
                raise HTTPException(400, f"Model group '{request.model_group}' not found")

        
        
        # Create analysis options with defaults
        options = {
            "calculate_token_salience": request.calculate_token_salience,
            "calculate_salience": request.calculate_token_salience,  # Both flags
            "optimize_explanations_config": {
                "best_of_k": request.best_of_k,
                "use_batched": True,
                "temperature": 1.0,
                "n_groups_per_rollout": None,
                "num_samples_per_iteration": 16,
                "salience_pct_threshold": 0.0
            },
            # All other options use defaults
            "batch_size": None,  # Auto-calculate
            "temperature": 1.0,
            "seed": 42,
            "no_eval": False,
            "tuned_lens": False,
            "logit_lens_analysis": False,
            "do_hard_tokens": False,
            "return_structured": True,
            "move_devices": False,
            "no_kl": True,
            "model_id": model_id,
            "use_chat_format": True
        }
        
        # Add last_n_messages if specified
        if request.last_n_messages is not None:
            options["last_n_messages"] = request.last_n_messages
        
        # Queue the request
        request_id = await inference_service.queue.add_request(
            text=request.messages,
            options={**options, **({"origin": req.headers.get("origin")} if req else {})},
            request_type="analyze"
        )
        
        # Get queue position
        queue_size = inference_service.queue.queue.qsize()
        
        return SlimResponse(
            request_id=request_id,
            status="queued",
            queue_position=queue_size,
            message=f"Analysis request queued at position {queue_size}"
        )
        
    except Exception as e:
        logger.error(f"Error in slim analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send_message", response_model=SlimResponse)
async def send_message(
    request: SendMessageRequest,
    authorized: bool = Depends(verify_api_key),
    req: Request = None,
):
    """
    Send a message and get a single response.
    
    Required:
    - messages: Chat-formatted conversation history
    - model_group: Which model group to use
    
    Optional:
    - temperature: Sampling temperature (default: 1.0)
    - max_tokens: Maximum tokens to generate (default: 1024)
    - model_id: Specific model to use
    """
    if not inference_service:
        raise HTTPException(500, "Inference service not initialized")
        
    try:
        # Use provided model_id or find the first model in the requested group
        if request.model_id:
            model_id = request.model_id
        else:
            model_id = None
            if grouped_model_manager:
                origin_header = req.headers.get('origin', '') if req else ''
                host_header = req.headers.get('host', '') if req else ''
                public_only = ("kitft.com" in origin_header) or ("kitft.com" in host_header)
                groups = grouped_model_manager.get_model_list(public_only=public_only)
                for group in groups:
                    if group['group_id'] == request.model_group:
                        if group['models']:
                            model_id = group['models'][0]['id']
                        break
            
            if not model_id:
                raise HTTPException(400, f"Model group '{request.model_group}' not found")
        
        # Create send_message options
        options = {
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "model_id": model_id,
            "use_cache": True
        }
        
        # Queue the request as send_message type
        request_id = await inference_service.queue.add_request(
            text="",  # Not used for send_message
            options={**options, **({"origin": req.headers.get("origin")} if req else {})},
            request_type="send_message"
        )
        
        # Get queue position
        queue_size = inference_service.queue.queue.qsize()
        
        return SlimResponse(
            request_id=request_id,
            status="queued",
            queue_position=queue_size,
            message=f"Message request queued at position {queue_size}"
        )
        
    except Exception as e:
        logger.error(f"Error in slim send_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{request_id}")
async def get_status(request_id: str, authorized: bool = Depends(verify_api_key)):
    """Get the status of a generation or analysis request."""
    if not inference_service:
        raise HTTPException(500, "Inference service not initialized")
        
    if request_id not in inference_service.queue.active_requests:
        raise HTTPException(status_code=404, detail="Request not found")
    
    request_data = inference_service.queue.active_requests[request_id]
    
    # Return simplified status
    return {
        "request_id": request_id,
        "status": request_data.get("status", "unknown"),
        "result": request_data.get("result") if request_data.get("status") == "completed" else None,
        "error": request_data.get("error") if request_data.get("status") == "failed" else None,
        "processing_time": (
            (request_data.get("completed_at") - request_data.get("started_at")).total_seconds()
            if request_data.get("completed_at") and request_data.get("started_at")
            else None
        )
    }


@router.get("/models")
async def list_model_groups(req: Request):
    """List available model groups for the slim API."""
    if not grouped_model_manager:
        return {"model_groups": []}
        
    origin_header = req.headers.get('origin', '')
    host_header = req.headers.get('host', '')
    public_only = ("kitft.com" in origin_header) or ("kitft.com" in host_header)

    groups = grouped_model_manager.get_model_list(public_only=public_only)
    
    # Get current loaded model info
    current_model_id = None
    if grouped_model_manager and hasattr(grouped_model_manager, 'current_model_id'):
        current_model_id = grouped_model_manager.current_model_id
    
    # Simplify the response for slim API
    return {
        "model_groups": [
            {
                "id": group['group_id'],
                "name": group['group_name'],
                "description": group.get('description', ''),
                "is_loaded": group.get('is_loaded', False),
                "default_layer": group['models'][0]['id'] if group.get('models') else None
            }
            for group in groups
        ],
        "current_model_id": current_model_id
    }


@router.get("/models/{group_id}/layers")
async def list_model_layers(group_id: str, req: Request):
    """List available layers/sub-models for a specific model group."""
    if not grouped_model_manager:
        raise HTTPException(404, "Model manager not initialized")
        
    origin_header = req.headers.get('origin', '')
    host_header = req.headers.get('host', '')
    public_only = ("kitft.com" in origin_header) or ("kitft.com" in host_header)
    groups = grouped_model_manager.get_model_list(public_only=public_only)
    
    # Debug: Log available groups
    available_groups = [g['group_id'] for g in groups]
    logger.info(f"Available groups: {available_groups}")
    logger.info(f"Requested group: {group_id}")
    
    # Get current loaded model info
    current_model_id = None
    if grouped_model_manager and hasattr(grouped_model_manager, 'current_model_id'):
        current_model_id = grouped_model_manager.current_model_id
    
    # Find the requested group
    for group in groups:
        if group['group_id'] == group_id:
            # Return simplified layer information
            return {
                "group_id": group_id,
                "group_name": group['group_name'],
                "is_loaded": group.get('is_loaded', False),
                "layers": [
                    {
                        "id": model['id'],
                        "name": model['name'],
                        "layer": model.get('layer'),
                        "description": model.get('description', ''),
                        "is_current": model['id'] == current_model_id
                    }
                    for model in group.get('models', [])
                ],
                "default_layer": group['models'][0]['id'] if group.get('models') else None
            }
    
    raise HTTPException(404, f"Model group '{group_id}' not found. Available groups: {available_groups}")


@router.get("/debug")
async def debug_endpoint():
    """Debug endpoint to verify changes are active"""
    return {"status": "slim_api updated", "timestamp": "2024-08-04", "changes": "get_model_list"}

@router.get("/")
async def slim_api_info():
    """Get information about the slim API."""
    return {
        "name": "Talkative Autoencoder Slim API",
        "version": "1.0.0",
        "description": "Simplified API for generation and analysis with minimal options",
        "endpoints": {
            "generate": {
                "path": "/api/slim/generate",
                "method": "POST",
                "description": "Generate text with minimal options"
            },
            "analyze": {
                "path": "/api/slim/analyze", 
                "method": "POST",
                "description": "Analyze text with token salience"
            },
            "send_message": {
                "path": "/api/slim/send_message",
                "method": "POST",
                "description": "Send a message and get a single response"
            },
            "status": {
                "path": "/api/slim/status/{request_id}",
                "method": "GET",
                "description": "Check request status"
            },
            "models": {
                "path": "/api/slim/models",
                "method": "GET",
                "description": "List available model groups"
            },
            "layers": {
                "path": "/api/slim/models/{group_id}/layers",
                "method": "GET",
                "description": "List layers for a specific model group"
            }
        }
    }