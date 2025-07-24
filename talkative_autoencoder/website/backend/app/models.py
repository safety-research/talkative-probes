from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any
from datetime import datetime

class OptimizeExplanationsConfig(BaseModel):
    """Configuration for explanation optimization"""
    use_batched: bool = True
    just_do_k_rollouts: int = Field(default=8, ge=1, le=64, description="Number of rollouts to perform")
    batch_size_for_rollouts: int = Field(default=8, ge=1, le=32, description="Batch size for rollouts")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    num_samples_per_iteration: int = Field(default=16, ge=1, le=64)
    salience_pct_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

class AnalyzeOptions(BaseModel):
    """Options for text analysis"""
    # Main batch size - will be auto-calculated based on just_do_k_rollouts if not provided
    batch_size: Optional[int] = Field(default=None, ge=1, le=256)
    
    # Core analysis options
    seed: int = Field(default=42)
    no_eval: bool = Field(default=False, description="Skip evaluation metrics (MSE, KL)")
    tuned_lens: bool = Field(default=False, description="Include TunedLens predictions")
    logit_lens_analysis: bool = Field(default=False, description="Add logit-lens predictions")
    
    # Generation parameters
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    do_hard_tokens: bool = Field(default=False)
    return_structured: bool = Field(default=True)
    move_devices: bool = Field(default=False)
    
    # Evaluation options
    no_kl: bool = Field(default=False, description="Skip KL divergence calculation")
    calculate_salience: bool = Field(default=True, description="Calculate salience scores")
    calculate_token_salience: bool = Field(default=True)
    
    # Token manipulation
    add_tokens: Optional[List[str]] = None
    replace_left: Optional[str] = None
    replace_right: Optional[str] = None
    
    # Optimization configuration
    optimize_explanations_config: Optional[OptimizeExplanationsConfig] = Field(
        default_factory=lambda: OptimizeExplanationsConfig()
    )
    
    @validator('batch_size', always=True)
    def calculate_batch_size(cls, v, values):
        """Auto-calculate batch size based on k_rollouts if not provided"""
        if v is None and 'optimize_explanations_config' in values:
            config = values['optimize_explanations_config']
            if config and config.just_do_k_rollouts:
                # Total batch size of 256, divided by k_rollouts
                v = max(1, 256 // config.just_do_k_rollouts)
        return v or 32  # Default to 32 if still None
    
class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoint"""
    text: str = Field(..., min_length=1, max_length=1000)
    options: Optional[AnalyzeOptions] = Field(default_factory=AnalyzeOptions)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or just whitespace')
        return v.strip()

class TokenAnalysis(BaseModel):
    """Individual token analysis result"""
    position: int
    token: str
    explanation: str
    explanation_structured: Optional[List[str]] = None
    token_salience: Optional[List[float]] = None
    mse: float
    kl_divergence: float
    relative_rmse: Optional[float] = None
    tuned_lens_top: Optional[str] = None
    logit_lens_top: Optional[str] = None
    layer: Optional[int] = None

class AnalyzeResponse(BaseModel):
    """Response model for analysis endpoint"""
    request_id: str
    status: str
    metadata: Dict[str, Any]
    data: Optional[List[TokenAnalysis]] = None
    error: Optional[str] = None
    queue_position: Optional[int] = None
    processing_time: Optional[float] = None
    
class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str  # 'analyze', 'status', 'result', 'error'
    request_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None