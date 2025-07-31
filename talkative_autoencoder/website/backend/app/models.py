from typing import Any

from pydantic import BaseModel, Field, validator

from .config import load_settings

settings = load_settings()


class OptimizeExplanationsConfig(BaseModel):
    """Configuration for explanation optimization"""

    use_batched: bool = True
    best_of_k: int = Field(default=8, ge=1, le=64, description="Number of rollouts to perform")
    n_groups_per_rollout: int = Field(default=8, ge=1, le=32, description="Batch size for rollouts")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    num_samples_per_iteration: int = Field(default=16, ge=1, le=64)
    salience_pct_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class AnalyzeOptions(BaseModel):
    """Options for text analysis"""

    # Main batch size - will be auto-calculated based on best_of_k if not provided
    # Note: max value validation removed - now handled dynamically per model
    batch_size: int | None = Field(default=None, ge=1)

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
    add_tokens: list[str] | None = None
    replace_left: str | None = None
    replace_right: str | None = None

    # Optimization configuration
    optimize_explanations_config: OptimizeExplanationsConfig | None = Field(
        default_factory=lambda: OptimizeExplanationsConfig()
    )

    @validator("batch_size", always=True)
    def calculate_batch_size(cls, v, values):
        """Auto-calculate batch size based on k_rollouts if not provided"""
        if v is None and "optimize_explanations_config" in values:
            config = values["optimize_explanations_config"]
            if config and config.best_of_k:
                # Total batch size of 256, divided by k_rollouts
                v = max(1, 256 // config.best_of_k)
        return v or 32  # Default to 32 if still None


class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoint"""

    text: str = Field(..., min_length=1, max_length=settings.max_text_length)
    options: AnalyzeOptions | None = Field(default_factory=AnalyzeOptions)

    @validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or just whitespace")
        return v.strip()


class TokenAnalysis(BaseModel):
    """Individual token analysis result"""

    position: int
    token: str
    explanation: str
    explanation_structured: list[str] | None = None
    token_salience: list[float] | None = None
    mse: float
    kl_divergence: float
    relative_rmse: float | None = None
    tuned_lens_top: str | None = None
    logit_lens_top: str | None = None
    layer: int | None = None


class AnalyzeResponse(BaseModel):
    """Response model for analysis endpoint"""

    request_id: str
    status: str
    metadata: dict[str, Any]
    data: list[TokenAnalysis] | None = None
    error: str | None = None
    queue_position: int | None = None
    processing_time: float | None = None


class WebSocketMessage(BaseModel):
    """WebSocket message format"""

    type: str  # 'analyze', 'status', 'result', 'error'
    request_id: str | None = None
    data: dict[str, Any] | None = None
