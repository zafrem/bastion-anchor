from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class SecurityOptions(BaseModel):
    noise_sigma: float = 0.0
    measure_bias: bool = False
    preserve_norm: bool = False
    timeout_ms: int = 0
    seed: int = 0


class SecureRequest(BaseModel):
    request_id: str = ""
    tenant_id: str = ""
    embedding: list[float]
    operation: str = "index"
    category: str = ""
    options: Optional[SecurityOptions] = None


class EmbedMetrics(BaseModel):
    noise_added: float = 0.0
    similarity_to_original: float = 0.0
    bias_score: float = 0.0
    strategy_used: str = ""
    original_norm: float = 0.0
    secured_norm: float = 0.0


class SecureResponse(BaseModel):
    request_id: str = ""
    secured_embedding: list[float]
    metrics: EmbedMetrics
    processing_time_ms: float = 0.0


class BatchSecureRequest(BaseModel):
    request_id: str = ""
    items: list[SecureRequest]


class BatchSecureResponse(BaseModel):
    request_id: str = ""
    items: list[SecureResponse]


class AttributePair(BaseModel):
    attribute_a: str
    attribute_b: str
    category: str = ""


class BiasRequest(BaseModel):
    tenant_id: str = ""
    test_words: list[str]
    attribute_pairs: list[AttributePair]
    word_embeddings: Optional[dict[str, list[float]]] = None


class BiasMeasurement(BaseModel):
    word: str
    category: str
    bias_score: float
    sim_a: float
    sim_b: float
    severity: str


class BiasResponse(BaseModel):
    measurements: list[BiasMeasurement]
    overall_bias_score: float
    interpretation: str


class EmbeddingPair(BaseModel):
    query_embedding: list[float]
    match_embedding: list[float]


class QualityRequest(BaseModel):
    tenant_id: str = ""
    noise_sigma: float = 0.0
    embedding_pairs: list[EmbeddingPair] = Field(default_factory=list)
    top_k: int = 10


class QualityResponse(BaseModel):
    noise_sigma: float
    recall_without: float
    recall_with: float
    recall_drop_percent: float
    avg_similarity_to_original: float
    status: str


class AnchorConfig(BaseModel):
    default_noise_sigma: float
    preserve_norm: bool
    strategy: str
    tenant_overrides: dict[str, float]
    bias_enabled: bool
    quality_enabled: bool


class HealthStatus(BaseModel):
    status: str
    checks: dict[str, str]
    version: str = "2.0.0"


class ErrorResponse(BaseModel):
    error: str
    code: int
    trace_id: str = ""


# ─── Phase 2: Output verification ─────────────────────────────────────────────

class UserContext(BaseModel):
    user_id: str = ""
    tenant_id: str = ""
    department: str = ""
    category: str = ""


class CategoryBaseline(BaseModel):
    category: str
    score: float
    measured_at: int = 0


class BiasBaseline(BaseModel):
    overall_score: float
    categories: list[CategoryBaseline] = Field(default_factory=list)


class VerifyOptions(BaseModel):
    check_bias: bool = False
    analyze_embedding: bool = False
    check_drift: bool = False
    measure_quality: bool = False
    bias_threshold: float = 0.0
    timeout_ms: int = 0


class VerifyRequest(BaseModel):
    request_id: str = ""
    trace_id: str = ""
    tenant_id: str = ""
    llm_response: str
    original_query: str = ""
    source_documents: list[str] = Field(default_factory=list)
    user: Optional[UserContext] = None
    phase1_baseline: Optional[BiasBaseline] = None
    options: Optional[VerifyOptions] = None


class ResponseCategoryBias(BaseModel):
    category: str
    score: float
    evidence: list[str] = Field(default_factory=list)


class ResponseBiasAnalysis(BaseModel):
    overall_score: float
    categories: list[ResponseCategoryBias] = Field(default_factory=list)
    severity: str
    exceeds_threshold: bool


class EmbeddingAnalysis(BaseModel):
    similarity_to_query: float
    similarity_to_context: float
    diversity_score: float
    semantic_drift_detected: bool


class DriftAnalysis(BaseModel):
    drift_detected: bool
    phase1_bias_score: float
    phase2_bias_score: float
    drift_amount: float
    drift_direction: str


class ResponseQuality(BaseModel):
    grounding_score: float
    coherence_score: float
    relevance_score: float
    overall_quality: float


class VerifyResponse(BaseModel):
    request_id: str = ""
    status: str = "UNKNOWN"
    bias_analysis: Optional[ResponseBiasAnalysis] = None
    embedding_analysis: Optional[EmbeddingAnalysis] = None
    drift_analysis: Optional[DriftAnalysis] = None
    quality: Optional[ResponseQuality] = None
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    processing_time_ms: float = 0.0


class EmbeddingAnalysisRequest(BaseModel):
    text: str
    comparison_texts: list[str] = Field(default_factory=list)


class EmbeddingAnalysisResponse(BaseModel):
    embedding: list[float] = Field(default_factory=list)
    similarities: list[float] = Field(default_factory=list)
    diversity: float = 0.0


class CategoryDrift(BaseModel):
    category: str
    phase1_score: float
    phase2_score: float
    change: float
    direction: str


class DriftRequest(BaseModel):
    phase1: Optional[BiasBaseline] = None
    phase2: Optional[ResponseBiasAnalysis] = None


class DriftResponse(BaseModel):
    drift_detected: bool
    drifts: list[CategoryDrift] = Field(default_factory=list)
    severity: str
