package models

// SecureRequest is the input for a single embedding security operation.
type SecureRequest struct {
	RequestID string          `json:"request_id"`
	TenantID  string          `json:"tenant_id"`
	Embedding []float32       `json:"embedding"`
	Operation string          `json:"operation"` // "index" | "search"
	Category  string          `json:"category"`
	Options   *SecurityOptions `json:"options,omitempty"`
}

// SecurityOptions overrides per-request security behaviour.
type SecurityOptions struct {
	NoiseSigma   float64 `json:"noise_sigma"`
	MeasureBias  bool    `json:"measure_bias"`
	PreserveNorm bool    `json:"preserve_norm"`
	TimeoutMs    int     `json:"timeout_ms"`
	Seed         int64   `json:"seed"` // 0 = non-deterministic
}

// SecureResponse carries the secured embedding and pipeline metrics.
type SecureResponse struct {
	RequestID        string   `json:"request_id"`
	SecuredEmbedding []float32 `json:"secured_embedding"`
	Metrics          EmbedMetrics `json:"metrics"`
	ProcessingTimeMs float64  `json:"processing_time_ms"`
}

// EmbedMetrics describes what the noise injector did.
type EmbedMetrics struct {
	NoiseAdded           float64 `json:"noise_added"`
	SimilarityToOriginal float64 `json:"similarity_to_original"`
	BiasScore            float64 `json:"bias_score,omitempty"`
	StrategyUsed         string  `json:"strategy_used"`
	OriginalNorm         float64 `json:"original_norm"`
	SecuredNorm          float64 `json:"secured_norm"`
}

// BatchSecureRequest batches multiple secure requests.
type BatchSecureRequest struct {
	RequestID string          `json:"request_id"`
	Items     []SecureRequest `json:"items"`
}

// BatchSecureResponse batches multiple secure responses.
type BatchSecureResponse struct {
	RequestID string           `json:"request_id"`
	Items     []SecureResponse `json:"items"`
}

// AttributePair defines a pair of contrasting attribute words (e.g. male/female).
type AttributePair struct {
	AttributeA string `json:"attribute_a"`
	AttributeB string `json:"attribute_b"`
	Category   string `json:"category"` // "gender", "ethnicity", "age"
}

// BiasRequest asks for a bias measurement run.
type BiasRequest struct {
	TenantID       string          `json:"tenant_id"`
	TestWords      []string        `json:"test_words"`
	AttributePairs []AttributePair `json:"attribute_pairs"`
	// Embeddings are optional; if nil the analyzer uses zero/random vectors in demo mode.
	WordEmbeddings map[string][]float32 `json:"word_embeddings,omitempty"`
}

// BiasMeasurement is the result for a single (word, attribute-pair) combination.
type BiasMeasurement struct {
	Word       string  `json:"word"`
	Category   string  `json:"category"`
	BiasScore  float64 `json:"bias_score"`
	SimA       float64 `json:"sim_a"`
	SimB       float64 `json:"sim_b"`
	Severity   string  `json:"severity"` // "low" | "medium" | "high"
}

// BiasResponse aggregates all measurements.
type BiasResponse struct {
	Measurements     []BiasMeasurement `json:"measurements"`
	OverallBiasScore float64           `json:"overall_bias_score"`
	Interpretation   string            `json:"interpretation"`
}

// QualityRequest asks for a quality impact measurement.
type QualityRequest struct {
	TenantID   string    `json:"tenant_id"`
	NoiseSigma float64   `json:"noise_sigma"`
	// Pairs of (original, expected-match) embeddings for recall simulation.
	EmbeddingPairs []EmbeddingPair `json:"embedding_pairs"`
	TopK           int             `json:"top_k"`
}

// EmbeddingPair links a query embedding to its ground-truth match embedding.
type EmbeddingPair struct {
	QueryEmbedding    []float32 `json:"query_embedding"`
	MatchEmbedding    []float32 `json:"match_embedding"`
}

// QualityResponse reports the recall and similarity impact.
type QualityResponse struct {
	NoiseSigma              float64 `json:"noise_sigma"`
	RecallWithout           float64 `json:"recall_without"`
	RecallWith              float64 `json:"recall_with"`
	RecallDropPercent       float64 `json:"recall_drop_percent"`
	AvgSimilarityToOriginal float64 `json:"avg_similarity_to_original"`
	Status                  string  `json:"status"` // "acceptable" | "warning" | "critical"
}

// AnchorConfig is the runtime configuration returned by the config endpoints.
type AnchorConfig struct {
	DefaultNoiseSigma float64            `json:"default_noise_sigma"`
	PreserveNorm      bool               `json:"preserve_norm"`
	Strategy          string             `json:"strategy"`
	TenantOverrides   map[string]float64 `json:"tenant_overrides"`
	BiasEnabled       bool               `json:"bias_enabled"`
	QualityEnabled    bool               `json:"quality_enabled"`
}

// HealthStatus is the standard health envelope.
type HealthStatus struct {
	Status  string            `json:"status"`
	Checks  map[string]string `json:"checks"`
	Version string            `json:"version"`
}

// ErrorResponse is the standard error envelope.
type ErrorResponse struct {
	Error   string `json:"error"`
	Code    int    `json:"code"`
	TraceID string `json:"trace_id,omitempty"`
}

// BastionEvent is the Foundation-standard event emitted to NATS (doc 02).
// Subject: bastion.events.anchor.{event_type}
type BastionEvent struct {
	EventID       string                 `json:"event_id"`        // UUID
	EventType     string                 `json:"event_type"`
	SchemaVersion string                 `json:"schema_version"`
	TraceID       string                 `json:"trace_id"`
	SpanID        string                 `json:"span_id"`
	ParentSpanID  string                 `json:"parent_span_id,omitempty"`
	Module        string                 `json:"module"`
	ModuleVersion string                 `json:"module_version"`
	Timestamp     int64                  `json:"timestamp"`       // Unix nano
	DurationMs    int64                  `json:"duration_ms,omitempty"`
	TenantID      string                 `json:"tenant_id"`
	UserID        string                 `json:"user_id,omitempty"`
	RequestID     string                 `json:"request_id,omitempty"`
	Severity      string                 `json:"severity"`        // info, warning, error, critical
	Category      string                 `json:"category"`        // operational, security, performance, audit
	Data          map[string]interface{} `json:"data,omitempty"`
	Status        string                 `json:"status,omitempty"`
	ActionTaken   string                 `json:"action_taken,omitempty"`
}

// TrackerEvent is kept for backwards compatibility; prefer BastionEvent.
type TrackerEvent = BastionEvent

// gRPC empty request types
type ConfigRequest struct{}
type HealthRequest struct{}

// ─── Phase 2: Output verification types (Anchor-OUT) ─────────────────────────

// VerifyRequest asks Anchor-OUT to verify an LLM response.
type VerifyRequest struct {
	RequestID       string         `json:"request_id"`
	TraceID         string         `json:"trace_id"`
	TenantID        string         `json:"tenant_id"`
	LLMResponse     string         `json:"llm_response"`
	OriginalQuery   string         `json:"original_query"`
	SourceDocuments []string       `json:"source_documents"`
	User            *UserContext   `json:"user,omitempty"`
	Phase1Baseline  *BiasBaseline  `json:"phase1_baseline,omitempty"`
	Options         *VerifyOptions `json:"options,omitempty"`
}

// UserContext carries user metadata for category-specific bias checks.
type UserContext struct {
	UserID     string `json:"user_id"`
	TenantID   string `json:"tenant_id"`
	Department string `json:"department"`
	Category   string `json:"category"`
}

// BiasBaseline carries Phase 1 bias measurements used for drift comparison.
type BiasBaseline struct {
	OverallScore float64            `json:"overall_score"`
	Categories   []CategoryBaseline `json:"categories"`
}

// CategoryBaseline is a single category's Phase 1 bias score.
type CategoryBaseline struct {
	Category   string  `json:"category"`
	Score      float64 `json:"score"`
	MeasuredAt int64   `json:"measured_at"`
}

// VerifyOptions controls which Phase 2 checks Anchor-OUT performs.
// If nil or all flags false, all checks are enabled.
type VerifyOptions struct {
	CheckBias        bool    `json:"check_bias"`
	AnalyzeEmbedding bool    `json:"analyze_embedding"`
	CheckDrift       bool    `json:"check_drift"`
	MeasureQuality   bool    `json:"measure_quality"`
	BiasThreshold    float64 `json:"bias_threshold"`
	TimeoutMs        int     `json:"timeout_ms"`
}

// VerifyStatus is the overall outcome of a Phase 2 verification run.
type VerifyStatus string

const (
	VerifyStatusUnknown  VerifyStatus = "UNKNOWN"
	VerifyStatusPassed   VerifyStatus = "PASSED"
	VerifyStatusWarning  VerifyStatus = "WARNING"
	VerifyStatusCritical VerifyStatus = "CRITICAL"
)

// VerifyResponse is the result of a Phase 2 verification run.
type VerifyResponse struct {
	RequestID         string                `json:"request_id"`
	Status            VerifyStatus          `json:"status"`
	BiasAnalysis      *ResponseBiasAnalysis `json:"bias_analysis,omitempty"`
	EmbeddingAnalysis *EmbeddingAnalysis    `json:"embedding_analysis,omitempty"`
	DriftAnalysis     *DriftAnalysis        `json:"drift_analysis,omitempty"`
	Quality           *ResponseQuality      `json:"quality,omitempty"`
	Warnings          []string              `json:"warnings,omitempty"`
	Recommendations   []string              `json:"recommendations,omitempty"`
	ProcessingTimeMs  float64               `json:"processing_time_ms"`
}

// ResponseBiasAnalysis aggregates bias findings for an LLM response.
type ResponseBiasAnalysis struct {
	OverallScore     float64               `json:"overall_score"`
	Categories       []ResponseCategoryBias `json:"categories,omitempty"`
	Severity         string                `json:"severity"`
	ExceedsThreshold bool                  `json:"exceeds_threshold"`
}

// ResponseCategoryBias is bias detected for one category in the response.
type ResponseCategoryBias struct {
	Category string   `json:"category"`
	Score    float64  `json:"score"`
	Evidence []string `json:"evidence,omitempty"`
}

// EmbeddingAnalysis captures response embedding characteristics.
// In PoC mode this uses word-overlap (Jaccard) as a similarity proxy.
type EmbeddingAnalysis struct {
	SimilarityToQuery     float64 `json:"similarity_to_query"`
	SimilarityToContext   float64 `json:"similarity_to_context"`
	DiversityScore        float64 `json:"diversity_score"`
	SemanticDriftDetected bool    `json:"semantic_drift_detected"`
}

// DriftAnalysis compares Phase 1 embedding bias vs Phase 2 response bias.
type DriftAnalysis struct {
	DriftDetected   bool    `json:"drift_detected"`
	Phase1BiasScore float64 `json:"phase1_bias_score"`
	Phase2BiasScore float64 `json:"phase2_bias_score"`
	DriftAmount     float64 `json:"drift_amount"`
	// DriftDirection is one of "amplification", "introduction", "reduction", "none".
	DriftDirection string `json:"drift_direction"`
}

// ResponseQuality measures end-to-end RAG response quality.
type ResponseQuality struct {
	GroundingScore float64 `json:"grounding_score"`
	CoherenceScore float64 `json:"coherence_score"`
	RelevanceScore float64 `json:"relevance_score"`
	OverallQuality float64 `json:"overall_quality"`
}

// EmbeddingAnalysisRequest asks for a standalone embedding comparison.
type EmbeddingAnalysisRequest struct {
	Text            string   `json:"text"`
	ComparisonTexts []string `json:"comparison_texts"`
}

// EmbeddingAnalysisResponse carries embedding similarities and diversity.
type EmbeddingAnalysisResponse struct {
	Embedding    []float32 `json:"embedding"`
	Similarities []float64 `json:"similarities"`
	Diversity    float64   `json:"diversity"`
}

// DriftRequest asks for a standalone Phase 1 vs Phase 2 drift comparison.
type DriftRequest struct {
	Phase1 *BiasBaseline         `json:"phase1"`
	Phase2 *ResponseBiasAnalysis `json:"phase2"`
}

// DriftResponse reports drift between Phase 1 and Phase 2 bias scores.
type DriftResponse struct {
	DriftDetected bool            `json:"drift_detected"`
	Drifts        []CategoryDrift `json:"drifts,omitempty"`
	Severity      string          `json:"severity"`
}

// CategoryDrift is the drift for a single bias category.
type CategoryDrift struct {
	Category    string  `json:"category"`
	Phase1Score float64 `json:"phase1_score"`
	Phase2Score float64 `json:"phase2_score"`
	Change      float64 `json:"change"`
	// Direction is one of "amplification", "introduction", "reduction", "none".
	Direction string `json:"direction"`
}
