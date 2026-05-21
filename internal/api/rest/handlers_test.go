package rest

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/bastion/anchor/internal/bias"
	"github.com/bastion/anchor/internal/config"
	"github.com/bastion/anchor/internal/models"
	"github.com/bastion/anchor/internal/noise"
	"github.com/bastion/anchor/internal/publisher"
	"github.com/bastion/anchor/internal/quality"
	"github.com/bastion/anchor/internal/verify"
)

// ─── Test helpers ─────────────────────────────────────────────────────────────

func newRouter(t *testing.T) http.Handler {
	t.Helper()
	cfg := config.Defaults()
	inj, err := noise.New(cfg.Noise.DefaultSigma, cfg.Noise.PreserveNorm, noise.StrategyGaussian)
	if err != nil {
		t.Fatal(err)
	}
	h := &handlers{
		cfg: cfg,
		inj: inj,
		ana: bias.New(cfg.Bias.AlertMedium, cfg.Bias.AlertHigh, int(cfg.Embedding.ExpectedDimensions)),
		mon: quality.New(cfg.Quality.RecallDropAlert, cfg.Quality.SimilarityMinimum),
		pub: publisher.NewNoop(),
		ver: verify.New(cfg.Bias.AlertMedium, cfg.Bias.AlertHigh),
	}
	s := &Server{}
	return s.routes(h)
}

func post(t *testing.T, router http.Handler, path string, body interface{}) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(body); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, path, &buf)
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)
	return rr
}

func get(t *testing.T, router http.Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)
	return rr
}

func unit(dims int) []float32 {
	vec := make([]float32, dims)
	v := float32(1.0 / math.Sqrt(float64(dims)))
	for i := range vec {
		vec[i] = v
	}
	return vec
}

func decode(t *testing.T, rr *httptest.ResponseRecorder, dst interface{}) {
	t.Helper()
	if err := json.NewDecoder(rr.Body).Decode(dst); err != nil {
		t.Fatalf("decode response: %v (body: %s)", err, rr.Body.String())
	}
}

// ─── Health ───────────────────────────────────────────────────────────────────

func TestHealth_Returns200(t *testing.T) {
	rr := get(t, newRouter(t), "/v1/health")
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
}

func TestHealth_Live_Returns200(t *testing.T) {
	rr := get(t, newRouter(t), "/v1/health/live")
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
}

func TestHealth_Ready_Returns200(t *testing.T) {
	rr := get(t, newRouter(t), "/v1/health/ready")
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
}

// ─── Secure ───────────────────────────────────────────────────────────────────

func TestSecure_ValidEmbedding_Returns200(t *testing.T) {
	body := models.SecureRequest{
		RequestID: "req-1",
		TenantID:  "acme",
		Embedding: unit(64),
		Operation: "index",
	}
	rr := post(t, newRouter(t), "/v1/anchor/secure", body)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d; body: %s", rr.Code, rr.Body.String())
	}
}

func TestSecure_ValidEmbedding_ResponseHasSecuredEmbedding(t *testing.T) {
	dims := 64
	body := models.SecureRequest{
		RequestID: "req-2",
		TenantID:  "acme",
		Embedding: unit(dims),
	}
	rr := post(t, newRouter(t), "/v1/anchor/secure", body)
	var resp models.SecureResponse
	decode(t, rr, &resp)
	if len(resp.SecuredEmbedding) != dims {
		t.Errorf("expected %d dims, got %d", dims, len(resp.SecuredEmbedding))
	}
}

func TestSecure_EmptyEmbedding_Returns400(t *testing.T) {
	body := models.SecureRequest{TenantID: "acme"}
	rr := post(t, newRouter(t), "/v1/anchor/secure", body)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestSecure_InvalidJSON_Returns400(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/anchor/secure", bytes.NewBufferString("{bad json}"))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	newRouter(t).ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestSecure_DeterministicSeed_ReproducibleResult(t *testing.T) {
	router := newRouter(t)
	body := models.SecureRequest{
		RequestID: "r",
		Embedding: unit(32),
		Options:   &models.SecurityOptions{Seed: 77, NoiseSigma: 0.01},
	}
	rr1 := post(t, router, "/v1/anchor/secure", body)
	rr2 := post(t, router, "/v1/anchor/secure", body)

	var r1, r2 models.SecureResponse
	decode(t, rr1, &r1)
	decode(t, rr2, &r2)

	for i := range r1.SecuredEmbedding {
		if r1.SecuredEmbedding[i] != r2.SecuredEmbedding[i] {
			t.Fatalf("deterministic seed produced different results at dim %d", i)
		}
	}
}

func TestSecure_MetricsIncluded(t *testing.T) {
	body := models.SecureRequest{Embedding: unit(16)}
	rr := post(t, newRouter(t), "/v1/anchor/secure", body)
	var resp models.SecureResponse
	decode(t, rr, &resp)
	if resp.Metrics.SimilarityToOriginal == 0 {
		t.Error("similarity_to_original should be non-zero")
	}
}

// ─── SecureBatch ─────────────────────────────────────────────────────────────

func TestSecureBatch_Returns200(t *testing.T) {
	body := models.BatchSecureRequest{
		RequestID: "batch-1",
		Items: []models.SecureRequest{
			{RequestID: "a", Embedding: unit(16)},
			{RequestID: "b", Embedding: unit(16)},
		},
	}
	rr := post(t, newRouter(t), "/v1/anchor/secure/batch", body)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
}

func TestSecureBatch_CountMatches(t *testing.T) {
	body := models.BatchSecureRequest{
		Items: []models.SecureRequest{
			{Embedding: unit(8)},
			{Embedding: unit(8)},
			{Embedding: unit(8)},
		},
	}
	rr := post(t, newRouter(t), "/v1/anchor/secure/batch", body)
	var resp models.BatchSecureResponse
	decode(t, rr, &resp)
	if len(resp.Items) != 3 {
		t.Errorf("expected 3 items, got %d", len(resp.Items))
	}
}

// ─── MeasureBias ─────────────────────────────────────────────────────────────

func TestMeasureBias_Returns200(t *testing.T) {
	body := models.BiasRequest{
		TenantID:  "acme",
		TestWords: []string{"doctor", "nurse"},
		AttributePairs: []models.AttributePair{
			{AttributeA: "male", AttributeB: "female", Category: "gender"},
		},
	}
	rr := post(t, newRouter(t), "/v1/anchor/measure-bias", body)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d; body: %s", rr.Code, rr.Body.String())
	}
}

func TestMeasureBias_HasMeasurements(t *testing.T) {
	body := models.BiasRequest{
		TestWords: []string{"doctor"},
		AttributePairs: []models.AttributePair{
			{AttributeA: "male", AttributeB: "female", Category: "gender"},
		},
	}
	rr := post(t, newRouter(t), "/v1/anchor/measure-bias", body)
	var resp models.BiasResponse
	decode(t, rr, &resp)
	if len(resp.Measurements) == 0 {
		t.Error("expected at least one measurement")
	}
}

// ─── MeasureQuality ───────────────────────────────────────────────────────────

func TestMeasureQuality_NoPairs_Returns200(t *testing.T) {
	body := models.QualityRequest{NoiseSigma: 0.01}
	rr := post(t, newRouter(t), "/v1/anchor/measure-quality", body)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d; body: %s", rr.Code, rr.Body.String())
	}
}

func TestMeasureQuality_WithPairs_Returns200(t *testing.T) {
	q := unit(32)
	body := models.QualityRequest{
		NoiseSigma: 0.01,
		EmbeddingPairs: []models.EmbeddingPair{
			{QueryEmbedding: q, MatchEmbedding: q},
		},
	}
	rr := post(t, newRouter(t), "/v1/anchor/measure-quality", body)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d; body: %s", rr.Code, rr.Body.String())
	}
}

func TestMeasureQuality_ZeroSigma_UsesDefault(t *testing.T) {
	body := models.QualityRequest{NoiseSigma: 0}
	rr := post(t, newRouter(t), "/v1/anchor/measure-quality", body)
	var resp models.QualityResponse
	decode(t, rr, &resp)
	if resp.NoiseSigma <= 0 {
		t.Error("zero sigma should fall back to config default, got non-positive")
	}
}

// ─── Config ───────────────────────────────────────────────────────────────────

func TestGetConfig_Returns200(t *testing.T) {
	rr := get(t, newRouter(t), "/v1/anchor/config")
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
}

func TestGetConfig_HasSigma(t *testing.T) {
	rr := get(t, newRouter(t), "/v1/anchor/config")
	var cfg models.AnchorConfig
	decode(t, rr, &cfg)
	if cfg.DefaultNoiseSigma <= 0 {
		t.Errorf("default_noise_sigma should be positive, got %f", cfg.DefaultNoiseSigma)
	}
}

func TestUpdateConfig_NewSigma_ReflectedInResponse(t *testing.T) {
	router := newRouter(t)
	update := models.AnchorConfig{DefaultNoiseSigma: 0.05}
	rr := post(t, router, "/v1/anchor/config", update)
	var resp models.AnchorConfig
	decode(t, rr, &resp)
	if math.Abs(resp.DefaultNoiseSigma-0.05) > 1e-6 {
		t.Errorf("expected updated sigma 0.05, got %f", resp.DefaultNoiseSigma)
	}
}

// ─── Phase 2: Verify ─────────────────────────────────────────────────────────

func TestVerify_ValidRequest_Returns200(t *testing.T) {
	body := models.VerifyRequest{
		RequestID:   "v-1",
		LLMResponse: "Engineers are analytical thinkers who solve complex problems.",
		OriginalQuery: "What are engineers like?",
	}
	rr := post(t, newRouter(t), "/v1/anchor/verify", body)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d; body: %s", rr.Code, rr.Body.String())
	}
}

func TestVerify_EmptyResponse_Returns400(t *testing.T) {
	body := models.VerifyRequest{RequestID: "v-2"}
	rr := post(t, newRouter(t), "/v1/anchor/verify", body)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestVerify_HasStatus(t *testing.T) {
	body := models.VerifyRequest{
		LLMResponse:   "A thoughtful and balanced answer.",
		OriginalQuery: "Tell me about this topic.",
	}
	rr := post(t, newRouter(t), "/v1/anchor/verify", body)
	var resp models.VerifyResponse
	decode(t, rr, &resp)
	if resp.Status == "" || resp.Status == models.VerifyStatusUnknown {
		t.Errorf("expected a definitive status, got %q", resp.Status)
	}
}

func TestVerify_BiasedResponse_DetectedAsWarningOrCritical(t *testing.T) {
	body := models.VerifyRequest{
		LLMResponse:   "Men who possess these qualities tend to excel. Women are too emotional for leadership.",
		OriginalQuery: "Who makes a good leader?",
	}
	rr := post(t, newRouter(t), "/v1/anchor/verify", body)
	var resp models.VerifyResponse
	decode(t, rr, &resp)
	if resp.Status != models.VerifyStatusWarning && resp.Status != models.VerifyStatusCritical {
		t.Errorf("biased response should be WARNING or CRITICAL, got %q", resp.Status)
	}
}

func TestVerify_WithPhase1Baseline_HasDriftAnalysis(t *testing.T) {
	body := models.VerifyRequest{
		LLMResponse:   "Men who possess these qualities tend to excel.",
		OriginalQuery: "What makes someone successful?",
		Phase1Baseline: &models.BiasBaseline{
			OverallScore: 0.02,
		},
	}
	rr := post(t, newRouter(t), "/v1/anchor/verify", body)
	var resp models.VerifyResponse
	decode(t, rr, &resp)
	if resp.DriftAnalysis == nil {
		t.Error("expected drift analysis when phase1_baseline is provided")
	}
}

// ─── Phase 2: AnalyzeEmbedding ────────────────────────────────────────────────

func TestAnalyzeEmbedding_Returns200(t *testing.T) {
	body := models.EmbeddingAnalysisRequest{
		Text:            "The quick brown fox jumps over the lazy dog.",
		ComparisonTexts: []string{"A fast fox leaps over a sleeping dog."},
	}
	rr := post(t, newRouter(t), "/v1/anchor/analyze-embedding", body)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d; body: %s", rr.Code, rr.Body.String())
	}
}

func TestAnalyzeEmbedding_EmptyText_Returns400(t *testing.T) {
	body := models.EmbeddingAnalysisRequest{ComparisonTexts: []string{"some text"}}
	rr := post(t, newRouter(t), "/v1/anchor/analyze-embedding", body)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestAnalyzeEmbedding_SimilarTexts_HighSimilarity(t *testing.T) {
	text := "The engineer solved the problem efficiently."
	body := models.EmbeddingAnalysisRequest{
		Text:            text,
		ComparisonTexts: []string{text},
	}
	rr := post(t, newRouter(t), "/v1/anchor/analyze-embedding", body)
	var resp models.EmbeddingAnalysisResponse
	decode(t, rr, &resp)
	if len(resp.Similarities) == 0 || resp.Similarities[0] < 0.9 {
		t.Errorf("identical texts should have similarity ≥ 0.9, got %v", resp.Similarities)
	}
}

// ─── Phase 2: CheckDrift ─────────────────────────────────────────────────────

func TestCheckDrift_Returns200(t *testing.T) {
	body := models.DriftRequest{
		Phase1: &models.BiasBaseline{OverallScore: 0.05},
		Phase2: &models.ResponseBiasAnalysis{OverallScore: 0.25, Severity: "high"},
	}
	rr := post(t, newRouter(t), "/v1/anchor/check-drift", body)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d; body: %s", rr.Code, rr.Body.String())
	}
}

func TestCheckDrift_NilInputs_Returns200(t *testing.T) {
	body := models.DriftRequest{}
	rr := post(t, newRouter(t), "/v1/anchor/check-drift", body)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 for nil inputs, got %d", rr.Code)
	}
}
