package rest

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/google/uuid"

	"github.com/bastion/anchor/internal/bias"
	"github.com/bastion/anchor/internal/config"
	"github.com/bastion/anchor/internal/hooks"
	"github.com/bastion/anchor/internal/metrics"
	"github.com/bastion/anchor/internal/models"
	"github.com/bastion/anchor/internal/noise"
	"github.com/bastion/anchor/internal/publisher"
	"github.com/bastion/anchor/internal/quality"
	"github.com/bastion/anchor/internal/verify"
)

type handlers struct {
	cfg *config.Config
	inj *noise.Injector
	ana *bias.Analyzer
	mon *quality.Monitor
	pub publisher.Publisher
	ver *verify.Verifier
	hm  *hooks.Manager
}

// ─── Health ───────────────────────────────────────────────────────────────────

func (h *handlers) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, models.HealthStatus{
		Status:  "ok",
		Version: "1.0.0",
		Checks:  map[string]string{"service": "up"},
	})
}

func (h *handlers) Live(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "alive"})
}

func (h *handlers) Ready(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// ─── Secure ───────────────────────────────────────────────────────────────────

func (h *handlers) Secure(w http.ResponseWriter, r *http.Request) {
	var req models.SecureRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if len(req.Embedding) == 0 {
		writeError(w, http.StatusBadRequest, "embedding must not be empty")
		return
	}

	// Apply per-tenant sigma override if options don't set it explicitly.
	inj := h.injectorFor(req.TenantID, req.Options)

	start := time.Now()
	result := inj.Inject(req.Embedding)
	elapsed := time.Since(start).Seconds() * 1000

	metrics.EmbeddingsSecured.WithLabelValues(req.TenantID, req.Operation).Inc()
	metrics.SecureDuration.Observe(time.Since(start).Seconds())
	metrics.NoiseMagnitude.Observe(result.NoiseMagnitude)
	metrics.SimilarityToOriginal.Observe(result.SimilarityToOriginal)

	tc := traceFrom(r, req.TenantID, "", req.RequestID)
	go h.pub.Publish(publisher.EmbeddingSecured(tc, req.Operation, result.NoiseMagnitude, result.SimilarityToOriginal))
	h.hm.Fire(hooks.Event{
		Type:      hooks.EventEmbeddingSecured,
		RequestID: req.RequestID,
		TenantID:  req.TenantID,
		TraceID:   tc.TraceID,
		SpanID:    tc.SpanID,
		Data: map[string]interface{}{
			"noise":      result.NoiseMagnitude,
			"similarity": result.SimilarityToOriginal,
			"operation":  req.Operation,
		},
		Ctx: r.Context(),
	})

	writeJSON(w, http.StatusOK, models.SecureResponse{
		RequestID:        req.RequestID,
		SecuredEmbedding: result.Secured,
		Metrics: models.EmbedMetrics{
			NoiseAdded:           result.NoiseMagnitude,
			SimilarityToOriginal: result.SimilarityToOriginal,
			StrategyUsed:         result.Strategy,
			OriginalNorm:         result.OriginalNorm,
			SecuredNorm:          result.SecuredNorm,
		},
		ProcessingTimeMs: elapsed,
	})
}

func (h *handlers) SecureBatch(w http.ResponseWriter, r *http.Request) {
	var req models.BatchSecureRequest
	if !decodeJSON(w, r, &req) {
		return
	}

	responses := make([]models.SecureResponse, 0, len(req.Items))
	for _, item := range req.Items {
		inj := h.injectorFor(item.TenantID, item.Options)
		start := time.Now()
		result := inj.Inject(item.Embedding)
		elapsed := time.Since(start).Seconds() * 1000

		metrics.EmbeddingsSecured.WithLabelValues(item.TenantID, item.Operation).Inc()
		metrics.SecureDuration.Observe(time.Since(start).Seconds())

		responses = append(responses, models.SecureResponse{
			RequestID:        item.RequestID,
			SecuredEmbedding: result.Secured,
			Metrics: models.EmbedMetrics{
				NoiseAdded:           result.NoiseMagnitude,
				SimilarityToOriginal: result.SimilarityToOriginal,
				StrategyUsed:         result.Strategy,
				OriginalNorm:         result.OriginalNorm,
				SecuredNorm:          result.SecuredNorm,
			},
			ProcessingTimeMs: elapsed,
		})
	}
	writeJSON(w, http.StatusOK, models.BatchSecureResponse{
		RequestID: req.RequestID,
		Items:     responses,
	})
}

// ─── Bias ─────────────────────────────────────────────────────────────────────

func (h *handlers) MeasureBias(w http.ResponseWriter, r *http.Request) {
	var req models.BiasRequest
	if !decodeJSON(w, r, &req) {
		return
	}

	resp := h.ana.Analyze(req.TestWords, req.AttributePairs, req.WordEmbeddings)

	metrics.BiasAnalysesTotal.Add(1)
	// Emit bias_detected events for high-severity findings.
	categoryTerms := make(map[string][]string)
	for _, m := range resp.Measurements {
		if m.Severity == bias.SeverityHigh {
			categoryTerms[m.Category] = append(categoryTerms[m.Category], m.Word)
		}
	}
	tc := traceFrom(r, req.TenantID, "", "")
	for cat, terms := range categoryTerms {
		metrics.AlertsTotal.WithLabelValues("bias_detected").Inc()
		go h.pub.Publish(publisher.BiasDetected(tc, cat, resp.OverallBiasScore, terms))
		h.hm.Fire(hooks.Event{
			Type:     hooks.EventBiasDetected,
			TenantID: req.TenantID,
			TraceID:  tc.TraceID,
			SpanID:   tc.SpanID,
			Data: map[string]interface{}{
				"category":   cat,
				"bias_score": resp.OverallBiasScore,
				"terms":      terms,
			},
			Ctx: r.Context(),
		})
	}

	writeJSON(w, http.StatusOK, resp)
}

// ─── Quality ──────────────────────────────────────────────────────────────────

func (h *handlers) MeasureQuality(w http.ResponseWriter, r *http.Request) {
	var req models.QualityRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.NoiseSigma == 0 {
		req.NoiseSigma = h.cfg.Noise.DefaultSigma
	}

	resp := h.mon.Estimate(req.NoiseSigma, req.EmbeddingPairs)
	metrics.QualityChecksTotal.Add(1)
	if resp.RecallWithout > 0 {
		metrics.RecallPreservation.Set(resp.RecallWith / resp.RecallWithout)
	}

	if resp.Status != "acceptable" {
		tc := traceFrom(r, req.TenantID, "", "")
		metrics.AlertsTotal.WithLabelValues("quality_degraded").Inc()
		go h.pub.Publish(publisher.QualityDegraded(tc, resp.RecallWithout, resp.RecallWith, req.NoiseSigma))
	}

	writeJSON(w, http.StatusOK, resp)
}

// ─── Phase 2: Output verification ─────────────────────────────────────────────

func (h *handlers) Verify(w http.ResponseWriter, r *http.Request) {
	var req models.VerifyRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.LLMResponse == "" {
		writeError(w, http.StatusBadRequest, "llm_response must not be empty")
		return
	}

	start := time.Now()
	resp := h.ver.Verify(&req)
	elapsed := time.Since(start).Seconds()

	metrics.VerificationsTotal.WithLabelValues(string(resp.Status)).Inc()
	metrics.VerifyDuration.Observe(elapsed)

	tc := traceFrom(r, req.TenantID, "", req.RequestID)

	if resp.BiasAnalysis != nil && resp.BiasAnalysis.ExceedsThreshold {
		metrics.BiasDetectedTotal.WithLabelValues(resp.BiasAnalysis.Severity).Inc()
		metrics.AlertsTotal.WithLabelValues("response_bias_detected").Inc()
		go h.pub.Publish(publisher.BiasDetectedResponse(tc, "response", resp.BiasAnalysis.Severity, resp.BiasAnalysis.OverallScore))
	}

	if resp.DriftAnalysis != nil && resp.DriftAnalysis.DriftDetected {
		metrics.DriftDetectedTotal.WithLabelValues(resp.DriftAnalysis.DriftDirection).Inc()
		switch resp.DriftAnalysis.DriftDirection {
		case "amplification":
			metrics.AlertsTotal.WithLabelValues("bias_amplification").Inc()
			go h.pub.Publish(publisher.BiasAmplificationDetected(tc, resp.DriftAnalysis.Phase1BiasScore, resp.DriftAnalysis.Phase2BiasScore))
		case "introduction":
			metrics.AlertsTotal.WithLabelValues("bias_introduction").Inc()
			go h.pub.Publish(publisher.BiasIntroductionDetected(tc, resp.DriftAnalysis.Phase1BiasScore, resp.DriftAnalysis.Phase2BiasScore))
		case "reduction":
			go h.pub.Publish(publisher.BiasReductionDetected(tc, resp.DriftAnalysis.Phase1BiasScore, resp.DriftAnalysis.Phase2BiasScore))
		}
	}

	if resp.EmbeddingAnalysis != nil && resp.EmbeddingAnalysis.SemanticDriftDetected {
		metrics.AlertsTotal.WithLabelValues("semantic_drift").Inc()
		go h.pub.Publish(publisher.SemanticDriftDetected(tc, resp.EmbeddingAnalysis.SimilarityToContext))
		go h.pub.Publish(publisher.AnomalyDetected(tc, "semantic_drift", resp.EmbeddingAnalysis.SimilarityToContext))
		h.hm.Fire(hooks.Event{
			Type:      hooks.EventAnomalyDetected,
			RequestID: req.RequestID,
			TenantID:  req.TenantID,
			TraceID:   tc.TraceID,
			SpanID:    tc.SpanID,
			Data: map[string]interface{}{
				"anomaly_type": "semantic_drift",
				"score":        resp.EmbeddingAnalysis.SimilarityToContext,
			},
			Ctx: r.Context(),
		})
	}

	if resp.Quality != nil && resp.Quality.OverallQuality < 0.5 {
		metrics.AlertsTotal.WithLabelValues("quality_low").Inc()
		go h.pub.Publish(publisher.QualityLow(tc, resp.Quality.OverallQuality))
	}

	go h.pub.Publish(publisher.ResponseVerified(tc, string(resp.Status), func() float64 {
		if resp.BiasAnalysis != nil {
			return resp.BiasAnalysis.OverallScore
		}
		return 0
	}()))

	writeJSON(w, http.StatusOK, resp)
}

func (h *handlers) AnalyzeEmbedding(w http.ResponseWriter, r *http.Request) {
	var req models.EmbeddingAnalysisRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.Text == "" {
		writeError(w, http.StatusBadRequest, "text must not be empty")
		return
	}
	writeJSON(w, http.StatusOK, h.ver.AnalyzeEmbedding(&req))
}

func (h *handlers) CheckDrift(w http.ResponseWriter, r *http.Request) {
	var req models.DriftRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	writeJSON(w, http.StatusOK, h.ver.CheckDrift(&req))
}

// ─── Config ───────────────────────────────────────────────────────────────────

func (h *handlers) GetConfig(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, models.AnchorConfig{
		DefaultNoiseSigma: h.cfg.Noise.DefaultSigma,
		PreserveNorm:      h.cfg.Noise.PreserveNorm,
		Strategy:          h.cfg.Noise.Strategy,
		TenantOverrides:   h.cfg.Noise.TenantOverrides,
		BiasEnabled:       h.cfg.Bias.Enabled,
		QualityEnabled:    h.cfg.Quality.Enabled,
	})
}

func (h *handlers) UpdateConfig(w http.ResponseWriter, r *http.Request) {
	var update models.AnchorConfig
	if !decodeJSON(w, r, &update) {
		return
	}
	if update.DefaultNoiseSigma > 0 {
		h.cfg.Noise.DefaultSigma = update.DefaultNoiseSigma
		h.inj.SetSigma(update.DefaultNoiseSigma)
	}
	if update.TenantOverrides != nil {
		h.cfg.Noise.TenantOverrides = update.TenantOverrides
	}
	h.GetConfig(w, r)
}

// ─── helpers ──────────────────────────────────────────────────────────────────

// traceFrom extracts Foundation trace context from request headers + body fields.
func traceFrom(r *http.Request, tenantID, userID, requestID string) publisher.TraceContext {
	traceID := r.Header.Get("X-Trace-ID")
	if traceID == "" {
		traceID = r.Header.Get("traceparent") // W3C trace context header
	}
	if traceID == "" {
		traceID = uuid.New().String()
	}
	spanID := r.Header.Get("X-Span-ID")
	if spanID == "" {
		spanID = uuid.New().String()[:16]
	}
	return publisher.TraceContext{
		TraceID:      traceID,
		SpanID:       spanID,
		ParentSpanID: r.Header.Get("X-Parent-Span-ID"),
		TenantID:     tenantID,
		UserID:       userID,
		RequestID:    requestID,
	}
}

// injectorFor returns the default injector or a per-request override.
func (h *handlers) injectorFor(tenantID string, opts *models.SecurityOptions) *noise.Injector {
	sigma := h.cfg.SigmaForTenant(tenantID)
	preserveNorm := h.cfg.Noise.PreserveNorm

	if opts != nil && opts.NoiseSigma > 0 {
		sigma = opts.NoiseSigma
	}
	if opts != nil && opts.Seed != 0 {
		return noise.NewDeterministic(sigma, preserveNorm, opts.Seed)
	}
	inj, err := noise.New(sigma, preserveNorm, h.cfg.Noise.Strategy)
	if err != nil {
		return h.inj // fallback to shared injector
	}
	return inj
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst interface{}) bool {
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, body interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, models.ErrorResponse{Error: msg, Code: status})
}
