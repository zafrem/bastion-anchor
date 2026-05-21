package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Embedding security operations
	EmbeddingsSecured = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "anchor_embeddings_secured_total",
		Help: "Total embeddings processed by Anchor.",
	}, []string{"tenant", "operation"})

	SecureDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "anchor_secure_duration_seconds",
		Help:    "Duration of noise injection per embedding.",
		Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05},
	})

	NoiseMagnitude = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "anchor_noise_magnitude",
		Help:    "Average noise magnitude added per embedding.",
		Buckets: []float64{0.001, 0.005, 0.01, 0.02, 0.05, 0.1},
	})

	SimilarityToOriginal = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "anchor_similarity_to_original",
		Help:    "Cosine similarity between original and secured embedding.",
		Buckets: []float64{0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0},
	})

	// Bias measurement
	BiasAnalysesTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "anchor_bias_analyses_total",
		Help: "Total bias analysis runs.",
	})

	BiasScoreGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "anchor_bias_score",
		Help: "Current overall bias score per category.",
	}, []string{"category"})

	// Quality monitoring
	QualityChecksTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "anchor_quality_checks_total",
		Help: "Total quality checks performed.",
	})

	RecallPreservation = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "anchor_recall_preservation_ratio",
		Help: "Ratio of recall preserved after noise injection (1.0 = no loss).",
	})

	// Alerts
	AlertsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "anchor_alerts_total",
		Help: "Total alerts raised by Anchor.",
	}, []string{"type"}) // "bias_detected", "quality_degraded"

	// Phase 2: Output verification
	VerificationsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "anchor_verifications_total",
		Help: "Total Phase 2 response verifications.",
	}, []string{"status"})

	BiasDetectedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "anchor_bias_detected_total",
		Help: "Total bias detections in LLM responses.",
	}, []string{"severity"})

	DriftDetectedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "anchor_drift_detected_total",
		Help: "Total bias drift events between Phase 1 and Phase 2.",
	}, []string{"direction"})

	VerifyDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "anchor_verify_duration_seconds",
		Help:    "Duration of Phase 2 response verification.",
		Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5},
	})
)
