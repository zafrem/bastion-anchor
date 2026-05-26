// Package publisher sends Anchor events to Module D (Tracker) via NATS,
// using the Foundation event schema (doc 02) with subject bastion.events.anchor.{event_type}.
// Falls back to stdout for standalone mode.
package publisher

import (
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
	"github.com/nats-io/nats.go"

	"github.com/bastion/anchor/internal/models"
)

const (
	moduleAnchor    = "anchor"
	moduleVersion   = "1.0.0"
	schemaVersion   = "1.0"
	subjectPrefix   = "bastion.events.anchor"

	catOperational = "operational"
	catSecurity    = "security"
	catPerformance = "performance"

	sevInfo     = "info"
	sevWarning  = "warning"
	sevCritical = "critical"
)

// TraceContext carries distributed trace metadata from a request.
type TraceContext struct {
	TraceID      string
	SpanID       string
	ParentSpanID string
	TenantID     string
	UserID       string
	RequestID    string
}

// Publisher sends Foundation-standard events asynchronously.
type Publisher interface {
	Publish(event models.BastionEvent) error
	Close()
}

// ─── NATS publisher ───────────────────────────────────────────────────────────

// NATSPublisher publishes events to NATS using the Foundation subject hierarchy.
type NATSPublisher struct {
	nc *nats.Conn
}

func NewNATS(url string) (*NATSPublisher, error) {
	nc, err := nats.Connect(url,
		nats.MaxReconnects(10),
		nats.ReconnectWait(2e9),
		nats.ErrorHandler(func(_ *nats.Conn, _ *nats.Subscription, err error) {
			log.Printf("[nats] error: %v", err)
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("nats connect %s: %w", url, err)
	}
	return &NATSPublisher{nc: nc}, nil
}

// NewNATSWithSubject exists for backwards compatibility; the subject param is ignored.
func NewNATSWithSubject(url, _ string) (*NATSPublisher, error) {
	return NewNATS(url)
}

func (p *NATSPublisher) Publish(event models.BastionEvent) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	subject := fmt.Sprintf("%s.%s", subjectPrefix, event.EventType)
	return p.nc.Publish(subject, data)
}

func (p *NATSPublisher) Close() {
	p.nc.Drain() //nolint:errcheck
}

// ─── Log publisher (standalone / dev) ────────────────────────────────────────

type LogPublisher struct{}

func NewLog() *LogPublisher { return &LogPublisher{} }

func (p *LogPublisher) Publish(event models.BastionEvent) error {
	data, _ := json.Marshal(event)
	log.Printf("[bastion-event] %s", data)
	return nil
}

func (p *LogPublisher) Close() {}

// ─── Noop publisher (testing) ─────────────────────────────────────────────────

type NoopPublisher struct{}

func NewNoop() *NoopPublisher { return &NoopPublisher{} }
func (p *NoopPublisher) Publish(_ models.BastionEvent) error { return nil }
func (p *NoopPublisher) Close()                               {}

// ─── Event constructors ───────────────────────────────────────────────────────

func newEvent(eventType, severity, category string, tc TraceContext, data map[string]interface{}) models.BastionEvent {
	return models.BastionEvent{
		EventID:       uuid.New().String(),
		EventType:     eventType,
		SchemaVersion: schemaVersion,
		TraceID:       tc.TraceID,
		SpanID:        tc.SpanID,
		ParentSpanID:  tc.ParentSpanID,
		Module:        moduleAnchor,
		ModuleVersion: moduleVersion,
		Timestamp:     time.Now().UnixNano(),
		TenantID:      tc.TenantID,
		UserID:        tc.UserID,
		RequestID:     tc.RequestID,
		Severity:      severity,
		Category:      category,
		Data:          data,
		Status:        "success",
	}
}

// EmbeddingSecured emits anchor.embedding_secured (Phase 1).
func EmbeddingSecured(tc TraceContext, operation string, noiseAdded, similarity float64) models.BastionEvent {
	return newEvent("embedding_secured", sevInfo, catOperational, tc, map[string]interface{}{
		"operation":            operation,
		"noise_added":          noiseAdded,
		"similarity_preserved": similarity,
	})
}

// BiasDetected emits anchor.bias_detected.
func BiasDetected(tc TraceContext, category string, biasScore float64, affectedTerms []string) models.BastionEvent {
	return newEvent("bias_detected", sevWarning, catSecurity, tc, map[string]interface{}{
		"category":       category,
		"bias_score":     biasScore,
		"affected_terms": affectedTerms,
	})
}

// QualityDegraded emits anchor.quality_degraded.
func QualityDegraded(tc TraceContext, expectedRecall, currentRecall, sigma float64) models.BastionEvent {
	return newEvent("quality_degraded", sevWarning, catPerformance, tc, map[string]interface{}{
		"expected_recall": expectedRecall,
		"current_recall":  currentRecall,
		"noise_sigma":     sigma,
	})
}

// ResponseVerified emits anchor.response_analyzed (Phase 2).
func ResponseVerified(tc TraceContext, status string, biasScore float64) models.BastionEvent {
	ev := newEvent("response_analyzed", sevInfo, catOperational, tc, map[string]interface{}{
		"bias_score": biasScore,
	})
	ev.Status = status
	return ev
}

// BiasDetectedResponse emits anchor.bias_detected for a response.
func BiasDetectedResponse(tc TraceContext, category, severity string, score float64) models.BastionEvent {
	return newEvent("bias_detected", sevWarning, catSecurity, tc, map[string]interface{}{
		"phase":    "response",
		"category": category,
		"severity": severity,
		"score":    score,
	})
}

// BiasAmplificationDetected emits anchor.bias_amplification_detected.
func BiasAmplificationDetected(tc TraceContext, phase1Score, phase2Score float64) models.BastionEvent {
	return newEvent("bias_amplification_detected", sevCritical, catSecurity, tc, map[string]interface{}{
		"phase1_score": phase1Score,
		"phase2_score": phase2Score,
		"drift_amount": phase2Score - phase1Score,
	})
}

// BiasIntroductionDetected emits anchor.bias_introduction_detected.
func BiasIntroductionDetected(tc TraceContext, phase1Score, phase2Score float64) models.BastionEvent {
	return newEvent("bias_introduction_detected", sevCritical, catSecurity, tc, map[string]interface{}{
		"phase1_score": phase1Score,
		"phase2_score": phase2Score,
	})
}

// QualityLow emits anchor.response_quality_low.
func QualityLow(tc TraceContext, overallQuality float64) models.BastionEvent {
	return newEvent("response_quality_low", sevWarning, catPerformance, tc, map[string]interface{}{
		"overall_quality": overallQuality,
	})
}

// SemanticDriftDetected emits anchor.semantic_drift_detected.
func SemanticDriftDetected(tc TraceContext, simToContext float64) models.BastionEvent {
	return newEvent("semantic_drift_detected", sevWarning, catSecurity, tc, map[string]interface{}{
		"sim_to_context": simToContext,
	})
}

// AnomalyDetected emits anchor.anomaly_detected (SRS doc 13 §6.4, via hooks).
func AnomalyDetected(tc TraceContext, anomalyType string, score float64) models.BastionEvent {
	return newEvent("anomaly_detected", sevWarning, catSecurity, tc, map[string]interface{}{
		"anomaly_type": anomalyType,
		"score":        score,
	})
}

// BiasReductionDetected emits anchor.bias_reduction_detected.
func BiasReductionDetected(tc TraceContext, phase1Score, phase2Score float64) models.BastionEvent {
	return newEvent("bias_reduction_detected", sevInfo, catOperational, tc, map[string]interface{}{
		"phase1_score": phase1Score,
		"phase2_score": phase2Score,
		"drift_amount": phase1Score - phase2Score,
	})
}
