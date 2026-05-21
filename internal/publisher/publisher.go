// Package publisher sends Anchor events to Module D (Tracker) via NATS,
// with a stdout fallback for standalone mode.
package publisher

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/nats-io/nats.go"

	"github.com/bastion/anchor/internal/models"
)

// Publisher sends events asynchronously.
type Publisher interface {
	Publish(event models.TrackerEvent) error
	Close()
}

// NATSPublisher publishes events to a NATS subject.
type NATSPublisher struct {
	nc      *nats.Conn
	subject string
}

func NewNATS(url, subject string) (*NATSPublisher, error) {
	nc, err := nats.Connect(url,
		nats.MaxReconnects(10),
		nats.ReconnectWait(2e9), // 2 seconds
		nats.ErrorHandler(func(_ *nats.Conn, _ *nats.Subscription, err error) {
			log.Printf("[nats] error: %v", err)
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("nats connect %s: %w", url, err)
	}
	return &NATSPublisher{nc: nc, subject: subject}, nil
}

func (p *NATSPublisher) Publish(event models.TrackerEvent) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	return p.nc.Publish(p.subject, data)
}

func (p *NATSPublisher) Close() {
	p.nc.Drain() //nolint:errcheck
}

// LogPublisher writes events to stdout (standalone / development mode).
type LogPublisher struct{}

func NewLog() *LogPublisher { return &LogPublisher{} }

func (p *LogPublisher) Publish(event models.TrackerEvent) error {
	data, _ := json.Marshal(event)
	log.Printf("[tracker-event] %s", data)
	return nil
}

func (p *LogPublisher) Close() {}

// NoopPublisher discards all events (testing).
type NoopPublisher struct{}

func NewNoop() *NoopPublisher { return &NoopPublisher{} }
func (p *NoopPublisher) Publish(_ models.TrackerEvent) error { return nil }
func (p *NoopPublisher) Close()                              {}

// Event constructors

func EmbeddingSecured(tenantID, operation string, noiseAdded, similarity float64) models.TrackerEvent {
	return models.TrackerEvent{
		EventType: "anchor.embedding_secured",
		Module:    "anchor",
		Severity:  "info",
		Data: map[string]interface{}{
			"tenant_id":            tenantID,
			"operation":            operation,
			"noise_added":          noiseAdded,
			"similarity_preserved": similarity,
		},
	}
}

func BiasDetected(category string, biasScore float64, affectedTerms []string) models.TrackerEvent {
	return models.TrackerEvent{
		EventType: "anchor.bias_detected",
		Module:    "anchor",
		Severity:  "warning",
		Data: map[string]interface{}{
			"category":       category,
			"bias_score":     biasScore,
			"affected_terms": affectedTerms,
		},
	}
}

func QualityDegraded(expectedRecall, currentRecall, sigma float64) models.TrackerEvent {
	return models.TrackerEvent{
		EventType: "anchor.quality_degraded",
		Module:    "anchor",
		Severity:  "warning",
		Data: map[string]interface{}{
			"expected_recall": expectedRecall,
			"current_recall":  currentRecall,
			"noise_sigma":     sigma,
		},
	}
}

// Phase 2 event constructors

func ResponseVerified(tenantID, status string, biasScore float64) models.TrackerEvent {
	return models.TrackerEvent{
		EventType: "anchor.response_verified",
		Module:    "anchor",
		Severity:  "info",
		Data: map[string]interface{}{
			"tenant_id":   tenantID,
			"status":      status,
			"bias_score":  biasScore,
		},
	}
}

func BiasDetectedResponse(tenantID, category, severity string, score float64) models.TrackerEvent {
	return models.TrackerEvent{
		EventType: "anchor.response_bias_detected",
		Module:    "anchor",
		Severity:  "warning",
		Data: map[string]interface{}{
			"tenant_id": tenantID,
			"category":  category,
			"severity":  severity,
			"score":     score,
		},
	}
}

func BiasAmplificationDetected(tenantID string, phase1Score, phase2Score float64) models.TrackerEvent {
	return models.TrackerEvent{
		EventType: "anchor.bias_amplification_detected",
		Module:    "anchor",
		Severity:  "critical",
		Data: map[string]interface{}{
			"tenant_id":    tenantID,
			"phase1_score": phase1Score,
			"phase2_score": phase2Score,
			"drift_amount": phase2Score - phase1Score,
		},
	}
}

func BiasIntroductionDetected(tenantID string, phase1Score, phase2Score float64) models.TrackerEvent {
	return models.TrackerEvent{
		EventType: "anchor.bias_introduction_detected",
		Module:    "anchor",
		Severity:  "critical",
		Data: map[string]interface{}{
			"tenant_id":    tenantID,
			"phase1_score": phase1Score,
			"phase2_score": phase2Score,
		},
	}
}

func QualityLow(tenantID string, overallQuality float64) models.TrackerEvent {
	return models.TrackerEvent{
		EventType: "anchor.response_quality_low",
		Module:    "anchor",
		Severity:  "warning",
		Data: map[string]interface{}{
			"tenant_id":       tenantID,
			"overall_quality": overallQuality,
		},
	}
}

func SemanticDriftDetected(tenantID string, simToContext float64) models.TrackerEvent {
	return models.TrackerEvent{
		EventType: "anchor.semantic_drift_detected",
		Module:    "anchor",
		Severity:  "warning",
		Data: map[string]interface{}{
			"tenant_id":      tenantID,
			"sim_to_context": simToContext,
		},
	}
}

func BiasReductionDetected(tenantID string, phase1Score, phase2Score float64) models.TrackerEvent {
	return models.TrackerEvent{
		EventType: "anchor.bias_reduction_detected",
		Module:    "anchor",
		Severity:  "info",
		Data: map[string]interface{}{
			"tenant_id":    tenantID,
			"phase1_score": phase1Score,
			"phase2_score": phase2Score,
			"drift_amount": phase1Score - phase2Score,
		},
	}
}
