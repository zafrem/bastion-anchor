package publisher

import (
	"testing"
)

// ─── Event constructors ───────────────────────────────────────────────────────

func TestEmbeddingSecured_EventType(t *testing.T) {
	e := EmbeddingSecured("tenant-a", "index", 0.03, 0.98)
	if e.EventType != "anchor.embedding_secured" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestEmbeddingSecured_SeverityInfo(t *testing.T) {
	e := EmbeddingSecured("t", "search", 0.01, 0.99)
	if e.Severity != "info" {
		t.Errorf("expected info severity, got %q", e.Severity)
	}
}

func TestEmbeddingSecured_DataFields(t *testing.T) {
	e := EmbeddingSecured("tenant-x", "index", 0.05, 0.95)
	if e.Data["tenant_id"] != "tenant-x" {
		t.Errorf("tenant_id mismatch: %v", e.Data["tenant_id"])
	}
	if e.Data["operation"] != "index" {
		t.Errorf("operation mismatch: %v", e.Data["operation"])
	}
	if e.Data["noise_added"] != 0.05 {
		t.Errorf("noise_added mismatch: %v", e.Data["noise_added"])
	}
	if e.Data["similarity_preserved"] != 0.95 {
		t.Errorf("similarity_preserved mismatch: %v", e.Data["similarity_preserved"])
	}
}

func TestBiasDetected_EventType(t *testing.T) {
	e := BiasDetected("gender", 0.18, []string{"nurse"})
	if e.EventType != "anchor.bias_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestBiasDetected_SeverityWarning(t *testing.T) {
	e := BiasDetected("gender", 0.2, nil)
	if e.Severity != "warning" {
		t.Errorf("expected warning, got %q", e.Severity)
	}
}

func TestBiasDetected_DataFields(t *testing.T) {
	e := BiasDetected("ethnicity", 0.22, []string{"foreigner"})
	if e.Data["category"] != "ethnicity" {
		t.Errorf("category mismatch: %v", e.Data["category"])
	}
	if e.Data["bias_score"] != 0.22 {
		t.Errorf("bias_score mismatch: %v", e.Data["bias_score"])
	}
}

func TestQualityDegraded_EventType(t *testing.T) {
	e := QualityDegraded(0.85, 0.78, 0.02)
	if e.EventType != "anchor.quality_degraded" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestQualityDegraded_SeverityWarning(t *testing.T) {
	e := QualityDegraded(0.85, 0.78, 0.02)
	if e.Severity != "warning" {
		t.Errorf("expected warning, got %q", e.Severity)
	}
}

func TestQualityDegraded_DataFields(t *testing.T) {
	e := QualityDegraded(0.85, 0.78, 0.02)
	if e.Data["expected_recall"] != 0.85 {
		t.Errorf("expected_recall mismatch: %v", e.Data["expected_recall"])
	}
	if e.Data["current_recall"] != 0.78 {
		t.Errorf("current_recall mismatch: %v", e.Data["current_recall"])
	}
	if e.Data["noise_sigma"] != 0.02 {
		t.Errorf("noise_sigma mismatch: %v", e.Data["noise_sigma"])
	}
}

// ─── Phase 2 event constructors ───────────────────────────────────────────────

func TestResponseVerified_EventType(t *testing.T) {
	e := ResponseVerified("tenant-a", "PASSED", 0.05)
	if e.EventType != "anchor.response_verified" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestResponseVerified_SeverityInfo(t *testing.T) {
	e := ResponseVerified("t", "PASSED", 0.05)
	if e.Severity != "info" {
		t.Errorf("expected info, got %q", e.Severity)
	}
}

func TestResponseVerified_DataFields(t *testing.T) {
	e := ResponseVerified("tenant-b", "WARNING", 0.18)
	if e.Data["tenant_id"] != "tenant-b" {
		t.Errorf("tenant_id mismatch: %v", e.Data["tenant_id"])
	}
	if e.Data["status"] != "WARNING" {
		t.Errorf("status mismatch: %v", e.Data["status"])
	}
	if e.Data["bias_score"] != 0.18 {
		t.Errorf("bias_score mismatch: %v", e.Data["bias_score"])
	}
}

func TestBiasDetectedResponse_EventType(t *testing.T) {
	e := BiasDetectedResponse("t", "gender", "high", 0.28)
	if e.EventType != "anchor.response_bias_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestBiasDetectedResponse_SeverityWarning(t *testing.T) {
	e := BiasDetectedResponse("t", "gender", "high", 0.28)
	if e.Severity != "warning" {
		t.Errorf("expected warning, got %q", e.Severity)
	}
}

func TestBiasAmplificationDetected_EventType(t *testing.T) {
	e := BiasAmplificationDetected("t", 0.05, 0.28)
	if e.EventType != "anchor.bias_amplification_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestBiasAmplificationDetected_SeverityCritical(t *testing.T) {
	e := BiasAmplificationDetected("t", 0.05, 0.28)
	if e.Severity != "critical" {
		t.Errorf("expected critical, got %q", e.Severity)
	}
}

func TestBiasAmplificationDetected_DriftAmount(t *testing.T) {
	e := BiasAmplificationDetected("t", 0.05, 0.28)
	want := 0.28 - 0.05
	if e.Data["drift_amount"] != want {
		t.Errorf("drift_amount: want %.4f, got %v", want, e.Data["drift_amount"])
	}
}

func TestBiasIntroductionDetected_EventType(t *testing.T) {
	e := BiasIntroductionDetected("t", 0.02, 0.20)
	if e.EventType != "anchor.bias_introduction_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestBiasIntroductionDetected_SeverityCritical(t *testing.T) {
	e := BiasIntroductionDetected("t", 0.02, 0.20)
	if e.Severity != "critical" {
		t.Errorf("expected critical, got %q", e.Severity)
	}
}

func TestBiasReductionDetected_EventType(t *testing.T) {
	e := BiasReductionDetected("t", 0.30, 0.08)
	if e.EventType != "anchor.bias_reduction_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestBiasReductionDetected_SeverityInfo(t *testing.T) {
	e := BiasReductionDetected("t", 0.30, 0.08)
	if e.Severity != "info" {
		t.Errorf("expected info, got %q", e.Severity)
	}
}

func TestBiasReductionDetected_DriftAmount(t *testing.T) {
	e := BiasReductionDetected("t", 0.30, 0.08)
	want := 0.30 - 0.08
	got, ok := e.Data["drift_amount"].(float64)
	if !ok {
		t.Fatalf("drift_amount is not float64: %T %v", e.Data["drift_amount"], e.Data["drift_amount"])
	}
	if abs(got-want) > 1e-9 {
		t.Errorf("drift_amount: want %.4f, got %.4f", want, got)
	}
}

func TestQualityLow_EventType(t *testing.T) {
	e := QualityLow("t", 0.42)
	if e.EventType != "anchor.response_quality_low" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestQualityLow_SeverityWarning(t *testing.T) {
	e := QualityLow("t", 0.42)
	if e.Severity != "warning" {
		t.Errorf("expected warning, got %q", e.Severity)
	}
}

func TestSemanticDriftDetected_EventType(t *testing.T) {
	e := SemanticDriftDetected("t", 0.45)
	if e.EventType != "anchor.semantic_drift_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestSemanticDriftDetected_SeverityWarning(t *testing.T) {
	e := SemanticDriftDetected("t", 0.45)
	if e.Severity != "warning" {
		t.Errorf("expected warning, got %q", e.Severity)
	}
}

func TestAllEvents_ModuleIsAnchor(t *testing.T) {
	events := []struct {
		name string
		fn   func() string
	}{
		{"EmbeddingSecured", func() string { return EmbeddingSecured("t", "index", 0, 0).Module }},
		{"BiasDetected", func() string { return BiasDetected("g", 0, nil).Module }},
		{"QualityDegraded", func() string { return QualityDegraded(0, 0, 0).Module }},
		{"ResponseVerified", func() string { return ResponseVerified("t", "PASSED", 0).Module }},
		{"BiasDetectedResponse", func() string { return BiasDetectedResponse("t", "g", "low", 0).Module }},
		{"BiasAmplificationDetected", func() string { return BiasAmplificationDetected("t", 0, 0).Module }},
		{"BiasIntroductionDetected", func() string { return BiasIntroductionDetected("t", 0, 0).Module }},
		{"BiasReductionDetected", func() string { return BiasReductionDetected("t", 0, 0).Module }},
		{"QualityLow", func() string { return QualityLow("t", 0).Module }},
		{"SemanticDriftDetected", func() string { return SemanticDriftDetected("t", 0).Module }},
	}
	for _, tc := range events {
		if got := tc.fn(); got != "anchor" {
			t.Errorf("%s: module should be 'anchor', got %q", tc.name, got)
		}
	}
}

// ─── Publishers ───────────────────────────────────────────────────────────────

func TestNoopPublisher_Publish_ReturnsNil(t *testing.T) {
	p := NewNoop()
	e := EmbeddingSecured("t", "index", 0.01, 0.99)
	if err := p.Publish(e); err != nil {
		t.Errorf("noop publisher should return nil, got %v", err)
	}
}

func TestNoopPublisher_Close_NoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Close() panicked: %v", r)
		}
	}()
	NewNoop().Close()
}

func TestLogPublisher_Publish_ReturnsNil(t *testing.T) {
	p := NewLog()
	e := BiasDetected("gender", 0.2, []string{"term"})
	if err := p.Publish(e); err != nil {
		t.Errorf("log publisher should return nil, got %v", err)
	}
}

func TestLogPublisher_Close_NoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Close() panicked: %v", r)
		}
	}()
	NewLog().Close()
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
