package publisher

import (
	"testing"
)

// tc is a zero-value trace context for tests that don't care about tracing.
var tc = TraceContext{TenantID: "tenant-test"}

// ─── Event constructors ───────────────────────────────────────────────────────

func TestEmbeddingSecured_EventType(t *testing.T) {
	e := EmbeddingSecured(tc, "index", 0.03, 0.98)
	if e.EventType != "embedding_secured" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestEmbeddingSecured_SeverityInfo(t *testing.T) {
	e := EmbeddingSecured(tc, "search", 0.01, 0.99)
	if e.Severity != "info" {
		t.Errorf("expected info severity, got %q", e.Severity)
	}
}

func TestEmbeddingSecured_DataFields(t *testing.T) {
	e := EmbeddingSecured(TraceContext{TenantID: "tenant-x"}, "index", 0.05, 0.95)
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

func TestEmbeddingSecured_TraceContext(t *testing.T) {
	ctx := TraceContext{TraceID: "trace-123", SpanID: "span-001", TenantID: "t1"}
	e := EmbeddingSecured(ctx, "index", 0.01, 0.99)
	if e.TraceID != "trace-123" {
		t.Errorf("TraceID mismatch: %q", e.TraceID)
	}
	if e.SpanID != "span-001" {
		t.Errorf("SpanID mismatch: %q", e.SpanID)
	}
	if e.TenantID != "t1" {
		t.Errorf("TenantID mismatch: %q", e.TenantID)
	}
	if e.EventID == "" {
		t.Error("EventID must not be empty")
	}
	if e.SchemaVersion != schemaVersion {
		t.Errorf("SchemaVersion: want %q, got %q", schemaVersion, e.SchemaVersion)
	}
}

func TestBiasDetected_EventType(t *testing.T) {
	e := BiasDetected(tc, "gender", 0.18, []string{"nurse"})
	if e.EventType != "bias_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestBiasDetected_SeverityWarning(t *testing.T) {
	e := BiasDetected(tc, "gender", 0.2, nil)
	if e.Severity != "warning" {
		t.Errorf("expected warning, got %q", e.Severity)
	}
}

func TestBiasDetected_DataFields(t *testing.T) {
	e := BiasDetected(tc, "ethnicity", 0.22, []string{"foreigner"})
	if e.Data["category"] != "ethnicity" {
		t.Errorf("category mismatch: %v", e.Data["category"])
	}
	if e.Data["bias_score"] != 0.22 {
		t.Errorf("bias_score mismatch: %v", e.Data["bias_score"])
	}
}

func TestQualityDegraded_EventType(t *testing.T) {
	e := QualityDegraded(tc, 0.85, 0.78, 0.02)
	if e.EventType != "quality_degraded" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestQualityDegraded_SeverityWarning(t *testing.T) {
	e := QualityDegraded(tc, 0.85, 0.78, 0.02)
	if e.Severity != "warning" {
		t.Errorf("expected warning, got %q", e.Severity)
	}
}

func TestQualityDegraded_DataFields(t *testing.T) {
	e := QualityDegraded(tc, 0.85, 0.78, 0.02)
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
	e := ResponseVerified(tc, "PASSED", 0.05)
	if e.EventType != "response_analyzed" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestResponseVerified_SeverityInfo(t *testing.T) {
	e := ResponseVerified(tc, "PASSED", 0.05)
	if e.Severity != "info" {
		t.Errorf("expected info, got %q", e.Severity)
	}
}

func TestResponseVerified_Status(t *testing.T) {
	e := ResponseVerified(tc, "WARNING", 0.18)
	if e.Status != "WARNING" {
		t.Errorf("status mismatch: %v", e.Status)
	}
}

func TestBiasDetectedResponse_EventType(t *testing.T) {
	e := BiasDetectedResponse(tc, "gender", "high", 0.28)
	if e.EventType != "bias_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestBiasDetectedResponse_SeverityWarning(t *testing.T) {
	e := BiasDetectedResponse(tc, "gender", "high", 0.28)
	if e.Severity != "warning" {
		t.Errorf("expected warning, got %q", e.Severity)
	}
}

func TestBiasAmplificationDetected_EventType(t *testing.T) {
	e := BiasAmplificationDetected(tc, 0.05, 0.28)
	if e.EventType != "bias_amplification_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestBiasAmplificationDetected_SeverityCritical(t *testing.T) {
	e := BiasAmplificationDetected(tc, 0.05, 0.28)
	if e.Severity != "critical" {
		t.Errorf("expected critical, got %q", e.Severity)
	}
}

func TestBiasAmplificationDetected_DriftAmount(t *testing.T) {
	e := BiasAmplificationDetected(tc, 0.05, 0.28)
	want := 0.28 - 0.05
	got := e.Data["drift_amount"].(float64)
	if abs(got-want) > 1e-9 {
		t.Errorf("drift_amount: want %.4f, got %v", want, got)
	}
}

func TestBiasIntroductionDetected_EventType(t *testing.T) {
	e := BiasIntroductionDetected(tc, 0.02, 0.20)
	if e.EventType != "bias_introduction_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestBiasIntroductionDetected_SeverityCritical(t *testing.T) {
	e := BiasIntroductionDetected(tc, 0.02, 0.20)
	if e.Severity != "critical" {
		t.Errorf("expected critical, got %q", e.Severity)
	}
}

func TestBiasReductionDetected_EventType(t *testing.T) {
	e := BiasReductionDetected(tc, 0.30, 0.08)
	if e.EventType != "bias_reduction_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestBiasReductionDetected_SeverityInfo(t *testing.T) {
	e := BiasReductionDetected(tc, 0.30, 0.08)
	if e.Severity != "info" {
		t.Errorf("expected info, got %q", e.Severity)
	}
}

func TestBiasReductionDetected_DriftAmount(t *testing.T) {
	e := BiasReductionDetected(tc, 0.30, 0.08)
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
	e := QualityLow(tc, 0.42)
	if e.EventType != "response_quality_low" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestQualityLow_SeverityWarning(t *testing.T) {
	e := QualityLow(tc, 0.42)
	if e.Severity != "warning" {
		t.Errorf("expected warning, got %q", e.Severity)
	}
}

func TestSemanticDriftDetected_EventType(t *testing.T) {
	e := SemanticDriftDetected(tc, 0.45)
	if e.EventType != "semantic_drift_detected" {
		t.Errorf("unexpected event type: %q", e.EventType)
	}
}

func TestSemanticDriftDetected_SeverityWarning(t *testing.T) {
	e := SemanticDriftDetected(tc, 0.45)
	if e.Severity != "warning" {
		t.Errorf("expected warning, got %q", e.Severity)
	}
}

func TestAllEvents_ModuleIsAnchor(t *testing.T) {
	events := []struct {
		name string
		fn   func() string
	}{
		{"EmbeddingSecured", func() string { return EmbeddingSecured(tc, "index", 0, 0).Module }},
		{"BiasDetected", func() string { return BiasDetected(tc, "g", 0, nil).Module }},
		{"QualityDegraded", func() string { return QualityDegraded(tc, 0, 0, 0).Module }},
		{"ResponseVerified", func() string { return ResponseVerified(tc, "PASSED", 0).Module }},
		{"BiasDetectedResponse", func() string { return BiasDetectedResponse(tc, "g", "low", 0).Module }},
		{"BiasAmplificationDetected", func() string { return BiasAmplificationDetected(tc, 0, 0).Module }},
		{"BiasIntroductionDetected", func() string { return BiasIntroductionDetected(tc, 0, 0).Module }},
		{"BiasReductionDetected", func() string { return BiasReductionDetected(tc, 0, 0).Module }},
		{"QualityLow", func() string { return QualityLow(tc, 0).Module }},
		{"SemanticDriftDetected", func() string { return SemanticDriftDetected(tc, 0).Module }},
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
	e := EmbeddingSecured(tc, "index", 0.01, 0.99)
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
	e := BiasDetected(tc, "gender", 0.2, []string{"term"})
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
