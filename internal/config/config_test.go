package config

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// ─── Defaults ─────────────────────────────────────────────────────────────────

func TestDefaults_RESTPort(t *testing.T) {
	cfg := Defaults()
	if cfg.Server.RESTPort != 8080 {
		t.Errorf("expected REST port 8080, got %d", cfg.Server.RESTPort)
	}
}

func TestDefaults_GRPCPort(t *testing.T) {
	cfg := Defaults()
	if cfg.Server.GRPCPort != 9090 {
		t.Errorf("expected gRPC port 9090, got %d", cfg.Server.GRPCPort)
	}
}

func TestDefaults_NoiseSigmaPositive(t *testing.T) {
	cfg := Defaults()
	if cfg.Noise.DefaultSigma <= 0 {
		t.Errorf("default sigma should be positive, got %f", cfg.Noise.DefaultSigma)
	}
}

func TestDefaults_PreserveNormTrue(t *testing.T) {
	cfg := Defaults()
	if !cfg.Noise.PreserveNorm {
		t.Error("preserve_norm should default to true")
	}
}

func TestDefaults_StrategyGaussian(t *testing.T) {
	cfg := Defaults()
	if cfg.Noise.Strategy != "gaussian" {
		t.Errorf("expected gaussian strategy, got %q", cfg.Noise.Strategy)
	}
}

func TestDefaults_BiasThresholdsOrdered(t *testing.T) {
	cfg := Defaults()
	if cfg.Bias.AlertMedium >= cfg.Bias.AlertHigh {
		t.Errorf("medium threshold (%.2f) should be less than high (%.2f)",
			cfg.Bias.AlertMedium, cfg.Bias.AlertHigh)
	}
}

func TestDefaults_EmbeddingDimensions(t *testing.T) {
	cfg := Defaults()
	if cfg.Embedding.ExpectedDimensions != 1024 {
		t.Errorf("expected 1024 dimensions, got %d", cfg.Embedding.ExpectedDimensions)
	}
}

func TestDefaults_TenantOverridesNotNil(t *testing.T) {
	cfg := Defaults()
	if cfg.Noise.TenantOverrides == nil {
		t.Error("tenant_overrides map should not be nil")
	}
}

func TestDefaults_QualityEnabled(t *testing.T) {
	cfg := Defaults()
	if !cfg.Quality.Enabled {
		t.Error("quality should be enabled by default")
	}
}

// ─── SigmaForTenant ───────────────────────────────────────────────────────────

func TestSigmaForTenant_UnknownTenant_ReturnsDefault(t *testing.T) {
	cfg := Defaults()
	got := cfg.SigmaForTenant("unknown-tenant")
	if math.Abs(got-cfg.Noise.DefaultSigma) > 1e-9 {
		t.Errorf("unknown tenant should return default sigma %.4f, got %.4f",
			cfg.Noise.DefaultSigma, got)
	}
}

func TestSigmaForTenant_KnownTenant_ReturnsOverride(t *testing.T) {
	cfg := Defaults()
	cfg.Noise.TenantOverrides["tenant-strict"] = 0.05
	got := cfg.SigmaForTenant("tenant-strict")
	if math.Abs(got-0.05) > 1e-9 {
		t.Errorf("known tenant should return override 0.05, got %.4f", got)
	}
}

func TestSigmaForTenant_EmptyTenantID_ReturnsDefault(t *testing.T) {
	cfg := Defaults()
	got := cfg.SigmaForTenant("")
	if math.Abs(got-cfg.Noise.DefaultSigma) > 1e-9 {
		t.Errorf("empty tenant ID should return default, got %.4f", got)
	}
}

func TestSigmaForTenant_MultipleOverrides_EachCorrect(t *testing.T) {
	cfg := Defaults()
	cfg.Noise.TenantOverrides["a"] = 0.001
	cfg.Noise.TenantOverrides["b"] = 0.1

	if got := cfg.SigmaForTenant("a"); math.Abs(got-0.001) > 1e-9 {
		t.Errorf("tenant a: expected 0.001, got %.4f", got)
	}
	if got := cfg.SigmaForTenant("b"); math.Abs(got-0.1) > 1e-9 {
		t.Errorf("tenant b: expected 0.1, got %.4f", got)
	}
}

// ─── Load ─────────────────────────────────────────────────────────────────────

func TestLoad_NonExistentFile_ReturnsDefaults(t *testing.T) {
	cfg, err := Load("/nonexistent/path/config.yaml")
	if err != nil {
		t.Fatalf("non-existent file should not error, got %v", err)
	}
	if cfg.Server.RESTPort != 8080 {
		t.Errorf("expected default REST port, got %d", cfg.Server.RESTPort)
	}
}

func TestLoad_ValidYAML_OverridesFields(t *testing.T) {
	yaml := `
server:
  rest_port: 9999
  grpc_port: 8888
noise:
  default_sigma: 0.05
  preserve_norm: false
  strategy: gaussian
`
	path := writeTempConfig(t, yaml)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if cfg.Server.RESTPort != 9999 {
		t.Errorf("expected REST port 9999, got %d", cfg.Server.RESTPort)
	}
	if cfg.Server.GRPCPort != 8888 {
		t.Errorf("expected gRPC port 8888, got %d", cfg.Server.GRPCPort)
	}
	if math.Abs(cfg.Noise.DefaultSigma-0.05) > 1e-9 {
		t.Errorf("expected sigma 0.05, got %.4f", cfg.Noise.DefaultSigma)
	}
	if cfg.Noise.PreserveNorm {
		t.Error("preserve_norm should be false")
	}
}

func TestLoad_PartialYAML_UnsetFieldsKeepDefaults(t *testing.T) {
	yaml := `
server:
  rest_port: 1234
`
	path := writeTempConfig(t, yaml)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if cfg.Server.RESTPort != 1234 {
		t.Errorf("expected 1234, got %d", cfg.Server.RESTPort)
	}
	// Fields not in YAML should keep defaults.
	if cfg.Noise.DefaultSigma <= 0 {
		t.Errorf("unset sigma should keep default, got %f", cfg.Noise.DefaultSigma)
	}
	if cfg.Embedding.ExpectedDimensions != 1024 {
		t.Errorf("unset dimensions should keep default 1024, got %d", cfg.Embedding.ExpectedDimensions)
	}
}

func TestLoad_InvalidYAML_ReturnsError(t *testing.T) {
	// Tabs are forbidden in YAML for indentation, and this malformed structure
	// should reliably fail unmarshaling.
	path := writeTempConfig(t, "noise:\n\tdefault_sigma: invalid")
	_, err := Load(path)
	if err == nil {
		t.Error("invalid YAML should return an error")
	}
}

func TestLoad_TenantOverrides_Parsed(t *testing.T) {
	yaml := `
noise:
  default_sigma: 0.01
  tenant_overrides:
    tenant-acme: 0.02
    tenant-public: 0.005
`
	path := writeTempConfig(t, yaml)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if math.Abs(cfg.SigmaForTenant("tenant-acme")-0.02) > 1e-9 {
		t.Errorf("tenant-acme override not applied, got %.4f", cfg.SigmaForTenant("tenant-acme"))
	}
	if math.Abs(cfg.SigmaForTenant("tenant-public")-0.005) > 1e-9 {
		t.Errorf("tenant-public override not applied, got %.4f", cfg.SigmaForTenant("tenant-public"))
	}
}

// ─── helpers ──────────────────────────────────────────────────────────────────

func writeTempConfig(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write temp config: %v", err)
	}
	return path
}
