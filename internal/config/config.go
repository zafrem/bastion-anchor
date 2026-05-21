package config

import (
	"fmt"
	"os"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// Config is the root configuration object.
type Config struct {
	mu      sync.RWMutex `yaml:"-"`
	Version string       `yaml:"version"`
	Server  ServerConfig `yaml:"server"`
	Noise   NoiseConfig  `yaml:"noise"`
	Bias    BiasConfig   `yaml:"bias"`
	Quality QualityConfig `yaml:"quality"`
	Embedding EmbeddingConfig `yaml:"embedding"`
	Tracker TrackerConfig `yaml:"tracker"`
	Logging LoggingConfig `yaml:"logging"`
	Metrics MetricsConfig `yaml:"metrics"`
}

type ServerConfig struct {
	RESTPort int `yaml:"rest_port"`
	GRPCPort int `yaml:"grpc_port"`
}

type NoiseConfig struct {
	DefaultSigma    float64            `yaml:"default_sigma"`
	PreserveNorm    bool               `yaml:"preserve_norm"`
	Strategy        string             `yaml:"strategy"` // "gaussian" | "laplacian"
	TenantOverrides map[string]float64 `yaml:"tenant_overrides"`
}

type BiasConfig struct {
	Enabled      bool          `yaml:"enabled"`
	CacheTTL     time.Duration `yaml:"cache_ttl"`
	AlertMedium  float64       `yaml:"alert_medium"`
	AlertHigh    float64       `yaml:"alert_high"`
	TestSetsPath string        `yaml:"test_sets_path"`
}

type QualityConfig struct {
	Enabled           bool    `yaml:"enabled"`
	SampleRate        float64 `yaml:"sample_rate"`
	RecallDropAlert   float64 `yaml:"recall_drop_percent"`
	SimilarityMinimum float64 `yaml:"similarity_minimum"`
}

type EmbeddingConfig struct {
	ExpectedDimensions int    `yaml:"expected_dimensions"`
	ExpectedModel      string `yaml:"expected_model"`
}

type TrackerConfig struct {
	Enabled      bool   `yaml:"enabled"`
	NATSUrl      string `yaml:"nats_url"`
	EventSubject string `yaml:"event_subject"`
}

type LoggingConfig struct {
	Level  string `yaml:"level"`
	Format string `yaml:"format"`
}

type MetricsConfig struct {
	Enabled bool `yaml:"enabled"`
	Port    int  `yaml:"port"`
}

// Defaults returns a Config with sensible PoC defaults.
func Defaults() *Config {
	return &Config{
		Version: "1.0",
		Server: ServerConfig{
			RESTPort: 8080,
			GRPCPort: 9090,
		},
		Noise: NoiseConfig{
			DefaultSigma: 0.01,
			PreserveNorm: true,
			Strategy:     "gaussian",
			TenantOverrides: map[string]float64{},
		},
		Bias: BiasConfig{
			Enabled:     true,
			CacheTTL:    time.Hour,
			AlertMedium: 0.15,
			AlertHigh:   0.25,
		},
		Quality: QualityConfig{
			Enabled:           true,
			SampleRate:        0.1,
			RecallDropAlert:   5.0,
			SimilarityMinimum: 0.95,
		},
		Embedding: EmbeddingConfig{
			ExpectedDimensions: 1024,
			ExpectedModel:      "BGE-M3",
		},
		Tracker: TrackerConfig{
			Enabled:      false,
			NATSUrl:      "nats://localhost:4222",
			EventSubject: "bastion.events.anchor",
		},
		Logging: LoggingConfig{Level: "info", Format: "json"},
		Metrics: MetricsConfig{Enabled: true, Port: 9091},
	}
}

// Load reads config from path, merging with defaults for missing fields.
func Load(path string) (*Config, error) {
	cfg := Defaults()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, fmt.Errorf("read config %s: %w", path, err)
	}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}
	return cfg, nil
}

// SigmaForTenant returns the configured noise sigma for the given tenant,
// falling back to the default.
func (c *Config) SigmaForTenant(tenantID string) float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if s, ok := c.Noise.TenantOverrides[tenantID]; ok {
		return s
	}
	return c.Noise.DefaultSigma
}
