package quality

import (
	"math"
	"testing"

	"github.com/bastion/anchor/internal/models"
)

func mon() *Monitor { return New(5.0, 0.95) }

// unit builds an L2-normalised vector where every component equals 1/√dims.
func unit(dims int) []float32 {
	vec := make([]float32, dims)
	v := float32(1.0 / math.Sqrt(float64(dims)))
	for i := range vec {
		vec[i] = v
	}
	return vec
}

// ─── Estimate without pairs (theoretical model) ───────────────────────────────

func TestEstimate_NoPairs_UsesTheoreticalModel(t *testing.T) {
	m := mon()
	resp := m.Estimate(0.005, nil)
	if resp.RecallWith <= 0 || resp.RecallWith > 1 {
		t.Errorf("recall should be in (0,1], got %.4f", resp.RecallWith)
	}
}

func TestEstimate_NoPairs_SmallSigma_HighRecall(t *testing.T) {
	m := mon()
	resp := m.Estimate(0.005, nil)
	if resp.RecallWith < 0.95 {
		t.Errorf("small sigma should give high recall preservation, got %.4f", resp.RecallWith)
	}
}

func TestEstimate_NoPairs_LargeSigma_LowRecall(t *testing.T) {
	m := mon()
	resp := m.Estimate(0.3, nil)
	if resp.RecallWith > 0.5 {
		t.Errorf("large sigma should give low recall, got %.4f", resp.RecallWith)
	}
}

func TestEstimate_NoPairs_SigmaPreserved(t *testing.T) {
	m := mon()
	resp := m.Estimate(0.07, nil)
	if math.Abs(resp.NoiseSigma-0.07) > 1e-6 {
		t.Errorf("sigma not preserved: got %.6f", resp.NoiseSigma)
	}
}

// ─── Estimate with pairs ──────────────────────────────────────────────────────

func TestEstimate_WithPairs_CountsHits(t *testing.T) {
	m := mon()
	q := unit(64)
	// Identical query and match → very high similarity → both should hit.
	pairs := []models.EmbeddingPair{
		{QueryEmbedding: q, MatchEmbedding: q},
		{QueryEmbedding: q, MatchEmbedding: q},
	}
	resp := m.Estimate(0.001, pairs)
	if resp.RecallWithout < 0.9 {
		t.Errorf("identical query/match should hit without noise, got %.4f", resp.RecallWithout)
	}
}

func TestEstimate_WithPairs_SmallSigma_HighRecall(t *testing.T) {
	m := mon()
	q := unit(128)
	pairs := []models.EmbeddingPair{
		{QueryEmbedding: q, MatchEmbedding: q},
		{QueryEmbedding: q, MatchEmbedding: q},
		{QueryEmbedding: q, MatchEmbedding: q},
	}
	resp := m.Estimate(0.001, pairs)
	if resp.RecallWith < 0.5 {
		t.Errorf("tiny sigma should keep recall high, got %.4f", resp.RecallWith)
	}
}

func TestEstimate_WithPairs_ReturnsPairs_Count(t *testing.T) {
	m := mon()
	q := unit(32)
	pairs := make([]models.EmbeddingPair, 5)
	for i := range pairs {
		pairs[i] = models.EmbeddingPair{QueryEmbedding: q, MatchEmbedding: q}
	}
	resp := m.Estimate(0.01, pairs)
	// RecallWithout and RecallWith are ratios in [0,1].
	if resp.RecallWithout < 0 || resp.RecallWithout > 1 {
		t.Errorf("recallWithout out of range: %.4f", resp.RecallWithout)
	}
	if resp.RecallWith < 0 || resp.RecallWith > 1 {
		t.Errorf("recallWith out of range: %.4f", resp.RecallWith)
	}
}

// ─── Status thresholds ────────────────────────────────────────────────────────

func TestBuildResponse_LowDrop_Acceptable(t *testing.T) {
	m := New(5.0, 0.95)
	resp := m.buildResponse(0.01, 1.0, 0.97, 0.98) // 3% drop, sim=0.98 > 0.95
	if resp.Status != "acceptable" {
		t.Errorf("expected acceptable, got %q", resp.Status)
	}
}

func TestBuildResponse_HighDrop_Critical(t *testing.T) {
	m := New(5.0, 0.95)
	// 80% recall drop (>10% = 2×threshold)
	resp := m.buildResponse(0.5, 1.0, 0.2, 0.98)
	if resp.Status != "critical" {
		t.Errorf("expected critical, got %q", resp.Status)
	}
}

func TestBuildResponse_ModerateDrop_Warning(t *testing.T) {
	m := New(5.0, 0.95)
	// 8% recall drop (>5% but ≤10%)
	resp := m.buildResponse(0.1, 1.0, 0.92, 0.98)
	if resp.Status != "warning" {
		t.Errorf("expected warning, got %q", resp.Status)
	}
}

func TestBuildResponse_LowSimilarity_Warning(t *testing.T) {
	m := New(5.0, 0.95)
	// sim below minimum threshold
	resp := m.buildResponse(0.1, 1.0, 0.98, 0.90)
	if resp.Status == "acceptable" {
		t.Error("low similarity should not be acceptable")
	}
}

// ─── SigmaVerdict ─────────────────────────────────────────────────────────────

func TestSigmaVerdict_TooWeak(t *testing.T) {
	if SigmaVerdict(0.003) != "Too weak" {
		t.Errorf("expected 'Too weak'")
	}
}

func TestSigmaVerdict_Recommended(t *testing.T) {
	if SigmaVerdict(0.01) != "Recommended" {
		t.Errorf("expected 'Recommended'")
	}
}

func TestSigmaVerdict_QualityImpact(t *testing.T) {
	if SigmaVerdict(0.02) != "Quality impact" {
		t.Errorf("expected 'Quality impact'")
	}
}

func TestSigmaVerdict_TooAggressive(t *testing.T) {
	if SigmaVerdict(0.1) != "Too aggressive" {
		t.Errorf("expected 'Too aggressive'")
	}
}
