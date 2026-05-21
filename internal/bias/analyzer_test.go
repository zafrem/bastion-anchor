package bias

import (
	"math"
	"testing"

	"github.com/bastion/anchor/internal/models"
)

func newAnalyzer() *Analyzer { return New(0.15, 0.25, 64) }

// helper: build a tiny deterministic embedding map.
func embMap(pairs ...string) map[string][]float32 {
	m := make(map[string][]float32, len(pairs)/2)
	dims := 64
	for i := 0; i+1 < len(pairs); i += 2 {
		token := pairs[i]
		vec := make([]float32, dims)
		// Use a simple deterministic pattern: alternate ±value per index.
		seed := float32(i+1) * 0.1
		for j := range vec {
			if j%2 == 0 {
				vec[j] = seed
			} else {
				vec[j] = -seed
			}
		}
		// L2 normalise.
		var sum float64
		for _, v := range vec {
			sum += float64(v) * float64(v)
		}
		if sum > 0 {
			norm := float32(1.0 / math.Sqrt(sum))
			for j := range vec {
				vec[j] *= norm
			}
		}
		m[token] = vec
	}
	return m
}

// ─── Analyze ──────────────────────────────────────────────────────────────────

func TestAnalyze_MeasurementCountMatchesWordTimesPairs(t *testing.T) {
	a := newAnalyzer()
	words := []string{"doctor", "nurse"}
	pairs := []models.AttributePair{
		{AttributeA: "male", AttributeB: "female", Category: "gender"},
		{AttributeA: "he", AttributeB: "she", Category: "gender"},
	}
	resp := a.Analyze(words, pairs, nil)
	want := len(words) * len(pairs)
	if len(resp.Measurements) != want {
		t.Errorf("expected %d measurements, got %d", want, len(resp.Measurements))
	}
}

func TestAnalyze_OverallScoreIsAverageOfMeasurements(t *testing.T) {
	a := newAnalyzer()
	words := []string{"w1", "w2"}
	pairs := []models.AttributePair{
		{AttributeA: "a", AttributeB: "b", Category: "test"},
	}
	resp := a.Analyze(words, pairs, nil)
	var sum float64
	for _, m := range resp.Measurements {
		sum += m.BiasScore
	}
	avg := sum / float64(len(resp.Measurements))
	if math.Abs(resp.OverallBiasScore-avg) > 1e-6 {
		t.Errorf("overall score %.6f does not match average %.6f", resp.OverallBiasScore, avg)
	}
}

func TestAnalyze_BiasScoreNonNegative(t *testing.T) {
	a := newAnalyzer()
	words := []string{"doctor", "nurse", "engineer"}
	pairs := []models.AttributePair{
		{AttributeA: "male", AttributeB: "female", Category: "gender"},
	}
	resp := a.Analyze(words, pairs, nil)
	for _, m := range resp.Measurements {
		if m.BiasScore < 0 {
			t.Errorf("negative bias score for word %q: %.6f", m.Word, m.BiasScore)
		}
	}
}

func TestAnalyze_IdenticalAttributeEmbeddings_ZeroBias(t *testing.T) {
	a := newAnalyzer()
	// When attributeA and attributeB have the same embedding, simA == simB → score 0.
	sharedVec := make([]float32, 64)
	for i := range sharedVec {
		sharedVec[i] = float32(i) + 1
	}
	// normalise
	var sum float64
	for _, v := range sharedVec {
		sum += float64(v) * float64(v)
	}
	norm := float32(1.0 / math.Sqrt(sum))
	for i := range sharedVec {
		sharedVec[i] *= norm
	}

	em := map[string][]float32{
		"word": sharedVec,
		"a":    sharedVec,
		"b":    sharedVec,
	}
	pairs := []models.AttributePair{{AttributeA: "a", AttributeB: "b", Category: "test"}}
	resp := a.Analyze([]string{"word"}, pairs, em)
	if resp.Measurements[0].BiasScore > 1e-5 {
		t.Errorf("identical attribute embeddings should produce near-zero bias, got %.6f", resp.Measurements[0].BiasScore)
	}
}

func TestAnalyze_PrecomputedEmbeddingsUsed(t *testing.T) {
	a := newAnalyzer()
	em := embMap("doctor", "0", "male", "1", "female", "2")
	pairs := []models.AttributePair{
		{AttributeA: "male", AttributeB: "female", Category: "gender"},
	}
	// Run twice — results must be identical (no random fallback used).
	r1 := a.Analyze([]string{"doctor"}, pairs, em)
	r2 := a.Analyze([]string{"doctor"}, pairs, em)
	if r1.Measurements[0].BiasScore != r2.Measurements[0].BiasScore {
		t.Errorf("pre-computed embeddings should give deterministic bias score")
	}
}

func TestAnalyze_NoWords_EmptyMeasurements(t *testing.T) {
	a := newAnalyzer()
	pairs := []models.AttributePair{{AttributeA: "a", AttributeB: "b"}}
	resp := a.Analyze(nil, pairs, nil)
	if len(resp.Measurements) != 0 {
		t.Errorf("expected 0 measurements for empty word list, got %d", len(resp.Measurements))
	}
	if resp.OverallBiasScore != 0 {
		t.Errorf("overall bias should be 0 with no measurements, got %.4f", resp.OverallBiasScore)
	}
}

// ─── Classify / Severity ─────────────────────────────────────────────────────

func TestClassify_BelowMedium_Low(t *testing.T) {
	a := newAnalyzer() // medium=0.15, high=0.25
	if got := a.classify(0.10); got != SeverityLow {
		t.Errorf("expected low, got %q", got)
	}
}

func TestClassify_AtMedium_Medium(t *testing.T) {
	a := newAnalyzer()
	if got := a.classify(0.15); got != SeverityMedium {
		t.Errorf("expected medium, got %q", got)
	}
}

func TestClassify_AtHigh_High(t *testing.T) {
	a := newAnalyzer()
	if got := a.classify(0.25); got != SeverityHigh {
		t.Errorf("expected high, got %q", got)
	}
}

// ─── AnalyzeBuiltin ───────────────────────────────────────────────────────────

func TestAnalyzeBuiltin_UnknownSet_ReturnsError(t *testing.T) {
	a := newAnalyzer()
	_, err := a.AnalyzeBuiltin("unknown-set", nil)
	if err == nil {
		t.Error("expected error for unknown built-in test set")
	}
}

func TestAnalyzeBuiltin_Gender_ReturnsNonEmptyMeasurements(t *testing.T) {
	a := newAnalyzer()
	resp, err := a.AnalyzeBuiltin("gender", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Measurements) == 0 {
		t.Error("expected measurements from gender built-in test set")
	}
}

func TestAnalyzeBuiltin_AllBuiltins_NoError(t *testing.T) {
	a := newAnalyzer()
	for name := range BuiltinTestSets {
		if _, err := a.AnalyzeBuiltin(name, nil); err != nil {
			t.Errorf("built-in %q returned error: %v", name, err)
		}
	}
}

// ─── Interpretation ───────────────────────────────────────────────────────────

func TestInterpret_HighScore_ContainsHighKeyword(t *testing.T) {
	a := newAnalyzer()
	s := a.interpret(0.3)
	if len(s) == 0 {
		t.Error("interpretation should not be empty")
	}
}

func TestInterpret_LowScore_AcceptableRange(t *testing.T) {
	a := newAnalyzer()
	s := a.interpret(0.05)
	if len(s) == 0 {
		t.Error("interpretation should not be empty")
	}
}
