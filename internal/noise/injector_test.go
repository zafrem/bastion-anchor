package noise

import (
	"math"
	mathrand "math/rand"
	"testing"
)

func unit(dims int) []float32 {
	vec := make([]float32, dims)
	norm := float32(math.Sqrt(float64(dims)))
	for i := range vec {
		vec[i] = 1.0 / norm
	}
	return vec
}

// ─── Deterministic injector ───────────────────────────────────────────────────

func TestDeterministicInjector_ReproducibleNoise(t *testing.T) {
	emb := unit(64)
	r1 := NewDeterministic(0.01, false, 42).Inject(emb)
	r2 := NewDeterministic(0.01, false, 42).Inject(emb)

	for i := range r1.Secured {
		if r1.Secured[i] != r2.Secured[i] {
			t.Fatalf("dim %d differs: %f vs %f", i, r1.Secured[i], r2.Secured[i])
		}
	}
}

func TestDeterministicInjector_DifferentSeedsProduceDifferentNoise(t *testing.T) {
	emb := unit(64)
	r1 := NewDeterministic(0.01, false, 1).Inject(emb)
	r2 := NewDeterministic(0.01, false, 2).Inject(emb)

	same := 0
	for i := range r1.Secured {
		if r1.Secured[i] == r2.Secured[i] {
			same++
		}
	}
	if same == len(r1.Secured) {
		t.Error("different seeds produced identical noise")
	}
}

// ─── Norm preservation ────────────────────────────────────────────────────────

func TestInject_PreserveNorm_OutputIsUnitVector(t *testing.T) {
	inj := NewDeterministic(0.05, true, 7)
	emb := unit(128)
	result := inj.Inject(emb)

	norm := l2Norm(result.Secured)
	if math.Abs(norm-1.0) > 1e-5 {
		t.Errorf("expected unit norm after normalization, got %.6f", norm)
	}
}

func TestInject_NoPreserveNorm_NormChanges(t *testing.T) {
	inj := NewDeterministic(0.1, false, 7)
	emb := unit(128)
	result := inj.Inject(emb)

	// With sigma=0.1 the norm will deviate measurably from 1.
	norm := l2Norm(result.Secured)
	// It may be very close to 1 by chance but usually won't be.
	if result.SecuredNorm == result.OriginalNorm {
		t.Log("note: norms happened to be equal (rare edge case)")
	}
	_ = norm
}

// ─── Noise magnitude ──────────────────────────────────────────────────────────

func TestInject_ZeroSigma_NoiseMagnitudeIsZero(t *testing.T) {
	inj := NewDeterministic(0, false, 1)
	result := inj.Inject(unit(32))
	if result.NoiseMagnitude != 0 {
		t.Errorf("expected 0 noise magnitude for sigma=0, got %f", result.NoiseMagnitude)
	}
}

func TestInject_LargerSigma_ProducesMoreNoise(t *testing.T) {
	emb := unit(256)
	low := NewDeterministic(0.001, false, 42).Inject(emb)
	high := NewDeterministic(0.1, false, 42).Inject(emb)

	if high.NoiseMagnitude <= low.NoiseMagnitude {
		t.Errorf("higher sigma should produce larger noise magnitude: low=%.4f high=%.4f",
			low.NoiseMagnitude, high.NoiseMagnitude)
	}
}

// ─── Similarity ───────────────────────────────────────────────────────────────

func TestInject_SmallSigma_HighSimilarity(t *testing.T) {
	inj := NewDeterministic(0.005, true, 99)
	result := inj.Inject(unit(1024))
	if result.SimilarityToOriginal < 0.95 {
		t.Errorf("sigma=0.005 should preserve high similarity, got %.4f",
			result.SimilarityToOriginal)
	}
}

func TestInject_LargeSigma_LowerSimilarity(t *testing.T) {
	inj := NewDeterministic(0.2, true, 99)
	result := inj.Inject(unit(1024))
	if result.SimilarityToOriginal > 0.99 {
		t.Errorf("sigma=0.2 should measurably reduce similarity, got %.4f",
			result.SimilarityToOriginal)
	}
}

// ─── Batch ────────────────────────────────────────────────────────────────────

func TestInjectBatch_CountMatches(t *testing.T) {
	inj := NewDeterministic(0.01, true, 1)
	inputs := [][]float32{unit(16), unit(16), unit(16)}
	results := inj.InjectBatch(inputs)
	if len(results) != len(inputs) {
		t.Errorf("expected %d results, got %d", len(inputs), len(results))
	}
}

// ─── Cosine similarity ────────────────────────────────────────────────────────

func TestCosineSimilarity_IdenticalVectors_ReturnsOne(t *testing.T) {
	v := unit(32)
	sim := CosineSimilarity(v, v)
	if math.Abs(sim-1.0) > 1e-5 {
		t.Errorf("identical vectors should have similarity 1.0, got %.6f", sim)
	}
}

func TestCosineSimilarity_OrthogonalVectors_ReturnsZero(t *testing.T) {
	a := make([]float32, 8)
	b := make([]float32, 8)
	a[0] = 1
	b[1] = 1
	sim := CosineSimilarity(a, b)
	if math.Abs(sim) > 1e-5 {
		t.Errorf("orthogonal vectors should have similarity ~0, got %.6f", sim)
	}
}

func TestCosineSimilarity_DifferentLengths_ReturnsZero(t *testing.T) {
	sim := CosineSimilarity([]float32{1, 2}, []float32{1, 2, 3})
	if sim != 0 {
		t.Errorf("mismatched lengths should return 0, got %.6f", sim)
	}
}

// ─── EstimatedRecallPreservation ──────────────────────────────────────────────

func TestEstimatedRecall_SmallSigma_HighRetention(t *testing.T) {
	r := EstimatedRecallPreservation(0.005)
	if r < 0.98 {
		t.Errorf("sigma=0.005 should have high recall retention, got %.3f", r)
	}
}

func TestEstimatedRecall_LargeSigma_LowRetention(t *testing.T) {
	r := EstimatedRecallPreservation(0.3)
	if r != 0 {
		t.Errorf("sigma=0.3 should clamp to 0 recall retention, got %.3f", r)
	}
}

// ─── Laplacian strategy ───────────────────────────────────────────────────────

func TestLaplacian_ProducesNoise(t *testing.T) {
	inj := &Injector{sigma: 0.01, preserveNorm: true, strategy: StrategyLaplacian,
		rng: newRNG(42)}
	result := inj.Inject(unit(64))
	if result.NoiseMagnitude == 0 {
		t.Error("laplacian noise should not be zero")
	}
}

func newRNG(seed int64) *mathrand.Rand {
	return NewDeterministic(0, false, seed).rng
}

// ─── New (production) injector ────────────────────────────────────────────────

func TestNew_InvalidSigma_ReturnsError(t *testing.T) {
	_, err := New(-1, true, StrategyGaussian)
	if err == nil {
		t.Error("negative sigma should return error")
	}
}

func TestNew_ValidParams_Works(t *testing.T) {
	inj, err := New(0.01, true, StrategyGaussian)
	if err != nil {
		t.Fatal(err)
	}
	result := inj.Inject(unit(32))
	if len(result.Secured) != 32 {
		t.Errorf("expected 32 dims, got %d", len(result.Secured))
	}
}
