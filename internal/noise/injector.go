// Package noise implements the Gaussian noise injection algorithm described in
// SRS §7.  Security randomness uses crypto/rand as the entropy source; a
// seeded math/rand is provided for deterministic testing (FR-NI-003).
package noise

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"math"
	mathrand "math/rand"
)

// Strategy names.
const (
	StrategyGaussian  = "gaussian"
	StrategyLaplacian = "laplacian"
)

// Result carries the secured embedding and per-request statistics.
type Result struct {
	Secured              []float32
	NoiseMagnitude       float64 // average |noise[i]|
	SimilarityToOriginal float64 // cosine similarity between original and secured
	OriginalNorm         float64
	SecuredNorm          float64
	Strategy             string
}

// Injector applies configurable noise to embeddings.
type Injector struct {
	sigma        float64
	preserveNorm bool
	strategy     string
	rng          *mathrand.Rand
}

// New creates a production Injector that seeds its RNG from crypto/rand.
func New(sigma float64, preserveNorm bool, strategy string) (*Injector, error) {
	if sigma < 0 {
		return nil, fmt.Errorf("sigma must be >= 0, got %f", sigma)
	}
	if strategy == "" {
		strategy = StrategyGaussian
	}
	seed, err := cryptoSeed()
	if err != nil {
		return nil, fmt.Errorf("seed from crypto/rand: %w", err)
	}
	return &Injector{
		sigma:        sigma,
		preserveNorm: preserveNorm,
		strategy:     strategy,
		rng:          mathrand.New(mathrand.NewSource(seed)), //nolint:gosec
	}, nil
}

// NewDeterministic creates an Injector with a fixed seed for reproducible tests (FR-NI-003).
func NewDeterministic(sigma float64, preserveNorm bool, seed int64) *Injector {
	return &Injector{
		sigma:        sigma,
		preserveNorm: preserveNorm,
		strategy:     StrategyGaussian,
		rng:          mathrand.New(mathrand.NewSource(seed)), //nolint:gosec
	}
}

// Inject applies noise to embedding and returns a Result.
// The input slice is never modified.
func (inj *Injector) Inject(embedding []float32) Result {
	origNorm := l2Norm(embedding)

	secured := make([]float32, len(embedding))
	var noiseMag float64

	switch inj.strategy {
	case StrategyLaplacian:
		for i, v := range embedding {
			n := float32(inj.laplacian())
			secured[i] = v + n
			noiseMag += math.Abs(float64(n))
		}
	default: // gaussian
		for i, v := range embedding {
			n := float32(inj.gaussian())
			secured[i] = v + n
			noiseMag += math.Abs(float64(n))
		}
	}

	if len(embedding) > 0 {
		noiseMag /= float64(len(embedding))
	}

	if inj.preserveNorm {
		l2Normalize(secured)
	}

	secNorm := l2Norm(secured)
	sim := cosineSimilarity(embedding, secured)

	return Result{
		Secured:              secured,
		NoiseMagnitude:       noiseMag,
		SimilarityToOriginal: sim,
		OriginalNorm:         origNorm,
		SecuredNorm:          secNorm,
		Strategy:             inj.strategy + "_noise",
	}
}

// InjectBatch applies noise to each embedding in the slice.
func (inj *Injector) InjectBatch(embeddings [][]float32) []Result {
	out := make([]Result, len(embeddings))
	for i, emb := range embeddings {
		out[i] = inj.Inject(emb)
	}
	return out
}

// SetSigma updates the noise level (supports hot-reload).
func (inj *Injector) SetSigma(sigma float64) { inj.sigma = sigma }

// Sigma returns the current noise level.
func (inj *Injector) Sigma() float64 { return inj.sigma }

// ─── Noise generators ─────────────────────────────────────────────────────────

// gaussian returns a sample from N(0, σ²) using the Box-Muller transform.
func (inj *Injector) gaussian() float64 {
	for {
		u1 := inj.rng.Float64()
		u2 := inj.rng.Float64()
		if u1 == 0 { // avoid log(0)
			continue
		}
		z := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		return z * inj.sigma
	}
}

// laplacian returns a sample from Laplace(0, b) where b = sigma/sqrt(2).
func (inj *Injector) laplacian() float64 {
	b := inj.sigma / math.Sqrt2
	u := inj.rng.Float64() - 0.5
	if u == 0 {
		return 0
	}
	sign := 1.0
	if u < 0 {
		sign = -1.0
	}
	return -b * sign * math.Log(1-2*math.Abs(u))
}

// ─── Vector math ──────────────────────────────────────────────────────────────

func l2Norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func l2Normalize(v []float32) {
	norm := l2Norm(v)
	if norm == 0 {
		return
	}
	for i := range v {
		v[i] = float32(float64(v[i]) / norm)
	}
}

// CosineSimilarity is exported for use by quality and bias packages.
func CosineSimilarity(a, b []float32) float64 {
	return cosineSimilarity(a, b)
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// cryptoSeed reads 8 bytes from crypto/rand to seed the PRNG.
func cryptoSeed() (int64, error) {
	var buf [8]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return 0, err
	}
	return int64(binary.LittleEndian.Uint64(buf[:])), nil
}

// EstimatedRecallPreservation returns the approximate Recall@K retention
// for a given sigma, based on the theoretical relationship between
// noise magnitude and cosine distance shift.
// This is a simplified linear model used for configuration guidance.
func EstimatedRecallPreservation(sigma float64) float64 {
	// Empirically: each 0.01 sigma costs ~4% recall at K=10.
	// Model: retention = 1 - 4*sigma, clamped to [0,1].
	// At sigma=0.005 → 0.98; sigma≥0.25 → 0 (clamped).
	retention := 1.0 - 4.0*sigma
	if retention < 0 {
		return 0
	}
	if retention > 1 {
		return 1
	}
	return retention
}
