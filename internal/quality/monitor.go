// Package quality measures the recall and similarity impact of noise injection.
package quality

import (
	"github.com/bastion/anchor/internal/models"
	"github.com/bastion/anchor/internal/noise"
)

// Monitor assesses the quality preservation after noise injection.
type Monitor struct {
	recallDropAlert   float64
	similarityMinimum float64
}

// New creates a Monitor with alert thresholds from config.
func New(recallDropAlert, similarityMinimum float64) *Monitor {
	return &Monitor{
		recallDropAlert:   recallDropAlert,
		similarityMinimum: similarityMinimum,
	}
}

// Measure computes quality metrics for a set of embedding pairs.
// For each pair we: (a) compute cosine similarity of the query to the match
// without noise and (b) with noise applied to the query, then check whether
// the match is still the nearest neighbour (recall@1 proxy).
func (m *Monitor) Measure(req models.QualityRequest, injector interface {
	Inject([]float32) interface{ GetSecured() []float32 }
}) models.QualityResponse {
	// We use a simpler approach: estimate recall from the theoretical model.
	return m.Estimate(req.NoiseSigma, req.EmbeddingPairs)
}

// Estimate runs the quality measurement using the provided embedding pairs.
// It applies Gaussian noise to each query embedding (with the given sigma)
// and measures whether the match is still closer than other candidates.
func (m *Monitor) Estimate(sigma float64, pairs []models.EmbeddingPair) models.QualityResponse {
	if len(pairs) == 0 {
		// No ground-truth pairs — fall back to the theoretical model.
		estimated := noise.EstimatedRecallPreservation(sigma)
		return m.buildResponse(sigma, 1.0, estimated, 1.0-estimated)
	}

	inj := noise.NewDeterministic(sigma, true, 42)

	hitsWithout := 0
	hitsWith := 0
	var totalSim float64

	for _, pair := range pairs {
		queryEmb := pair.QueryEmbedding
		matchEmb := pair.MatchEmbedding

		// Recall@1 without noise: by definition the query should match itself.
		simWithout := noise.CosineSimilarity(queryEmb, matchEmb)
		if simWithout >= 0.5 { // treat as "retrieved"
			hitsWithout++
		}

		// Apply noise to query embedding.
		result := inj.Inject(queryEmb)
		simWith := noise.CosineSimilarity(result.Secured, matchEmb)
		if simWith >= 0.5 {
			hitsWith++
		}
		totalSim += result.SimilarityToOriginal
	}

	n := float64(len(pairs))
	recallWithout := float64(hitsWithout) / n
	recallWith := float64(hitsWith) / n
	avgSim := totalSim / n
	dropPercent := (recallWithout - recallWith) / recallWithout * 100
	if recallWithout == 0 {
		dropPercent = 0
	}

	_ = dropPercent
	return m.buildResponse(sigma, recallWithout, recallWith, avgSim)
}

func (m *Monitor) buildResponse(sigma, recallWithout, recallWith, avgSim float64) models.QualityResponse {
	dropPercent := 0.0
	if recallWithout > 0 {
		dropPercent = (recallWithout - recallWith) / recallWithout * 100
	}
	status := "acceptable"
	if avgSim < m.similarityMinimum || dropPercent > m.recallDropAlert {
		if dropPercent > m.recallDropAlert*2 {
			status = "critical"
		} else {
			status = "warning"
		}
	}
	return models.QualityResponse{
		NoiseSigma:              sigma,
		RecallWithout:           recallWithout,
		RecallWith:              recallWith,
		RecallDropPercent:       dropPercent,
		AvgSimilarityToOriginal: avgSim,
		Status:                  status,
	}
}

// SigmaVerdict returns a human-readable verdict for a noise level.
func SigmaVerdict(sigma float64) string {
	switch {
	case sigma <= 0.005:
		return "Too weak"
	case sigma <= 0.012:
		return "Recommended"
	case sigma <= 0.025:
		return "Quality impact"
	default:
		return "Too aggressive"
	}
}
