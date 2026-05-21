// Package bias implements a simplified WEAT (Word Embedding Association Test)
// as described in SRS §8.  It measures embedding bias for attribute pairs such
// as gender or ethnicity by comparing cosine similarities.
package bias

import (
	"fmt"
	"math"
	mathrand "math/rand"

	"github.com/bastion/anchor/internal/models"
	"github.com/bastion/anchor/internal/noise"
)

// Severity thresholds.
const (
	SeverityLow    = "low"
	SeverityMedium = "medium"
	SeverityHigh   = "high"
)

// BuiltinTestSets are the default WEAT word lists from SRS §8.2.
var BuiltinTestSets = map[string]TestSet{
	"gender": {
		TestWords: []string{"doctor", "nurse", "engineer", "teacher", "manager",
			"의사", "간호사", "엔지니어", "교사", "매니저"},
		Pairs: []models.AttributePair{
			{AttributeA: "male", AttributeB: "female", Category: "gender"},
			{AttributeA: "he", AttributeB: "she", Category: "gender"},
			{AttributeA: "남성", AttributeB: "여성", Category: "gender"},
		},
	},
	"ethnicity": {
		TestWords: []string{"intelligent", "hardworking", "athletic",
			"똑똑한", "부지런한", "운동을 잘하는"},
		Pairs: []models.AttributePair{
			{AttributeA: "Korean", AttributeB: "Foreign", Category: "ethnicity"},
			{AttributeA: "한국인", AttributeB: "외국인", Category: "ethnicity"},
		},
	},
	"age": {
		TestWords: []string{"capable", "modern", "traditional",
			"능력있는", "현대적인", "전통적인"},
		Pairs: []models.AttributePair{
			{AttributeA: "young", AttributeB: "old", Category: "age"},
			{AttributeA: "젊은", AttributeB: "나이든", Category: "age"},
		},
	},
}

// TestSet groups a word list and attribute pairs for a single bias category.
type TestSet struct {
	TestWords []string
	Pairs     []models.AttributePair
}

// EmbeddingProvider retrieves a vector for a text token.
type EmbeddingProvider interface {
	Embed(text string) ([]float32, bool)
}

// Analyzer runs the WEAT bias tests.
type Analyzer struct {
	mediumThreshold float64
	highThreshold   float64
	dims            int
}

// New creates an Analyzer with the given severity thresholds.
func New(mediumThreshold, highThreshold float64, dims int) *Analyzer {
	if dims == 0 {
		dims = 1024
	}
	return &Analyzer{
		mediumThreshold: mediumThreshold,
		highThreshold:   highThreshold,
		dims:            dims,
	}
}

// Analyze runs WEAT measurements for the given words and attribute pairs.
// wordEmbeddings is a pre-computed lookup; missing words get random vectors
// (demo/standalone mode — the algorithm is correct, the numbers are illustrative).
func (a *Analyzer) Analyze(testWords []string, pairs []models.AttributePair, wordEmbeddings map[string][]float32) models.BiasResponse {
	measurements := make([]models.BiasMeasurement, 0, len(testWords)*len(pairs))
	var totalBias float64

	for _, word := range testWords {
		wordEmb := a.getOrRandom(word, wordEmbeddings)
		for _, pair := range pairs {
			attrAEmb := a.getOrRandom(pair.AttributeA, wordEmbeddings)
			attrBEmb := a.getOrRandom(pair.AttributeB, wordEmbeddings)

			simA := noise.CosineSimilarity(wordEmb, attrAEmb)
			simB := noise.CosineSimilarity(wordEmb, attrBEmb)
			biasScore := math.Abs(simA - simB)

			measurements = append(measurements, models.BiasMeasurement{
				Word:      word,
				Category:  pair.Category,
				BiasScore: biasScore,
				SimA:      simA,
				SimB:      simB,
				Severity:  a.classify(biasScore),
			})
			totalBias += biasScore
		}
	}

	var overall float64
	if len(measurements) > 0 {
		overall = totalBias / float64(len(measurements))
	}

	return models.BiasResponse{
		Measurements:     measurements,
		OverallBiasScore: overall,
		Interpretation:   a.interpret(overall),
	}
}

// AnalyzeBuiltin runs the built-in test set identified by name.
func (a *Analyzer) AnalyzeBuiltin(name string, wordEmbeddings map[string][]float32) (models.BiasResponse, error) {
	ts, ok := BuiltinTestSets[name]
	if !ok {
		return models.BiasResponse{}, fmt.Errorf("unknown built-in test set %q (available: gender, ethnicity, age)", name)
	}
	return a.Analyze(ts.TestWords, ts.Pairs, wordEmbeddings), nil
}

func (a *Analyzer) classify(score float64) string {
	switch {
	case score >= a.highThreshold:
		return SeverityHigh
	case score >= a.mediumThreshold:
		return SeverityMedium
	default:
		return SeverityLow
	}
}

func (a *Analyzer) interpret(overall float64) string {
	switch {
	case overall >= a.highThreshold:
		return fmt.Sprintf("High bias detected (score=%.3f). Immediate model review recommended.", overall)
	case overall >= a.mediumThreshold:
		return fmt.Sprintf("Moderate bias detected (score=%.3f). Consider reviewing embedding model.", overall)
	default:
		return fmt.Sprintf("Bias within acceptable range (score=%.3f). No immediate action required.", overall)
	}
}

// getOrRandom retrieves a pre-computed embedding or generates a unit random vector.
func (a *Analyzer) getOrRandom(token string, provided map[string][]float32) []float32 {
	if provided != nil {
		if emb, ok := provided[token]; ok {
			return emb
		}
	}
	return randomUnit(a.dims)
}

func randomUnit(dims int) []float32 {
	vec := make([]float32, dims)
	var sumSq float64
	for i := range vec {
		v := mathrand.Float32()*2 - 1 //nolint:gosec
		vec[i] = v
		sumSq += float64(v) * float64(v)
	}
	if sumSq > 0 {
		norm := float32(1.0 / math.Sqrt(sumSq))
		for i := range vec {
			vec[i] *= norm
		}
	}
	return vec
}
