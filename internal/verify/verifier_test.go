package verify

import (
	"testing"

	"github.com/bastion/anchor/internal/models"
)

func newVerifier() *Verifier { return New(0.15, 0.25) }

// ─── Verify ───────────────────────────────────────────────────────────────────

func TestVerify_NeutralResponse_StatusPassed(t *testing.T) {
	v := newVerifier()
	resp := v.Verify(&models.VerifyRequest{
		LLMResponse:   "A thoughtful and balanced perspective on the matter.",
		OriginalQuery: "Tell me about this topic.",
	})
	if resp.Status != models.VerifyStatusPassed {
		t.Errorf("expected PASSED, got %s", resp.Status)
	}
}

func TestVerify_BiasedResponse_NotPassed(t *testing.T) {
	v := newVerifier()
	resp := v.Verify(&models.VerifyRequest{
		LLMResponse:   "Women are too emotional for leadership. Men are better suited.",
		OriginalQuery: "Who makes a good leader?",
	})
	if resp.Status == models.VerifyStatusPassed {
		t.Error("biased response should not be PASSED")
	}
}

func TestVerify_HasBiasAnalysisWhenRunAll(t *testing.T) {
	v := newVerifier()
	resp := v.Verify(&models.VerifyRequest{
		LLMResponse:   "Some response text.",
		OriginalQuery: "A question.",
	})
	if resp.BiasAnalysis == nil {
		t.Error("expected bias analysis when no options set (run all)")
	}
}

func TestVerify_HasEmbeddingAnalysisWhenRunAll(t *testing.T) {
	v := newVerifier()
	resp := v.Verify(&models.VerifyRequest{
		LLMResponse:   "Some response text.",
		OriginalQuery: "A question.",
	})
	if resp.EmbeddingAnalysis == nil {
		t.Error("expected embedding analysis when no options set")
	}
}

func TestVerify_HasQualityWhenRunAll(t *testing.T) {
	v := newVerifier()
	resp := v.Verify(&models.VerifyRequest{
		LLMResponse:   "Some response text.",
		OriginalQuery: "A question.",
	})
	if resp.Quality == nil {
		t.Error("expected quality metrics when no options set")
	}
}

func TestVerify_DriftAnalysis_RequiresBaseline(t *testing.T) {
	v := newVerifier()
	resp := v.Verify(&models.VerifyRequest{
		LLMResponse:   "Some response text.",
		OriginalQuery: "A question.",
		// No Phase1Baseline
	})
	if resp.DriftAnalysis != nil {
		t.Error("drift analysis should be nil when no baseline provided")
	}
}

func TestVerify_DriftAnalysis_PresentWithBaseline(t *testing.T) {
	v := newVerifier()
	resp := v.Verify(&models.VerifyRequest{
		LLMResponse:    "Some response text.",
		OriginalQuery:  "A question.",
		Phase1Baseline: &models.BiasBaseline{OverallScore: 0.05},
	})
	if resp.DriftAnalysis == nil {
		t.Error("drift analysis should be present when baseline is provided")
	}
}

func TestVerify_SelectiveOptions_OnlyBias(t *testing.T) {
	v := newVerifier()
	resp := v.Verify(&models.VerifyRequest{
		LLMResponse:   "Some text.",
		OriginalQuery: "A query.",
		Options:       &models.VerifyOptions{CheckBias: true},
	})
	if resp.BiasAnalysis == nil {
		t.Error("expected bias analysis")
	}
	if resp.EmbeddingAnalysis != nil {
		t.Error("embedding analysis should be nil when only check_bias requested")
	}
	if resp.Quality != nil {
		t.Error("quality should be nil when only check_bias requested")
	}
}

func TestVerify_ProcessingTimePositive(t *testing.T) {
	v := newVerifier()
	resp := v.Verify(&models.VerifyRequest{
		LLMResponse:   "Some text.",
		OriginalQuery: "A query.",
	})
	if resp.ProcessingTimeMs < 0 {
		t.Errorf("processing time should be non-negative, got %f", resp.ProcessingTimeMs)
	}
}

func TestVerify_AmplificationDetected_StatusWarning(t *testing.T) {
	v := newVerifier()
	resp := v.Verify(&models.VerifyRequest{
		LLMResponse:    "Men who possess these qualities excel. Women are weak.",
		OriginalQuery:  "What traits matter?",
		Phase1Baseline: &models.BiasBaseline{OverallScore: 0.02},
	})
	if resp.Status == models.VerifyStatusPassed {
		t.Error("amplification should not result in PASSED")
	}
}

// ─── detectBias ───────────────────────────────────────────────────────────────

func TestDetectBias_NeutralText_LowScore(t *testing.T) {
	v := newVerifier()
	result := v.detectBias("The engineer solved the complex problem.", 0.15)
	if result.OverallScore > 0.15 {
		t.Errorf("neutral text should have low bias score, got %.3f", result.OverallScore)
	}
}

func TestDetectBias_GenderedText_NonZeroScore(t *testing.T) {
	v := newVerifier()
	result := v.detectBias("he he he him his his his man men men men men", 0.15)
	if result.OverallScore == 0 {
		t.Error("strongly gendered text should produce non-zero bias score")
	}
}

func TestDetectBias_StereotypePattern_BoostsScore(t *testing.T) {
	v := newVerifier()
	result := v.detectBias("Women are too emotional for this role.", 0.15)
	var found bool
	for _, cat := range result.Categories {
		if cat.Category == "gender" && len(cat.Evidence) > 0 {
			found = true
		}
	}
	if !found {
		t.Error("stereotype pattern should appear in evidence")
	}
}

func TestDetectBias_ExceedsThreshold(t *testing.T) {
	v := newVerifier()
	result := v.detectBias("Men are better leaders. Men excel. He is strong. Him only.", 0.15)
	if !result.ExceedsThreshold {
		t.Errorf("heavily gendered text should exceed threshold (score=%.3f)", result.OverallScore)
	}
}

func TestDetectBias_Severity_High(t *testing.T) {
	v := newVerifier()
	result := v.detectBias("Men are better leaders. Men excel. He is strong. Him only.", 0.15)
	if result.Severity != "high" {
		t.Errorf("expected high severity, got %q (score=%.3f)", result.Severity, result.OverallScore)
	}
}

// ─── analyzeEmbedding ─────────────────────────────────────────────────────────

func TestAnalyzeEmbedding_IdenticalTexts_HighSimilarity(t *testing.T) {
	v := newVerifier()
	text := "The quick brown fox jumps over the lazy dog."
	result := v.analyzeEmbedding(text, text, nil)
	if result.SimilarityToQuery < 0.9 {
		t.Errorf("identical text should have high query similarity, got %.3f", result.SimilarityToQuery)
	}
}

func TestAnalyzeEmbedding_DisjointTexts_LowSimilarity(t *testing.T) {
	v := newVerifier()
	result := v.analyzeEmbedding(
		"alpha beta gamma delta epsilon",
		"zeta eta theta iota kappa",
		nil,
	)
	if result.SimilarityToQuery > 0.3 {
		t.Errorf("disjoint texts should have low similarity, got %.3f", result.SimilarityToQuery)
	}
}

func TestAnalyzeEmbedding_NoSources_NoSemanticDrift(t *testing.T) {
	v := newVerifier()
	result := v.analyzeEmbedding("some text", "a query", nil)
	if result.SemanticDriftDetected {
		t.Error("should not detect semantic drift when no source documents provided")
	}
}

func TestAnalyzeEmbedding_LowContextSimilarity_DriftDetected(t *testing.T) {
	v := newVerifier()
	result := v.analyzeEmbedding(
		"completely unrelated response about planets and stars",
		"What is the capital of France?",
		[]string{"Paris is the capital of France.", "France is in Europe."},
	)
	if !result.SemanticDriftDetected {
		t.Errorf("low context similarity (%.3f) should trigger semantic drift", result.SimilarityToContext)
	}
}

// ─── evaluateQuality ──────────────────────────────────────────────────────────

func TestEvaluateQuality_WellGroundedResponse_HighGrounding(t *testing.T) {
	v := newVerifier()
	source := "Paris is the capital of France and a major European city."
	result := v.evaluateQuality(source, "What is Paris?", []string{source})
	if result.GroundingScore < 0.5 {
		t.Errorf("response identical to source should have high grounding, got %.3f", result.GroundingScore)
	}
}

func TestEvaluateQuality_RelevantResponse_HighRelevance(t *testing.T) {
	v := newVerifier()
	query := "What is Paris the capital of?"
	result := v.evaluateQuality("Paris is the capital of France.", query, nil)
	if result.RelevanceScore < 0.3 {
		t.Errorf("relevant response should have reasonable relevance score, got %.3f", result.RelevanceScore)
	}
}

func TestEvaluateQuality_MultiSentenceResponse_BetterCoherence(t *testing.T) {
	v := newVerifier()
	multi := "Paris is in France. It is a beautiful city. Many tourists visit it. The Eiffel Tower stands there. It is well known."
	single := "Paris."
	rMulti := v.evaluateQuality(multi, "Tell me about Paris.", nil)
	rSingle := v.evaluateQuality(single, "Tell me about Paris.", nil)
	if rMulti.CoherenceScore <= rSingle.CoherenceScore {
		t.Errorf("multi-sentence response should have higher coherence: %.3f vs %.3f",
			rMulti.CoherenceScore, rSingle.CoherenceScore)
	}
}

func TestEvaluateQuality_OverallIsBetweenZeroAndOne(t *testing.T) {
	v := newVerifier()
	result := v.evaluateQuality("A response.", "A query.", []string{"Some context."})
	if result.OverallQuality < 0 || result.OverallQuality > 1 {
		t.Errorf("overall quality should be in [0,1], got %.3f", result.OverallQuality)
	}
}

// ─── detectDrift ──────────────────────────────────────────────────────────────

func TestDetectDrift_Amplification(t *testing.T) {
	v := newVerifier()
	baseline := &models.BiasBaseline{OverallScore: 0.05}
	biasAnal := &models.ResponseBiasAnalysis{OverallScore: 0.20}
	result := v.detectDrift(baseline, biasAnal)
	if result.DriftDirection != "amplification" {
		t.Errorf("expected amplification, got %q", result.DriftDirection)
	}
	if !result.DriftDetected {
		t.Error("drift should be detected for amplification")
	}
}

func TestDetectDrift_Introduction(t *testing.T) {
	v := newVerifier()
	baseline := &models.BiasBaseline{OverallScore: 0.03}
	biasAnal := &models.ResponseBiasAnalysis{OverallScore: 0.20}
	result := v.detectDrift(baseline, biasAnal)
	if result.DriftDirection != "introduction" {
		t.Errorf("expected introduction, got %q", result.DriftDirection)
	}
}

func TestDetectDrift_Reduction(t *testing.T) {
	v := newVerifier()
	baseline := &models.BiasBaseline{OverallScore: 0.30}
	biasAnal := &models.ResponseBiasAnalysis{OverallScore: 0.05}
	result := v.detectDrift(baseline, biasAnal)
	if result.DriftDirection != "reduction" {
		t.Errorf("expected reduction, got %q", result.DriftDirection)
	}
}

func TestDetectDrift_None(t *testing.T) {
	v := newVerifier()
	baseline := &models.BiasBaseline{OverallScore: 0.10}
	biasAnal := &models.ResponseBiasAnalysis{OverallScore: 0.11}
	result := v.detectDrift(baseline, biasAnal)
	if result.DriftDirection != "none" {
		t.Errorf("expected none, got %q", result.DriftDirection)
	}
	if result.DriftDetected {
		t.Error("no significant drift should not set DriftDetected")
	}
}

func TestDetectDrift_AmountIsAbsoluteValue(t *testing.T) {
	v := newVerifier()
	baseline := &models.BiasBaseline{OverallScore: 0.30}
	biasAnal := &models.ResponseBiasAnalysis{OverallScore: 0.05}
	result := v.detectDrift(baseline, biasAnal)
	if result.DriftAmount < 0 {
		t.Errorf("drift amount should be non-negative, got %.3f", result.DriftAmount)
	}
}

// ─── CheckDrift ───────────────────────────────────────────────────────────────

func TestCheckDrift_NilInputs_ReturnsSafe(t *testing.T) {
	v := newVerifier()
	result := v.CheckDrift(&models.DriftRequest{})
	if result.DriftDetected {
		t.Error("nil inputs should not detect drift")
	}
}

func TestCheckDrift_AmplificationAcrossCategories(t *testing.T) {
	v := newVerifier()
	req := &models.DriftRequest{
		Phase1: &models.BiasBaseline{
			OverallScore: 0.05,
			Categories: []models.CategoryBaseline{
				{Category: "gender", Score: 0.05},
			},
		},
		Phase2: &models.ResponseBiasAnalysis{
			OverallScore: 0.25,
			Categories: []models.ResponseCategoryBias{
				{Category: "gender", Score: 0.25},
			},
		},
	}
	result := v.CheckDrift(req)
	if !result.DriftDetected {
		t.Error("expected drift detected")
	}
	if len(result.Drifts) == 0 {
		t.Error("expected at least one category drift")
	}
}

// ─── AnalyzeEmbedding ─────────────────────────────────────────────────────────

func TestAnalyzeEmbedding_Public_IdenticalText_HighSimilarity(t *testing.T) {
	v := newVerifier()
	text := "hello world foo bar baz"
	resp := v.AnalyzeEmbedding(&models.EmbeddingAnalysisRequest{
		Text:            text,
		ComparisonTexts: []string{text},
	})
	if len(resp.Similarities) != 1 {
		t.Fatalf("expected 1 similarity, got %d", len(resp.Similarities))
	}
	if resp.Similarities[0] < 0.9 {
		t.Errorf("identical text similarity should be ≥ 0.9, got %.3f", resp.Similarities[0])
	}
}

func TestAnalyzeEmbedding_Public_DiversityInverseOfSimilarity(t *testing.T) {
	v := newVerifier()
	text := "alpha beta gamma delta"
	resp := v.AnalyzeEmbedding(&models.EmbeddingAnalysisRequest{
		Text:            text,
		ComparisonTexts: []string{text},
	})
	expectedDiversity := 1.0 - resp.Similarities[0]
	if abs(resp.Diversity-expectedDiversity) > 0.001 {
		t.Errorf("diversity should be 1-similarity: want %.3f, got %.3f",
			expectedDiversity, resp.Diversity)
	}
}

// ─── utilities ────────────────────────────────────────────────────────────────

func TestJaccardSimilarity_IdenticalSets_ReturnsOne(t *testing.T) {
	tokens := []string{"a", "b", "c"}
	if sim := jaccardSimilarity(tokens, tokens); sim != 1.0 {
		t.Errorf("identical sets should have Jaccard = 1, got %.3f", sim)
	}
}

func TestJaccardSimilarity_DisjointSets_ReturnsZero(t *testing.T) {
	a := []string{"a", "b", "c"}
	b := []string{"d", "e", "f"}
	if sim := jaccardSimilarity(a, b); sim != 0.0 {
		t.Errorf("disjoint sets should have Jaccard = 0, got %.3f", sim)
	}
}

func TestJaccardSimilarity_BothEmpty_ReturnsOne(t *testing.T) {
	if sim := jaccardSimilarity(nil, nil); sim != 1.0 {
		t.Errorf("both empty should return 1.0, got %.3f", sim)
	}
}

func TestTokenize_SplitsOnSpaceAndPunct(t *testing.T) {
	tokens := tokenize("hello, world! foo.bar")
	expected := map[string]bool{"hello": true, "world": true, "foo": true, "bar": true}
	for _, tok := range tokens {
		if !expected[tok] {
			t.Errorf("unexpected token: %q", tok)
		}
	}
}

func TestDeduplicate_RemovesDuplicates(t *testing.T) {
	out := deduplicate([]string{"a", "b", "a", "c", "b"})
	if len(out) != 3 {
		t.Errorf("expected 3 unique tokens, got %d: %v", len(out), out)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
