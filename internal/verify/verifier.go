package verify

import (
	"math"
	"strings"
	"time"
	"unicode"

	"github.com/bastion/anchor/internal/models"
)

const (
	defaultBiasThreshold = 0.15
	highBiasThreshold    = 0.25
	ampThreshold         = 0.10
	introPhase1Max       = 0.05
	introPhase2Min       = 0.15
	reductionThresh      = 0.10
	contextSimMin        = 0.60
)

// Verifier implements Phase 2 (Anchor-OUT) response verification.
type Verifier struct {
	mediumThreshold float64
	highThreshold   float64
}

// New returns a Verifier with the given bias thresholds.
func New(mediumThreshold, highThreshold float64) *Verifier {
	return &Verifier{
		mediumThreshold: mediumThreshold,
		highThreshold:   highThreshold,
	}
}

// Verify runs a full Phase 2 verification on the given request.
func (v *Verifier) Verify(req *models.VerifyRequest) *models.VerifyResponse {
	start := time.Now()

	opts := req.Options
	runAll := opts == nil || (!opts.CheckBias && !opts.AnalyzeEmbedding && !opts.CheckDrift && !opts.MeasureQuality)

	biasThreshold := defaultBiasThreshold
	if opts != nil && opts.BiasThreshold > 0 {
		biasThreshold = opts.BiasThreshold
	}

	resp := &models.VerifyResponse{
		RequestID: req.RequestID,
		Status:    models.VerifyStatusUnknown,
	}

	var biasAnal *models.ResponseBiasAnalysis
	if runAll || opts.CheckBias {
		ba := v.detectBias(req.LLMResponse, biasThreshold)
		biasAnal = &ba
		resp.BiasAnalysis = biasAnal
	}

	if runAll || opts.AnalyzeEmbedding {
		ea := v.analyzeEmbedding(req.LLMResponse, req.OriginalQuery, req.SourceDocuments)
		resp.EmbeddingAnalysis = &ea
	}

	if (runAll || opts.CheckDrift) && req.Phase1Baseline != nil && biasAnal != nil {
		da := v.detectDrift(req.Phase1Baseline, biasAnal)
		resp.DriftAnalysis = &da
	}

	if runAll || opts.MeasureQuality {
		rq := v.evaluateQuality(req.LLMResponse, req.OriginalQuery, req.SourceDocuments)
		resp.Quality = &rq
	}

	resp.Status = v.determineStatus(biasAnal, resp.DriftAnalysis, resp.EmbeddingAnalysis)
	resp.Warnings, resp.Recommendations = v.generateInsights(resp)
	resp.ProcessingTimeMs = float64(time.Since(start).Microseconds()) / 1000.0

	return resp
}

// AnalyzeEmbedding performs a standalone embedding similarity analysis.
func (v *Verifier) AnalyzeEmbedding(req *models.EmbeddingAnalysisRequest) *models.EmbeddingAnalysisResponse {
	tokens := tokenize(req.Text)
	sims := make([]float64, len(req.ComparisonTexts))
	for i, ct := range req.ComparisonTexts {
		sims[i] = jaccardSimilarity(tokens, tokenize(ct))
	}

	diversity := 0.0
	if len(sims) > 0 {
		sum := 0.0
		for _, s := range sims {
			sum += s
		}
		diversity = 1.0 - sum/float64(len(sims))
		if diversity < 0 {
			diversity = 0
		}
	}

	return &models.EmbeddingAnalysisResponse{
		Embedding:    []float32{},
		Similarities: sims,
		Diversity:    diversity,
	}
}

// CheckDrift performs a standalone Phase 1 vs Phase 2 drift comparison.
func (v *Verifier) CheckDrift(req *models.DriftRequest) *models.DriftResponse {
	if req.Phase1 == nil || req.Phase2 == nil {
		return &models.DriftResponse{DriftDetected: false, Severity: "low"}
	}

	// Build a map of Phase 1 category scores.
	p1Map := make(map[string]float64, len(req.Phase1.Categories))
	for _, c := range req.Phase1.Categories {
		p1Map[c.Category] = c.Score
	}

	var drifts []models.CategoryDrift
	for _, c := range req.Phase2.Categories {
		p1Score := p1Map[c.Category]
		change := c.Score - p1Score
		direction := classifyDrift(p1Score, c.Score, change)
		if direction != "none" {
			drifts = append(drifts, models.CategoryDrift{
				Category:    c.Category,
				Phase1Score: p1Score,
				Phase2Score: c.Score,
				Change:      change,
				Direction:   direction,
			})
		}
	}

	detected := len(drifts) > 0
	severity := "low"
	for _, d := range drifts {
		if d.Direction == "amplification" || d.Direction == "introduction" {
			if math.Abs(d.Change) > 0.25 {
				severity = "high"
			} else if severity != "high" {
				severity = "medium"
			}
		}
	}

	return &models.DriftResponse{
		DriftDetected: detected,
		Drifts:        drifts,
		Severity:      severity,
	}
}

// ─── internal: bias detection ─────────────────────────────────────────────────

var (
	genderGroupA = wordSet([]string{
		"male", "man", "men", "he", "his", "him", "boy", "boys", "gentleman",
		"남성", "남자", "남자들", "남성들", "그는", "그의",
	})
	genderGroupB = wordSet([]string{
		"female", "woman", "women", "she", "her", "girl", "girls", "lady", "ladies",
		"여성", "여자", "여자들", "여성들", "그녀는", "그녀의",
	})
	ethnicityGroupA = wordSet([]string{
		"korean", "한국인", "한국인들",
	})
	ethnicityGroupB = wordSet([]string{
		"foreign", "foreigner", "foreigners", "외국인", "외국인들",
	})
	ageGroupA = wordSet([]string{
		"young", "youth", "junior", "젊은", "청년", "어린",
	})
	ageGroupB = wordSet([]string{
		"old", "elder", "elderly", "senior", "나이든", "노인",
	})

	biasPatterns = map[string][]string{
		"gender": {
			"women are too emotional", "men are better leaders",
			"여자는 감정적이다", "남자가 더 잘한다", "여성은 약하다", "남성이 더 적합하다",
			"여자는 감정적이라서 리더가 될 수 없다",
		},
		"ethnicity": {
			"foreigners cannot", "foreigners don't understand",
			"외국인은 못한다", "외국인은 이해 못한다", "한국인만 가능하다",
		},
		"age": {
			"too old to", "young people don't know",
			"나이가 많아서 못한다", "젊은 사람은 모른다", "노인은 배우기 힘들다",
		},
	}
)

func (v *Verifier) detectBias(text string, threshold float64) models.ResponseBiasAnalysis {
	lower := strings.ToLower(text)
	tokens := tokenize(lower)

	categories := []struct {
		name   string
		groupA map[string]struct{}
		groupB map[string]struct{}
	}{
		{"gender", genderGroupA, genderGroupB},
		{"ethnicity", ethnicityGroupA, ethnicityGroupB},
		{"age", ageGroupA, ageGroupB},
	}

	var catBias []models.ResponseCategoryBias
	overallMax := 0.0

	for _, cat := range categories {
		countA, countB := 0, 0
		for _, t := range tokens {
			if _, ok := cat.groupA[t]; ok {
				countA++
			}
			if _, ok := cat.groupB[t]; ok {
				countB++
			}
		}

		score := 0.0
		if total := countA + countB; total > 0 {
			score = math.Abs(float64(countA-countB)) / float64(total)
		}

		// Boost score on explicit stereotype patterns.
		var evidence []string
		for _, pat := range biasPatterns[cat.name] {
			if strings.Contains(lower, pat) {
				if score < 0.4 {
					score = 0.4
				}
				evidence = append(evidence, pat)
			}
		}

		if score > 0 || len(evidence) > 0 {
			catBias = append(catBias, models.ResponseCategoryBias{
				Category: cat.name,
				Score:    score,
				Evidence: evidence,
			})
		}

		if score > overallMax {
			overallMax = score
		}
	}

	severity := "low"
	exceeds := overallMax >= threshold
	if overallMax >= v.highThreshold {
		severity = "high"
	} else if overallMax >= v.mediumThreshold {
		severity = "medium"
	}

	return models.ResponseBiasAnalysis{
		OverallScore:     overallMax,
		Categories:       catBias,
		Severity:         severity,
		ExceedsThreshold: exceeds,
	}
}

// ─── internal: embedding analysis ─────────────────────────────────────────────

func (v *Verifier) analyzeEmbedding(response, query string, sources []string) models.EmbeddingAnalysis {
	respTokens := tokenize(strings.ToLower(response))
	queryTokens := tokenize(strings.ToLower(query))

	simToQuery := jaccardSimilarity(respTokens, queryTokens)

	simToContext := 0.0
	if len(sources) > 0 {
		combined := make([]string, 0)
		for _, s := range sources {
			combined = append(combined, tokenize(strings.ToLower(s))...)
		}
		combined = deduplicate(combined)
		simToContext = jaccardSimilarity(respTokens, combined)
	}

	diversity := 1.0 - simToContext
	if diversity < 0 {
		diversity = 0
	}

	return models.EmbeddingAnalysis{
		SimilarityToQuery:     simToQuery,
		SimilarityToContext:   simToContext,
		DiversityScore:        diversity,
		SemanticDriftDetected: simToContext < contextSimMin && len(sources) > 0,
	}
}

// ─── internal: quality evaluation ─────────────────────────────────────────────

func (v *Verifier) evaluateQuality(response, query string, sources []string) models.ResponseQuality {
	respTokens := tokenize(strings.ToLower(response))
	respSet := wordSet(respTokens)

	// Grounding: fraction of unique response words found in source docs.
	grounding := 0.0
	if len(sources) > 0 && len(respSet) > 0 {
		sourceTokens := make([]string, 0)
		for _, s := range sources {
			sourceTokens = append(sourceTokens, tokenize(strings.ToLower(s))...)
		}
		sourceSet := wordSet(sourceTokens)
		found := 0
		for w := range respSet {
			if _, ok := sourceSet[w]; ok {
				found++
			}
		}
		grounding = float64(found) / float64(len(respSet))
	}

	// Coherence: sentence structure heuristic (more sentences = more coherent up to a cap).
	sentences := strings.FieldsFunc(response, func(r rune) bool {
		return r == '.' || r == '!' || r == '?' || r == '。' || r == '！' || r == '？'
	})
	nonEmpty := 0
	for _, s := range sentences {
		if strings.TrimSpace(s) != "" {
			nonEmpty++
		}
	}
	coherence := math.Min(float64(nonEmpty)/5.0, 1.0)

	// Relevance: Jaccard between response and query.
	relevance := jaccardSimilarity(respTokens, tokenize(strings.ToLower(query)))

	overall := 0.4*grounding + 0.3*coherence + 0.3*relevance
	return models.ResponseQuality{
		GroundingScore: grounding,
		CoherenceScore: coherence,
		RelevanceScore: relevance,
		OverallQuality: overall,
	}
}

// ─── internal: drift detection ─────────────────────────────────────────────────

func (v *Verifier) detectDrift(baseline *models.BiasBaseline, biasAnal *models.ResponseBiasAnalysis) models.DriftAnalysis {
	p1 := baseline.OverallScore
	p2 := biasAnal.OverallScore
	diff := p2 - p1
	direction := classifyDrift(p1, p2, diff)

	return models.DriftAnalysis{
		DriftDetected:   direction != "none",
		Phase1BiasScore: p1,
		Phase2BiasScore: p2,
		DriftAmount:     math.Abs(diff),
		DriftDirection:  direction,
	}
}

func classifyDrift(p1, p2, diff float64) string {
	switch {
	case p1 < introPhase1Max && p2 > introPhase2Min:
		return "introduction"
	case diff > ampThreshold:
		return "amplification"
	case diff < -reductionThresh:
		return "reduction"
	default:
		return "none"
	}
}

// ─── internal: status & insights ──────────────────────────────────────────────

func (v *Verifier) determineStatus(
	bias *models.ResponseBiasAnalysis,
	drift *models.DriftAnalysis,
	embed *models.EmbeddingAnalysis,
) models.VerifyStatus {
	if bias != nil {
		if bias.OverallScore >= v.highThreshold {
			return models.VerifyStatusCritical
		}
		if bias.ExceedsThreshold {
			return models.VerifyStatusWarning
		}
	}
	if drift != nil {
		if drift.DriftDirection == "amplification" || drift.DriftDirection == "introduction" {
			return models.VerifyStatusWarning
		}
		if drift.DriftDetected {
			return models.VerifyStatusWarning
		}
	}
	if embed != nil && embed.SemanticDriftDetected {
		return models.VerifyStatusWarning
	}
	return models.VerifyStatusPassed
}

func (v *Verifier) generateInsights(resp *models.VerifyResponse) (warnings, recommendations []string) {
	if resp.BiasAnalysis != nil {
		ba := resp.BiasAnalysis
		if ba.ExceedsThreshold {
			warnings = append(warnings, "Response bias score exceeds threshold ("+ba.Severity+")")
		}
		for _, cat := range ba.Categories {
			if len(cat.Evidence) > 0 {
				warnings = append(warnings, "Stereotype pattern detected in category: "+cat.Category)
				recommendations = append(recommendations, "Review and rephrase content related to "+cat.Category+" stereotypes")
			}
		}
	}

	if resp.DriftAnalysis != nil {
		da := resp.DriftAnalysis
		switch da.DriftDirection {
		case "amplification":
			warnings = append(warnings, "Bias amplified from Phase 1 to Phase 2 response")
			recommendations = append(recommendations, "Inspect retrieval pipeline for biased source documents")
		case "introduction":
			warnings = append(warnings, "New bias introduced in Phase 2 not present in Phase 1")
			recommendations = append(recommendations, "Review LLM generation step for introduced bias")
		case "reduction":
			recommendations = append(recommendations, "Bias reduced in response — no action required")
		}
	}

	if resp.EmbeddingAnalysis != nil && resp.EmbeddingAnalysis.SemanticDriftDetected {
		warnings = append(warnings, "Semantic drift detected: response may not be grounded in source documents")
		recommendations = append(recommendations, "Verify retrieval results are relevant to the query")
	}

	if resp.Quality != nil && resp.Quality.OverallQuality < 0.5 {
		warnings = append(warnings, "Low overall response quality score")
		recommendations = append(recommendations, "Improve grounding by adding more relevant source documents")
	}

	return warnings, recommendations
}

// ─── utilities ────────────────────────────────────────────────────────────────

func tokenize(text string) []string {
	return strings.FieldsFunc(text, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r)
	})
}

func wordSet(tokens []string) map[string]struct{} {
	m := make(map[string]struct{}, len(tokens))
	for _, t := range tokens {
		m[t] = struct{}{}
	}
	return m
}

func jaccardSimilarity(a, b []string) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	setA := wordSet(a)
	setB := wordSet(b)

	intersection := 0
	for k := range setA {
		if _, ok := setB[k]; ok {
			intersection++
		}
	}
	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func deduplicate(tokens []string) []string {
	seen := make(map[string]struct{}, len(tokens))
	out := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if _, ok := seen[t]; !ok {
			seen[t] = struct{}{}
			out = append(out, t)
		}
	}
	return out
}
