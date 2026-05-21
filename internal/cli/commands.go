package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/bastion/anchor/internal/bias"
	"github.com/bastion/anchor/internal/config"
	"github.com/bastion/anchor/internal/models"
	"github.com/bastion/anchor/internal/noise"
	"github.com/bastion/anchor/internal/quality"
	"github.com/bastion/anchor/internal/verify"
)

// Build constructs the root command tree.
func Build(cfg *config.Config, inj *noise.Injector, ana *bias.Analyzer, mon *quality.Monitor, ver *verify.Verifier) *cobra.Command {
	root := &cobra.Command{
		Use:   "anchor-cli",
		Short: "Bastion-Anchor CLI — embedding security operations",
	}

	root.AddCommand(
		buildSecure(inj),
		buildMeasureBias(ana),
		buildTestQuality(cfg, mon),
		buildCompareNoise(cfg, mon),
		buildDemo(cfg, inj, ana, mon, ver),
		buildInteractive(cfg, inj, ana, ver),
		buildVerify(ver),
		buildAnalyzeEmbedding(ver),
		buildCompare(ver),
	)
	return root
}

// ─── secure command ───────────────────────────────────────────────────────────

func buildSecure(inj *noise.Injector) *cobra.Command {
	var embFile, outFile string
	var sigma float64
	var seed int64

	cmd := &cobra.Command{
		Use:   "secure",
		Short: "Apply noise to an embedding",
		Example: `  anchor-cli secure --embedding-file query.json --noise-sigma 0.01
  anchor-cli secure --embedding-file query.json --output secured.json`,
		RunE: func(cmd *cobra.Command, args []string) error {
			data, err := os.ReadFile(embFile)
			if err != nil {
				return err
			}
			var embedding []float32
			if err := json.Unmarshal(data, &embedding); err != nil {
				return fmt.Errorf("embedding file must be a JSON float array: %w", err)
			}

			activeInj := inj
			if sigma > 0 || seed != 0 {
				if seed != 0 {
					activeInj = noise.NewDeterministic(sigma, true, seed)
				} else {
					activeInj, err = noise.New(sigma, true, noise.StrategyGaussian)
					if err != nil {
						return err
					}
				}
			}

			result := activeInj.Inject(embedding)
			printSecureResult(embedding, result)

			if outFile != "" {
				out, _ := json.Marshal(result.Secured)
				return os.WriteFile(outFile, out, 0o644)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&embFile, "embedding-file", "", "JSON file containing float32 array")
	cmd.Flags().StringVar(&outFile, "output", "", "Output file for secured embedding")
	cmd.Flags().Float64Var(&sigma, "noise-sigma", 0, "Noise sigma (default: from config)")
	cmd.Flags().Int64Var(&seed, "seed", 0, "Deterministic seed (0 = random)")
	_ = cmd.MarkFlagRequired("embedding-file")
	return cmd
}

// ─── measure-bias command ─────────────────────────────────────────────────────

func buildMeasureBias(ana *bias.Analyzer) *cobra.Command {
	var words, pairsFlag, testSet, outFile string

	cmd := &cobra.Command{
		Use:   "measure-bias",
		Short: "Measure embedding bias using WEAT",
		Example: `  anchor-cli measure-bias --test-set gender
  anchor-cli measure-bias --test-words doctor,nurse --pairs gender:male,female`,
		RunE: func(cmd *cobra.Command, args []string) error {
			var resp models.BiasResponse
			var err error

			if testSet != "" {
				resp, err = ana.AnalyzeBuiltin(testSet, nil)
				if err != nil {
					return err
				}
			} else {
				testWords := splitTrim(words)
				pairs := parsePairs(pairsFlag)
				resp = ana.Analyze(testWords, pairs, nil)
			}

			printBiasReport(resp)
			if outFile != "" {
				data, _ := json.MarshalIndent(resp, "", "  ")
				return os.WriteFile(outFile, data, 0o644)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&testSet, "test-set", "", "Built-in test set: gender|ethnicity|age")
	cmd.Flags().StringVar(&words, "test-words", "", "Comma-separated test words")
	cmd.Flags().StringVar(&pairsFlag, "pairs", "", "Attribute pairs: category:a,b (e.g. gender:male,female)")
	cmd.Flags().StringVar(&outFile, "output", "", "Output JSON file")
	return cmd
}

// ─── test-quality command ─────────────────────────────────────────────────────

func buildTestQuality(cfg *config.Config, mon *quality.Monitor) *cobra.Command {
	var queriesFile string
	var sigma float64

	cmd := &cobra.Command{
		Use:   "test-quality",
		Short: "Measure recall impact of noise injection",
		RunE: func(cmd *cobra.Command, args []string) error {
			if sigma == 0 {
				sigma = cfg.Noise.DefaultSigma
			}
			var pairs []models.EmbeddingPair
			if queriesFile != "" {
				var err error
				pairs, err = loadEmbeddingPairs(queriesFile)
				if err != nil {
					return err
				}
			}

			resp := mon.Estimate(sigma, pairs)
			printQualityReport(sigma, resp)
			return nil
		},
	}
	cmd.Flags().StringVar(&queriesFile, "queries", "", "JSONL file with EmbeddingPair records")
	cmd.Flags().Float64Var(&sigma, "noise-sigma", 0, "Noise sigma to test")
	return cmd
}

// ─── compare-noise command ────────────────────────────────────────────────────

func buildCompareNoise(cfg *config.Config, mon *quality.Monitor) *cobra.Command {
	var queriesFile, sigmasFlag string

	cmd := &cobra.Command{
		Use:   "compare-noise",
		Short: "Compare recall impact across noise levels",
		RunE: func(cmd *cobra.Command, args []string) error {
			var pairs []models.EmbeddingPair
			if queriesFile != "" {
				var err error
				pairs, err = loadEmbeddingPairs(queriesFile)
				if err != nil {
					return err
				}
			}

			sigmas := parseFloats(sigmasFlag)
			if len(sigmas) == 0 {
				sigmas = []float64{0.005, 0.01, 0.02, 0.05}
			}
			_ = cfg

			sep := strings.Repeat("─", 55)
			fmt.Println("Noise Level Comparison:")
			fmt.Println(sep)
			fmt.Printf("%-8s  %-12s  %-12s  %s\n", "Sigma", "Recall@K", "Similarity", "Verdict")
			fmt.Println(sep)
			for _, s := range sigmas {
				resp := mon.Estimate(s, pairs)
				verdict := quality.SigmaVerdict(s)
				fmt.Printf("%-8.3f  %-12.3f  %-12.3f  %s\n",
					s, resp.RecallWith, resp.AvgSimilarityToOriginal, verdict)
			}
			fmt.Println(sep)
			return nil
		},
	}
	cmd.Flags().StringVar(&queriesFile, "queries", "", "JSONL file with EmbeddingPair records")
	cmd.Flags().StringVar(&sigmasFlag, "sigmas", "0.005,0.01,0.02,0.05", "Comma-separated sigma values")
	return cmd
}

// ─── demo command ─────────────────────────────────────────────────────────────

func buildDemo(cfg *config.Config, inj *noise.Injector, ana *bias.Analyzer, mon *quality.Monitor, ver *verify.Verifier) *cobra.Command {
	var phase int
	var scenario string
	cmd := &cobra.Command{
		Use:   "demo",
		Short: "Run a 2-minute Anchor security demonstration",
		RunE: func(cmd *cobra.Command, args []string) error {
			if phase == 2 {
				runDemoPhase2(ver, scenario)
			} else {
				runDemo(cfg, inj, ana, mon)
			}
			return nil
		},
	}
	cmd.Flags().IntVar(&phase, "phase", 1, "Demo phase: 1 (embedding security) or 2 (output verification)")
	cmd.Flags().StringVar(&scenario, "scenario", "bias-amplification", "Phase 2 scenario: bias-amplification|bias-introduction|quality-verification")
	return cmd
}

func runDemoPhase2(ver *verify.Verifier, scenario string) {
	sep := strings.Repeat("═", 54)
	thin := strings.Repeat("─", 54)

	fmt.Println(sep)
	fmt.Println("  Anchor-OUT Phase 2 Demo")
	fmt.Println(sep)

	switch scenario {
	case "bias-introduction":
		fmt.Println("Scenario: Bias Introduction Detection")
		fmt.Println(thin)
		fmt.Println("Source documents: neutral technical content")
		fmt.Println("Phase 1 baseline: 0.02 (near zero)")
		resp := ver.Verify(&models.VerifyRequest{
			LLMResponse:     "Foreigners cannot fully understand our local culture and business practices.",
			OriginalQuery:   "What skills are important for this role?",
			SourceDocuments: []string{"Strong analytical skills are essential.", "Teamwork and communication matter."},
			Phase1Baseline:  &models.BiasBaseline{OverallScore: 0.02},
		})
		printVerifyReport(resp)

	case "quality-verification":
		fmt.Println("Scenario: Quality Verification")
		fmt.Println(thin)
		fmt.Println("Source documents: specific facts about a product")
		resp := ver.Verify(&models.VerifyRequest{
			LLMResponse:     "The product costs $999 and was invented in 1850 by Napoleon.",
			OriginalQuery:   "What is the product price?",
			SourceDocuments: []string{"The product costs $499.", "Released in 2023."},
		})
		printVerifyReport(resp)

	default: // "bias-amplification"
		fmt.Println("Scenario: Bias Amplification Detection")
		fmt.Println(thin)
		fmt.Println("Source documents: gender-neutral")
		fmt.Println("Phase 1 baseline: 0.05 (low)")
		resp := ver.Verify(&models.VerifyRequest{
			LLMResponse:     "Engineers are typically analytical thinkers. Men who possess these qualities tend to excel in this field.",
			OriginalQuery:   "What are the characteristics of engineers?",
			SourceDocuments: []string{"Engineers need analytical and problem-solving skills.", "Engineering is a field where people from diverse backgrounds excel."},
			Phase1Baseline:  &models.BiasBaseline{OverallScore: 0.05},
		})
		printVerifyReport(resp)
	}
}

func runDemo(cfg *config.Config, inj *noise.Injector, ana *bias.Analyzer, mon *quality.Monitor) {
	sep := strings.Repeat("═", 54)
	thin := strings.Repeat("─", 54)

	fmt.Println(sep)
	fmt.Println("  Anchor Security Demo")
	fmt.Println(sep)

	// Demo 1: Noise injection
	fmt.Println("\nDemo 1: Embedding Noise Injection")
	fmt.Println(thin)
	sampleEmb := sampleEmbedding(16) // show first 16 dims
	result := inj.Inject(sampleEmb)
	fmt.Printf("Original embedding:  %s\n", formatVec(sampleEmb[:8]))
	fmt.Printf("Applying noise (σ=%.3f)...\n", cfg.Noise.DefaultSigma)
	fmt.Printf("Secured embedding:   %s\n", formatVec(result.Secured[:8]))
	fmt.Printf("Cosine similarity:   %.4f  %s\n", result.SimilarityToOriginal, qualIcon(result.SimilarityToOriginal))
	fmt.Printf("Noise magnitude:     %.4f\n", result.NoiseMagnitude)
	fmt.Println("Status: Embedding secured!")

	// Demo 2: Bias measurement
	fmt.Println("\nDemo 2: Bias Measurement")
	fmt.Println(thin)
	resp, _ := ana.AnalyzeBuiltin("gender", nil)
	printBiasReport(resp)

	// Demo 3: Quality impact
	fmt.Println("\nDemo 3: Quality Impact")
	fmt.Println(thin)
	qResp := mon.Estimate(cfg.Noise.DefaultSigma, nil)
	printQualityReport(cfg.Noise.DefaultSigma, qResp)

	fmt.Println("\n" + sep)
}

// ─── interactive command ──────────────────────────────────────────────────────

func buildInteractive(cfg *config.Config, inj *noise.Injector, ana *bias.Analyzer, ver *verify.Verifier) *cobra.Command {
	return &cobra.Command{
		Use:   "interactive",
		Short: "Start an interactive Anchor shell",
		RunE: func(cmd *cobra.Command, args []string) error {
			runInteractive(cfg, inj, ana, ver)
			return nil
		},
	}
}

func runInteractive(cfg *config.Config, inj *noise.Injector, ana *bias.Analyzer, ver *verify.Verifier) {
	fmt.Println("Bastion-Anchor interactive mode. Type 'help' for commands.")
	mode := "input" // "input" | "output"
	var verifyCount, biasDetectedCount, driftCount int

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("anchor[%s]> ", mode)
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		switch parts[0] {
		case "exit", "quit":
			fmt.Println("Goodbye.")
			return
		case "help":
			fmt.Println(`Commands (Phase 1 - input):
  config              Show current configuration
  secure-test         Apply noise to a random embedding
  bias <set>          Run built-in bias test (gender|ethnicity|age)
  sigma <value>       Change noise sigma
  mode <input|output> Switch phase
Commands (Phase 2 - output):
  verify <text>       Verify an LLM response text for bias
  stats               Show session statistics
  exit                Quit`)
		case "mode":
			if len(parts) < 2 || (parts[1] != "input" && parts[1] != "output") {
				fmt.Println("Usage: mode <input|output>")
				continue
			}
			mode = parts[1]
			fmt.Printf("Mode: %s (Phase %s)\n", mode, map[string]string{"input": "1", "output": "2"}[mode])
		case "config":
			fmt.Printf("{\n  \"default_noise_sigma\": %.4f,\n  \"preserve_norm\": %v,\n  \"strategy\": %q\n}\n",
				cfg.Noise.DefaultSigma, cfg.Noise.PreserveNorm, cfg.Noise.Strategy)
		case "secure-test":
			emb := sampleEmbedding(1024)
			result := inj.Inject(emb)
			fmt.Printf("Original:   [%.4f, %.4f, ...]\n", emb[0], emb[1])
			fmt.Printf("Secured:    [%.4f, %.4f, ...]\n", result.Secured[0], result.Secured[1])
			fmt.Printf("Similarity: %.4f\n", result.SimilarityToOriginal)
		case "bias":
			set := "gender"
			if len(parts) > 1 {
				set = parts[1]
			}
			resp, err := ana.AnalyzeBuiltin(set, nil)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}
			fmt.Printf("Overall bias score: %.4f (%s)\n", resp.OverallBiasScore, resp.Interpretation)
		case "sigma":
			if len(parts) < 2 {
				fmt.Printf("Current sigma: %.4f\n", inj.Sigma())
				continue
			}
			s, err := strconv.ParseFloat(parts[1], 64)
			if err != nil {
				fmt.Println("Invalid sigma value")
				continue
			}
			inj.SetSigma(s)
			cfg.Noise.DefaultSigma = s
			fmt.Printf("Sigma updated to %.4f\n", s)
		case "verify":
			if mode != "output" {
				fmt.Println("Switch to output mode first: mode output")
				continue
			}
			text := strings.Join(parts[1:], " ")
			if text == "" {
				fmt.Println("Usage: verify <response text>")
				continue
			}
			resp := ver.Verify(&models.VerifyRequest{LLMResponse: text})
			verifyCount++
			if resp.BiasAnalysis != nil && resp.BiasAnalysis.ExceedsThreshold {
				biasDetectedCount++
			}
			if resp.DriftAnalysis != nil && resp.DriftAnalysis.DriftDetected {
				driftCount++
			}
			statusIcon := "✅"
			switch resp.Status {
			case models.VerifyStatusWarning:
				statusIcon = "⚠️ "
			case models.VerifyStatusCritical:
				statusIcon = "❌"
			}
			fmt.Printf("%s Status: %s\n", statusIcon, resp.Status)
			if resp.BiasAnalysis != nil {
				fmt.Printf("   Bias detected: %.3f  severity=%s\n",
					resp.BiasAnalysis.OverallScore, resp.BiasAnalysis.Severity)
				for _, cat := range resp.BiasAnalysis.Categories {
					if len(cat.Evidence) > 0 {
						fmt.Printf("   Evidence [%s]: %s\n", cat.Category, strings.Join(cat.Evidence, ", "))
					}
				}
			}
			for _, rec := range resp.Recommendations {
				fmt.Printf("   → %s\n", rec)
			}
		case "stats":
			fmt.Printf("Session Statistics (phase %s):\n", mode)
			fmt.Printf("  Verifications:  %d\n", verifyCount)
			fmt.Printf("  Bias detected:  %d\n", biasDetectedCount)
			fmt.Printf("  Drift alerts:   %d\n", driftCount)
		default:
			fmt.Printf("Unknown command: %s (type 'help')\n", parts[0])
		}
	}
}

// ─── verify command ───────────────────────────────────────────────────────────

func buildVerify(ver *verify.Verifier) *cobra.Command {
	var responseFile, queryFile, sourcesFlag, outFile string

	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Verify an LLM response for bias and quality (Phase 2)",
		Example: `  anchor-cli verify --response response.txt --query "What is justice?"
  anchor-cli verify --response response.txt --query query.txt --sources doc1.txt,doc2.txt`,
		RunE: func(cmd *cobra.Command, args []string) error {
			response, err := os.ReadFile(responseFile)
			if err != nil {
				return fmt.Errorf("reading response file: %w", err)
			}

			query := queryFile
			if _, err2 := os.Stat(queryFile); err2 == nil {
				data, err3 := os.ReadFile(queryFile)
				if err3 != nil {
					return err3
				}
				query = strings.TrimSpace(string(data))
			}

			var sources []string
			for _, s := range splitTrim(sourcesFlag) {
				data, err2 := os.ReadFile(s)
				if err2 == nil {
					sources = append(sources, string(data))
				} else {
					sources = append(sources, s)
				}
			}

			req := &models.VerifyRequest{
				LLMResponse:     strings.TrimSpace(string(response)),
				OriginalQuery:   query,
				SourceDocuments: sources,
			}

			resp := ver.Verify(req)
			printVerifyReport(resp)

			if outFile != "" {
				data, _ := json.MarshalIndent(resp, "", "  ")
				return os.WriteFile(outFile, data, 0o644)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&responseFile, "response", "", "File containing the LLM response text")
	cmd.Flags().StringVar(&queryFile, "query", "", "Original query text or file path")
	cmd.Flags().StringVar(&sourcesFlag, "sources", "", "Comma-separated source document files or texts")
	cmd.Flags().StringVar(&outFile, "output", "", "Output JSON file for full report")
	_ = cmd.MarkFlagRequired("response")
	_ = cmd.MarkFlagRequired("query")
	return cmd
}

// ─── analyze-embedding command ────────────────────────────────────────────────

func buildAnalyzeEmbedding(ver *verify.Verifier) *cobra.Command {
	var textFlag, compareFlag string

	cmd := &cobra.Command{
		Use:   "analyze-embedding",
		Short: "Analyze response embedding similarity against comparison texts (Phase 2)",
		Example: `  anchor-cli analyze-embedding --text "response text" --compare-against doc1.txt,doc2.txt`,
		RunE: func(cmd *cobra.Command, args []string) error {
			comparisonTexts := make([]string, 0)
			for _, path := range splitTrim(compareFlag) {
				data, err := os.ReadFile(path)
				if err == nil {
					comparisonTexts = append(comparisonTexts, string(data))
				} else {
					comparisonTexts = append(comparisonTexts, path)
				}
			}

			resp := ver.AnalyzeEmbedding(&models.EmbeddingAnalysisRequest{
				Text:            textFlag,
				ComparisonTexts: comparisonTexts,
			})

			thin := strings.Repeat("─", 48)
			fmt.Println("Embedding Similarity Analysis:")
			fmt.Println(thin)
			for i, sim := range resp.Similarities {
				label := fmt.Sprintf("doc-%d", i+1)
				if i < len(comparisonTexts) && len(comparisonTexts[i]) < 30 {
					label = strings.TrimSpace(comparisonTexts[i])
				}
				fmt.Printf("  %-30s  %.3f\n", truncate(label, 30), sim)
			}
			fmt.Println(thin)
			fmt.Printf("Diversity score: %.3f\n", resp.Diversity)
			return nil
		},
	}
	cmd.Flags().StringVar(&textFlag, "text", "", "Response text to analyze")
	cmd.Flags().StringVar(&compareFlag, "compare-against", "", "Comma-separated comparison file paths or texts")
	_ = cmd.MarkFlagRequired("text")
	return cmd
}

// ─── compare command ──────────────────────────────────────────────────────────

func buildCompare(ver *verify.Verifier) *cobra.Command {
	var baselineFile, responseFile string

	cmd := &cobra.Command{
		Use:   "compare",
		Short: "Compare Phase 1 bias baseline vs Phase 2 response bias (Phase 2)",
		Example: `  anchor-cli compare --phase1-baseline baseline.json --response-file response.txt`,
		RunE: func(cmd *cobra.Command, args []string) error {
			baselineData, err := os.ReadFile(baselineFile)
			if err != nil {
				return fmt.Errorf("reading baseline file: %w", err)
			}
			var baseline models.BiasBaseline
			if err := json.Unmarshal(baselineData, &baseline); err != nil {
				return fmt.Errorf("parsing baseline: %w", err)
			}

			responseData, err := os.ReadFile(responseFile)
			if err != nil {
				return fmt.Errorf("reading response file: %w", err)
			}

			verResp := ver.Verify(&models.VerifyRequest{
				LLMResponse:    strings.TrimSpace(string(responseData)),
				Phase1Baseline: &baseline,
			})

			thin := strings.Repeat("─", 48)
			fmt.Printf("Phase 1 baseline: %.3f\n", baseline.OverallScore)
			if verResp.BiasAnalysis != nil {
				fmt.Printf("Phase 2 detected: %.3f\n", verResp.BiasAnalysis.OverallScore)
			}
			if verResp.DriftAnalysis != nil {
				da := verResp.DriftAnalysis
				sign := "+"
				if da.DriftDirection == "reduction" {
					sign = "-"
				}
				fmt.Printf("Drift: %s%.3f (%s)\n", sign, da.DriftAmount, da.DriftDirection)
				fmt.Println(thin)
				if da.DriftDetected {
					switch da.DriftDirection {
					case "amplification":
						fmt.Println("⚠ LLM AMPLIFIED bias beyond Phase 1 baseline")
					case "introduction":
						fmt.Println("⚠ LLM INTRODUCED bias not present in Phase 1")
					case "reduction":
						fmt.Println("✅ Bias REDUCED in response (positive case)")
					}
				} else {
					fmt.Println("✅ No significant drift detected")
				}
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&baselineFile, "phase1-baseline", "", "JSON file containing BiasBaseline")
	cmd.Flags().StringVar(&responseFile, "response-file", "", "File containing the LLM response text")
	_ = cmd.MarkFlagRequired("phase1-baseline")
	_ = cmd.MarkFlagRequired("response-file")
	return cmd
}

func printVerifyReport(resp *models.VerifyResponse) {
	sep := strings.Repeat("═", 54)
	thin := strings.Repeat("─", 54)
	statusIcon := "✅"
	switch resp.Status {
	case models.VerifyStatusWarning:
		statusIcon = "⚠️ "
	case models.VerifyStatusCritical:
		statusIcon = "❌"
	}

	fmt.Println(sep)
	fmt.Println("  Anchor Phase 2 Verification Report")
	fmt.Println(sep)
	fmt.Printf("Status:  %s %s\n", statusIcon, resp.Status)
	fmt.Printf("Elapsed: %.1f ms\n", resp.ProcessingTimeMs)

	if resp.BiasAnalysis != nil {
		fmt.Println(thin)
		fmt.Println("Bias Analysis:")
		fmt.Printf("  Overall score: %.3f  severity=%s  exceeds_threshold=%v\n",
			resp.BiasAnalysis.OverallScore, resp.BiasAnalysis.Severity, resp.BiasAnalysis.ExceedsThreshold)
		for _, c := range resp.BiasAnalysis.Categories {
			fmt.Printf("  [%s] score=%.3f  evidence=%d patterns\n", c.Category, c.Score, len(c.Evidence))
		}
	}

	if resp.EmbeddingAnalysis != nil {
		fmt.Println(thin)
		fmt.Println("Embedding Analysis:")
		fmt.Printf("  Sim to query:   %.3f\n", resp.EmbeddingAnalysis.SimilarityToQuery)
		fmt.Printf("  Sim to context: %.3f\n", resp.EmbeddingAnalysis.SimilarityToContext)
		fmt.Printf("  Diversity:      %.3f\n", resp.EmbeddingAnalysis.DiversityScore)
		fmt.Printf("  Semantic drift: %v\n", resp.EmbeddingAnalysis.SemanticDriftDetected)
	}

	if resp.DriftAnalysis != nil {
		fmt.Println(thin)
		fmt.Println("Drift Analysis:")
		fmt.Printf("  Phase1 bias: %.3f  Phase2 bias: %.3f\n",
			resp.DriftAnalysis.Phase1BiasScore, resp.DriftAnalysis.Phase2BiasScore)
		fmt.Printf("  Direction: %s  amount=%.3f\n",
			resp.DriftAnalysis.DriftDirection, resp.DriftAnalysis.DriftAmount)
	}

	if resp.Quality != nil {
		fmt.Println(thin)
		fmt.Println("Response Quality:")
		fmt.Printf("  Grounding:  %.3f\n", resp.Quality.GroundingScore)
		fmt.Printf("  Coherence:  %.3f\n", resp.Quality.CoherenceScore)
		fmt.Printf("  Relevance:  %.3f\n", resp.Quality.RelevanceScore)
		fmt.Printf("  Overall:    %.3f\n", resp.Quality.OverallQuality)
	}

	if len(resp.Warnings) > 0 {
		fmt.Println(thin)
		fmt.Println("Warnings:")
		for _, w := range resp.Warnings {
			fmt.Printf("  ⚠ %s\n", w)
		}
	}

	if len(resp.Recommendations) > 0 {
		fmt.Println(thin)
		fmt.Println("Recommendations:")
		for _, r := range resp.Recommendations {
			fmt.Printf("  → %s\n", r)
		}
	}

	fmt.Println(sep)
}

// ─── output helpers ───────────────────────────────────────────────────────────

func printSecureResult(original []float32, result noise.Result) {
	sep := strings.Repeat("═", 54)
	thin := strings.Repeat("─", 54)
	fmt.Println(sep)
	fmt.Println("  Anchor Security Report")
	fmt.Println(sep)
	fmt.Printf("Strategy:        %s\n", result.Strategy)
	fmt.Printf("Noise magnitude: %.4f (avg)\n", result.NoiseMagnitude)
	fmt.Printf("Original norm:   %.4f\n", result.OriginalNorm)
	fmt.Printf("Secured norm:    %.4f\n", result.SecuredNorm)
	fmt.Println(thin)
	fmt.Printf("Cosine sim:      %.4f  %s\n", result.SimilarityToOriginal, qualIcon(result.SimilarityToOriginal))
	fmt.Printf("\nOriginal[0:4]: %s\n", formatVec(original[:min(4, len(original))]))
	fmt.Printf("Secured [0:4]: %s\n", formatVec(result.Secured[:min(4, len(result.Secured))]))
	fmt.Println(sep)
}

func printBiasReport(resp models.BiasResponse) {
	thin := strings.Repeat("─", 48)
	fmt.Println("Bias Analysis Report:")
	fmt.Println(thin)
	fmt.Printf("%-20s %-12s %-8s  %s\n", "Word", "Category", "Score", "Severity")
	fmt.Println(thin)
	for _, m := range resp.Measurements {
		icon := "🟢"
		if m.Severity == bias.SeverityHigh {
			icon = "🔴"
		} else if m.Severity == bias.SeverityMedium {
			icon = "🟡"
		}
		fmt.Printf("%-20s %-12s %-8.3f  %s %s\n",
			truncate(m.Word, 20), m.Category, m.BiasScore, icon, m.Severity)
	}
	fmt.Println(thin)
	fmt.Printf("Overall: %.3f\n", resp.OverallBiasScore)
	fmt.Println(resp.Interpretation)
}

func printQualityReport(sigma float64, resp models.QualityResponse) {
	thin := strings.Repeat("─", 48)
	statusIcon := "✅"
	if resp.Status == "warning" {
		statusIcon = "⚠️"
	} else if resp.Status == "critical" {
		statusIcon = "❌"
	}
	fmt.Println("Quality Impact Test:")
	fmt.Println(thin)
	fmt.Printf("Sigma:                   %.4f\n", sigma)
	fmt.Printf("Recall without noise:    %.3f\n", resp.RecallWithout)
	fmt.Printf("Recall with noise:       %.3f\n", resp.RecallWith)
	fmt.Printf("Noise impact:            %.1f%% degradation\n", resp.RecallDropPercent)
	fmt.Printf("Avg similarity:          %.4f\n", resp.AvgSimilarityToOriginal)
	fmt.Printf("Status: %s %s\n", statusIcon, resp.Status)
	fmt.Println(thin)
}

func formatVec(v []float32) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = fmt.Sprintf("%.4f", x)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func qualIcon(sim float64) string {
	if sim >= 0.95 {
		return "✅ high preservation"
	}
	if sim >= 0.85 {
		return "⚠️  moderate"
	}
	return "❌ low"
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ─── parsing helpers ──────────────────────────────────────────────────────────

func splitTrim(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}

func parseFloats(s string) []float64 {
	var out []float64
	for _, p := range splitTrim(s) {
		f, err := strconv.ParseFloat(p, 64)
		if err == nil {
			out = append(out, f)
		}
	}
	return out
}

// parsePairs parses "gender:male,female" into AttributePairs.
func parsePairs(s string) []models.AttributePair {
	if s == "" {
		return nil
	}
	var pairs []models.AttributePair
	for _, segment := range splitTrim(s) {
		colonIdx := strings.Index(segment, ":")
		if colonIdx < 0 {
			continue
		}
		cat := segment[:colonIdx]
		rest := strings.SplitN(segment[colonIdx+1:], ",", 2)
		if len(rest) < 2 {
			continue
		}
		pairs = append(pairs, models.AttributePair{
			AttributeA: strings.TrimSpace(rest[0]),
			AttributeB: strings.TrimSpace(rest[1]),
			Category:   cat,
		})
	}
	return pairs
}

func loadEmbeddingPairs(path string) ([]models.EmbeddingPair, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var pairs []models.EmbeddingPair
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var pair models.EmbeddingPair
		if err := json.Unmarshal([]byte(line), &pair); err != nil {
			return nil, fmt.Errorf("parse pair: %w", err)
		}
		pairs = append(pairs, pair)
	}
	return pairs, nil
}

// sampleEmbedding returns a random unit vector of length dims.
func sampleEmbedding(dims int) []float32 {
	inj := noise.NewDeterministic(0, false, 99)
	// Inject zero noise to get a normalised random vector.
	zeroNoise := noise.NewDeterministic(0.0001, true, 12345)
	base := make([]float32, dims)
	result := zeroNoise.Inject(base)
	_ = inj
	return result.Secured
}

// Ensure context is used to satisfy the build.
var _ = context.Background
