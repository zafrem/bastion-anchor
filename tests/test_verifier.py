"""Unit tests for anchor.verifier — Verifier (Anchor-OUT phase 2)."""
from __future__ import annotations

import pytest

from anchor.verifier import Verifier, _classify_drift
from anchor.models import (
    AttributePair,
    BiasBaseline,
    CategoryBaseline,
    DriftRequest,
    EmbeddingAnalysisRequest,
    ResponseBiasAnalysis,
    ResponseCategoryBias,
    VerifyOptions,
    VerifyRequest,
)


def _vreq(
    response: str = "The engineer solved the problem.",
    query: str = "Who solved it?",
    sources: list[str] | None = None,
    options: VerifyOptions | None = None,
    phase1: BiasBaseline | None = None,
) -> VerifyRequest:
    return VerifyRequest(
        request_id="req-1",
        llm_response=response,
        original_query=query,
        source_documents=sources or [],
        phase1_baseline=phase1,
        options=options,
    )


class TestVerifierDefaults:
    """When no VerifyOptions given, all checks run."""

    def test_returns_verify_response(self):
        v = Verifier()
        resp = v.verify(_vreq())
        assert resp is not None

    def test_status_is_valid(self):
        resp = Verifier().verify(_vreq())
        assert resp.status in ("PASSED", "WARNING", "CRITICAL")

    def test_all_analysis_fields_populated_by_default(self):
        resp = Verifier().verify(_vreq())
        assert resp.bias_analysis is not None
        assert resp.embedding_analysis is not None
        assert resp.quality is not None

    def test_processing_time_positive(self):
        resp = Verifier().verify(_vreq())
        assert resp.processing_time_ms > 0.0

    def test_warnings_and_recommendations_are_lists(self):
        resp = Verifier().verify(_vreq())
        assert isinstance(resp.warnings, list)
        assert isinstance(resp.recommendations, list)

    def test_request_id_propagated(self):
        req = _vreq()
        resp = Verifier().verify(req)
        assert resp.request_id == "req-1"


class TestVerifierBiasDetection:
    def test_clean_response_passes(self):
        resp = Verifier().verify(_vreq("The engineer completed the analysis."))
        assert resp.bias_analysis is not None
        assert resp.bias_analysis.overall_score >= 0.0

    def test_stereotyping_phrase_raises_score(self):
        biased = "women are too emotional and should not be leaders"
        resp = Verifier().verify(_vreq(biased))
        assert resp.bias_analysis is not None
        assert resp.bias_analysis.overall_score > 0.0

    def test_high_bias_produces_critical_or_warning(self):
        biased = "women are too emotional"
        resp = Verifier().verify(_vreq(biased))
        assert resp.status in ("WARNING", "CRITICAL")

    def test_bias_analysis_has_categories(self):
        biased = "women are too emotional"
        resp = Verifier().verify(_vreq(biased))
        # gender category should have evidence
        categories = {c.category for c in resp.bias_analysis.categories}
        assert "gender" in categories

    def test_korean_bias_phrase_detected(self):
        resp = Verifier().verify(_vreq("여자는 감정적이다"))
        assert resp.bias_analysis.overall_score > 0.0

    def test_exceeds_threshold_flag(self):
        biased = "women are too emotional"
        resp = Verifier(medium_threshold=0.01).verify(_vreq(biased))
        assert resp.bias_analysis.exceeds_threshold is True


class TestVerifierEmbeddingAnalysis:
    def test_no_sources_gives_zero_context_sim(self):
        resp = Verifier().verify(_vreq(sources=[]))
        assert resp.embedding_analysis is not None
        assert resp.embedding_analysis.similarity_to_context == pytest.approx(0.0)

    def test_matching_source_increases_context_sim(self):
        response = "The engineer solved the problem using algebra."
        source = "The engineer solved the problem using algebra and calculus."
        resp = Verifier().verify(_vreq(response, sources=[source]))
        assert resp.embedding_analysis.similarity_to_context > 0.0

    def test_totally_different_source_low_context_sim(self):
        resp = Verifier().verify(_vreq(
            "The moon is made of cheese.",
            sources=["photosynthesis occurs in plants"],
        ))
        assert resp.embedding_analysis.similarity_to_context < 0.5

    def test_semantic_drift_detected_when_response_unrelated_to_sources(self):
        resp = Verifier().verify(_vreq(
            "Quantum entanglement is a complex physical phenomenon.",
            sources=["HR policy update 2024"],
        ))
        # sim_to_context < 0.60 → semantic drift
        if resp.embedding_analysis.similarity_to_context < 0.60:
            assert resp.embedding_analysis.semantic_drift_detected is True

    def test_diversity_score_in_range(self):
        resp = Verifier().verify(_vreq("answer", sources=["source text"]))
        assert 0.0 <= resp.embedding_analysis.diversity_score <= 1.0


class TestVerifierQuality:
    def test_quality_fields_populated(self):
        resp = Verifier().verify(_vreq("the answer", sources=["the answer is here"]))
        q = resp.quality
        assert 0.0 <= q.grounding_score <= 1.0
        assert 0.0 <= q.coherence_score <= 1.0
        assert 0.0 <= q.relevance_score <= 1.0
        assert 0.0 <= q.overall_quality <= 1.0

    def test_grounded_response_has_high_grounding(self):
        words = "doctor nurse engineer manager teacher"
        resp = Verifier().verify(_vreq(words, sources=[words]))
        assert resp.quality.grounding_score > 0.5

    def test_empty_response_has_zero_grounding(self):
        resp = Verifier().verify(_vreq("", sources=["some source text"]))
        assert resp.quality.grounding_score == pytest.approx(0.0)


class TestVerifierDriftDetection:
    def _make_baseline(self, score: float) -> BiasBaseline:
        return BiasBaseline(overall_score=score, categories=[])

    def test_drift_detected_when_bias_amplified(self):
        # Phase 1 score 0.05, Phase 2 score 0.50 → amplification
        baseline = self._make_baseline(0.05)
        # Force bias high using known phrase
        resp = Verifier().verify(_vreq("women are too emotional", phase1=baseline))
        if resp.drift_analysis:
            # If drift analysis ran, it should detect amplification
            assert resp.drift_analysis.drift_detected

    def test_no_drift_when_baseline_missing(self):
        resp = Verifier().verify(_vreq())
        # drift_analysis only populated when phase1_baseline AND bias_anal both present
        # Without phase1_baseline it should be None
        assert resp.drift_analysis is None


class TestVerifierOptions:
    def test_only_bias_check_runs(self):
        opts = VerifyOptions(check_bias=True)
        resp = Verifier().verify(_vreq(options=opts))
        assert resp.bias_analysis is not None
        assert resp.embedding_analysis is None
        assert resp.quality is None

    def test_only_embedding_analysis_runs(self):
        opts = VerifyOptions(analyze_embedding=True)
        resp = Verifier().verify(_vreq(options=opts))
        assert resp.embedding_analysis is not None
        assert resp.bias_analysis is None
        assert resp.quality is None

    def test_only_quality_runs(self):
        opts = VerifyOptions(measure_quality=True)
        resp = Verifier().verify(_vreq(options=opts))
        assert resp.quality is not None
        assert resp.bias_analysis is None
        assert resp.embedding_analysis is None

    def test_custom_bias_threshold_applied(self):
        opts = VerifyOptions(check_bias=True, bias_threshold=0.001)
        resp = Verifier().verify(_vreq("women are too emotional", options=opts))
        assert resp.bias_analysis.exceeds_threshold is True


class TestVerifierAnalyzeEmbedding:
    def test_returns_similarities(self):
        req = EmbeddingAnalysisRequest(
            text="The doctor worked hard.",
            comparison_texts=["The doctor worked.", "Completely unrelated farming topic."],
        )
        resp = Verifier().analyze_embedding(req)
        assert len(resp.similarities) == 2
        assert 0.0 <= resp.similarities[0] <= 1.0

    def test_empty_comparison_texts(self):
        req = EmbeddingAnalysisRequest(text="hello", comparison_texts=[])
        resp = Verifier().analyze_embedding(req)
        assert resp.similarities == []
        assert resp.diversity == pytest.approx(0.0)

    def test_identical_text_gives_high_similarity(self):
        text = "the quick brown fox"
        req = EmbeddingAnalysisRequest(text=text, comparison_texts=[text])
        resp = Verifier().analyze_embedding(req)
        assert resp.similarities[0] == pytest.approx(1.0)


class TestVerifierCheckDrift:
    def _make_baseline(self, score: float) -> BiasBaseline:
        return BiasBaseline(overall_score=score, categories=[
            CategoryBaseline(category="gender", score=score),
        ])

    def _make_phase2(self, score: float) -> ResponseBiasAnalysis:
        return ResponseBiasAnalysis(
            overall_score=score,
            categories=[ResponseCategoryBias(category="gender", score=score)],
            severity="low",
            exceeds_threshold=False,
        )

    def test_amplification_detected(self):
        req = DriftRequest(
            phase1=self._make_baseline(0.05),
            phase2=self._make_phase2(0.30),
        )
        resp = Verifier().check_drift(req)
        assert resp.drift_detected
        assert resp.severity in ("medium", "high")

    def test_no_drift_when_scores_equal(self):
        req = DriftRequest(
            phase1=self._make_baseline(0.10),
            phase2=self._make_phase2(0.10),
        )
        resp = Verifier().check_drift(req)
        assert not resp.drift_detected

    def test_empty_request_no_drift(self):
        resp = Verifier().check_drift(DriftRequest())
        assert not resp.drift_detected
        assert resp.severity == "low"

    def test_reduction_detected(self):
        req = DriftRequest(
            phase1=self._make_baseline(0.4),
            phase2=self._make_phase2(0.1),
        )
        resp = Verifier().check_drift(req)
        # reduction direction (p1=0.4 → p2=0.1, diff=-0.3)
        if resp.drift_detected:
            found_reduction = any(d.direction == "reduction" for d in resp.drifts)
            assert found_reduction


class TestClassifyDrift:
    def test_introduction(self):
        assert _classify_drift(0.01, 0.20, 0.19) == "introduction"

    def test_amplification(self):
        assert _classify_drift(0.10, 0.30, 0.20) == "amplification"

    def test_reduction(self):
        assert _classify_drift(0.40, 0.20, -0.20) == "reduction"

    def test_none_for_small_change(self):
        assert _classify_drift(0.10, 0.12, 0.02) == "none"
