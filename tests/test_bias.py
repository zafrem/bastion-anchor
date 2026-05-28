"""Unit tests for anchor.bias — BiasAnalyzer and WEAT measurements."""
from __future__ import annotations

import numpy as np
import pytest

from anchor.bias import BiasAnalyzer, SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, BUILTIN_TEST_SETS
from anchor.models import AttributePair


def _make_analyzer(**kw) -> BiasAnalyzer:
    return BiasAnalyzer(medium_threshold=0.15, high_threshold=0.25, dims=8, **kw)


def _vec(dim: int = 8, val: float = 1.0) -> list[float]:
    """Unit vector with first component set to val."""
    v = np.zeros(dim)
    v[0] = val
    n = np.linalg.norm(v)
    return (v / n).tolist()


# ─── basic analysis ───────────────────────────────────────────────────────────

class TestBiasAnalyzerBasic:
    def test_returns_measurement_per_word_per_pair(self):
        ana = _make_analyzer()
        words = ["doctor", "nurse"]
        pairs = [AttributePair(attribute_a="male", attribute_b="female", category="gender")]
        resp = ana.analyze(words, pairs)
        assert len(resp.measurements) == 2  # 2 words × 1 pair

    def test_multiple_pairs(self):
        ana = _make_analyzer()
        words = ["doctor"]
        pairs = [
            AttributePair(attribute_a="male", attribute_b="female", category="gender"),
            AttributePair(attribute_a="young", attribute_b="old", category="age"),
        ]
        resp = ana.analyze(words, pairs)
        assert len(resp.measurements) == 2

    def test_overall_score_in_range(self):
        ana = _make_analyzer()
        resp = ana.analyze(["doctor"], [AttributePair(attribute_a="a", attribute_b="b", category="test")])
        assert 0.0 <= resp.overall_bias_score <= 1.0

    def test_interpretation_is_string(self):
        ana = _make_analyzer()
        resp = ana.analyze(["word"], [AttributePair(attribute_a="a", attribute_b="b", category="x")])
        assert isinstance(resp.interpretation, str)
        assert len(resp.interpretation) > 0

    def test_empty_words_list_returns_zero_score(self):
        ana = _make_analyzer()
        resp = ana.analyze([], [AttributePair(attribute_a="a", attribute_b="b", category="x")])
        assert resp.overall_bias_score == 0.0
        assert len(resp.measurements) == 0


class TestBiasAnalyzerSeverity:
    def test_high_bias_classified_correctly_with_embeddings(self):
        """Provide embeddings where word is very close to attr_a, far from attr_b."""
        ana = _make_analyzer()
        # attr_a points to +x, attr_b points to -x, word is on +x → max bias
        embs = {
            "word": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "yes":  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "no":   [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        resp = ana.analyze(
            ["word"],
            [AttributePair(attribute_a="yes", attribute_b="no", category="test")],
            word_embeddings=embs,
        )
        assert len(resp.measurements) == 1
        # sim_a ≈ 1.0, sim_b ≈ -1.0 → score ≈ 2.0, clamped to high
        assert resp.measurements[0].severity == SEVERITY_HIGH

    def test_zero_bias_gives_low_severity(self):
        """Provide identical attribute embeddings → sim_a == sim_b → score 0."""
        ana = _make_analyzer()
        embs = {
            "word": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "attr": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        resp = ana.analyze(
            ["word"],
            [AttributePair(attribute_a="attr", attribute_b="attr", category="test")],
            word_embeddings=embs,
        )
        assert resp.measurements[0].severity == SEVERITY_LOW
        assert resp.measurements[0].bias_score == pytest.approx(0.0, abs=1e-9)

    def test_low_medium_high_thresholds(self):
        ana = BiasAnalyzer(medium_threshold=0.3, high_threshold=0.6, dims=8)
        assert ana._classify(0.1) == SEVERITY_LOW
        assert ana._classify(0.4) == SEVERITY_MEDIUM
        assert ana._classify(0.7) == SEVERITY_HIGH


class TestBiasAnalyzerWithProvidedEmbeddings:
    def test_uses_provided_embeddings_instead_of_random(self):
        """Two runs with same provided embeddings must produce identical results."""
        ana = BiasAnalyzer(dims=8)
        embs = {
            "doctor": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "male":   [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "female": [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        pairs = [AttributePair(attribute_a="male", attribute_b="female", category="gender")]
        r1 = ana.analyze(["doctor"], pairs, word_embeddings=embs)
        r2 = ana.analyze(["doctor"], pairs, word_embeddings=embs)
        assert r1.overall_bias_score == r2.overall_bias_score

    def test_measurement_fields_populated(self):
        ana = BiasAnalyzer(dims=4)
        embs = {"w": [1.0, 0.0, 0.0, 0.0], "a": [0.7, 0.7, 0.0, 0.0], "b": [0.0, 0.0, 1.0, 0.0]}
        resp = ana.analyze(
            ["w"],
            [AttributePair(attribute_a="a", attribute_b="b", category="x")],
            word_embeddings=embs,
        )
        m = resp.measurements[0]
        assert m.word == "w"
        assert m.category == "x"
        assert 0.0 <= m.sim_a <= 1.0
        assert 0.0 <= m.sim_b <= 1.0
        assert m.bias_score >= 0.0


class TestBuiltinTestSets:
    def test_builtin_sets_have_required_keys(self):
        for name, entry in BUILTIN_TEST_SETS.items():
            assert "test_words" in entry, f"{name} missing test_words"
            assert "pairs" in entry, f"{name} missing pairs"

    def test_gender_set_non_empty(self):
        gs = BUILTIN_TEST_SETS["gender"]
        assert len(gs["test_words"]) > 0
        assert len(gs["pairs"]) > 0

    def test_builtin_sets_can_be_analyzed(self):
        ana = BiasAnalyzer(dims=8)
        for name, entry in BUILTIN_TEST_SETS.items():
            resp = ana.analyze(entry["test_words"], entry["pairs"])
            assert resp.overall_bias_score >= 0.0, f"{name} gave negative bias"
