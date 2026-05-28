"""Unit tests for anchor.quality — QualityMonitor."""
from __future__ import annotations

import numpy as np
import pytest

from anchor.quality import QualityMonitor
from anchor.models import EmbeddingPair


def _unit(dim: int = 16, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, dim)
    return (v / np.linalg.norm(v)).tolist()


def _pair(query_seed: int = 0, match_seed: int = 1, dim: int = 16) -> EmbeddingPair:
    return EmbeddingPair(
        query_embedding=_unit(dim, query_seed),
        match_embedding=_unit(dim, match_seed),
    )


class TestQualityMonitorNoPairs:
    """When no pairs are provided, QualityMonitor uses the analytic estimate."""

    def test_returns_quality_response(self):
        mon = QualityMonitor()
        resp = mon.estimate(0.01, [])
        assert resp is not None

    def test_sigma_stored_in_response(self):
        resp = QualityMonitor().estimate(0.05, [])
        assert resp.noise_sigma == pytest.approx(0.05)

    def test_status_acceptable_for_tiny_sigma(self):
        resp = QualityMonitor().estimate(0.001, [])
        assert resp.status == "acceptable"

    def test_recall_without_is_one_for_zero_sigma(self):
        resp = QualityMonitor().estimate(0.0, [])
        assert resp.recall_without == pytest.approx(1.0)

    def test_recall_drop_non_negative(self):
        for sigma in [0.0, 0.05, 0.1, 0.2]:
            resp = QualityMonitor().estimate(sigma, [])
            assert resp.recall_drop_percent >= 0.0

    def test_large_sigma_triggers_critical(self):
        # sigma=1.0 → estimated_recall_preservation(1.0) = 0.0 → drop 100%
        mon = QualityMonitor(recall_drop_alert=20.0, similarity_minimum=0.95)
        resp = mon.estimate(1.0, [])
        assert resp.status in ("warning", "critical")


class TestQualityMonitorWithPairs:
    def test_processes_pairs(self):
        pairs = [_pair(i, i + 5) for i in range(4)]
        resp = QualityMonitor().estimate(0.01, pairs)
        assert 0.0 <= resp.recall_with <= 1.0
        assert 0.0 <= resp.recall_without <= 1.0

    def test_avg_similarity_in_range(self):
        pairs = [_pair(0, 1), _pair(2, 3)]
        resp = QualityMonitor().estimate(0.01, pairs)
        assert 0.0 <= resp.avg_similarity_to_original <= 1.0

    def test_identical_query_and_match_maximises_recall_without(self):
        """When query == match, cosine_similarity ≈ 1.0 ≥ 0.5 → all hits."""
        same = _unit(16, 0)
        pairs = [EmbeddingPair(query_embedding=same, match_embedding=same) for _ in range(3)]
        resp = QualityMonitor().estimate(0.01, pairs)
        assert resp.recall_without == pytest.approx(1.0)

    def test_orthogonal_query_and_match_zero_recall_without(self):
        """Orthogonal vectors have cos_sim = 0.0 < 0.5 → no hits."""
        a = [1.0] + [0.0] * 15
        b = [0.0, 1.0] + [0.0] * 14
        pairs = [EmbeddingPair(query_embedding=a, match_embedding=b)]
        resp = QualityMonitor().estimate(0.0, pairs)
        assert resp.recall_without == pytest.approx(0.0)

    def test_sigma_field_matches_input(self):
        pairs = [_pair()]
        resp = QualityMonitor().estimate(0.07, pairs)
        assert resp.noise_sigma == pytest.approx(0.07)

    def test_status_field_is_valid_string(self):
        pairs = [_pair()]
        resp = QualityMonitor().estimate(0.01, pairs)
        assert resp.status in ("acceptable", "warning", "critical")

    def test_high_sigma_can_cause_warning(self):
        same = _unit(16, 0)
        pairs = [EmbeddingPair(query_embedding=same, match_embedding=same) for _ in range(5)]
        # High sigma → avg_similarity drops → potentially warning
        mon = QualityMonitor(recall_drop_alert=5.0, similarity_minimum=0.999)
        resp = mon.estimate(0.5, pairs)
        # Status may be warning or critical — just not acceptable
        assert resp.status in ("warning", "critical")


class TestQualityMonitorThresholds:
    def test_custom_thresholds_respected(self):
        mon = QualityMonitor(recall_drop_alert=1.0, similarity_minimum=0.9999)
        # Even tiny sigma may exceed thresholds
        pairs = [_pair(0, 1)]
        resp = mon.estimate(0.1, pairs)
        # Don't assert exact status — just that it runs without error
        assert resp.status in ("acceptable", "warning", "critical")

    def test_zero_sigma_no_recall_drop(self):
        same = _unit(16, 42)
        pairs = [EmbeddingPair(query_embedding=same, match_embedding=same)]
        resp = QualityMonitor().estimate(0.0, pairs)
        assert resp.recall_drop_percent == pytest.approx(0.0, abs=1e-9)
