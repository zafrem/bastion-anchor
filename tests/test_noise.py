"""Unit tests for anchor.noise — NoiseInjector."""
from __future__ import annotations

import math

import numpy as np
import pytest

from anchor.noise import (
    NoiseInjector,
    STRATEGY_GAUSSIAN,
    STRATEGY_LAPLACIAN,
    cosine_similarity,
    estimated_recall_preservation,
)


def _unit_vec(dim: int = 16, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, dim)
    return (v / np.linalg.norm(v)).tolist()


# ─── NoiseInjector basics ─────────────────────────────────────────────────────

class TestNoiseInjectorConstruction:
    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            NoiseInjector(-0.1, preserve_norm=True)

    def test_zero_sigma_is_valid(self):
        inj = NoiseInjector(0.0, preserve_norm=False, seed=1)
        result = inj.inject(_unit_vec())
        # Zero sigma → noise_magnitude is approximately 0
        assert result.noise_magnitude < 1e-9

    def test_explicit_seed_is_deterministic(self):
        vec = _unit_vec()
        r1 = NoiseInjector(0.05, preserve_norm=False, seed=42).inject(vec)
        r2 = NoiseInjector(0.05, preserve_norm=False, seed=42).inject(vec)
        assert r1.secured == r2.secured

    def test_different_seeds_differ(self):
        vec = _unit_vec()
        r1 = NoiseInjector(0.05, preserve_norm=False, seed=1).inject(vec)
        r2 = NoiseInjector(0.05, preserve_norm=False, seed=2).inject(vec)
        assert r1.secured != r2.secured


class TestNoiseInjectorOutput:
    def test_output_length_matches_input(self):
        vec = _unit_vec(32)
        result = NoiseInjector(0.01, preserve_norm=False, seed=7).inject(vec)
        assert len(result.secured) == 32

    def test_strategy_field_set_correctly(self):
        r = NoiseInjector(0.01, preserve_norm=False, strategy=STRATEGY_GAUSSIAN, seed=0).inject(_unit_vec())
        assert "gaussian" in r.strategy

        r2 = NoiseInjector(0.01, preserve_norm=False, strategy=STRATEGY_LAPLACIAN, seed=0).inject(_unit_vec())
        assert "laplacian" in r2.strategy

    def test_similarity_below_one_with_nonzero_sigma(self):
        vec = _unit_vec(64)
        r = NoiseInjector(0.1, preserve_norm=False, seed=5).inject(vec)
        assert r.similarity_to_original < 1.0

    def test_similarity_very_high_with_tiny_sigma(self):
        vec = _unit_vec(64)
        r = NoiseInjector(1e-6, preserve_norm=False, seed=5).inject(vec)
        assert r.similarity_to_original > 0.999

    def test_noise_magnitude_positive(self):
        r = NoiseInjector(0.05, preserve_norm=False, seed=3).inject(_unit_vec())
        assert r.noise_magnitude > 0.0

    def test_original_norm_recorded(self):
        vec = _unit_vec(16)
        r = NoiseInjector(0.02, preserve_norm=False, seed=0).inject(vec)
        expected_norm = float(np.linalg.norm(vec))
        assert abs(r.original_norm - expected_norm) < 1e-9


class TestPreserveNorm:
    def test_preserve_norm_keeps_unit_length(self):
        vec = _unit_vec(32)
        r = NoiseInjector(0.1, preserve_norm=True, seed=1).inject(vec)
        assert abs(float(np.linalg.norm(r.secured)) - 1.0) < 1e-6

    def test_no_preserve_norm_changes_length(self):
        vec = _unit_vec(32)
        r = NoiseInjector(0.5, preserve_norm=False, seed=1).inject(vec)
        # With large sigma the norm will almost certainly drift
        assert abs(float(np.linalg.norm(r.secured)) - 1.0) > 1e-4


class TestBatchInjection:
    def test_batch_length_matches(self):
        vecs = [_unit_vec(16, seed=i) for i in range(5)]
        inj = NoiseInjector(0.02, preserve_norm=False, seed=9)
        results = inj.inject_batch(vecs)
        assert len(results) == 5

    def test_each_batch_result_has_correct_dim(self):
        vecs = [_unit_vec(8) for _ in range(3)]
        results = NoiseInjector(0.02, preserve_norm=False, seed=0).inject_batch(vecs)
        for r in results:
            assert len(r.secured) == 8


class TestSetSigma:
    def test_set_sigma_updates_injector(self):
        inj = NoiseInjector(0.0, preserve_norm=False, seed=10)
        inj.set_sigma(0.5)
        r = inj.inject(_unit_vec())
        assert r.noise_magnitude > 0.0


# ─── Laplacian vs Gaussian ────────────────────────────────────────────────────

class TestStrategies:
    def test_laplacian_produces_results(self):
        vec = _unit_vec(32)
        r = NoiseInjector(0.05, preserve_norm=False, strategy=STRATEGY_LAPLACIAN, seed=42).inject(vec)
        assert len(r.secured) == 32
        assert r.noise_magnitude > 0.0

    def test_gaussian_and_laplacian_differ(self):
        vec = _unit_vec(32)
        rg = NoiseInjector(0.05, preserve_norm=False, strategy=STRATEGY_GAUSSIAN, seed=1).inject(vec)
        rl = NoiseInjector(0.05, preserve_norm=False, strategy=STRATEGY_LAPLACIAN, seed=1).inject(vec)
        # Same seed, different distribution — outputs should differ
        assert rg.secured != rl.secured


# ─── cosine_similarity helper ─────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0])
        assert abs(cosine_similarity(v, -v) - (-1.0)) < 1e-9

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-9

    def test_zero_vector_returns_zero(self):
        v = np.array([1.0, 0.0])
        z = np.array([0.0, 0.0])
        assert cosine_similarity(v, z) == 0.0

    def test_shape_mismatch_returns_zero(self):
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0])) == 0.0

    def test_empty_returns_zero(self):
        assert cosine_similarity(np.array([]), np.array([])) == 0.0


# ─── estimated_recall_preservation ───────────────────────────────────────────

class TestEstimatedRecall:
    def test_zero_sigma_full_recall(self):
        assert estimated_recall_preservation(0.0) == 1.0

    def test_high_sigma_clamps_to_zero(self):
        assert estimated_recall_preservation(1.0) == 0.0

    def test_output_in_range(self):
        for s in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            r = estimated_recall_preservation(s)
            assert 0.0 <= r <= 1.0

    def test_monotone_decreasing(self):
        prev = 1.0
        for s in [0.01, 0.05, 0.1, 0.2]:
            r = estimated_recall_preservation(s)
            assert r <= prev
            prev = r
