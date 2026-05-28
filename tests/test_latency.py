"""NFR latency tests for Anchor — all operations measured against SRS targets.

NFR-PE-001: Noise injection p95 < 5 ms
NFR-PE-002: Response verify p95 < 50 ms
NFR-PE-003: Bias analysis p95 < 10 ms

Skipped when pytest is run with --no-nfr or in environments without numpy.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np
import pytest

from anchor.bias import BiasAnalyzer
from anchor.models import (
    AttributePair,
    VerifyRequest,
)
from anchor.noise import NoiseInjector
from anchor.verifier import Verifier

ITERATIONS = 200


def _percentile(durations: list[float], n: int) -> float:
    """Return the n-th percentile of a list of durations (ms)."""
    sorted_d = sorted(durations)
    idx = (n * len(sorted_d)) // 100
    if idx >= len(sorted_d):
        idx = len(sorted_d) - 1
    return sorted_d[idx]


def _unit_vec(dim: int = 128, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, dim)
    return (v / np.linalg.norm(v)).tolist()


def _measure(fn: Callable[[], None], n: int = ITERATIONS) -> list[float]:
    """Run fn n times and return elapsed times in milliseconds."""
    results = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        results.append((time.perf_counter() - t0) * 1000.0)
    return results


# ── NFR-PE-001: Noise injection ────────────────────────────────────────────────

class TestNFR_NoiseInjection_P95_Under5ms:
    def test_gaussian(self):
        inj = NoiseInjector(sigma=0.1, preserve_norm=True, strategy="gaussian")
        vec = _unit_vec(384)
        durations = _measure(lambda: inj.inject(vec))
        p50 = _percentile(durations, 50)
        p95 = _percentile(durations, 95)
        print(f"\nNoise injection (gaussian 384-d): p50={p50:.3f}ms p95={p95:.3f}ms (target <5ms)")
        assert p95 < 5.0, f"NFR-PE-001 FAIL: noise injection p95={p95:.3f}ms exceeds 5ms"

    def test_laplacian(self):
        inj = NoiseInjector(sigma=0.1, preserve_norm=True, strategy="laplacian")
        vec = _unit_vec(384)
        durations = _measure(lambda: inj.inject(vec))
        p95 = _percentile(durations, 95)
        print(f"\nNoise injection (laplacian 384-d): p95={p95:.3f}ms (target <5ms)")
        assert p95 < 5.0, f"NFR-PE-001 FAIL: noise injection (laplacian) p95={p95:.3f}ms exceeds 5ms"


# ── NFR-PE-002: Response verification ─────────────────────────────────────────

class TestNFR_Verify_P95_Under50ms:
    def _make_req(self) -> VerifyRequest:
        return VerifyRequest(
            llm_response=(
                "The capital of South Korea is Seoul. It is the largest city "
                "in the country and home to over 10 million people."
            ),
            original_query="What is the capital of South Korea?",
            source_documents=[
                "Seoul is the capital and largest metropolis of South Korea.",
                "The city has a population of approximately 10 million.",
            ],
        )

    def test_verify_p95(self):
        ver = Verifier(medium_threshold=0.15, high_threshold=0.25)
        req = self._make_req()
        durations = _measure(lambda: ver.verify(req))
        p50 = _percentile(durations, 50)
        p95 = _percentile(durations, 95)
        print(f"\nVerify: p50={p50:.3f}ms p95={p95:.3f}ms (target <50ms)")
        assert p95 < 50.0, f"NFR-PE-002 FAIL: verify p95={p95:.3f}ms exceeds 50ms"


# ── NFR-PE-003: Bias analysis ──────────────────────────────────────────────────

class TestNFR_Bias_P95_Under10ms:
    def _make_inputs(self):
        words = ["doctor", "nurse", "engineer", "teacher"]
        pairs = [
            AttributePair(attribute_a="male", attribute_b="female", category="gender"),
        ]
        rng = np.random.default_rng(42)
        embeddings = {w: rng.normal(0, 1, 64).tolist() for w in words + ["male", "female"]}
        return words, pairs, embeddings

    def test_bias_p95(self):
        ana = BiasAnalyzer(medium_threshold=0.15, high_threshold=0.25)
        words, pairs, embeddings = self._make_inputs()
        durations = _measure(lambda: ana.analyze(words, pairs, embeddings))
        p50 = _percentile(durations, 50)
        p95 = _percentile(durations, 95)
        print(f"\nBias analysis: p50={p50:.3f}ms p95={p95:.3f}ms (target <10ms)")
        assert p95 < 10.0, f"NFR-PE-003 FAIL: bias analysis p95={p95:.3f}ms exceeds 10ms"
