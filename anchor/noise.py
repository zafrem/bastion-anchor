"""Gaussian / Laplacian noise injection (SRS §7)."""
from __future__ import annotations

import secrets
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

STRATEGY_GAUSSIAN = "gaussian"
STRATEGY_LAPLACIAN = "laplacian"


@dataclass
class NoiseResult:
    secured: list[float]
    noise_magnitude: float
    similarity_to_original: float
    original_norm: float
    secured_norm: float
    strategy: str


def _crypto_seed() -> int:
    raw = secrets.token_bytes(8)
    seed = struct.unpack("<Q", raw)[0]
    return int(seed % (2**31))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or len(a) == 0:
        return 0.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class NoiseInjector:
    def __init__(
        self,
        sigma: float,
        preserve_norm: bool,
        strategy: str = STRATEGY_GAUSSIAN,
        seed: Optional[int] = None,
    ) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = sigma
        self.preserve_norm = preserve_norm
        self.strategy = strategy or STRATEGY_GAUSSIAN
        rng_seed = seed if seed is not None else _crypto_seed()
        self.rng = np.random.default_rng(rng_seed)

    def inject(self, embedding: list[float]) -> NoiseResult:
        orig = np.asarray(embedding, dtype=np.float64)
        original_norm = float(np.linalg.norm(orig))

        if self.strategy == STRATEGY_LAPLACIAN:
            b = self.sigma / np.sqrt(2)
            noise = self.rng.laplace(0.0, b, size=len(orig))
        else:
            # Box-Muller is done internally by numpy's normal()
            noise = self.rng.normal(0.0, self.sigma, size=len(orig))

        secured = orig + noise
        noise_magnitude = float(np.mean(np.abs(noise)))

        if self.preserve_norm:
            norm = np.linalg.norm(secured)
            if norm > 0:
                secured = secured / norm

        secured_norm = float(np.linalg.norm(secured))
        similarity = cosine_similarity(orig, secured)

        return NoiseResult(
            secured=secured.tolist(),
            noise_magnitude=noise_magnitude,
            similarity_to_original=similarity,
            original_norm=original_norm,
            secured_norm=secured_norm,
            strategy=self.strategy + "_noise",
        )

    def inject_batch(self, embeddings: list[list[float]]) -> list[NoiseResult]:
        return [self.inject(emb) for emb in embeddings]

    def set_sigma(self, sigma: float) -> None:
        self.sigma = sigma


def estimated_recall_preservation(sigma: float) -> float:
    """Approximate Recall@K retention for a given sigma (linear model)."""
    return max(0.0, min(1.0, 1.0 - 4.0 * sigma))
