"""Quality monitor: recall & similarity impact of noise injection (SRS §9)."""
from __future__ import annotations

import numpy as np

from .models import EmbeddingPair, QualityResponse
from .noise import NoiseInjector, cosine_similarity, estimated_recall_preservation


class QualityMonitor:
    def __init__(self, recall_drop_alert: float = 20.0, similarity_minimum: float = 0.95) -> None:
        self.recall_drop_alert = recall_drop_alert
        self.similarity_minimum = similarity_minimum

    def estimate(self, sigma: float, pairs: list[EmbeddingPair]) -> QualityResponse:
        if not pairs:
            estimated = estimated_recall_preservation(sigma)
            return self._build(sigma, 1.0, estimated, 1.0 - estimated)

        inj = NoiseInjector(sigma, preserve_norm=True, seed=42)
        hits_without = hits_with = 0
        total_sim = 0.0

        for pair in pairs:
            q = np.asarray(pair.query_embedding, dtype=np.float64)
            m = np.asarray(pair.match_embedding, dtype=np.float64)
            if cosine_similarity(q, m) >= 0.5:
                hits_without += 1
            result = inj.inject(pair.query_embedding)
            secured = np.asarray(result.secured, dtype=np.float64)
            if cosine_similarity(secured, m) >= 0.5:
                hits_with += 1
            total_sim += result.similarity_to_original

        n = float(len(pairs))
        return self._build(sigma, hits_without / n, hits_with / n, total_sim / n)

    def _build(
        self,
        sigma: float,
        recall_without: float,
        recall_with: float,
        avg_sim: float,
    ) -> QualityResponse:
        drop_pct = 0.0
        if recall_without > 0:
            drop_pct = (recall_without - recall_with) / recall_without * 100

        status = "acceptable"
        if avg_sim < self.similarity_minimum or drop_pct > self.recall_drop_alert:
            status = "critical" if drop_pct > self.recall_drop_alert * 2 else "warning"

        return QualityResponse(
            noise_sigma=sigma,
            recall_without=recall_without,
            recall_with=recall_with,
            recall_drop_percent=drop_pct,
            avg_similarity_to_original=avg_sim,
            status=status,
        )
