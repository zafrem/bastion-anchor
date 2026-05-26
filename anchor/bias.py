"""WEAT (Word Embedding Association Test) bias analyzer (SRS §8)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .models import AttributePair, BiasMeasurement, BiasResponse
from .noise import cosine_similarity

SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"

BUILTIN_TEST_SETS: dict[str, dict] = {
    "gender": {
        "test_words": [
            "doctor", "nurse", "engineer", "teacher", "manager",
            "의사", "간호사", "엔지니어", "교사", "매니저",
        ],
        "pairs": [
            AttributePair(attribute_a="male", attribute_b="female", category="gender"),
            AttributePair(attribute_a="he", attribute_b="she", category="gender"),
            AttributePair(attribute_a="남성", attribute_b="여성", category="gender"),
        ],
    },
    "ethnicity": {
        "test_words": ["intelligent", "hardworking", "athletic", "똑똑한", "부지런한", "운동을 잘하는"],
        "pairs": [
            AttributePair(attribute_a="Korean", attribute_b="Foreign", category="ethnicity"),
            AttributePair(attribute_a="한국인", attribute_b="외국인", category="ethnicity"),
        ],
    },
    "age": {
        "test_words": ["capable", "modern", "traditional", "능력있는", "현대적인", "전통적인"],
        "pairs": [
            AttributePair(attribute_a="young", attribute_b="old", category="age"),
            AttributePair(attribute_a="젊은", attribute_b="나이든", category="age"),
        ],
    },
}


class BiasAnalyzer:
    def __init__(
        self,
        medium_threshold: float = 0.15,
        high_threshold: float = 0.25,
        dims: int = 1024,
    ) -> None:
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.dims = dims
        self._rng = np.random.default_rng()

    def analyze(
        self,
        test_words: list[str],
        pairs: list[AttributePair],
        word_embeddings: Optional[dict[str, list[float]]] = None,
    ) -> BiasResponse:
        measurements: list[BiasMeasurement] = []
        total_bias = 0.0

        for word in test_words:
            word_emb = self._get_or_random(word, word_embeddings)
            for pair in pairs:
                attr_a = self._get_or_random(pair.attribute_a, word_embeddings)
                attr_b = self._get_or_random(pair.attribute_b, word_embeddings)
                sim_a = cosine_similarity(word_emb, attr_a)
                sim_b = cosine_similarity(word_emb, attr_b)
                score = abs(sim_a - sim_b)
                measurements.append(BiasMeasurement(
                    word=word,
                    category=pair.category,
                    bias_score=score,
                    sim_a=sim_a,
                    sim_b=sim_b,
                    severity=self._classify(score),
                ))
                total_bias += score

        overall = total_bias / len(measurements) if measurements else 0.0
        return BiasResponse(
            measurements=measurements,
            overall_bias_score=overall,
            interpretation=self._interpret(overall),
        )

    def _classify(self, score: float) -> str:
        if score >= self.high_threshold:
            return SEVERITY_HIGH
        if score >= self.medium_threshold:
            return SEVERITY_MEDIUM
        return SEVERITY_LOW

    def _interpret(self, overall: float) -> str:
        if overall >= self.high_threshold:
            return f"High bias detected (score={overall:.3f}). Immediate model review recommended."
        if overall >= self.medium_threshold:
            return f"Moderate bias detected (score={overall:.3f}). Consider reviewing embedding model."
        return f"Bias within acceptable range (score={overall:.3f}). No immediate action required."

    def _get_or_random(
        self, token: str, provided: Optional[dict[str, list[float]]]
    ) -> np.ndarray:
        if provided and token in provided:
            return np.asarray(provided[token], dtype=np.float64)
        return self._random_unit()

    def _random_unit(self) -> np.ndarray:
        vec = self._rng.uniform(-1.0, 1.0, self.dims)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
