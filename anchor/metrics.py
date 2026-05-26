"""Prometheus metrics for Anchor."""
from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge

requests_total = Counter(
    "anchor_requests_total", "Total requests by operation", ["operation"]
)
request_duration_seconds = Histogram(
    "anchor_request_duration_seconds", "Request duration", ["operation"]
)
noise_magnitude = Histogram(
    "anchor_noise_magnitude", "Noise magnitude added to embeddings",
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
)
similarity_to_original = Histogram(
    "anchor_similarity_to_original", "Cosine similarity between original and secured embedding",
    buckets=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
bias_score = Histogram(
    "anchor_bias_score", "Bias score from WEAT analysis",
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
)
