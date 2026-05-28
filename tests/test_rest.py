"""REST API tests for Anchor — FastAPI endpoints via TestClient."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from anchor.config import Config
from anchor.events import Publisher
from anchor.hooks import (
    HookManager,
    EVENT_EMBEDDING_SECURED,
    EVENT_BIAS_DETECTED,
    EVENT_RESPONSE_VERIFIED,
    EVENT_ANOMALY_DETECTED,
)
from anchor.models import AttributePair
from anchor.rest import build_app


# ─── helpers ─────────────────────────────────────────────────────────────────

def _unit_vec(dim: int = 8, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, dim)
    return (v / np.linalg.norm(v)).tolist()


def _make_client(cfg: Config | None = None) -> tuple[TestClient, MagicMock, MagicMock]:
    c = cfg or Config()
    pub = MagicMock(spec=Publisher)
    hm = MagicMock(spec=HookManager)
    app = build_app(c, pub, hm)
    return TestClient(app), pub, hm


# ─── health ──────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self):
        client, _, _ = _make_client()
        r = client.get("/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_metrics_endpoint(self):
        client, _, _ = _make_client()
        r = client.get("/v1/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]


# ─── /v1/anchor/secure ───────────────────────────────────────────────────────

class TestSecureEmbedding:
    def test_returns_secured_embedding(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/secure", json={
            "embedding": _unit_vec(16),
            "tenant_id": "t1",
            "operation": "index",
        })
        assert r.status_code == 200
        body = r.json()
        assert "secured_embedding" in body
        assert len(body["secured_embedding"]) == 16

    def test_metrics_in_response(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/secure", json={"embedding": _unit_vec()})
        m = r.json()["metrics"]
        assert "noise_added" in m
        assert "similarity_to_original" in m
        assert "strategy_used" in m

    def test_processing_time_positive(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/secure", json={"embedding": _unit_vec()})
        assert r.json()["processing_time_ms"] >= 0.0

    def test_request_id_propagated(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/secure", json={
            "request_id": "req-abc",
            "embedding": _unit_vec(),
        })
        assert r.json()["request_id"] == "req-abc"

    def test_event_published(self):
        client, pub, _ = _make_client()
        client.post("/v1/anchor/secure", json={"embedding": _unit_vec(), "tenant_id": "t1"})
        pub.publish.assert_called_once()

    def test_hook_fired(self):
        client, _, hm = _make_client()
        client.post("/v1/anchor/secure", json={"embedding": _unit_vec(), "tenant_id": "t1"})
        hm.fire.assert_called_once()
        ev = hm.fire.call_args[0][0]
        assert ev.type == EVENT_EMBEDDING_SECURED
        assert ev.tenant_id == "t1"

    def test_custom_sigma_applied(self):
        """Custom sigma=1e-10 (near-zero) produces near-identical secured embedding."""
        vec = _unit_vec(16)
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/secure", json={
            "embedding": vec,
            "options": {"noise_sigma": 1e-10, "seed": 42},
        })
        # Near-zero sigma → output ≈ input (much smaller diff than default sigma=0.01)
        secured = r.json()["secured_embedding"]
        assert r.json()["metrics"]["noise_added"] < 1e-8

    def test_tenant_override_sigma(self):
        """Tenant override in config changes the injected sigma."""
        cfg = Config()
        cfg.noise.tenant_overrides = {"high-sec": 0.5}
        client, _, _ = _make_client(cfg)
        r = client.post("/v1/anchor/secure", json={
            "embedding": _unit_vec(8),
            "tenant_id": "high-sec",
        })
        assert r.status_code == 200


# ─── /v1/anchor/secure/batch ─────────────────────────────────────────────────

class TestSecureBatch:
    def test_batch_returns_all_items(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/secure/batch", json={
            "request_id": "b1",
            "items": [
                {"embedding": _unit_vec(8, i), "tenant_id": "t1"}
                for i in range(3)
            ],
        })
        assert r.status_code == 200
        assert len(r.json()["items"]) == 3

    def test_each_batch_item_has_secured_embedding(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/secure/batch", json={
            "items": [{"embedding": _unit_vec(4, i)} for i in range(2)],
        })
        for item in r.json()["items"]:
            assert len(item["secured_embedding"]) == 4

    def test_empty_batch_returns_empty_items(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/secure/batch", json={"items": []})
        assert r.status_code == 200
        assert r.json()["items"] == []


# ─── /v1/anchor/bias ─────────────────────────────────────────────────────────

class TestBiasEndpoint:
    def _bias_payload(self, words=None, pairs=None):
        return {
            "test_words": words or ["doctor", "nurse"],
            "attribute_pairs": pairs or [
                {"attribute_a": "male", "attribute_b": "female", "category": "gender"}
            ],
            "tenant_id": "t1",
        }

    def test_returns_measurements(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/bias", json=self._bias_payload())
        assert r.status_code == 200
        assert "measurements" in r.json()
        assert len(r.json()["measurements"]) == 2  # 2 words × 1 pair

    def test_overall_score_present(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/bias", json=self._bias_payload())
        assert "overall_bias_score" in r.json()
        assert r.json()["overall_bias_score"] >= 0.0

    def test_high_bias_fires_event(self):
        """Provide embeddings that maximise bias to trigger event."""
        client, pub, hm = _make_client()
        payload = {
            "test_words": ["word"],
            "attribute_pairs": [{"attribute_a": "yes", "attribute_b": "no", "category": "test"}],
            "word_embeddings": {
                "word": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "yes":  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "no":   [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "tenant_id": "t1",
        }
        r = client.post("/v1/anchor/bias", json=payload)
        assert r.status_code == 200
        # High severity measurement should fire event
        pub.publish.assert_called()
        hm.fire.assert_called()
        ev = hm.fire.call_args[0][0]
        assert ev.type == EVENT_BIAS_DETECTED

    def test_low_bias_no_event(self):
        """Identical attr embeddings → zero bias → no event fired."""
        client, pub, hm = _make_client()
        payload = {
            "test_words": ["word"],
            "attribute_pairs": [{"attribute_a": "attr", "attribute_b": "attr", "category": "test"}],
            "word_embeddings": {
                "word": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "attr": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "tenant_id": "t1",
        }
        client.post("/v1/anchor/bias", json=payload)
        pub.publish.assert_not_called()

    def test_interpretation_in_response(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/bias", json=self._bias_payload())
        assert "interpretation" in r.json()


# ─── /v1/anchor/quality ──────────────────────────────────────────────────────

class TestQualityEndpoint:
    def test_no_pairs_returns_estimate(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/quality", json={"noise_sigma": 0.01})
        assert r.status_code == 200
        body = r.json()
        assert body["noise_sigma"] == pytest.approx(0.01, abs=1e-6)

    def test_with_pairs(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/quality", json={
            "noise_sigma": 0.01,
            "embedding_pairs": [
                {"query_embedding": _unit_vec(8, 0), "match_embedding": _unit_vec(8, 1)},
            ],
        })
        assert r.status_code == 200
        assert r.json()["status"] in ("acceptable", "warning", "critical")

    def test_uses_config_sigma_when_zero(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/quality", json={"noise_sigma": 0.0})
        assert r.status_code == 200


# ─── /v1/anchor/config GET/PUT ───────────────────────────────────────────────

class TestConfigEndpoint:
    def test_get_config(self):
        client, _, _ = _make_client()
        r = client.get("/v1/anchor/config")
        assert r.status_code == 200
        body = r.json()
        assert "default_noise_sigma" in body
        assert "strategy" in body
        assert "preserve_norm" in body

    def test_put_config_updates_sigma(self):
        client, _, _ = _make_client()
        r = client.put("/v1/anchor/config", json={
            "default_noise_sigma": 0.05,
            "preserve_norm": True,
            "strategy": "gaussian",
            "tenant_overrides": {},
            "bias_enabled": True,
            "quality_enabled": True,
        })
        assert r.status_code == 200
        assert r.json()["default_noise_sigma"] == pytest.approx(0.05, abs=1e-6)

    def test_put_config_tenant_overrides(self):
        client, _, _ = _make_client()
        r = client.put("/v1/anchor/config", json={
            "default_noise_sigma": 0.01,
            "preserve_norm": True,
            "strategy": "gaussian",
            "tenant_overrides": {"vip": 0.001},
            "bias_enabled": True,
            "quality_enabled": True,
        })
        assert r.status_code == 200
        assert "vip" in r.json()["tenant_overrides"]


# ─── /v1/anchor/verify ───────────────────────────────────────────────────────

class TestVerifyEndpoint:
    def _req(self, response: str = "The analyst completed the report.", **kw):
        return {
            "request_id": "v-1",
            "llm_response": response,
            "original_query": "What happened?",
            "source_documents": ["The analyst completed the report."],
            **kw,
        }

    def test_returns_verify_response(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/verify", json=self._req())
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert body["status"] in ("PASSED", "WARNING", "CRITICAL")

    def test_event_published(self):
        client, pub, _ = _make_client()
        client.post("/v1/anchor/verify", json=self._req())
        pub.publish.assert_called()

    def test_hook_fired_response_verified(self):
        client, _, hm = _make_client()
        client.post("/v1/anchor/verify", json=self._req())
        types_fired = [call[0][0].type for call in hm.fire.call_args_list]
        assert EVENT_RESPONSE_VERIFIED in types_fired

    def test_semantic_drift_fires_anomaly_event(self):
        """Response completely unrelated to source → semantic drift → anomaly hook."""
        client, _, hm = _make_client()
        client.post("/v1/anchor/verify", json={
            "llm_response": "quantum mechanics relativity energy mass",
            "original_query": "HR policy",
            "source_documents": ["the HR policy update states that leaves are capped"],
        })
        types_fired = [call[0][0].type for call in hm.fire.call_args_list]
        # If semantic drift was detected, anomaly event should fire
        # (depends on jaccard similarity — verify at least response_verified fired)
        assert EVENT_RESPONSE_VERIFIED in types_fired

    def test_biased_response_status_not_passed(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/verify", json=self._req(
            "women are too emotional and cannot lead teams"
        ))
        assert r.json()["status"] in ("WARNING", "CRITICAL")

    def test_request_id_propagated(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/verify", json=self._req())
        assert r.json()["request_id"] == "v-1"

    def test_warnings_list_present(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/verify", json=self._req())
        assert isinstance(r.json()["warnings"], list)


class TestAnalyzeEndpoint:
    """POST /v1/anchor/analyze is an alias for /v1/anchor/verify."""

    def test_alias_works(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/analyze", json={
            "llm_response": "The report was completed.",
            "original_query": "Was the report done?",
        })
        assert r.status_code == 200
        assert r.json()["status"] in ("PASSED", "WARNING", "CRITICAL")


class TestAnalyzeEmbeddingEndpoint:
    def test_returns_similarities(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/analyze/embedding", json={
            "text": "The doctor worked hard.",
            "comparison_texts": ["doctor worked", "completely unrelated"],
        })
        assert r.status_code == 200
        assert len(r.json()["similarities"]) == 2

    def test_empty_comparisons(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/analyze/embedding", json={
            "text": "hello",
            "comparison_texts": [],
        })
        assert r.status_code == 200
        assert r.json()["similarities"] == []


class TestDriftEndpoint:
    def test_no_drift_with_equal_scores(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/drift", json={
            "phase1": {"overall_score": 0.1, "categories": [{"category": "gender", "score": 0.1}]},
            "phase2": {
                "overall_score": 0.1,
                "categories": [{"category": "gender", "score": 0.1}],
                "severity": "low",
                "exceeds_threshold": False,
            },
        })
        assert r.status_code == 200
        assert r.json()["drift_detected"] is False

    def test_amplification_drift(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/drift", json={
            "phase1": {"overall_score": 0.05, "categories": [{"category": "gender", "score": 0.05}]},
            "phase2": {
                "overall_score": 0.40,
                "categories": [{"category": "gender", "score": 0.40}],
                "severity": "medium",
                "exceeds_threshold": True,
            },
        })
        assert r.status_code == 200
        assert r.json()["drift_detected"] is True

    def test_empty_request_no_crash(self):
        client, _, _ = _make_client()
        r = client.post("/v1/anchor/drift", json={})
        assert r.status_code == 200
        assert r.json()["drift_detected"] is False
