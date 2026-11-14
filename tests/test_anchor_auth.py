"""Anchor caller authentication (REST + gRPC) and startup validation."""
from __future__ import annotations

import grpc
import numpy as np
import pytest
from fastapi.testclient import TestClient

from anchor.config import Config
from anchor.events import Publisher
from anchor.grpc_server import GRPCServer
from anchor.hooks import HookManager
from anchor.rest import build_app

TOKEN = "anchor-shared-token-0123456789"


def _unit_vec(dim: int = 8) -> list[float]:
    v = np.ones(dim, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


def _cfg(token: str = "") -> Config:
    cfg = Config()
    cfg.server.auth_token = token
    return cfg


def _client(token: str = "") -> TestClient:
    return TestClient(build_app(_cfg(token), Publisher(""), HookManager()))


SECURE = {"request_id": "r1", "tenant_id": "acme", "embedding": _unit_vec()}


# ─── REST ─────────────────────────────────────────────────────────────────────

def test_rest_requires_the_token_when_configured():
    c = _client(TOKEN)
    assert c.post("/v1/anchor/secure", json=SECURE).status_code == 401
    assert c.post("/v1/anchor/secure", json=SECURE, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/v1/anchor/secure", json=SECURE, headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
    assert c.post("/v1/anchor/secure", json=SECURE, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_rest_config_and_verify_endpoints_are_protected_too():
    c = _client(TOKEN)
    assert c.get("/v1/anchor/config").status_code == 401
    assert c.put("/v1/anchor/config", json={"default_noise_sigma": 0.05}).status_code == 401
    assert c.post("/v1/anchor/verify", json={"llm_response": "hi", "original_query": "q"}).status_code == 401
    assert c.get("/v1/anchor/config", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_rest_probes_stay_public():
    c = _client(TOKEN)
    for path in ("/v1/health", "/v1/health/live", "/v1/health/ready", "/v1/metrics"):
        assert c.get(path).status_code == 200, path


def test_rest_is_open_when_no_token_is_configured(monkeypatch):
    monkeypatch.delenv("ANCHOR_AUTH_TOKEN", raising=False)
    assert _client("").post("/v1/anchor/secure", json=SECURE).status_code == 200


def test_rest_token_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("ANCHOR_AUTH_TOKEN", TOKEN)
    c = _client("")
    assert c.post("/v1/anchor/secure", json=SECURE).status_code == 401
    assert c.post("/v1/anchor/secure", json=SECURE, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


# ─── gRPC ─────────────────────────────────────────────────────────────────────

SECURE_RPC = "/bastion.anchor.v1.AnchorService/SecureEmbedding"
CONFIG_RPC = "/bastion.anchor.v1.AnchorService/GetConfig"
HEALTH_RPC = "/bastion.anchor.v1.AnchorService/Health"


@pytest.fixture()
def grpc_channel():
    servers = []

    def make(token: str):
        srv = GRPCServer(_cfg(token), Publisher(""), HookManager())._make_server()
        port = srv.add_insecure_port("localhost:0")
        srv.start()
        servers.append(srv)
        return grpc.insecure_channel(f"localhost:{port}")

    yield make
    for s in servers:
        s.stop(0)


def _call(channel, method, metadata=None):
    return channel.unary_unary(method)(b"{}", metadata=metadata or [], timeout=5)


def test_grpc_requires_the_token_when_configured(grpc_channel):
    ch = grpc_channel(TOKEN)
    for md in (None, [("authorization", "Bearer nope")], [("authorization", f"Basic {TOKEN}")]):
        with pytest.raises(grpc.RpcError) as e:
            _call(ch, CONFIG_RPC, md)
        assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert _call(ch, CONFIG_RPC, [("authorization", f"Bearer {TOKEN}")])


def test_grpc_health_stays_public(grpc_channel):
    assert _call(grpc_channel(TOKEN), HEALTH_RPC)


def test_grpc_is_open_when_no_token_is_configured(grpc_channel, monkeypatch):
    monkeypatch.delenv("ANCHOR_AUTH_TOKEN", raising=False)
    assert _call(grpc_channel(""), CONFIG_RPC)


# ─── startup validation ───────────────────────────────────────────────────────

def test_startup_refuses_without_an_auth_token(monkeypatch):
    monkeypatch.delenv("ANCHOR_AUTH_TOKEN", raising=False)
    with pytest.raises(ValueError, match="auth_token"):
        Config().validate_security()


def test_startup_accepts_a_configured_or_environment_token(monkeypatch):
    monkeypatch.delenv("ANCHOR_AUTH_TOKEN", raising=False)
    cfg = Config()
    cfg.server.auth_token = TOKEN
    cfg.validate_security()

    monkeypatch.setenv("ANCHOR_AUTH_TOKEN", TOKEN)
    Config().validate_security()


def test_unauthenticated_mode_needs_an_explicit_dev_opt_in(monkeypatch):
    monkeypatch.delenv("ANCHOR_AUTH_TOKEN", raising=False)
    cfg = Config()
    assert cfg.server.allow_unauthenticated is False
    cfg.server.allow_unauthenticated = True
    cfg.validate_security()


def test_the_shipped_sample_config_requires_the_operator_to_provide_a_token(monkeypatch):
    monkeypatch.delenv("ANCHOR_AUTH_TOKEN", raising=False)
    cfg = Config.load("config/config.yaml")
    assert cfg.server.auth_token == "" and cfg.server.allow_unauthenticated is False
    with pytest.raises(ValueError):
        cfg.validate_security()
