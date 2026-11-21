from __future__ import annotations

import yaml
from pydantic import BaseModel, Field

from .auth import resolve_token


class ServerConfig(BaseModel):
    rest_port: int = 8083
    grpc_port: int = 9093
    workers: int = 4
    # Shared bearer token callers must present on REST and gRPC (or env ANCHOR_AUTH_TOKEN).
    # Required: Anchor refuses to start without it unless allow_unauthenticated is set.
    auth_token: str = ""
    # Development only: start with no auth token and accept unauthenticated callers.
    allow_unauthenticated: bool = False
    # gRPC TLS: PEM files. Set cert + key to serve gRPC over TLS (files are re-read when they change, so
    # rotation needs no restart); add client_ca to require client certificates (mutual TLS). Empty = plaintext.
    grpc_tls_cert: str = ""
    grpc_tls_key: str = ""
    grpc_tls_client_ca: str = ""


class NoiseConfig(BaseModel):
    default_sigma: float = 0.01
    preserve_norm: bool = True
    strategy: str = "gaussian"
    tenant_overrides: dict[str, float] = Field(default_factory=dict)


class BiasConfig(BaseModel):
    enabled: bool = True
    medium_threshold: float = 0.15
    high_threshold: float = 0.25
    dims: int = 1024


class QualityConfig(BaseModel):
    enabled: bool = True
    recall_drop_alert: float = 20.0
    similarity_minimum: float = 0.95


class EventsConfig(BaseModel):
    nats_url: str = ""
    enabled: bool = True


class ExceptionsConfig(BaseModel):
    """Scoped exceptions signed by Tracker. Off unless a public key is set (env ANCHOR_EXCEPTIONS_PUBLIC_KEY).

    Several comma-separated keys are accepted while a key is rotated. ``file`` is where verified
    exceptions are kept, so Anchor enforces from its own copy even when Tracker or NATS is down.
    """
    public_key: str = ""
    file: str = ""


class Config(BaseModel):
    version: str = "3.0"
    server: ServerConfig = Field(default_factory=ServerConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    bias: BiasConfig = Field(default_factory=BiasConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    exceptions: ExceptionsConfig = Field(default_factory=ExceptionsConfig)

    def sigma_for_tenant(self, tenant_id: str) -> float:
        return self.noise.tenant_overrides.get(tenant_id, self.noise.default_sigma)

    @classmethod
    def load(cls, path: str) -> "Config":
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            return cls.model_validate(data or {})
        except FileNotFoundError:
            return cls()

    def validate_security(self) -> None:
        """Refuse configurations that accept unauthenticated callers without an explicit dev opt-in."""
        if not resolve_token(self.server.auth_token) and not self.server.allow_unauthenticated:
            raise ValueError(
                "server.auth_token (or ANCHOR_AUTH_TOKEN) is required: Anchor governs per-tenant "
                "embedding noise and runtime config, so it must authenticate its callers. For local "
                "development only, set server.allow_unauthenticated: true"
            )
