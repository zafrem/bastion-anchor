from __future__ import annotations

import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    rest_port: int = 8083
    grpc_port: int = 9093
    workers: int = 4


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


class Config(BaseModel):
    version: str = "3.0"
    server: ServerConfig = Field(default_factory=ServerConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    bias: BiasConfig = Field(default_factory=BiasConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)

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
