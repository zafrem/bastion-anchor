"""Caller authentication for Anchor's REST and gRPC APIs (shared bearer token).

Anchor controls per-tenant differential privacy noise and output verification thresholds, so only
callers holding the shared token may invoke its endpoints or update its configuration (health and
metrics probes excepted).
"""
from __future__ import annotations

import hmac
import os
from typing import Optional

import grpc

# Environment fallback when server.auth_token is empty.
ENV_TOKEN = "ANCHOR_AUTH_TOKEN"

# Reachable without a token so orchestrators and Prometheus can probe the service.
PUBLIC_REST_PATHS = frozenset({"/v1/health", "/v1/health/live", "/v1/health/ready", "/v1/metrics"})
GRPC_SERVICE_NAME = "bastion.anchor.v1.AnchorService"
PUBLIC_GRPC_METHODS = frozenset({f"/{GRPC_SERVICE_NAME}/Health"})


def resolve_token(configured: str = "") -> str:
    """The shared token: the configured value, else ANCHOR_AUTH_TOKEN, else "" (unauthenticated)."""
    return configured or os.environ.get(ENV_TOKEN, "")


def token_matches(header_value: Optional[str], token: str) -> bool:
    """True if ``header_value`` is ``Bearer <token>``, compared in constant time."""
    if not header_value or not header_value.startswith("Bearer "):
        return False
    return hmac.compare_digest(header_value[len("Bearer "):].encode(), token.encode())


class GrpcAuthInterceptor(grpc.ServerInterceptor):
    """Rejects gRPC calls that lack the bearer token (``authorization`` metadata)."""

    def __init__(self, token: str) -> None:
        self._token = token

    @staticmethod
    def _abort(_request, context):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "missing or invalid bearer token")

    @classmethod
    def _deny_like(cls, handler):
        """A handler that rejects with UNAUTHENTICATED but has the same cardinality as ``handler``,
        so unary and streaming methods are all refused with the proper status."""
        if handler is None or (not handler.request_streaming and not handler.response_streaming):
            return grpc.unary_unary_rpc_method_handler(cls._abort)
        if not handler.request_streaming:
            return grpc.unary_stream_rpc_method_handler(cls._abort)
        if not handler.response_streaming:
            return grpc.stream_unary_rpc_method_handler(cls._abort)
        return grpc.stream_stream_rpc_method_handler(cls._abort)

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if not self._token or handler_call_details.method in PUBLIC_GRPC_METHODS:
            return handler
        for key, value in handler_call_details.invocation_metadata or ():
            if key.lower() == "authorization" and token_matches(value, self._token):
                return handler
        return self._deny_like(handler)
