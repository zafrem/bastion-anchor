"""Foundation-standard NATS event publisher for Anchor (doc 02)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

log = logging.getLogger(__name__)

_MODULE = "anchor"
_MODULE_VERSION = "3.0.0"
_SCHEMA_VER = "1.0"
_SUBJECT_PFX = "bastion.events.anchor"


@dataclass
class TraceContext:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_span_id: str = ""
    tenant_id: str = ""
    user_id: str = ""
    request_id: str = ""


@dataclass
class AnchorEvent:
    event_id: str
    event_type: str
    schema_version: str
    trace_id: str
    span_id: str
    parent_span_id: str
    module: str
    module_version: str
    timestamp: int
    tenant_id: str
    severity: str
    category: str
    user_id: str = ""
    request_id: str = ""
    duration_ms: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    status: str = ""
    action_taken: str = ""


def _new_event(
    tc: TraceContext,
    event_type: str,
    severity: str,
    category: str,
    data: dict[str, Any],
) -> AnchorEvent:
    return AnchorEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        schema_version=_SCHEMA_VER,
        trace_id=tc.trace_id,
        span_id=tc.span_id,
        parent_span_id=tc.parent_span_id,
        module=_MODULE,
        module_version=_MODULE_VERSION,
        timestamp=time.time_ns(),
        tenant_id=tc.tenant_id,
        user_id=tc.user_id,
        request_id=tc.request_id,
        severity=severity,
        category=category,
        data=data,
    )


def event_embedding_secured(tc: TraceContext, operation: str, noise_magnitude: float, similarity: float) -> AnchorEvent:
    ev = _new_event(tc, "embedding_secured", "info", "security", {
        "operation": operation,
        "noise_magnitude": noise_magnitude,
        "similarity_to_original": similarity,
    })
    ev.status = "secured"
    ev.action_taken = "noise_inject"
    return ev


def event_bias_detected(tc: TraceContext, category: str, score: float) -> AnchorEvent:
    ev = _new_event(tc, "bias_detected", "warning", "security", {
        "bias_category": category,
        "bias_score": score,
    })
    ev.status = "detected"
    ev.action_taken = "flag"
    return ev


def event_response_verified(tc: TraceContext, status: str, bias_score: float) -> AnchorEvent:
    ev = _new_event(tc, "response_verified", "info", "security", {
        "verify_status": status,
        "bias_score": bias_score,
    })
    ev.status = status.lower()
    ev.action_taken = "verify"
    return ev


def event_anomaly_detected(tc: TraceContext, anomaly_type: str, score: float) -> AnchorEvent:
    ev = _new_event(tc, "anomaly_detected", "warning", "security", {
        "anomaly_type": anomaly_type,
        "score": score,
    })
    ev.status = "detected"
    ev.action_taken = "flag"
    return ev


class Publisher:
    """Thread-safe async NATS publisher. Fire-and-forget; never blocks callers."""

    def __init__(self, nats_url: str) -> None:
        self._url = nats_url
        self._nc = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        if nats_url:
            self._start()

    def _start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="anchor-nats")
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop)

    async def _connect(self) -> None:
        try:
            import nats
            self._nc = await nats.connect(
                self._url,
                max_reconnect_attempts=5,
                reconnect_time_wait=2,
                error_cb=lambda e: log.warning("[anchor-events] nats error: %s", e),
            )
        except Exception as exc:
            log.warning("[anchor-events] nats unavailable (%s), events disabled", exc)

    def publish(self, ev: AnchorEvent) -> None:
        if not self._loop or not self._nc:
            return
        asyncio.run_coroutine_threadsafe(self._publish(ev), self._loop)

    async def _publish(self, ev: AnchorEvent) -> None:
        if not self._nc:
            return
        try:
            subject = f"{_SUBJECT_PFX}.{ev.event_type}"
            payload = json.dumps(asdict(ev)).encode()
            await self._nc.publish(subject, payload)
        except Exception as exc:
            log.debug("[anchor-events] publish failed: %s", exc)

    def close(self) -> None:
        if self._nc and self._loop:
            asyncio.run_coroutine_threadsafe(self._nc.drain(), self._loop)
