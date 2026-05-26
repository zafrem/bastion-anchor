"""Lightweight hook/extension-point system for Anchor (doc 01 §2.3)."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

EVENT_EMBEDDING_SECURED = "embedding_secured"
EVENT_BIAS_DETECTED = "bias_detected"
EVENT_RESPONSE_VERIFIED = "response_verified"
EVENT_ANOMALY_DETECTED = "anomaly_detected"


@dataclass
class HookEvent:
    type: str
    tenant_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    request_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)


HookHandler = Callable[[HookEvent], None]


class HookManager:
    """Thread-safe hook registry. Registered handlers are called asynchronously."""

    def __init__(self) -> None:
        self._mu = threading.RLock()
        self._handlers: dict[str, list[HookHandler]] = {}

    def register(self, event_type: str, handler: HookHandler) -> None:
        with self._mu:
            self._handlers.setdefault(event_type, []).append(handler)

    def fire(self, ev: HookEvent) -> None:
        with self._mu:
            handlers = list(self._handlers.get(ev.type, []))
        for h in handlers:
            t = threading.Thread(target=self._safe_call, args=(h, ev), daemon=True)
            t.start()

    @staticmethod
    def _safe_call(h: HookHandler, ev: HookEvent) -> None:
        try:
            h(ev)
        except Exception as exc:
            log.warning("[hooks] handler error for %s: %s", ev.type, exc)
