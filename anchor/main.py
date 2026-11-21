"""Anchor service entry point: starts FastAPI (REST) + gRPC in parallel."""
from __future__ import annotations

import argparse
import logging
import os
import threading

import uvicorn

from .config import Config
from .events import Publisher
from .exceptions import Client as ExceptionsClient, ExceptionsError
from .verifier import set_exceptions
from .hooks import HookManager
from .rest import build_app
from .grpc_server import GRPCServer
from .tls import TLSConfigError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bastion Anchor service")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    try:
        cfg.validate_security()
    except ValueError as exc:
        raise SystemExit(f"refusing to start: {exc}")
    if not cfg.server.auth_token and cfg.server.allow_unauthenticated:
        log.warning("[security] server.allow_unauthenticated is enabled: callers are not authenticated — do not use in production")
    log.info("[anchor] starting v%s (REST :%d, gRPC :%d)", cfg.version, cfg.server.rest_port, cfg.server.grpc_port)

    pub = Publisher(cfg.events.nats_url)
    hm = HookManager()

    # Scoped exceptions: verified against Tracker's public key and kept in a local file, so Anchor
    # enforces from its own copy and keeps working with Tracker or NATS down. No key = feature off.
    pub_keys = os.environ.get("ANCHOR_EXCEPTIONS_PUBLIC_KEY", cfg.exceptions.public_key)
    exc_file = cfg.exceptions.file or ("anchor-exceptions.json" if pub_keys else "")
    try:
        exc_client = ExceptionsClient("anchor", pub_keys, exc_file)
    except ExceptionsError as exc:
        raise SystemExit(f"refusing to start: exceptions: {exc}")
    if exc_client.enabled:
        set_exceptions(exc_client)
        exc_client.start(cfg.events.nats_url)
        log.info("[anchor] scoped exceptions enabled (file %s)", exc_file)

    # gRPC server runs in a background thread
    try:
        grpc_srv = GRPCServer(cfg, pub, hm)
    except TLSConfigError as exc:
        raise SystemExit(f"refusing to start: gRPC TLS: {exc}")
    grpc_thread = threading.Thread(
        target=grpc_srv.serve,
        args=(cfg.server.grpc_port,),
        daemon=True,
        name="anchor-grpc",
    )
    grpc_thread.start()

    # REST server runs in the main thread (blocks).
    # NOTE: uvicorn only honours workers>1 when given an import string, but the
    # app is built in-process with live state (publisher, hooks, gRPC thread),
    # so it must run single-worker. cfg.server.workers is intentionally ignored
    # here (matching the navigator module); scale out with multiple processes.
    app = build_app(cfg, pub, hm)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=cfg.server.rest_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
