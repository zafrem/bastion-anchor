"""Anchor service entry point: starts FastAPI (REST) + gRPC in parallel."""
from __future__ import annotations

import argparse
import logging
import threading

import uvicorn

from .config import Config
from .events import Publisher
from .hooks import HookManager
from .rest import build_app
from .grpc_server import GRPCServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bastion Anchor service")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    log.info("[anchor] starting v%s (REST :%d, gRPC :%d)", cfg.version, cfg.server.rest_port, cfg.server.grpc_port)

    pub = Publisher(cfg.events.nats_url)
    hm = HookManager()

    # gRPC server runs in a background thread
    grpc_srv = GRPCServer(cfg, pub, hm)
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
