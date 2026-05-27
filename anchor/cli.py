"""anchor-cli — Bastion Anchor command-line interface (SRS §6.3)."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


# ─── helpers ─────────────────────────────────────────────────────────────────

def _api_get(base_url: str, path: str) -> Any:
    import urllib.request
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=10) as resp:
        return json.loads(resp.read())


def _api_post(base_url: str, path: str, payload: dict) -> Any:
    import urllib.request
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ─── commands ────────────────────────────────────────────────────────────────

def cmd_server(args: argparse.Namespace) -> int:
    """Start the Anchor REST + gRPC server."""
    import sys
    argv = ["anchor"]
    if args.config:
        argv += ["--config", args.config]
    sys.argv = argv
    from anchor.main import main as _main
    _main()
    return 0


def cmd_secure(args: argparse.Namespace) -> int:
    """Inject noise into an embedding vector."""
    if args.embedding_file:
        with open(args.embedding_file) as f:
            data = json.load(f)
        embedding = data if isinstance(data, list) else data.get("embedding", data)
    elif args.embedding:
        embedding = json.loads(args.embedding)
    else:
        print("error: provide --embedding or --embedding-file", file=sys.stderr)
        return 1

    payload: dict[str, Any] = {
        "embedding": embedding,
        "strategy": args.strategy,
        "preserve_norm": not args.no_preserve_norm,
    }
    if args.tenant:
        payload["tenant_id"] = args.tenant
    if args.sigma is not None:
        payload["sigma"] = args.sigma

    result = _api_post(args.api, "/v1/anchor/secure", payload)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        mag = result.get("noise_magnitude", 0.0)
        sim = result.get("similarity_to_original", 0.0)
        strategy = result.get("strategy_used", args.strategy)
        print(f"Secured  strategy={strategy}  noise_magnitude={mag:.6f}  similarity={sim:.6f}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify an LLM response for bias and quality."""
    payload: dict[str, Any] = {
        "response": args.response,
    }
    if args.query:
        payload["query"] = args.query
    if args.tenant:
        payload["tenant_id"] = args.tenant

    result = _api_post(args.api, "/v1/anchor/verify", payload)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = result.get("status", "unknown")
        bias_score = result.get("bias_score", 0.0)
        print(f"Verify  status={status}  bias_score={bias_score:.4f}")
        for insight in result.get("insights", []):
            print(f"  insight: {insight}")
    return 0


def cmd_bias(args: argparse.Namespace) -> int:
    """Run WEAT bias analysis on an embedding."""
    if args.embedding_file:
        with open(args.embedding_file) as f:
            data = json.load(f)
        embedding = data if isinstance(data, list) else data.get("embedding", data)
    else:
        embedding = json.loads(args.embedding)

    payload: dict[str, Any] = {"embedding": embedding}
    if args.test_set:
        payload["test_set"] = args.test_set

    result = _api_post(args.api, "/v1/anchor/bias", payload)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        severity = result.get("severity", "unknown")
        score = result.get("bias_score", 0.0)
        print(f"Bias  severity={severity}  score={score:.4f}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Check the health of a running Anchor server."""
    result = _api_get(args.api, "/v1/health")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = result.get("status", "unknown")
        version = result.get("version", "?")
        print(f"Anchor  status={status}  version={version}")
    ok = result.get("status") in ("ok", "ready")
    return 0 if ok else 1


# ─── parser ───────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anchor-cli",
        description="Bastion Anchor — embedding security and response verification",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # server
    srv = sub.add_parser("server", help="Start the Anchor server")
    srv.add_argument("--config", default="", help="Path to config YAML")
    srv.set_defaults(func=cmd_server)

    # secure
    sec = sub.add_parser("secure", help="Inject noise into an embedding")
    grp = sec.add_mutually_exclusive_group()
    grp.add_argument("--embedding-file", "-f", help="JSON file containing embedding array")
    grp.add_argument("--embedding", "-e", help="Embedding as JSON array string")
    sec.add_argument("--tenant", "-t", default="", help="Tenant ID (determines sigma override)")
    sec.add_argument("--strategy", default="gaussian", choices=["gaussian", "laplacian"],
                     help="Noise strategy (default: gaussian)")
    sec.add_argument("--sigma", type=float, default=None, help="Override noise sigma")
    sec.add_argument("--no-preserve-norm", action="store_true",
                     help="Do not preserve the original L2 norm")
    sec.add_argument("--api", default="http://localhost:8083", help="Anchor API base URL")
    sec.add_argument("--json", action="store_true", help="Output raw JSON")
    sec.set_defaults(func=cmd_secure)

    # verify
    ver = sub.add_parser("verify", help="Verify an LLM response")
    ver.add_argument("--response", "-r", required=True, help="LLM response text to verify")
    ver.add_argument("--query", "-q", default="", help="Original user query (improves analysis)")
    ver.add_argument("--tenant", "-t", default="", help="Tenant ID")
    ver.add_argument("--api", default="http://localhost:8083", help="Anchor API base URL")
    ver.add_argument("--json", action="store_true", help="Output raw JSON")
    ver.set_defaults(func=cmd_verify)

    # bias
    bi = sub.add_parser("bias", help="Run WEAT bias analysis on an embedding")
    grp2 = bi.add_mutually_exclusive_group(required=True)
    grp2.add_argument("--embedding-file", "-f", help="JSON file containing embedding array")
    grp2.add_argument("--embedding", "-e", help="Embedding as JSON array string")
    bi.add_argument("--test-set", default="", help="Specific WEAT test set to run")
    bi.add_argument("--api", default="http://localhost:8083", help="Anchor API base URL")
    bi.add_argument("--json", action="store_true", help="Output raw JSON")
    bi.set_defaults(func=cmd_bias)

    # health
    h = sub.add_parser("health", help="Check server health")
    h.add_argument("--api", default="http://localhost:8083", help="Anchor API base URL")
    h.add_argument("--json", action="store_true", help="Output raw JSON")
    h.set_defaults(func=cmd_health)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        sys.exit(args.func(args))
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
