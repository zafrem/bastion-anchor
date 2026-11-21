"""Anchor gRPC over TLS: server credentials, mutual TLS and rotation."""
from __future__ import annotations

import hashlib
import os
import socket
import ssl
import time

import grpc
import pytest

from anchor.config import Config
from anchor.events import Publisher
from anchor.grpc_server import GRPCServer
from anchor.hooks import HookManager
from anchor.tls import TLSConfigError, server_credentials
from tests._pki import PKI, needs_openssl

pytestmark = needs_openssl

HEALTH = "/bastion.anchor.v1.AnchorService/Health"


@pytest.fixture()
def pki(tmp_path):
    return PKI(tmp_path)


@pytest.fixture()
def serve():
    servers = []

    def start(*, cert="", key="", client_ca="") -> int:
        cfg = Config()
        cfg.server.grpc_tls_cert = str(cert) if cert else ""
        cfg.server.grpc_tls_key = str(key) if key else ""
        cfg.server.grpc_tls_client_ca = str(client_ca) if client_ca else ""
        gs = GRPCServer(cfg, Publisher(""), HookManager())
        srv = gs._make_server()
        port = gs._add_port(srv, "localhost:0")
        srv.start()
        servers.append(srv)
        return port

    yield start
    for s in servers:
        s.stop(0)


def _read(p) -> bytes:
    with open(p, "rb") as f:
        return f.read()


def _tls_channel(port, pki, *, ca=None, client=False, cert=None, key=None):
    creds = grpc.ssl_channel_credentials(
        root_certificates=_read(ca or pki.ca_cert),
        private_key=_read(key or pki.client_key) if (client or key) else None,
        certificate_chain=_read(cert or pki.client_cert) if (client or cert) else None,
    )
    return grpc.secure_channel(f"localhost:{port}", creds)


def _call(channel):
    return channel.unary_unary(HEALTH)(b"{}", timeout=5)


def _served_cert_digest(port) -> str:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.set_alpn_protocols(["h2"])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock, ctx.wrap_socket(sock) as tls:
        return hashlib.sha256(tls.getpeercert(binary_form=True)).hexdigest()


def test_no_tls_settings_means_plaintext(serve):
    assert server_credentials("", "", "") is None
    assert _call(grpc.insecure_channel(f"localhost:{serve()}"))


def test_misconfiguration_raises(pki, tmp_path):
    garbage = tmp_path / "garbage.pem"
    garbage.write_text("not a pem")
    for args in [
        (str(pki.server_cert), "", ""),
        ("", str(pki.server_key), ""),
        ("", "", str(pki.ca_cert)),
        (str(tmp_path / "nope.pem"), str(pki.server_key), ""),
        (str(pki.server_cert), str(pki.client_key), ""),
        (str(pki.server_cert), str(pki.server_key), str(garbage)),
    ]:
        with pytest.raises(TLSConfigError):
            server_credentials(*args)


def test_grpc_server_refuses_bad_tls_settings_at_construction(pki):
    cfg = Config()
    cfg.server.grpc_tls_cert = str(pki.server_cert)  # key missing
    with pytest.raises(TLSConfigError):
        GRPCServer(cfg, Publisher(""), HookManager())


def test_serves_over_tls_and_rejects_plaintext(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key)
    assert _call(_tls_channel(port, pki))
    with pytest.raises(grpc.RpcError) as e:
        _call(grpc.insecure_channel(f"localhost:{port}"))
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE


def test_client_that_does_not_trust_the_server_certificate_fails(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key)
    other_ca, _ = pki.other_ca()
    with pytest.raises(grpc.RpcError):
        _call(_tls_channel(port, pki, ca=other_ca))


def test_mutual_tls_requires_a_trusted_client_certificate(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key, client_ca=pki.ca_cert)
    with pytest.raises(grpc.RpcError):
        _call(_tls_channel(port, pki))
    assert _call(_tls_channel(port, pki, client=True))
    rc, rk = pki.rogue_client()
    with pytest.raises(grpc.RpcError):
        _call(_tls_channel(port, pki, cert=rc, key=rk))


def test_rotated_certificate_is_picked_up_without_a_restart(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key)
    before = _served_cert_digest(port)
    pki.rotate_server()
    assert _served_cert_digest(port) != before
    assert _call(_tls_channel(port, pki))


def test_broken_replacement_keeps_the_working_certificate(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key)
    before = _served_cert_digest(port)
    pki.server_cert.write_text("half-written")
    later = time.time() + 5
    os.utime(pki.server_cert, (later, later))
    assert _served_cert_digest(port) == before
