"""Scoped exceptions in Anchor: the verifier can exempt one bias category for one tenant."""
from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from anchor.exceptions import Client, ExceptionsError, parse_public_keys, verify
from anchor.models import UserContext, VerifyRequest
from anchor.verifier import Verifier, set_exceptions

DOMAIN = b"bastion-exceptions-v1\n"
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "exceptions_go_envelope.json")
BIASED = "Women are too emotional to lead, and men are better leaders."


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def _keypair():
    sk = Ed25519PrivateKey.generate()
    pub = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sk, base64.b64encode(pub).decode()


def _signed(sk, exc: dict) -> dict:
    payload = json.dumps({"kind": "update", "issued_at": _iso(datetime.now(timezone.utc)), "exceptions": [exc]}).encode()
    return {"payload": base64.b64encode(payload).decode(), "sig": base64.b64encode(sk.sign(DOMAIN + payload)).decode()}


def _exc(version=1, tenant="acme", revoked=False, category="gender", modules=("anchor",)):
    now = datetime.now(timezone.utc)
    e = {
        "id": "exc-bias", "version": version, "rule": f"anchor.bias:{category}", "modules": list(modules),
        "tenant_id": tenant, "justification": "research on bias tooling", "created_by": "admin",
        "created_at": _iso(now), "expires_at": _iso(now + timedelta(hours=1)),
    }
    if revoked:
        e["revoked"] = True
    return e


@pytest.fixture(autouse=True)
def _reset():
    yield
    set_exceptions(None)


def _verify(tenant="acme"):
    return Verifier().verify(VerifyRequest(request_id="r1", tenant_id=tenant, llm_response=BIASED,
                                           user=UserContext(user_id="alice", tenant_id=tenant)))


def test_bias_flagged_without_exceptions():
    resp = _verify()
    assert resp.status in ("WARNING", "CRITICAL")
    assert any(c.category == "gender" for c in resp.bias_analysis.categories)
    assert resp.exceptions_applied == []


def test_exception_exempts_category_for_its_tenant_only():
    sk, pub = _keypair()
    client = Client("anchor", pub)
    client.apply(_signed(sk, _exc()))
    set_exceptions(client)

    resp = _verify("acme")
    assert not any(c.category == "gender" for c in resp.bias_analysis.categories)
    assert resp.exceptions_applied == ["exc-bias"]  # recorded because the category really showed bias
    assert resp.status == "PASSED"

    other = _verify("globex")
    assert any(c.category == "gender" for c in other.bias_analysis.categories)  # still enforced
    assert other.exceptions_applied == []


def test_other_categories_stay_enforced():
    sk, pub = _keypair()
    client = Client("anchor", pub)
    client.apply(_signed(sk, _exc(category="age")))  # exempts age, not gender
    set_exceptions(client)
    assert any(c.category == "gender" for c in _verify().bias_analysis.categories)


def test_use_is_not_recorded_when_nothing_to_exempt():
    sk, pub = _keypair()
    client = Client("anchor", pub)
    client.apply(_signed(sk, _exc()))
    set_exceptions(client)
    resp = Verifier().verify(VerifyRequest(request_id="r2", tenant_id="acme", llm_response="The sky is blue."))
    assert resp.exceptions_applied == []


def test_revocation_restores_enforcement():
    sk, pub = _keypair()
    client = Client("anchor", pub)
    client.apply(_signed(sk, _exc(version=1)))
    set_exceptions(client)
    assert _verify().exceptions_applied == ["exc-bias"]
    client.apply(_signed(sk, _exc(version=2, revoked=True)))
    assert any(c.category == "gender" for c in _verify().bias_analysis.categories)


def test_exception_for_another_module_is_not_honoured():
    sk, pub = _keypair()
    client = Client("anchor", pub)
    client.apply(_signed(sk, _exc(modules=("navigator",))))
    set_exceptions(client)
    assert any(c.category == "gender" for c in _verify().bias_analysis.categories)


def test_forged_exception_is_rejected():
    _, pub = _keypair()
    attacker, _ = _keypair()
    client = Client("anchor", pub)
    with pytest.raises(ExceptionsError):
        client.apply(_signed(attacker, _exc()))
    set_exceptions(client)
    assert any(c.category == "gender" for c in _verify().bias_analysis.categories)


def test_works_standalone_without_any_exception_store():
    set_exceptions(None)
    assert any(c.category == "gender" for c in _verify().bias_analysis.categories)


def test_accepts_an_envelope_signed_by_the_go_implementation():
    with open(FIXTURE) as fh:
        fx = json.load(fh)
    payload = verify(parse_public_keys(fx["public_key"]), fx["envelope"])
    ids = {e["id"]: e for e in payload["exceptions"]}
    assert ids[fx["revoked_id"]]["rule"] == "anchor.bias:gender" and ids[fx["revoked_id"]]["revoked"] is True
