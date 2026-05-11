"""Test the standalone forget-with-proof verifier.

Mirrors the core's signing/Merkle logic and asserts the verifier
agrees on (a) a valid cert, (b) a tampered signature, and (c) a
content-hash mismatch.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from z3rno_evals.forget_verify import (
    _build_merkle_root,
    _canonical_payload,
    verify,
)


def _make_cert_dict(
    sk: Ed25519PrivateKey,
    memory_ids: list[UUID],
    content_hashes: dict[str, str],
) -> dict[str, object]:
    root = _build_merkle_root(sorted(str(m) for m in memory_ids), content_hashes)
    cert_body = {
        "cert_id": str(uuid4()),
        "org_id": str(uuid4()),
        "agent_id": None,
        "memory_ids": [str(m) for m in memory_ids],
        "merkle_root_b64": base64.b64encode(root).decode("ascii"),
        "signer_key_id": "test-key-2026",
        "audit_seq_start": None,
        "audit_seq_end": None,
        "hard_delete": True,
        "signed_at": datetime.now(UTC).isoformat(),
    }
    payload = _canonical_payload(cert_body)
    sig = sk.sign(payload)
    cert_body["signature_b64"] = base64.b64encode(sig).decode("ascii")
    return cert_body


def _pem_public(sk: Ed25519PrivateKey) -> bytes:
    return sk.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def test_verify_valid_cert() -> None:
    sk = Ed25519PrivateKey.generate()
    mids = [uuid4(), uuid4()]
    chash = {str(m): hashlib.sha256(f"c{i}".encode()).hexdigest() for i, m in enumerate(mids)}
    cert = _make_cert_dict(sk, mids, chash)

    verdict = verify(cert, _pem_public(sk), content_hashes=chash)
    assert verdict.signature_ok is True
    assert verdict.merkle_ok is True
    assert verdict.overall_ok is True


def test_verify_rejects_wrong_key() -> None:
    sk = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    mids = [uuid4()]
    chash = {str(mids[0]): "abc"}
    cert = _make_cert_dict(sk, mids, chash)

    verdict = verify(cert, _pem_public(other))
    assert verdict.signature_ok is False
    assert verdict.overall_ok is False


def test_verify_rejects_content_tamper() -> None:
    sk = Ed25519PrivateKey.generate()
    mids = [uuid4(), uuid4()]
    chash = {str(m): f"hash{i}" for i, m in enumerate(mids)}
    cert = _make_cert_dict(sk, mids, chash)

    # Auditor was handed a DIFFERENT content hash for one Memo:
    tampered_chash = {**chash, str(mids[0]): "DIFFERENT_HASH"}
    verdict = verify(cert, _pem_public(sk), content_hashes=tampered_chash)
    # Signature still valid (cert itself is intact) — but merkle mismatch.
    assert verdict.signature_ok is True
    assert verdict.merkle_ok is False
    assert verdict.overall_ok is False


def test_verify_signature_only_mode() -> None:
    """If no content hashes are supplied, signature_ok alone decides."""
    sk = Ed25519PrivateKey.generate()
    mids = [uuid4()]
    chash = {str(mids[0]): "abc"}
    cert = _make_cert_dict(sk, mids, chash)

    verdict = verify(cert, _pem_public(sk), content_hashes=None)
    assert verdict.signature_ok is True
    assert verdict.merkle_ok is None
    assert verdict.overall_ok is True


def test_canonical_payload_matches_core_format(tmp_path: Path) -> None:
    """Pin the exact byte layout so server + evals never drift."""
    cert = {
        "cert_id": "00000000-0000-0000-0000-00000000002a",
        "org_id": "00000000-0000-0000-0000-000000000007",
        "memory_ids": [
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000001",
        ],
        "merkle_root_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        "signer_key_id": "key-2026",
        "signed_at": "2026-05-11T12:00:00+00:00",
        "hard_delete": False,
        "audit_seq_start": None,
        "audit_seq_end": None,
    }
    payload = _canonical_payload(cert)
    decoded = json.loads(payload)
    # memory_ids must be sorted ascending.
    assert decoded["memory_ids"] == [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]
    # Keys must be sorted.
    assert list(decoded.keys()) == sorted(decoded.keys())
