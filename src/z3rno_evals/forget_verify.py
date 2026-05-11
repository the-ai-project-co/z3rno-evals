"""Phase F slice 5 — forget-with-proof verifier.

Standalone verifier: given a certificate JSON (as returned by
``GET /v1/forget/{cert_id}``) and an operator's PEM public key, check
that the ed25519 signature was made over the canonical payload.

Optionally, given a ``--content-hashes`` JSON mapping
``{memory_id: sha256hex}``, also recomputes the Merkle root and
asserts equality with the cert. That catches the case where the cert
is internally valid but the underlying Memos were tampered with
post-erasure (or were never deleted at all and the operator silently
ran a no-op forget()).

Pure-Python; ``z3rno-evals`` deliberately doesn't import z3rno-core
so the verifier can be distributed as part of the eval kit to
auditors who don't have the server source on their machines.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


@dataclass(frozen=True)
class VerifyVerdict:
    signature_ok: bool
    merkle_ok: bool | None  # None when content hashes weren't supplied
    reason: str = ""

    @property
    def overall_ok(self) -> bool:
        if not self.signature_ok:
            return False
        return self.merkle_ok in (True, None)


def _canonical_payload(cert: dict[str, Any]) -> bytes:
    """Mirror of ``z3rno_core.forget_proof.canonical_payload``.

    Deliberately re-implemented so this CLI has zero z3rno-core
    dependency. Any drift in the core key set must be mirrored here
    — the unit tests pin both sides to identical bytes.
    """
    body = {
        "cert_id": str(cert["cert_id"]),
        "org_id": str(cert["org_id"]),
        "memory_ids": sorted(str(m) for m in cert["memory_ids"]),
        "merkle_root": cert["merkle_root_b64"],
        "signer_key_id": cert["signer_key_id"],
        "signed_at": cert["signed_at"],
        "hard_delete": bool(cert.get("hard_delete", False)),
        "audit_seq_start": cert.get("audit_seq_start"),
        "audit_seq_end": cert.get("audit_seq_end"),
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _build_merkle_root(memory_ids: list[str], content_hashes: dict[str, str]) -> bytes:
    """Mirror of ``z3rno_core.forget_proof.build_merkle_root`` for the
    case where the auditor knows the original ``content_hash`` for
    every covered Memo. Audit hashes default to ``-`` since the
    auditor typically doesn't have audit-row access — that matches
    the engine's behavior when audit hashes aren't supplied."""
    leaves = [
        hashlib.sha256(f"{mid}:{content_hashes.get(mid, '-')}:-".encode()).digest()
        for mid in memory_ids
    ]
    if not leaves:
        raise ValueError("Merkle tree requires at least one leaf")
    level: list[bytes] = sorted(leaves)
    while len(level) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(hashlib.sha256(left + right).digest())
        level = nxt
    return level[0]


def verify(
    cert: dict[str, Any],
    public_key_pem: bytes,
    content_hashes: dict[str, str] | None = None,
) -> VerifyVerdict:
    pk = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(pk, Ed25519PublicKey):
        return VerifyVerdict(False, None, reason="public key is not Ed25519")

    signature = base64.b64decode(cert["signature_b64"])
    payload = _canonical_payload(cert)
    try:
        pk.verify(signature, payload)
        sig_ok = True
    except InvalidSignature:
        return VerifyVerdict(False, None, reason="ed25519 signature invalid")

    merkle_ok: bool | None = None
    if content_hashes is not None:
        recomputed = _build_merkle_root(
            sorted(str(m) for m in cert["memory_ids"]),
            content_hashes,
        )
        recomputed_b64 = base64.b64encode(recomputed).decode("ascii")
        merkle_ok = recomputed_b64 == cert["merkle_root_b64"]

    return VerifyVerdict(sig_ok, merkle_ok)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="z3rno-evals-verify-forget",
        description="Verify a Z3rno forget-with-proof certificate.",
    )
    parser.add_argument("--cert", required=True, type=Path, help="Path to certificate JSON")
    parser.add_argument(
        "--public-key",
        required=True,
        type=Path,
        help="Path to PEM-encoded ed25519 public key",
    )
    parser.add_argument(
        "--content-hashes",
        type=Path,
        default=None,
        help="Optional JSON file: {memory_id: sha256hex} for Merkle re-check",
    )
    args = parser.parse_args(argv)

    cert = json.loads(args.cert.read_text())
    pk_pem = args.public_key.read_bytes()
    content_hashes = json.loads(args.content_hashes.read_text()) if args.content_hashes else None

    verdict = verify(cert, pk_pem, content_hashes)
    print(  # noqa: T201 — CLI output is the whole point
        json.dumps(
            {
                "ok": verdict.overall_ok,
                "signature_ok": verdict.signature_ok,
                "merkle_ok": verdict.merkle_ok,
                "reason": verdict.reason,
            },
            indent=2,
        )
    )
    return 0 if verdict.overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
