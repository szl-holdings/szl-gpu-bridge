"""Ed25519 envelopes. Every receipt is {payload, payload_sha256, sig: {alg, public_key, signature}}.

Keys here are EPHEMERAL and generated in memory for local tests and demos. The private key is
never written to disk. A production deployment swaps Signer for Amaru's key or Sigstore keyless
(see README). If `cryptography` is missing, signing is impossible and verification FAILS closed.
"""
from __future__ import annotations

import base64
from typing import Any

from .canonical import ProofError, canonical_bytes, sha256_bytes

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (Ed25519PrivateKey,
                                                                   Ed25519PublicKey)
    HAVE_CRYPTO = True
except Exception:  # pragma: no cover
    HAVE_CRYPTO = False

ALG = "ed25519"


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    try:
        return base64.b64decode(s.encode("ascii"), validate=True)
    except Exception:
        raise ProofError("bad base64 in signature envelope")


def key_fingerprint(public_key_b64: str) -> str:
    return "ed25519:" + sha256_bytes(_unb64(public_key_b64))[:16]


class Signer:
    """Ephemeral in-memory ed25519 signer (test/demo only)."""

    def __init__(self, role: str = "test-ephemeral") -> None:
        if not HAVE_CRYPTO:
            raise ProofError("cryptography not installed: cannot sign")
        self._sk = Ed25519PrivateKey.generate()
        self.role = role
        raw = self._sk.public_key().public_bytes(serialization.Encoding.Raw,
                                                 serialization.PublicFormat.Raw)
        self.public_key_b64 = _b64(raw)
        self.fingerprint = key_fingerprint(self.public_key_b64)

    def envelope(self, kind: str, payload: dict) -> dict:
        body = {"kind": kind, "payload": payload}
        msg = canonical_bytes(body)
        return {"kind": kind, "payload": payload, "payload_sha256": sha256_bytes(msg),
                "sig": {"alg": ALG, "public_key": self.public_key_b64, "role": self.role,
                        "signature": _b64(self._sk.sign(msg))}}


def open_envelope(env: Any, kind: str, trusted: set[str] | None = None) -> tuple[dict, str]:
    """Verify an envelope and return (payload, signer_fingerprint). Raises ProofError on ANY issue."""
    if not isinstance(env, dict) or env.get("kind") != kind:
        raise ProofError(f"not a {kind} envelope")
    payload = env.get("payload")
    if not isinstance(payload, dict):
        raise ProofError(f"{kind}: payload missing")
    msg = canonical_bytes({"kind": kind, "payload": payload})
    if env.get("payload_sha256") != sha256_bytes(msg):
        raise ProofError(f"{kind}: payload_sha256 mismatch (payload edited after signing)")
    sig = env.get("sig")
    if not isinstance(sig, dict) or sig.get("alg") != ALG:
        raise ProofError(f"{kind}: unsigned or unsupported signature algorithm")
    if not HAVE_CRYPTO:
        raise ProofError("cryptography not installed: signature cannot be verified (fail closed)")
    pk_b64 = sig.get("public_key", "")
    try:
        pk = Ed25519PublicKey.from_public_bytes(_unb64(pk_b64))
        pk.verify(_unb64(sig.get("signature", "")), msg)
    except InvalidSignature:
        raise ProofError(f"{kind}: signature INVALID")
    except ProofError:
        raise
    except Exception as e:
        raise ProofError(f"{kind}: signature unverifiable ({e.__class__.__name__})")
    fp = key_fingerprint(pk_b64)
    if trusted is not None and fp not in trusted:
        raise ProofError(f"{kind}: signer {fp} is not in the trusted key set")
    return payload, fp
