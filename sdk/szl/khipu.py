"""khipu — signed, hash-chained receipts. Fail closed, tamper-evident.

A receipt is canonical JSON. Its signature is HMAC-SHA256 over the
canonical body hash, stored FULL length (never truncated). Receipts chain:
each carries the sha256 of the previous receipt's body, so any reorder,
edit, or deletion is detectable (Kamachikuna Law 5).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

KIND = "szl.receipt/v1"


def canonical(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace. The signing basis."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sign(body_sha256: str, secret: str) -> str:
    """Full-length HMAC signature. Never truncate a signature."""
    return hmac.new(secret.encode(), body_sha256.encode(), hashlib.sha256).hexdigest()


def verify(body_sha256: str, signature: str, secret: str) -> bool:
    if not isinstance(signature, str) or len(signature) != 64:
        return False
    return hmac.compare_digest(sign(body_sha256, secret), signature)


def mint(
    claim: dict[str, Any],
    secret: str,
    prev_body_sha256: str | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """Mint one signed receipt for a claim. The claim must already be
    honest: no receipt can make an unverified claim true (Law 7)."""
    body = {
        "kind": KIND,
        "ts": ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prev_receipt_sha256": prev_body_sha256,
        "claim": claim,
    }
    body_sha256 = sha256_hex(canonical(body))
    return {**body, "body_sha256": body_sha256, "signature": sign(body_sha256, secret)}


def check(receipt: dict[str, Any], secret: str) -> bool:
    """Verify one receipt's integrity and origin. Proves the receipt is
    intact and signed by the holder of `secret` — never that the claim
    inside is accurate."""
    if not isinstance(receipt, dict):
        return False
    body = {k: v for k, v in receipt.items() if k not in ("body_sha256", "signature")}
    if body.get("kind") != KIND:
        return False
    expected = sha256_hex(canonical(body))
    if receipt.get("body_sha256") != expected:
        return False
    return verify(expected, receipt.get("signature", ""), secret)


def check_chain(
    receipts: list[dict[str, Any]], secret: str, expected_head: str | None = None
) -> bool:
    """Verify a full chain: every receipt valid, every link intact.
    A prefix of the chain is internally valid — truncation is only
    detectable against `expected_head` (the latest known body_sha256).
    Pass it to prove completeness, not just integrity."""
    prev: str | None = None
    for r in receipts:
        if not check(r, secret):
            return False
        if r.get("prev_receipt_sha256") != prev:
            return False
        prev = r["body_sha256"]
    if expected_head is not None and prev != expected_head:
        return False  # truncated chain
    return True
