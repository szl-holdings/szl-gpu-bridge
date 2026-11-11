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
import math
import re
import time
from typing import Any

KIND = "szl.receipt/v1"


def canonical(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace. The signing basis."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sign(body_sha256: str, secret: str) -> str:
    """Full-length HMAC signature. Never truncate a signature."""
    if not isinstance(secret, str) or not secret:
        raise ValueError("A nonempty signing secret is required")
    return hmac.new(secret.encode(), body_sha256.encode(), hashlib.sha256).hexdigest()


def verify(body_sha256: str, signature: str, secret: str) -> bool:
    if not isinstance(secret, str) or not secret:
        return False
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
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
    if not isinstance(claim, dict):
        raise ValueError("A receipt claim must be a mapping")
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
    inside is accurate. An unsigned or empty receipt NEVER verifies."""
    if not isinstance(receipt, dict) or not isinstance(secret, str) or not secret:
        return False
    if not isinstance(receipt.get("claim"), dict):
        return False
    previous = receipt.get("prev_receipt_sha256")
    if "prev_receipt_sha256" not in receipt or (
        previous is not None
        and (
            not isinstance(previous, str) or not re.fullmatch(r"[0-9a-f]{64}", previous)
        )
    ):
        return False
    if not isinstance(receipt.get("ts"), str) or not receipt["ts"]:
        return False
    import datetime

    try:
        dt = datetime.datetime.fromisoformat(str(receipt["ts"]).replace("Z", "+00:00"))
        if dt.tzinfo is None:  # timezone-free = unverifiable
            return False
        if dt > datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            minutes=5
        ):
            return False  # future timestamp = unverifiable
    except (ValueError, TypeError):
        return False
    body = {k: v for k, v in receipt.items() if k not in ("body_sha256", "signature")}
    if body.get("kind") != KIND:
        return False
    try:
        expected = sha256_hex(canonical(body))
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False
    if receipt.get("body_sha256") != expected:
        return False
    return verify(expected, receipt.get("signature", ""), secret)


def check_chain(
    receipts: list[dict[str, Any]],
    secret: str,
    expected_head: str | None = None,
    max_age_days: float | None = None,
) -> bool:
    """Verify a full chain: every receipt valid, every link intact.
    A prefix of the chain is internally valid — truncation is only
    detectable against `expected_head` (the latest known body_sha256).
    Pass it to prove completeness, not just integrity.

    `max_age_days` implements the freshness rule: a chain whose newest
    receipt is older than the window is STALE, not green. A stale record
    that still verifies is worse than no record, because it is believed.
    (Vidal-Gadea Lab roadmap, Nov 2026: cached values sat untouched 57
    days and a program read them with no error and no warning.)"""
    if not isinstance(receipts, list):
        return False
    if max_age_days is not None:
        if type(max_age_days) not in (int, float) or max_age_days < 0:
            return False
        try:
            if not math.isfinite(max_age_days) or not receipts:
                return False
        except OverflowError:
            return False
    prev: str | None = None
    newest_ts: str | None = None
    for r in receipts:
        if not check(r, secret):
            return False
        if r.get("prev_receipt_sha256") != prev:
            return False
        prev = r["body_sha256"]
        newest_ts = r.get("ts") or newest_ts
    if expected_head is not None and prev != expected_head:
        return False  # truncated chain
    if max_age_days is not None and newest_ts is not None:
        import datetime

        try:
            dt = datetime.datetime.fromisoformat(str(newest_ts).replace("Z", "+00:00"))
        except ValueError:
            return False  # unreadable timestamp: cannot prove freshness
        age = (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds()
        if age > max_age_days * 86400:
            return False  # stale chain: valid, but no longer current
    return True
