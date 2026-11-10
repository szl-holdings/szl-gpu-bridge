"""attest — attestation receipts: bind hardware/model/input claims to khipu.

Evolution of the receipts layer toward verifiable inference. A receipt of
kind szl.attestation/v1 binds: the committed model weights hash, a hardware
attestation digest (TEE quote, e.g. NVIDIA Confidential Compute on
H100/H200/B200, or a Proof-of-Sampling challenge), and the input/output
digests — so "this output came from this model on this attested hardware"
becomes a checkable claim, not a trust assumption.

Honesty: the khipu signature proves integrity/origin of the *attestation
record*. It does not itself verify the TEE quote — that check happens against
the silicon vendor's root cert out of band. We record the quote digest and
the verification status honestly: UNVERIFIED until checked (Law 6/8).
"""
from __future__ import annotations

from typing import Any

from . import khipu

KIND = "szl.attestation/v1"

ATTESTATION_SOURCES = ("tee-quote", "proof-of-sampling", "none")


def mint_attestation(
    *,
    model_sha256: str,
    input_sha256: str,
    output_sha256: str,
    attestation_source: str,
    attestation_digest: str | None,
    tee_quote_verified: bool,
    secret: str,
    prev_body_sha256: str | None = None,
) -> dict[str, Any]:
    """Mint a signed attestation receipt. Fail closed: an unverified TEE
    quote is recorded as such, never upgraded to VERIFIED (Law 8)."""
    if attestation_source not in ATTESTATION_SOURCES:
        raise ValueError(f"unknown attestation_source: {attestation_source}")
    if attestation_source != "none" and not attestation_digest:
        raise ValueError("attestation_digest required unless source is 'none'")
    claim = {
        "kind": KIND,
        "model_sha256": model_sha256,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "attestation": {
            "source": attestation_source,
            "digest": attestation_digest,
            "teeQuoteVerified": bool(tee_quote_verified) if attestation_source == "tee-quote" else None,
        },
        "note": "khipu signature proves integrity/origin of this record; TEE quote validity is verified against the vendor root cert out of band.",
    }
    return khipu.mint(claim, secret, prev_body_sha256=prev_body_sha256)


def attestation_state(receipt: dict[str, Any], secret: str) -> str:
    """Honest state: VERIFIED only if the receipt verifies AND the embedded
    attestation was verified; otherwise UNAVAILABLE — never a claim."""
    if not khipu.check(receipt, secret):
        return "UNAVAILABLE"
    att = receipt.get("claim", {}).get("attestation", {})
    if att.get("source") == "tee-quote" and att.get("teeQuoteVerified") is not True:
        return "UNAVAILABLE"
    if att.get("source") == "none":
        return "UNATTESTED"
    return "VERIFIED"
