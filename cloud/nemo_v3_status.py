#!/usr/bin/env python3
"""Evaluate the one reviewed SZL-Nemo v3 attempt without running it.

The controller verifies the plaintext reviewed spec, an optional engine-signed
queue envelope, and an optional laptop-signed terminal receipt. It performs no
training, signing, queue mutation, candidate upload, publication, deployment, or
promotion. Missing evidence remains an explicit waiting state.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAPTOP = ROOT / "laptop"
if str(LAPTOP) not in sys.path:
    sys.path.insert(0, str(LAPTOP))

from frontier_contract import canonicalize, derive_key_id, verify_envelope  # noqa: E402
from nemo_v3_contract import NEMO_V3_PAYLOAD_TYPE, validate_nemo_v3_spec  # noqa: E402

SPEC_PATH = ROOT / "jobspecs" / "nemo-v3-20260722-reviewed.json"
ENGINE_PIN_PATH = ROOT / "keys" / "engine_pubkey.json"
RECEIPTS_REPO = "SZLHOLDINGS/szl-training-receipts"
TERMINAL_FILENAMES = (
    "nemo-v3-qualified.signed.json",
    "nemo-v3-terminal.signed.json",
    "blocked_receipt.signed.json",
)


@dataclass(frozen=True)
class QueueEvidence:
    present: bool
    valid: bool
    path: str
    payload_sha256: str | None = None
    engine_key_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ReceiptEvidence:
    present: bool
    valid: bool
    path: str | None = None
    observed_key_id: str | None = None
    identity_pinned: bool = False
    body_sha256: str | None = None
    state: str | None = None
    payload_binding: str | None = None
    error: str | None = None


class StatusError(RuntimeError):
    """A signed queue or receipt violated the governed attempt contract."""


def load_reviewed_spec(root: pathlib.Path = ROOT) -> dict[str, Any]:
    path = root / SPEC_PATH.relative_to(ROOT)
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_nemo_v3_spec(value)
    return value


def queue_path(spec: dict[str, Any], root: pathlib.Path = ROOT) -> pathlib.Path:
    return root / "queue" / "pending" / f"{spec['jobId']}.json"


def verify_queue(spec: dict[str, Any], root: pathlib.Path = ROOT) -> QueueEvidence:
    path = queue_path(spec, root)
    if not path.is_file():
        return QueueEvidence(False, False, str(path.relative_to(root)))
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        pin = json.loads((root / "keys" / "engine_pubkey.json").read_text(encoding="utf-8"))
        signed_spec, exact_payload, payload_type = verify_envelope(
            envelope,
            pin,
            allowed_payload_types=(NEMO_V3_PAYLOAD_TYPE,),
        )
        validate_nemo_v3_spec(signed_spec)
        if payload_type != NEMO_V3_PAYLOAD_TYPE:
            raise StatusError(f"unexpected payload type {payload_type!r}")
        if signed_spec != spec:
            raise StatusError("signed queue payload differs from the reviewed jobspec")
        canonical = canonicalize(spec).encode("utf-8")
        if exact_payload != canonical:
            raise StatusError("signed queue bytes are not the canonical reviewed jobspec")
        return QueueEvidence(
            True,
            True,
            str(path.relative_to(root)),
            hashlib.sha256(exact_payload).hexdigest(),
            str(pin.get("keyId") or ""),
        )
    except Exception as exc:  # noqa: BLE001
        return QueueEvidence(
            True,
            False,
            str(path.relative_to(root)),
            error=f"{type(exc).__name__}: {exc}",
        )


def _decode(value: Any, field: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise StatusError(f"{field} must be non-empty base64")
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise StatusError(f"{field} is invalid base64: {exc}") from exc


def verify_receipt(
    signed: dict[str, Any],
    *,
    spec: dict[str, Any],
    queue: QueueEvidence,
    expected_laptop_key_id: str,
    path: str,
) -> ReceiptEvidence:
    try:
        if not queue.valid or not queue.payload_sha256:
            raise StatusError("a trusted queue envelope is required before any receipt")
        spki = _decode(signed.get("publicKeySpkiBase64"), "publicKeySpkiBase64")
        observed_key_id = derive_key_id(spki)
        if signed.get("keyId") != observed_key_id:
            raise StatusError("receipt keyId differs from its embedded public key")
        expected = expected_laptop_key_id.strip().lower()
        identity_pinned = bool(expected)
        if expected and observed_key_id != expected:
            raise StatusError(
                f"receipt keyId {observed_key_id} differs from enrolled laptop key {expected}"
            )
        if signed.get("scheme") != "ed25519-over-exact-bytes-v2":
            raise StatusError("unsupported laptop receipt signing scheme")
        body = _decode(signed.get("bodyBase64"), "bodyBase64")
        signature = _decode(signed.get("signatureBase64"), "signatureBase64")
        try:
            from nacl.signing import VerifyKey

            if len(spki) != 44:
                raise StatusError(f"unexpected Ed25519 SPKI length {len(spki)}")
            VerifyKey(spki[-32:]).verify(body, signature)
        except StatusError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StatusError(f"laptop receipt signature verification failed: {exc}") from exc
        receipt = json.loads(body)
        if not isinstance(receipt, dict):
            raise StatusError("signed receipt body is not an object")
        if canonicalize(receipt) != canonicalize(signed.get("receipt")):
            raise StatusError("display receipt differs from the exact signed bytes")
        if receipt.get("jobId") != spec["jobId"]:
            raise StatusError("receipt jobId differs from the reviewed attempt")

        payload_binding = "JOB_ID_ONLY"
        signed_job_sha = receipt.get("signed_job_payload_sha256")
        if signed_job_sha is not None:
            if signed_job_sha != queue.payload_sha256:
                raise StatusError("receipt does not bind the exact signed queue payload")
            payload_binding = "EXACT_SIGNED_PAYLOAD_SHA256"

        kind = receipt.get("kind")
        state = str(receipt.get("state") or receipt.get("verdict") or "UNKNOWN")
        if kind == "szl-nemo-v3-governed-training":
            effects = receipt.get("effects")
            if not isinstance(effects, dict) or set(effects) != {
                "candidate_uploaded",
                "published",
                "deployed",
                "promoted",
            } or any(value is not False for value in effects.values()):
                raise StatusError("receipt effects do not preserve the no-publication boundary")
            if payload_binding != "EXACT_SIGNED_PAYLOAD_SHA256":
                raise StatusError("Nemo terminal receipt lacks exact queue payload binding")
            if state == "QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW":
                evaluation = receipt.get("evaluation") or {}
                if (
                    evaluation.get("state") != "PASS"
                    or evaluation.get("pass_rate") != 1.0
                    or evaluation.get("degenerate") != 0
                    or evaluation.get("passes") != evaluation.get("rows")
                    or receipt.get("decision") != "SEPARATE_PROMOTION_REVIEW_REQUIRED"
                ):
                    raise StatusError("qualified receipt does not prove the preregistered all-pass gate")
            elif state != "EVALUATION_FAILED_NOT_PROMOTED_NOT_SIGNED":
                raise StatusError(f"unsupported Nemo terminal state {state!r}")
        elif kind == "szl-frontier-training-blocked":
            if state != "BLOCKED":
                raise StatusError("generic blocked receipt lacks verdict=BLOCKED")
        else:
            raise StatusError(f"unsupported terminal receipt kind {kind!r}")

        if not identity_pinned:
            return ReceiptEvidence(
                True,
                False,
                path,
                observed_key_id,
                False,
                hashlib.sha256(body).hexdigest(),
                state,
                payload_binding,
                "laptop receipt is cryptographically valid but its keyId is not enrolled",
            )
        return ReceiptEvidence(
            True,
            True,
            path,
            observed_key_id,
            True,
            hashlib.sha256(body).hexdigest(),
            state,
            payload_binding,
        )
    except Exception as exc:  # noqa: BLE001
        return ReceiptEvidence(
            True,
            False,
            path,
            error=f"{type(exc).__name__}: {exc}",
        )


def default_receipt_loader(
    spec: dict[str, Any], token: str
) -> tuple[str, dict[str, Any]] | None:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token or None)
    if not api.repo_exists(RECEIPTS_REPO, repo_type="dataset"):
        return None
    files = set(api.list_repo_files(RECEIPTS_REPO, repo_type="dataset"))
    prefix = f"{spec['jobId']}/"
    found = [prefix + name for name in TERMINAL_FILENAMES if prefix + name in files]
    if len(found) > 1:
        raise StatusError(f"multiple terminal receipts exist for one attempt: {found}")
    if not found:
        return None
    local = hf_hub_download(
        repo_id=RECEIPTS_REPO,
        repo_type="dataset",
        filename=found[0],
        token=token or None,
        force_download=True,
    )
    return found[0], json.loads(pathlib.Path(local).read_text(encoding="utf-8"))


def evaluate(
    *,
    root: pathlib.Path = ROOT,
    expected_laptop_key_id: str = "",
    hf_token: str = "",
    receipt_loader: Callable[[dict[str, Any], str], tuple[str, dict[str, Any]] | None]
    | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    spec = load_reviewed_spec(root)
    queue = verify_queue(spec, root)
    now = now or datetime.now(timezone.utc)
    expires = datetime.fromisoformat(spec["expiresAt"].replace("Z", "+00:00"))
    loader = receipt_loader or default_receipt_loader

    receipt = ReceiptEvidence(False, False)
    if queue.valid:
        try:
            loaded = loader(spec, hf_token)
        except Exception as exc:  # noqa: BLE001
            receipt = ReceiptEvidence(
                False,
                False,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            if loaded is not None:
                receipt = verify_receipt(
                    loaded[1],
                    spec=spec,
                    queue=queue,
                    expected_laptop_key_id=expected_laptop_key_id,
                    path=loaded[0],
                )

    if queue.present and not queue.valid:
        status = "INVALID_QUEUE_ENVELOPE"
    elif not queue.present:
        status = "EXPIRED_AWAITING_ENGINE_SIGNATURE" if now > expires else "AWAITING_ENGINE_SIGNATURE"
    elif receipt.error and not receipt.present:
        status = "RECEIPT_DISCOVERY_ERROR"
    elif not receipt.present:
        status = "QUEUED_AWAITING_GPU_RECEIPT"
    elif not receipt.valid and not receipt.identity_pinned and receipt.observed_key_id:
        status = "AWAITING_LAPTOP_RECEIPT_KEY_ENROLLMENT"
    elif not receipt.valid:
        status = "INVALID_TERMINAL_RECEIPT"
    elif receipt.state == "QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW":
        status = receipt.state
    else:
        status = "TERMINAL_FAILURE"

    terminal = status in {
        "QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW",
        "TERMINAL_FAILURE",
    }
    report = {
        "schema": "szl.nemo-v3-attempt-status/v1",
        "generated_at": now.isoformat(),
        "job_id": spec["jobId"],
        "status": status,
        "terminal": terminal,
        "reviewed_spec": {
            "path": str((root / SPEC_PATH.relative_to(ROOT)).relative_to(root)),
            "sha256": hashlib.sha256(canonicalize(spec).encode("utf-8")).hexdigest(),
            "source_revision": spec["source"]["revision"],
            "base_repo_id": spec["base"]["repoId"],
            "base_revision": spec["base"]["revision"],
            "expires_at": spec["expiresAt"],
            "candidate_publication_enabled": spec["outputs"]["publishCandidate"],
        },
        "queue": asdict(queue),
        "receipt": asdict(receipt),
        "boundaries": [
            "This controller performs no training, signing, queue mutation, candidate upload, publication, deployment, or promotion.",
            "A plaintext reviewed jobspec is not executable; only the pinned engine key can authorize a queue envelope.",
            "A laptop receipt is not trusted until its derived keyId matches the explicitly enrolled owner-host keyId.",
            "QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW is not a release or deployment claim.",
            "Receipts are attestations and not cryptographic proof of computation.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/nemo-v3-attempt-status.json")
    parser.add_argument(
        "--expect-laptop-keyid",
        default=os.environ.get("SZL_LAPTOP_RECEIPT_KEY_ID") or "",
    )
    args = parser.parse_args()
    report = evaluate(
        expected_laptop_key_id=args.expect_laptop_keyid,
        hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HF_ORG_TOKEN") or "",
    )
    path = pathlib.Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["status"].startswith("INVALID_") or report["status"] == "RECEIPT_DISCOVERY_ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
