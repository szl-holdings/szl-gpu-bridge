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
import math
import os
import pathlib
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAPTOP = ROOT / "laptop"
if str(LAPTOP) not in sys.path:
    sys.path.insert(0, str(LAPTOP))

from frontier_contract import (  # noqa: E402
    canonicalize,
    derive_key_id,
    load_engine_pin_for_envelope,
    verify_envelope,
)
from nemo_v3_contract import (  # noqa: E402
    NEMO_V3_PAYLOAD_TYPE,
    expected_engine_key_id,
    quarantine_policy,
    validate_nemo_v3_spec,
)

SPEC_PATH = ROOT / "jobspecs" / "nemo-v3-20260722-reviewed.json"
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


@dataclass(frozen=True)
class QuarantineEvidence:
    present: bool
    valid: bool
    path: str | None = None
    statuses: tuple[str, ...] = ()
    error: str | None = None


class StatusError(RuntimeError):
    """A signed queue or receipt violated the governed attempt contract."""


def signer_canonicalize(value: Any) -> str:
    """Match the JavaScript signer canonicalizer for the reviewed jobspec.

    ``JSON.stringify`` emits integral floating-point values as integers (``1``
    rather than Python's ``1.0``). The queue verifier must compare against the
    bytes produced by ``cloud/sign-nemo-v3-job.mjs``, not a Python-specific JSON
    spelling.
    """

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StatusError("non-finite number cannot be signed")
        if value == 0:
            return "0"
        if value.is_integer():
            return str(int(value))
        rendered = json.dumps(value, allow_nan=False, separators=(",", ":"))
        return re.sub(r"e([+-])0+(\d+)", r"e\1\2", rendered)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(signer_canonicalize(item) for item in value) + "]"
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise StatusError("jobspec object keys must be strings")
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=False)}:{signer_canonicalize(value[key])}"
                for key in sorted(value)
            )
            + "}"
        )
    raise StatusError(f"unsupported jobspec value type {type(value).__name__}")


def resolve_spec_path(
    root: pathlib.Path,
    spec_path: pathlib.Path | str = SPEC_PATH,
) -> pathlib.Path:
    candidate = pathlib.Path(spec_path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(ROOT)
        except ValueError as exc:
            raise StatusError(
                "reviewed spec must be inside the bridge repository"
            ) from exc
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise StatusError("reviewed spec path escapes the bridge repository") from exc
    if resolved.parent.name != "jobspecs":
        raise StatusError("reviewed spec must be directly under jobspecs/")
    return resolved


def load_reviewed_spec(
    root: pathlib.Path = ROOT,
    spec_path: pathlib.Path | str = SPEC_PATH,
) -> dict[str, Any]:
    path = resolve_spec_path(root, spec_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_nemo_v3_spec(value)
    return value


def queue_path(spec: dict[str, Any], root: pathlib.Path = ROOT) -> pathlib.Path:
    return root / "queue" / "pending" / f"{spec['jobId']}.json"


def verify_queue(spec: dict[str, Any], root: pathlib.Path = ROOT) -> QueueEvidence:
    path = queue_path(spec, root)
    if not path.is_file():
        return QueueEvidence(False, False, path.relative_to(root).as_posix())
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        pin = load_engine_pin_for_envelope(root / "keys", envelope)
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
        if (
            "authorization" in signed_spec
            or (root / "keys" / "engine_keyring.json").is_file()
        ) and pin.get("keyId") != expected_engine_key_id(signed_spec):
            raise StatusError("signed queue uses the wrong engine authorization key")
        canonical = signer_canonicalize(spec).encode("utf-8")
        if exact_payload != canonical:
            raise StatusError(
                "signed queue bytes are not the canonical reviewed jobspec"
            )
        return QueueEvidence(
            True,
            True,
            path.relative_to(root).as_posix(),
            hashlib.sha256(exact_payload).hexdigest(),
            str(pin.get("keyId") or ""),
        )
    except Exception as exc:  # noqa: BLE001
        return QueueEvidence(
            True,
            False,
            path.relative_to(root).as_posix(),
            error=f"{type(exc).__name__}: {exc}",
        )


def verify_quarantine(
    spec: dict[str, Any],
    queue: QueueEvidence,
    root: pathlib.Path = ROOT,
) -> QuarantineEvidence:
    policy = quarantine_policy(spec)
    path = root / "queue" / "quarantine" / f"{spec['jobId']}.json"
    if policy is None:
        if path.is_file():
            return QuarantineEvidence(
                True,
                False,
                path.relative_to(root).as_posix(),
                error="unexpected quarantine record for dispatchable job",
            )
        return QuarantineEvidence(False, False)
    if not path.is_file():
        return QuarantineEvidence(
            False,
            False,
            error="required immutable quarantine record is missing",
        )

    relative = path.relative_to(root).as_posix()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "kind",
            "v",
            "jobId",
            "recordedAt",
            "status",
            "queuePath",
            "queueFileSha256",
            "signedPayloadSha256",
            "engineKeyId",
            "sourceRevision",
            "preserveEnvelope",
            "dispatchAuthorized",
            "replacement",
            "reason",
        }
        expected_execution_evidence = policy.get("execution_evidence")
        if expected_execution_evidence is not None:
            required.add("executionEvidence")
        statuses = tuple(policy["statuses"])
        expected_queue_path = f"queue/pending/{spec['jobId']}.json"
        queue_file = root / expected_queue_path
        replacement = record.get("replacement")
        expected_replacement = policy["replacement"]
        if (
            not isinstance(record, dict)
            or set(record) != required
            or record.get("kind") != "szl-nemo-v3-queue-quarantine"
            or record.get("v") != 1
            or record.get("jobId") != spec["jobId"]
            or tuple(record.get("status") or ()) != statuses
            or record.get("queuePath") != expected_queue_path
            or not queue_file.is_file()
            or record.get("queueFileSha256")
            != hashlib.sha256(queue_file.read_bytes()).hexdigest()
            or record.get("queueFileSha256") != policy["queue_file_sha256"]
            or not queue.valid
            or record.get("signedPayloadSha256") != queue.payload_sha256
            or record.get("signedPayloadSha256") != policy["payload_sha256"]
            or record.get("engineKeyId") != policy["engine_key_id"]
            or record.get("engineKeyId") != queue.engine_key_id
            or record.get("sourceRevision") != policy["source_revision"]
            or record.get("sourceRevision") != spec["source"]["revision"]
            or record.get("preserveEnvelope") is not True
            or record.get("dispatchAuthorized") is not False
            or replacement != expected_replacement
            or record.get("executionEvidence") != expected_execution_evidence
            or not isinstance(record.get("reason"), str)
            or not record["reason"].strip()
        ):
            raise StatusError("quarantine record does not bind immutable evidence")
        datetime.fromisoformat(str(record["recordedAt"]).replace("Z", "+00:00"))
        return QuarantineEvidence(True, True, relative, statuses)
    except Exception as exc:  # noqa: BLE001
        return QuarantineEvidence(
            True,
            False,
            relative,
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
            raise StatusError(
                f"laptop receipt signature verification failed: {exc}"
            ) from exc
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
                raise StatusError(
                    "receipt does not bind the exact signed queue payload"
                )
            payload_binding = "EXACT_SIGNED_PAYLOAD_SHA256"

        kind = receipt.get("kind")
        state = str(receipt.get("state") or receipt.get("verdict") or "UNKNOWN")
        if kind == "szl-nemo-v3-governed-training":
            effects = receipt.get("effects")
            if (
                not isinstance(effects, dict)
                or set(effects)
                != {
                    "candidate_uploaded",
                    "published",
                    "deployed",
                    "promoted",
                }
                or any(value is not False for value in effects.values())
            ):
                raise StatusError(
                    "receipt effects do not preserve the no-publication boundary"
                )
            if payload_binding != "EXACT_SIGNED_PAYLOAD_SHA256":
                raise StatusError(
                    "Nemo terminal receipt lacks exact queue payload binding"
                )
            if state == "QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW":
                evaluation = receipt.get("evaluation") or {}
                if (
                    evaluation.get("state") != "PASS"
                    or evaluation.get("pass_rate") != 1.0
                    or evaluation.get("degenerate") != 0
                    or evaluation.get("passes") != evaluation.get("rows")
                    or receipt.get("decision") != "SEPARATE_PROMOTION_REVIEW_REQUIRED"
                ):
                    raise StatusError(
                        "qualified receipt does not prove the preregistered all-pass gate"
                    )
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
    token = token.strip()
    if not token:
        raise StatusError(
            "HF_TOKEN is required to inspect the private authoritative receipt repository"
        )

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    if not api.repo_exists(RECEIPTS_REPO, repo_type="dataset"):
        raise StatusError(
            "the authoritative receipt repository is not visible with the configured "
            "HF_TOKEN"
        )
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
        token=token,
        force_download=True,
    )
    return found[0], json.loads(pathlib.Path(local).read_text(encoding="utf-8"))


def evaluate(
    *,
    root: pathlib.Path = ROOT,
    spec_path: pathlib.Path | str = SPEC_PATH,
    expected_laptop_key_id: str = "",
    hf_token: str = "",
    receipt_loader: Callable[[dict[str, Any], str], tuple[str, dict[str, Any]] | None]
    | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    reviewed_path = resolve_spec_path(root, spec_path)
    spec = load_reviewed_spec(root, spec_path)
    queue = verify_queue(spec, root)
    policy = quarantine_policy(spec)
    quarantine = verify_quarantine(spec, queue, root)
    now = now or datetime.now(timezone.utc)
    expires = datetime.fromisoformat(spec["expiresAt"].replace("Z", "+00:00"))
    loader = receipt_loader or default_receipt_loader

    receipt = ReceiptEvidence(False, False)
    if queue.valid and policy is None:
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

    if policy is not None and not quarantine.valid:
        status = "INVALID_QUARANTINE_RECORD"
    elif quarantine.valid:
        status = "QUARANTINED_NEVER_DISPATCH"
    elif queue.present and not queue.valid:
        status = "INVALID_QUEUE_ENVELOPE"
    elif not queue.present:
        status = (
            "EXPIRED_AWAITING_ENGINE_SIGNATURE"
            if now > expires
            else "AWAITING_ENGINE_SIGNATURE"
        )
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
        "QUARANTINED_NEVER_DISPATCH",
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
            "path": reviewed_path.relative_to(root.resolve()).as_posix(),
            "sha256": hashlib.sha256(
                signer_canonicalize(spec).encode("utf-8")
            ).hexdigest(),
            "source_revision": spec["source"]["revision"],
            "base_repo_id": spec["base"]["repoId"],
            "base_revision": spec["base"]["revision"],
            "expires_at": spec["expiresAt"],
            "candidate_publication_enabled": spec["outputs"]["publishCandidate"],
        },
        "queue": asdict(queue),
        "quarantine": asdict(quarantine),
        "receipt": asdict(receipt),
        "boundaries": [
            "This controller performs no training, signing, queue mutation, candidate upload, publication, deployment, or promotion.",
            "A plaintext reviewed jobspec is not executable; only the pinned engine key can authorize a queue envelope.",
            "A laptop receipt is not trusted until its derived keyId matches the explicitly enrolled owner-host keyId.",
            "A valid quarantine record preserves its signed envelope as evidence and is never dispatch authority.",
            "QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW is not a release or deployment claim.",
            "Receipts are attestations and not cryptographic proof of computation.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/nemo-v3-attempt-status.json")
    parser.add_argument(
        "--spec",
        default=SPEC_PATH.relative_to(ROOT).as_posix(),
        help="reviewed jobspec path under jobspecs/",
    )
    parser.add_argument(
        "--expect-laptop-keyid",
        default=os.environ.get("SZL_LAPTOP_RECEIPT_KEY_ID") or "",
    )
    args = parser.parse_args()
    report = evaluate(
        spec_path=args.spec,
        expected_laptop_key_id=args.expect_laptop_keyid,
        hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HF_ORG_TOKEN") or "",
    )
    path = pathlib.Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return (
        1
        if report["status"].startswith("INVALID_")
        or report["status"] == "RECEIPT_DISCOVERY_ERROR"
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
