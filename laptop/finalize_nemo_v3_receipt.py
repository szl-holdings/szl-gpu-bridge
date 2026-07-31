#!/usr/bin/env python3
"""Sign, upload, and immutably read back one isolated Nemo v3 receipt intent."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import re
from datetime import datetime, timezone
from typing import Any

import frontier_job
from frontier_contract import canonicalize, load_pin, verify_envelope
from frontier_runtime import artifact_manifest, manifest_digest
from nemo_v3_contract import (
    NEMO_V3_PAYLOAD_TYPE,
    expected_engine_key_id,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)


ALLOWED_NAMES = {
    "blocked_receipt.signed.json",
    "nemo-v3-qualified.signed.json",
    "nemo-v3-terminal.signed.json",
}


def parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def load_verified_job(
    spec_path: pathlib.Path, engine_key_path: pathlib.Path
) -> tuple[dict[str, Any], bytes]:
    envelope = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    pin = load_pin(engine_key_path)
    spec, exact_payload, payload_type = verify_envelope(
        envelope,
        pin,
        allowed_payload_types=(NEMO_V3_PAYLOAD_TYPE,),
    )
    if payload_type != NEMO_V3_PAYLOAD_TYPE:
        raise ValueError("signed job is not a Nemo v3 payload")
    validate_nemo_v3_spec(spec)
    if "authorization" in spec and pin.get("keyId") != expected_engine_key_id(spec):
        raise ValueError("Nemo v3 engine authorization key mismatch")
    return spec, exact_payload


def validate_intent(
    intent_path: pathlib.Path,
    spec: dict[str, Any],
    exact_payload: bytes,
    not_before: datetime,
    bridge_root: pathlib.Path,
    attempt_claim: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, str]:
    intent = json.loads(intent_path.read_text(encoding="utf-8-sig"))
    if (
        intent.get("kind") != "szl-receipt-signing-intent"
        or intent.get("v") != 1
        or intent.get("transport") != frontier_job.UNSIGNED_OUTBOX_TRANSPORT
        or intent.get("jobId") != spec["jobId"]
    ):
        raise ValueError("receipt intent contract is invalid")
    requested_name = intent.get("requestedReceiptName")
    if requested_name not in ALLOWED_NAMES:
        raise ValueError("receipt intent requested an unapproved file name")

    receipt = intent.get("receipt")
    if not isinstance(receipt, dict) or receipt.get("jobId") != spec["jobId"]:
        raise ValueError("receipt intent does not bind the signed job")
    if parse_timestamp(receipt.get("at")) < not_before:
        raise ValueError("receipt intent predates this execution")

    if receipt.get("kind") == "szl-frontier-training-blocked":
        if (
            receipt.get("verdict") != "BLOCKED"
            or not isinstance(receipt.get("stage"), str)
            or not receipt["stage"]
            or not isinstance(receipt.get("reason"), str)
            or not receipt["reason"]
            or requested_name != "blocked_receipt.signed.json"
        ):
            raise ValueError("blocked receipt intent is incomplete")
        return receipt, "BLOCKED", requested_name

    if receipt.get("kind") != "szl-nemo-v3-governed-training":
        raise ValueError("receipt intent kind is not admitted")
    if attempt_claim is None:
        raise ValueError("Nemo receipt has no trusted attempt claim")
    if receipt.get("candidateId") != spec["outputs"]["candidateId"]:
        raise ValueError("receipt candidate does not match the signed job")
    if receipt.get("source") != spec["source"]:
        raise ValueError("receipt source does not match the signed job")
    expected_base = {
        key: spec["base"][key] for key in ("repoId", "revision", "licenseId")
    }
    if receipt.get("base") != expected_base:
        raise ValueError("receipt base does not match the signed job")
    if receipt.get("training_rights_basis") != spec["dataset"]["rightsBasis"]:
        raise ValueError("receipt rights basis does not match the signed job")
    if (
        receipt.get("signed_job_payload_sha256")
        != hashlib.sha256(exact_payload).hexdigest()
    ):
        raise ValueError("receipt does not bind the exact signed job bytes")
    if receipt.get("effects") != {
        "candidate_uploaded": False,
        "published": False,
        "deployed": False,
        "promoted": False,
    }:
        raise ValueError("receipt effects exceed the signed no-publication boundary")

    state = receipt.get("state")
    evaluation = receipt.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("receipt evaluation evidence is missing")
    expected_container_image = claim_container_image(attempt_claim)
    stack = evaluation.get("stack")
    observed_container_image = (
        stack.get("containerImage") if isinstance(stack, dict) else None
    )
    if observed_container_image != expected_container_image:
        raise ValueError("receipt container image does not match the trusted claim")
    observed_bridge_execution = (
        stack.get("bridgeExecution") if isinstance(stack, dict) else None
    )
    if observed_bridge_execution != claim_bridge_execution(attempt_claim):
        raise ValueError("receipt Bridge revisions do not match the trusted claim")
    observed_launcher_sha256 = (
        stack.get("launcherSha256") if isinstance(stack, dict) else None
    )
    if observed_launcher_sha256 != attempt_claim["launcherSha256"]:
        raise ValueError("receipt launcher does not match the trusted claim")
    if state == "QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW":
        if (
            requested_name != "nemo-v3-qualified.signed.json"
            or receipt.get("decision") != "SEPARATE_PROMOTION_REVIEW_REQUIRED"
            or evaluation.get("state") != "PASS"
            or evaluation.get("pass_rate") != spec["evaluation"]["requiredPassRate"]
            or evaluation.get("degenerate") != 0
        ):
            raise ValueError(
                "qualified receipt does not satisfy signed evaluation gates"
            )
        candidate = bridge_root / "jobs" / spec["jobId"] / "candidate"
        observed_manifest = artifact_manifest(candidate)
        if observed_manifest != evaluation.get("adapter_files") or manifest_digest(
            observed_manifest
        ) != evaluation.get("adapter_manifest_sha256"):
            raise ValueError("qualified receipt artifact manifest is not reproducible")
    elif state == "EVALUATION_FAILED_NOT_PROMOTED_NOT_SIGNED":
        if (
            requested_name != "nemo-v3-terminal.signed.json"
            or receipt.get("decision") != "TERMINAL_FAILURE_NO_AUTOMATIC_RETRY"
            or evaluation.get("state") != "FAIL"
        ):
            raise ValueError("terminal evaluation receipt is inconsistent")
    else:
        raise ValueError("Nemo receipt intent is not terminal")
    return receipt, state, requested_name


def validate_attempt_claim(
    claim_path: pathlib.Path,
    spec_path: pathlib.Path,
    spec: dict[str, Any],
    not_before: datetime,
) -> dict[str, Any]:
    claim = json.loads(claim_path.read_text(encoding="utf-8-sig"))
    required_fields = {
        "kind",
        "v",
        "jobId",
        "jobEnvelopeSha256",
        "bridgeRevision",
        "envelopeRevision",
        "executionBridgeRevision",
        "launcherSha256",
        "trainingImage",
        "observedImageId",
        "environmentProbeSha256",
        "githubRunId",
        "claimedAt",
    }
    if (
        not isinstance(claim, dict)
        or set(claim) != required_fields
        or claim.get("kind") != "szl-nemo-v3-attempt-claim"
        or claim.get("v") != 3
        or claim.get("jobId") != spec["jobId"]
    ):
        raise ValueError("one-attempt claim contract is invalid")
    expected_envelope_sha256 = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    if claim.get("jobEnvelopeSha256") != expected_envelope_sha256:
        raise ValueError("one-attempt claim does not bind the signed job envelope")
    claimed_at = parse_timestamp(claim.get("claimedAt"))
    if claimed_at != not_before:
        raise ValueError("one-attempt claim timestamp does not bind this execution")
    bridge_revision = claim.get("bridgeRevision")
    envelope_revision = claim.get("envelopeRevision")
    execution_bridge_revision = claim.get("executionBridgeRevision")
    training_image = claim.get("trainingImage")
    observed_image_id = claim.get("observedImageId")
    environment_probe_sha256 = claim.get("environmentProbeSha256")
    launcher_sha256 = claim.get("launcherSha256")
    if (
        not isinstance(bridge_revision, str)
        or len(bridge_revision) != 40
        or any(character not in "0123456789abcdef" for character in bridge_revision)
        or not isinstance(envelope_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", envelope_revision) is None
        or not isinstance(execution_bridge_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", execution_bridge_revision) is None
        or bridge_revision != execution_bridge_revision
        or envelope_revision == execution_bridge_revision
        or not isinstance(training_image, str)
        or re.fullmatch(
            r"unsloth/unsloth@sha256:[0-9a-f]{64}",
            training_image,
        )
        is None
        or not isinstance(observed_image_id, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", observed_image_id) is None
        or observed_image_id != training_image.rsplit("@", 1)[1]
        or not isinstance(environment_probe_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", environment_probe_sha256) is None
    ):
        raise ValueError("one-attempt claim has no immutable execution identity")
    if (
        not isinstance(launcher_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", launcher_sha256) is None
    ):
        raise ValueError("one-attempt claim has no approved launcher binding")
    authorization = spec.get("authorization")
    if isinstance(authorization, dict) and (
        authorization.get("correctedBridgeRevision") != execution_bridge_revision
    ):
        raise ValueError("one-attempt claim differs from signed Bridge authority")
    owner_dispatch = spec.get("ownerDispatch")
    if isinstance(owner_dispatch, dict) and (
        owner_dispatch.get("trainingImage") != training_image
    ):
        raise ValueError("one-attempt claim differs from signed training image")
    return claim


def require_claim_bound_dispatchable(
    spec: dict[str, Any], attempt_claim: dict[str, Any]
) -> None:
    """Bind finalization to the execution revision proven by the durable claim."""
    require_nemo_v3_dispatchable(
        spec,
        expected_execution_bridge_revision=attempt_claim["executionBridgeRevision"],
    )


def claim_container_image(claim: dict[str, Any]) -> dict[str, Any]:
    """Return the exact trusted image identity that a receipt must reproduce."""
    result: dict[str, Any] = {
        "reference": claim["trainingImage"],
        "id": claim["observedImageId"],
        "environmentProbeSha256": claim["environmentProbeSha256"],
    }
    return result


def claim_bridge_execution(claim: dict[str, Any]) -> dict[str, str]:
    """Return the exact envelope-data and executable Bridge revisions."""

    return {
        "envelopeRevision": claim["envelopeRevision"],
        "executionBridgeRevision": claim["executionBridgeRevision"],
    }


def immutable_readback(
    local_path: pathlib.Path,
    repo_id: str,
    remote_path: str,
    revision: str,
    token: str,
) -> str:
    from huggingface_hub import hf_hub_download

    downloaded = pathlib.Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=revision,
            token=token,
            force_download=True,
        )
    )
    local_bytes = local_path.read_bytes()
    if downloaded.read_bytes() != local_bytes:
        raise ValueError("immutable Hub readback differs from the signed local receipt")
    return hashlib.sha256(local_bytes).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--intent", type=pathlib.Path, required=True)
    parser.add_argument("--engine-key", type=pathlib.Path, required=True)
    parser.add_argument("--bridge-root", type=pathlib.Path, required=True)
    parser.add_argument("--not-before", required=True)
    parser.add_argument("--claim", type=pathlib.Path, required=True)
    parser.add_argument("--ledger", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if os.environ.get(frontier_job.RECEIPT_TRANSPORT_ENV):
        raise RuntimeError("trusted finalizer refuses an alternate receipt transport")
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for trusted receipt finalization")

    spec, exact_payload = load_verified_job(args.spec, args.engine_key)
    seen = set()
    if args.ledger.exists():
        seen = {
            line.split("#", 1)[0].strip()
            for line in args.ledger.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        }
    if spec["jobId"] in seen:
        raise RuntimeError("one-attempt job is already present in the terminal ledger")

    not_before = parse_timestamp(args.not_before)
    attempt_claim = validate_attempt_claim(args.claim, args.spec, spec, not_before)
    require_claim_bound_dispatchable(spec, attempt_claim)
    receipt, state, requested_name = validate_intent(
        args.intent,
        spec,
        exact_payload,
        not_before,
        args.bridge_root,
        attempt_claim,
    )

    frontier_job.ROOT = args.bridge_root
    signed = frontier_job.sign_receipt(receipt)
    body = base64.b64decode(signed["bodyBase64"], validate=True)
    if body != canonicalize(receipt).encode("utf-8"):
        raise RuntimeError("trusted signer did not preserve canonical receipt bytes")
    result = frontier_job.upload_receipt(signed, requested_name, spec)
    revision = getattr(result, "oid", None)
    if not revision:
        raise RuntimeError("receipt upload returned no immutable revision")

    local_path = args.bridge_root / "jobs" / spec["jobId"] / "receipts" / requested_name
    remote_path = f"{spec['jobId']}/{requested_name}"
    digest = immutable_readback(
        local_path,
        spec["outputs"]["receiptsRepoId"],
        remote_path,
        str(revision),
        token,
    )

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{spec['jobId']}  # VERIFIED_IMMUTABLE_SIGNED_RECEIPT {revision}\n"
        )

    print(
        json.dumps(
            {
                "jobId": spec["jobId"],
                "keyId": signed["keyId"],
                "receiptPath": remote_path,
                "receiptRevision": str(revision),
                "receiptSha256": digest,
                "state": state,
                "verdict": "VERIFIED_IMMUTABLE_SIGNED_RECEIPT",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
