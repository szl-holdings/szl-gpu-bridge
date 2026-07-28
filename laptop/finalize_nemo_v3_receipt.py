#!/usr/bin/env python3
"""Sign, upload, and immutably read back one isolated Nemo v3 receipt intent."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any

import frontier_job
from frontier_contract import canonicalize, load_pin, verify_envelope
from frontier_runtime import artifact_manifest, manifest_digest
from nemo_v3_contract import NEMO_V3_PAYLOAD_TYPE, validate_nemo_v3_spec


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
    spec, exact_payload, payload_type = verify_envelope(
        envelope,
        load_pin(engine_key_path),
        allowed_payload_types=(NEMO_V3_PAYLOAD_TYPE,),
    )
    if payload_type != NEMO_V3_PAYLOAD_TYPE:
        raise ValueError("signed job is not a Nemo v3 payload")
    validate_nemo_v3_spec(spec)
    return spec, exact_payload


def validate_intent(
    intent_path: pathlib.Path,
    spec: dict[str, Any],
    exact_payload: bytes,
    not_before: datetime,
    bridge_root: pathlib.Path,
) -> tuple[dict[str, Any], str]:
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
        return receipt, "BLOCKED"

    if receipt.get("kind") != "szl-nemo-v3-governed-training":
        raise ValueError("receipt intent kind is not admitted")
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
    return receipt, state


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

    receipt, state = validate_intent(
        args.intent,
        spec,
        exact_payload,
        parse_timestamp(args.not_before),
        args.bridge_root,
    )

    frontier_job.ROOT = args.bridge_root
    signed = frontier_job.sign_receipt(receipt)
    body = base64.b64decode(signed["bodyBase64"], validate=True)
    if body != canonicalize(receipt).encode("utf-8"):
        raise RuntimeError("trusted signer did not preserve canonical receipt bytes")
    result = frontier_job.upload_receipt(
        signed, args.intent.name.replace(".intent.json", ".signed.json"), spec
    )
    revision = getattr(result, "oid", None)
    if not revision:
        raise RuntimeError("receipt upload returned no immutable revision")

    signed_name = args.intent.name.replace(".intent.json", ".signed.json")
    local_path = args.bridge_root / "jobs" / spec["jobId"] / "receipts" / signed_name
    remote_path = f"{spec['jobId']}/{signed_name}"
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
