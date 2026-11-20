"""Non-authorizing CPU-only preflight for an SZL model-training wave.

The connected Salad admission runner is the ONLY prospective mutation path.
This module MUST NEVER invoke it, read credentials, issue receipts, reserve
capacity, submit jobs, or infer that a model is trained from Hugging Face.
The input is operator-REPORTED data, not authenticated provider evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

SCHEMA = "szl.training-wave-plan/v0"
RESULT_SCHEMA = "szl.training-wave-preflight/v0"
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_MODELS = 500
MAX_MICRO_USD = 10**13
SHA = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA40 = re.compile(r"^[0-9a-f]{40}$")
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
LANE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,96}$")
SKU = re.compile(r"^[a-zA-Z0-9._-]{2,64}$")

PLAN_FIELDS = {
    "schema",
    "mode",
    "catalog_sha256",
    "auth_witness_state",
    "owner_budget_approved",
    "owner_budget_micro_usd",
    "inflight_lane_ids",
    "models",
}
ROW_FIELDS = {
    "model_id",
    "lane_id",
    "intent",
    "artifact_kind",
    "base_model_id",
    "base_revision",
    "dataset_id",
    "dataset_revision",
    "recipe_sha256",
    "trainer_commit_sha",
    "eval_receipt_sha256",
    "canary_receipt_sha256",
    "license_rights_state",
    "dependencies_confirmed",
    "gpu_sku",
    "min_vram_gib",
    "replicas",
    "max_attempts",
    "max_runtime_minutes",
    "hourly_gpu_price_micro_usd",
    "checkpoint_interval_steps",
    "checkpoint_target",
}
AUTH = {
    "MISSING",
    "DENIED_401",
    "READ_ONLY",
    "UNKNOWN",
    "EXTERNALLY_REPORTED_VALID",
}
ARTIFACTS = {"adapter", "merged", "gguf", "checkpoint", "evaluation"}
RIGHTS = {"UNKNOWN", "RESTRICTED", "OWNER_REPORTED_REVIEWED"}
INTENTS = {"train", "evaluate"}
# Report only fixed reason labels and counts; NEVER echo model IDs or credentials.
REASONS = {
    "INPUT_INVALID",
    "AUTH_MISSING",
    "AUTH_DENIED_401",
    "AUTH_READ_ONLY",
    "AUTH_UNVERIFIED",
    "BUDGET_UNAPPROVED",
    "BUDGET_INSUFFICIENT",
    "INFLIGHT_LANE",
    "DUPLICATE_LANE",
    "DUPLICATE_OUTPUT_REPO",
    "DUPLICATE_TRAINING_INTENT",
    "LICENSE_RIGHTS_UNVERIFIED",
    "LINEAGE_INCOMPLETE",
    "CANARY_OR_EVAL_MISSING",
    "CHECKPOINT_PLAN_INVALID",
    "GPU_PLAN_INVALID",
    "BUDGET_ARITHMETIC_INVALID",
}


class PreflightError(ValueError):
    """Non-secret, fixed-schema diagnostic."""


def _raise(reason):
    if reason not in REASONS:
        reason = "INPUT_INVALID"
    raise PreflightError(reason)


def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            _raise("INPUT_INVALID")
        out[key] = value
    return out


def _nonfinite(_):
    _raise("INPUT_INVALID")


def _bounded_json(path):
    f = Path(path)
    if f.is_symlink() or not f.is_file():
        _raise("INPUT_INVALID")
    with f.open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if not 0 < len(raw) <= MAX_INPUT_BYTES:
        _raise("INPUT_INVALID")
    try:
        obj = json.loads(
            raw, object_pairs_hook=_unique_pairs, parse_constant=_nonfinite
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        _raise("INPUT_INVALID")
    return obj, hashlib.sha256(raw).hexdigest()


def _sha(text):
    """SHA-256 for content fingerprints, manifests and evidence receipts."""
    return isinstance(text, str) and SHA.fullmatch(text) is not None


def _git_revision(text):
    """Git object ID syntax: immutable 40-hex commit, not SHA-256 digest.

    Existence, signer, branch ownership and provider readback remain unverified.
    """
    return type(text) is str and GIT_SHA40.fullmatch(text) is not None


def _id(text):
    return isinstance(text, str) and MODEL_ID.fullmatch(text) is not None


def _int(text, lo, hi):
    return type(text) is int and lo <= text <= hi


def _label(text):
    return isinstance(text, str) and LANE_ID.fullmatch(text) is not None


def _fields(value, fields):
    if type(value) is not dict or set(value) != fields:
        _raise("INPUT_INVALID")


def _validate(plan):
    _fields(plan, PLAN_FIELDS)
    if (
        plan["schema"] != SCHEMA
        or plan["mode"] != "DRY_RUN"
        or not _sha(plan["catalog_sha256"])
    ):
        _raise("INPUT_INVALID")
    if (
        type(plan["auth_witness_state"]) is not str
        or plan["auth_witness_state"] not in AUTH
    ):
        _raise("INPUT_INVALID")
    if type(plan["owner_budget_approved"]) is not bool:
        _raise("INPUT_INVALID")
    if not _int(plan["owner_budget_micro_usd"], 0, MAX_MICRO_USD):
        _raise("INPUT_INVALID")
    active = plan["inflight_lane_ids"]
    if (
        type(active) is not list
        or len(active) > MAX_MODELS
        or not all(_label(x) for x in active)
    ):
        _raise("INPUT_INVALID")
    if len(set(active)) != len(active):
        _raise("INPUT_INVALID")
    models = plan["models"]
    if type(models) is not list or not 1 <= len(models) <= MAX_MODELS:
        _raise("INPUT_INVALID")
    return models, set(active)


def _row(row):
    _fields(row, ROW_FIELDS)
    if not (
        _id(row["model_id"])
        and _label(row["lane_id"])
        and type(row["intent"]) is str
        and row["intent"] in INTENTS
        and type(row["artifact_kind"]) is str
        and row["artifact_kind"] in ARTIFACTS
        and _id(row["base_model_id"])
        and _git_revision(row["base_revision"])
        and _id(row["dataset_id"])
        and _git_revision(row["dataset_revision"])
        and _sha(row["recipe_sha256"])
        and _git_revision(row["trainer_commit_sha"])
    ):
        _raise("LINEAGE_INCOMPLETE")
    if (
        type(row["license_rights_state"]) is not str
        or row["license_rights_state"] not in RIGHTS
        or type(row["dependencies_confirmed"]) is not bool
    ):
        _raise("LINEAGE_INCOMPLETE")
    for field in ("canary_receipt_sha256", "eval_receipt_sha256"):
        if row[field] is not None and not _sha(row[field]):
            _raise("LINEAGE_INCOMPLETE")
    if not isinstance(row["gpu_sku"], str) or SKU.fullmatch(row["gpu_sku"]) is None:
        _raise("GPU_PLAN_INVALID")
    if not (
        _int(row["min_vram_gib"], 1, 1024)
        and _int(row["replicas"], 1, 20)
        and _int(row["max_attempts"], 1, 3)
        and _int(row["max_runtime_minutes"], 1, 60 * 24 * 14)
        and _int(row["hourly_gpu_price_micro_usd"], 1, 10**10)
    ):
        _raise("GPU_PLAN_INVALID")
    if not _int(row["checkpoint_interval_steps"], 1, 100_000):
        _raise("CHECKPOINT_PLAN_INVALID")
    if not _id(row["checkpoint_target"]):
        _raise("CHECKPOINT_PLAN_INVALID")
    return row


def evaluate(plan):
    """Compute a non-authorizing cost upper bound from UNVERIFIED estimates.

    Reject malformed manifests outright; track per-row blockers by index only.
    Even 'EXTERNALLY_REPORTED_VALID' auth or owner budget claim is NOT proof.
    """
    models, in_flight = _validate(plan)
    reasons = set()
    global_reasons = set()
    auth = plan["auth_witness_state"]
    global_reasons.add(
        {
            "MISSING": "AUTH_MISSING",
            "DENIED_401": "AUTH_DENIED_401",
            "READ_ONLY": "AUTH_READ_ONLY",
            "UNKNOWN": "AUTH_UNVERIFIED",
            "EXTERNALLY_REPORTED_VALID": "AUTH_UNVERIFIED",
        }[auth]
    )
    if not plan["owner_budget_approved"]:
        global_reasons.add("BUDGET_UNAPPROVED")
    rows = []
    total_cost = 0
    seen_lane, seen_output, seen_train = set(), set(), set()
    for index, original in enumerate(models):
        row = _row(original)
        issues = set()
        if row["lane_id"] in in_flight:
            issues.add("INFLIGHT_LANE")
        if row["lane_id"] in seen_lane:
            issues.add("DUPLICATE_LANE")
        seen_lane.add(row["lane_id"])
        if row["model_id"] in seen_output:
            issues.add("DUPLICATE_OUTPUT_REPO")
        seen_output.add(row["model_id"])
        key = (
            row["base_model_id"],
            row["base_revision"],
            row["dataset_id"],
            row["dataset_revision"],
            row["recipe_sha256"],
            row["artifact_kind"],
            row["intent"],
        )
        if key in seen_train:
            issues.add("DUPLICATE_TRAINING_INTENT")
        seen_train.add(key)
        if row["license_rights_state"] != "OWNER_REPORTED_REVIEWED":
            issues.add("LICENSE_RIGHTS_UNVERIFIED")
        if (
            not row["dependencies_confirmed"]
            or row["eval_receipt_sha256"] is None
            or row["canary_receipt_sha256"] is None
        ):
            issues.add("CANARY_OR_EVAL_MISSING")
        # Integer-only CEIL(cost/hour * replicas * runtime * max_attempts / 60).
        # Attempts bound a scenario, they do NOT authorize retrying 401 errors.
        numerator = (
            row["hourly_gpu_price_micro_usd"]
            * row["replicas"]
            * row["max_runtime_minutes"]
            * row["max_attempts"]
        )
        upper = (numerator + 59) // 60
        if upper > MAX_MICRO_USD or total_cost + upper > MAX_MICRO_USD:
            _raise("BUDGET_ARITHMETIC_INVALID")
        total_cost += upper
        reasons.update(issues)
        rows.append(
            {
                "index": index,
                "state": "HOLD" if issues else "STRUCTURAL_CANDIDATE_ONLY",
                "reasons": sorted(issues),
                "estimated_upper_cost_micro_usd": upper,
            }
        )
    if total_cost > plan["owner_budget_micro_usd"]:
        global_reasons.add("BUDGET_INSUFFICIENT")
    # Structural candidate status never authorizes GPU execution.
    return {
        "schema": RESULT_SCHEMA,
        "decision": "HOLD_NO_COMPUTE",
        "provider_writes_authorized": False,
        "training_authorized": False,
        "model_training_verified": False,
        "source_data_authenticated": False,
        "credential_checked": False,
        "receipt_signatures_checked": False,
        "plan_model_count": len(models),
        "estimated_upper_cost_micro_usd": total_cost,
        "owner_budget_reported_micro_usd": plan["owner_budget_micro_usd"],
        "global_reasons": sorted(global_reasons),
        "row_reasons": sorted(reasons),
        "row_results": rows,
        "note": "Operator-reported inputs and estimates; a separate signer, exact artifact verification and single-writer Salad gate are mandatory.",
    }


def _atomic_report(destination, record):
    """Publish one complete report without replacing a user-owned file.

    A same-directory temporary file is synced before the final hard link.
    Linking is exclusive: an existing destination, including a symlink,
    is never truncated, overwritten, or removed.
    """
    path = Path(destination)
    tmp = None
    try:
        raw = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8")
        fd, name = tempfile.mkstemp(
            dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp"
        )
        tmp = Path(name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
        if path.read_bytes() != raw:
            raise OSError("REPORT_READBACK_MISMATCH")
    finally:
        # Only this call's freshly created scratch is cleaned.
        if tmp:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_json", help="local operator-reported planning manifest")
    ap.add_argument("report_json", help="local redacted, non-authorizing output")
    args = ap.parse_args(argv)
    # An output alias must never replace the operator's original private plan.
    try:
        source, output = Path(args.input_json), Path(args.report_json)
        if source.resolve() == output.resolve() or (
            source.exists() and output.exists() and os.path.samefile(source, output)
        ):
            print("OUTPUT_OVERLAPS_INPUT", file=sys.stderr)
            return 1
    except OSError:
        print("PATH_CHECK_UNAVAILABLE", file=sys.stderr)
        return 1
    try:
        doc, source_sha = _bounded_json(args.input_json)
        report = evaluate(doc)
        report["input_sha256"] = source_sha
    except (PreflightError, OSError, ValueError):
        report = {
            "schema": RESULT_SCHEMA,
            "decision": "HOLD_NO_COMPUTE",
            "reason": "INPUT_INVALID",
            "training_authorized": False,
            "provider_writes_authorized": False,
        }
    try:
        _atomic_report(args.report_json, report)
    except OSError:
        print("REPORT_UNAVAILABLE", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "models": report.get("plan_model_count"),
                "reasons": report.get("global_reasons", [report.get("reason")]),
            },
            sort_keys=True,
        )
    )
    # A plan is *never* an approval or executable command.
    return 1


if __name__ == "__main__":
    sys.exit(main())
