"""Read-only, offline public Stack400 canary receipt consistency verifier.

A matching content self-hash is not a digital signature, a Salad instance
attestation, an independently retrieved adapter, or a 54-hour completion.
No HTTP, credentials, job submission, signature minting, or weight loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

SCHEMA = "szl.stack400-public-canary-review/v1"
EXPECTED_RECEIPT = "szl.receipt/v1"
REPO = "SZLHOLDINGS/stack400-runs"
ACTOR = "szl-stack400-canary-worker/v1"
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
RUN = re.compile(r"stack400-[a-z0-9-]{2,80}\Z")
BASE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@(.+)\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
MAX_RECEIPT_BYTES = 16_384
MAX_MARKERS_BYTES = 65_536
MAX_MARKERS = 100
REQUIRED_STAGES = (
    "BOOT_OK",
    "IMPORTS_OK",
    "FETCH_OK",
    "LOAD_OK",
    "GPU_ATTEST",
    "TRAIN_OK",
    "PROBE_MEASURED",
    "EVAL_OK",
    "RECEIPT_MINTED",
    "UPLOAD_OK",
)
RECEIPT_KEYS = {
    "schema",
    "seq",
    "actor",
    "repo",
    "head",
    "base",
    "policy_digest",
    "verifier_digest",
    "artifact_digest",
    "effects",
    "prev",
    "training",
    "recorded_hash",
}


class ReceiptHold(ValueError):
    """Fixed non-secret denial code, never evidence of a passed release."""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ReceiptHold(reason)


def pairs_strict(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def bad_constant(_):
    raise ReceiptHold("NONFINITE_JSON_NUMBER")


def parse_bounded(raw: bytes, limit: int):
    require(type(raw) is bytes and 0 < len(raw) <= limit, "INPUT_BYTE_BUDGET")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_strict,
            parse_constant=bad_constant,
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ReceiptHold("INVALID_JSON") from exc


def number(v: object, reason: str) -> float:
    require(type(v) in (int, float) and math.isfinite(v), reason)
    return float(v)


def digest_ok(v: object) -> bool:
    return type(v) is str and SHA256.fullmatch(v) is not None


def read_local(path: Path, cap: int) -> bytes:
    require(path.is_file() and not path.is_symlink(), "LOCAL_INPUT_NOT_FILE")
    require(0 < path.stat().st_size <= cap, "INPUT_BYTE_BUDGET")
    raw = path.read_bytes()
    require(len(raw) <= cap, "INPUT_BYTE_BUDGET")
    return raw


def verify(receipt_raw: bytes, markers_raw: bytes) -> dict:
    """Validate public receipt and marker *consistency*, never authenticity."""
    obj = parse_bounded(receipt_raw, MAX_RECEIPT_BYTES)
    require(type(obj) is dict and set(obj) == RECEIPT_KEYS, "RECEIPT_SCHEMA")
    require(obj["schema"] == EXPECTED_RECEIPT, "RECEIPT_SCHEMA")
    require(type(obj["seq"]) is int and obj["seq"] == 1, "SEQUENCE_UNSUPPORTED")
    require(obj["actor"] == ACTOR and obj["repo"] == REPO, "RECEIPT_SOURCE_MISMATCH")
    require(obj["prev"] == "genesis", "RECEIPT_HISTORY_UNRECOGNIZED")
    runhead = obj["head"]
    require(type(runhead) is str and runhead.startswith("salad:"), "RUN_ID")
    run_id = runhead.removeprefix("salad:")
    require(RUN.fullmatch(run_id) is not None, "RUN_ID")

    for key in ("recorded_hash", "policy_digest", "verifier_digest", "artifact_digest"):
        require(digest_ok(obj[key]), "DIGEST_FORMAT_INVALID")
    self_hash = obj["recorded_hash"]
    without_hash = {k: v for k, v in obj.items() if k != "recorded_hash"}
    serialized = json.dumps(
        without_hash,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    require(
        "sha256:" + hashlib.sha256(serialized).hexdigest() == self_hash,
        "RECEIPT_SELF_HASH_MISMATCH",
    )

    effects = obj["effects"]
    require(type(effects) is list and len(effects) == 2, "EFFECT_LIST_INVALID")
    expected_files = [
        "receipts/" + run_id + "/receipt.json",
        "receipts/" + run_id + "/markers.txt",
    ]
    require(
        all(type(e) is dict and set(e) == {"path", "action"} for e in effects)
        and [e["path"] for e in effects] == expected_files
        and all(e["action"] == "create" for e in effects),
        "EFFECT_SOURCE_MISMATCH",
    )
    training = obj["training"]
    require(
        type(training) is dict
        and set(training)
        == {
            "job_seed",
            "no_op_inject",
            "mean_delta",
            "max_delta",
            "adapter_trainable_l1",
            "eval_score",
            "loss_first",
            "loss_last",
        },
        "TRAINING_RECORD_INVALID",
    )
    require(type(training["job_seed"]) is int, "SEED_INVALID")
    require(type(training["no_op_inject"]) is bool, "NOOP_FLAG_INVALID")
    for key in (
        "mean_delta",
        "max_delta",
        "adapter_trainable_l1",
        "eval_score",
        "loss_first",
        "loss_last",
    ):
        number(training[key], "TRAINING_NUMBER_INVALID")
    # These are only worker-reported values, but a claimed successful
    # adapter update cannot also report no trainable weights or no movement.
    # A pass here is consistency only, never independently proved training.
    require(
        training["mean_delta"] >= 0
        and training["max_delta"] > 0
        and training["adapter_trainable_l1"] > 0
        and training["mean_delta"] <= training["max_delta"],
        "TRAIN_MOVEMENT_INCONSISTENT",
    )
    require(0 <= training["eval_score"] <= 1, "EVAL_SCORE_INVALID")
    require(training["loss_first"] > 0 and training["loss_last"] >= 0, "LOSS_INVALID")

    require(
        type(markers_raw) is bytes and 0 < len(markers_raw) <= MAX_MARKERS_BYTES,
        "MARKERS_BYTE_BUDGET",
    )
    lines = markers_raw.splitlines()
    require(0 < len(lines) <= MAX_MARKERS, "MARKER_COUNT_INVALID")
    markers = [parse_bounded(line, 8192) for line in lines]
    require(
        all(type(m) is dict and type(m.get("tag")) is str for m in markers),
        "MARKER_SCHEMA_INVALID",
    )
    tags = [m["tag"] for m in markers]
    indices = []
    for tag in REQUIRED_STAGES:
        require(tags.count(tag) == 1, "STAGE_MISSING_OR_DUPLICATE")
        indices.append(tags.index(tag))
    require(indices == sorted(indices), "STAGE_ORDER_INVALID")
    require(tags[-1] == "UPLOAD_OK", "UPLOAD_IS_NOT_TERMINAL")
    bytag = {m["tag"]: m for m in markers}
    require(bytag["BOOT_OK"].get("run") == run_id, "MARKER_RUN_MISMATCH")
    require(
        bytag["RECEIPT_MINTED"].get("recorded_hash") == self_hash,
        "MARKER_SELF_HASH_MISMATCH",
    )
    require(
        bytag["UPLOAD_OK"].get("path") == expected_files[0],
        "UPLOAD_PATH_MISMATCH",
    )
    steps = bytag["TRAIN_OK"].get("steps")
    require(type(steps) is int and 0 < steps <= 1000, "TRAIN_STEP_COUNT_INVALID")
    require(
        abs(
            number(bytag["TRAIN_OK"].get("loss_first"), "MARKER_LOSS_INVALID")
            - training["loss_first"]
        )
        < 1e-3
        and abs(
            number(bytag["TRAIN_OK"].get("loss_last"), "MARKER_LOSS_INVALID")
            - training["loss_last"]
        )
        < 1e-3,
        "RECEIPT_MARKER_LOSS_MISMATCH",
    )
    probe = bytag["PROBE_MEASURED"]
    require(probe.get("no_op") is False, "NOOP_QUARANTINE")
    require(
        abs(
            number(probe.get("mean_delta"), "PROBE_DELTA_INVALID")
            - training["mean_delta"]
        )
        < 1e-3
        and abs(
            number(probe.get("max_delta"), "PROBE_DELTA_INVALID")
            - training["max_delta"]
        )
        < 1e-3,
        "PROBE_RECEIPT_MISMATCH",
    )
    require(
        abs(
            number(bytag["EVAL_OK"].get("score"), "MARKER_EVAL_INVALID")
            - training["eval_score"]
        )
        < 1e-6,
        "EVAL_RECEIPT_MISMATCH",
    )
    base = obj["base"]
    require(type(base) is str, "BASE_NOT_IDENTIFIED")
    match = BASE.fullmatch(base)
    reasons = [
        "RECEIPT_SELF_HASH_ONLY_NOT_SIGNED",
        "WORKER_SOURCE_NOT_PROVED_FOR_THIS_RUN",
        "ADAPTER_BYTES_NOT_INDEPENDENTLY_VERIFIED",
        "SALAD_GROUP_INSTANCE_NOT_AUTHENTICATED",
        "EVALUATION_NOT_INDEPENDENT_HELDOUT",
        "DATA_RIGHTS_AND_OWNER_LEASE_UNVERIFIED",
        "54H_DURATION_NOT_ESTABLISHED",
    ]
    if not match or COMMIT.fullmatch(match.group(1)) is None:
        reasons.append("BASE_IMMUTABLE_REVISION_UNAVAILABLE")
    return {
        "schema": SCHEMA,
        "source_repo": REPO,
        "run_id": run_id,
        "public_receipt_self_hash_match": True,
        "source_markers_consistent": True,
        "worker_claimed_steps": steps,
        "worker_claimed_loss_first": training["loss_first"],
        "worker_claimed_loss_last": training["loss_last"],
        "worker_claimed_eval_score": training["eval_score"],
        "artifact_sha256_declared_not_verified": obj["artifact_digest"],
        "verdict": "REVIEW_ONLY_HOLD",
        "signed_receipt_verified": False,
        "gpu_instance_verified": False,
        "adapter_bytes_verified": False,
        "independent_holdout_verified": False,
        "54h_training_complete_verified": False,
        "reward_eligible": False,
        "provider_mutations": 0,
        "reasons": reasons,
    }


def create_report(path: Path, report: dict) -> str:
    require(path.parent.is_dir() and not path.is_symlink(), "OUTPUT_PARENT_INVALID")
    raw = (
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        read_local(args.receipt, MAX_RECEIPT_BYTES),
        read_local(args.markers, MAX_MARKERS_BYTES),
    )
    sha = create_report(args.output, result)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "claimed_steps": result["worker_claimed_steps"],
                "signed": False,
                "provider_mutations": 0,
                "report_sha256": sha,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
