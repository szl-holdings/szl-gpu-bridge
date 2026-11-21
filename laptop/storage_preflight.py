"""SZL workstation capacity admission: bounded, read-only, fail-closed.

This is NOT the training dispatcher, a watchdog, a signed job, a disk
cleaner, or evidence that a GPU service is healthy. It never deletes
files, touches running jobs, reads credentials or modifies Windows tasks.

Use immediately before a future owner-authorized local training launch.
An unstable disk free-space reading refuses admission even if one sample
is healthy. On success, only the *storage* gate passes; all other model,
dataset, source, budget, signer and control-plane gates remain separate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Callable

GIB = 1024**3
SCHEMA = "szl.windows_training_storage_preflight.v1"
CODE_VERSION = "storage-capacity-sampled-20261117"
MIN_SAMPLES = 3
MAX_SAMPLES = 12
MAX_INTERVAL_SEC = 5.0
DEFAULT_MINIMUM_FREE_GIB = 50
DEFAULT_MAX_SWING_GIB = 2
REFUSED_EXIT = 78


@dataclass(frozen=True)
class Reading:
    observed_utc: str
    total_bytes: int
    free_bytes: int
    used_bytes: int


def _canonical(obj: object) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def classify(
    samples: list[Reading],
    *,
    min_free_gib: int = DEFAULT_MINIMUM_FREE_GIB,
    max_swing_gib: int = DEFAULT_MAX_SWING_GIB,
) -> dict:
    """Classify independently recorded samples; no fresh I/O or side effects."""
    if (
        type(min_free_gib) is not int
        or not 1 <= min_free_gib <= 2048
        or type(max_swing_gib) is not int
        or not 0 <= max_swing_gib <= 128
    ):
        raise ValueError("INVALID_FIXED_STORAGE_POLICY")
    if not MIN_SAMPLES <= len(samples) <= MAX_SAMPLES:
        raise ValueError("INSUFFICIENT_OR_UNBOUNDED_SAMPLE_COUNT")
    for sample in samples:
        if not isinstance(sample, Reading):
            raise ValueError("BAD_READING_TYPE")
        if (
            not all(
                _finite_nonnegative_int(x)
                for x in (sample.total_bytes, sample.free_bytes, sample.used_bytes)
            )
            or sample.total_bytes == 0
            or sample.free_bytes > sample.total_bytes
            or sample.used_bytes > sample.total_bytes
        ):
            raise ValueError("INVALID_CAPACITY_READING")
        if not isinstance(sample.observed_utc, str) or not sample.observed_utc:
            raise ValueError("MISSING_OBSERVATION_TIMESTAMP")
    free = [x.free_bytes for x in samples]
    total = [x.total_bytes for x in samples]
    if max(total) != min(total):
        raise ValueError("FILESYSTEM_IDENTITY_OR_SIZE_CHANGED")
    minimum = min(free)
    maximum = max(free)
    swing = maximum - minimum
    reasons = []
    if minimum < min_free_gib * GIB:
        reasons.append("HOLD_LOW_MINIMUM_FREE_SPACE")
    if swing > max_swing_gib * GIB:
        reasons.append("HOLD_UNSTABLE_FREE_SPACE")
    status = "CAPACITY_GATE_PASS_ONLY" if not reasons else "HOLD"
    return {
        "state": status,
        "reasons": reasons,
        "sample_count": len(samples),
        "total_capacity_bytes": total[0],
        "minimum_observed_free_bytes": minimum,
        "maximum_observed_free_bytes": maximum,
        "observed_free_swing_bytes": swing,
        "last_observed_free_bytes": free[-1],
        "policy": {
            "minimum_free_gib": min_free_gib,
            "maximum_observed_swing_gib": max_swing_gib,
        },
        "storage_only_passed": not reasons,
        "other_training_qualification_gates_checked": False,
    }


def probe(
    root: Path,
    *,
    samples: int = 5,
    interval_sec: float = 2.0,
    min_free_gib: int = DEFAULT_MINIMUM_FREE_GIB,
    max_swing_gib: int = DEFAULT_MAX_SWING_GIB,
    usage: Callable = shutil.disk_usage,
    sleeper: Callable = time.sleep,
    clock: Callable = _timestamp,
) -> dict:
    if type(samples) is not int or not MIN_SAMPLES <= samples <= MAX_SAMPLES:
        raise ValueError("INVALID_SAMPLE_COUNT")
    if not 0 <= interval_sec <= MAX_INTERVAL_SEC:
        raise ValueError("INVALID_INTERVAL")
    observations = []
    root = Path(root)
    failure = None
    try:
        if not root.is_dir():
            raise OSError("STORAGE_ROOT_NOT_DIRECTORY")
        for i in range(samples):
            snapshot = usage(root)
            # Preserve provider-native exact bytes. int() would silently turn
            # bool/float/numeric strings into apparent disk measurements,
            # bypassing classify()'s strict integer type validation.
            total, free, used = snapshot.total, snapshot.free, snapshot.used
            if not all(_finite_nonnegative_int(x) for x in (total, free, used)):
                raise ValueError("INVALID_CAPACITY_READING")
            read = Reading(
                observed_utc=clock(),
                total_bytes=total,
                free_bytes=free,
                used_bytes=used,
            )
            observations.append(read)
            if i < samples - 1:
                sleeper(interval_sec)
        verdict = classify(
            observations, min_free_gib=min_free_gib, max_swing_gib=max_swing_gib
        )
    except (OSError, ValueError, AttributeError, OverflowError) as error:
        failure = type(error).__name__
        verdict = {
            "state": "HOLD_UNAVAILABLE_STORAGE_PROBE",
            "reasons": ["STORAGE_OBSERVATION_UNAVAILABLE_OR_INCONSISTENT"],
            "storage_only_passed": False,
            "other_training_qualification_gates_checked": False,
        }
    record = {
        "schema": SCHEMA,
        "source_version": CODE_VERSION,
        "generated_utc": clock(),
        "scope": "read-only sampled workstation disk capacity, not cloud GPU/storage",
        "target_drive_root": str(root),
        "observed_samples": [o.__dict__ for o in observations],
        "verdict": verdict,
        "observation_error_type": failure,
        "training_job_started": False,
        "credential_values_read": False,
        "files_removed": False,
        "watchdog_tasks_changed": False,
        "existing_training_generation_modified": False,
        "authorizes_flagship_dispatch": False,
        "minimum_threshold_is_a_storage_gate_not_a_production_claim": True,
        "receipt_is_cryptographic_signature": False,
    }
    return record


def write_exclusive_receipt(path: Path, report: dict) -> str:
    """Write once with fsync/readback; never replace or mutate prior evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical(report)
    with path.open("xb") as fd:
        fd.write(raw)
        fd.flush()
        os.fsync(fd.fileno())
    if path.read_bytes() != raw:
        raise OSError("STORAGE_RECEIPT_READBACK_FAILED")
    return sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="C:\\" if os.name == "nt" else "/",
        help="Existing local filesystem root to measure (never modified).",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--interval-sec", type=float, default=2.0)
    parser.add_argument(
        "--min-free-gib",
        type=int,
        default=DEFAULT_MINIMUM_FREE_GIB,
    )
    parser.add_argument(
        "--max-swing-gib",
        type=int,
        default=DEFAULT_MAX_SWING_GIB,
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Optional new owner-private JSON evidence path; exclusive creation.",
    )
    args = parser.parse_args(argv)
    try:
        report = probe(
            Path(args.root),
            samples=args.samples,
            interval_sec=args.interval_sec,
            min_free_gib=args.min_free_gib,
            max_swing_gib=args.max_swing_gib,
        )
        if args.receipt is not None:
            # The external SHA covers final file bytes, not a self-referential
            # digest inside the artifact. A SHA-256 is not a signature.
            hash_value = write_exclusive_receipt(args.receipt, report)
        else:
            hash_value = sha256(_canonical(report)).hexdigest()
        public = {
            "state": report["verdict"]["state"],
            "reasons": report["verdict"]["reasons"],
            "sample_count": len(report["observed_samples"]),
            "minimum_free_gib": round(
                report["verdict"].get("minimum_observed_free_bytes", 0) / GIB, 4
            ),
            "maximum_free_gib": round(
                report["verdict"].get("maximum_observed_free_bytes", 0) / GIB, 4
            ),
            "storage_only_passed": report["verdict"]["storage_only_passed"],
            "other_training_gates_verified": False,
            "receipt_sha256": hash_value,
            "receipt_path": str(args.receipt) if args.receipt is not None else None,
        }
        print(json.dumps(public, sort_keys=True))
        return 0 if report["verdict"]["storage_only_passed"] else REFUSED_EXIT
    except (ValueError, OSError, FileExistsError) as error:
        print(
            json.dumps(
                {
                    "state": "HOLD_INVALID_CONFIGURATION_OR_RECEIPT",
                    "reason": type(error).__name__,
                    "storage_only_passed": False,
                    "other_training_gates_verified": False,
                },
                sort_keys=True,
            )
        )
        return REFUSED_EXIT


if __name__ == "__main__":
    sys.exit(main())
