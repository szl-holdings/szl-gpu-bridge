"""Fail-closed, SOURCE-ONLY longitudinal capacity advisory for existing bridge receipts.

Consumes bounded historical laptop/storage_preflight.py JSON receipts.
Never measures a live drive, performs provider/network calls, launches training,
changes dispatcher state, signs evidence, or claims an approved training owner.
A forecast is descriptive extrapolation, NOT a confidence/probability bound.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

GIB = 1024**3
INPUT_SCHEMA = "szl.windows_training_storage_preflight.v1"
EXPECTED_VERSION = "storage-capacity-sampled-20261117"
OUTPUT_SCHEMA = "szl.storage-horizon-advisory/v1"
MIN_WINDOWS = 3
MAX_WINDOWS = 24
MIN_SPAN_SECONDS = 15 * 60
MAX_GAP_SECONDS = 2 * 60 * 60
MAX_AGE_SECONDS = 5 * 60
MAX_FUTURE_SECONDS = 5
MAX_SOURCE_BYTES = 64_000
RESERVE_GIB = 50
SWING_GIB = 2
HORIZON_SECONDS = 4 * 60 * 60
BURST_BUFFER_BYTES = 2 * GIB


class Hold(ValueError):
    """Stable reason code; never untrusted source text/path in exceptions."""


def _require(cond: bool, code: str) -> None:
    if not cond:
        raise Hold(code)


def _ts(value: Any) -> datetime:
    _require(type(value) is str and 15 <= len(value) <= 40, "TIMESTAMP_INVALID")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise Hold("TIMESTAMP_INVALID") from None
    _require(stamp.utcoffset() is not None, "TIMESTAMP_UNZONED")
    return stamp.astimezone(timezone.utc)


def _integer(v: Any, *, low: int, high: int, code: str) -> int:
    _require(type(v) is int and low <= v <= high, code)
    return v


def _object(v: Any, code: str) -> Mapping[str, Any]:
    _require(type(v) is dict, code)
    return v


def parse_receipt(raw: bytes) -> Mapping[str, Any]:
    """Parse only a small UTF-8 JSON receipt; reject duplicate keys and NaN."""
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_SOURCE_BYTES, "RECEIPT_BYTE_BUDGET")

    def distinct(pairs):
        d = {}
        for k, v in pairs:
            _require(k not in d, "DUPLICATE_RECEIPT_FIELD")
            d[k] = v
        return d

    try:
        result = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=distinct,
            parse_constant=lambda _: (_ for _ in ()).throw(Hold("NONFINITE_NUMBER")),
        )
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, Hold):
            raise
        raise Hold("RECEIPT_INVALID_JSON") from None
    return _object(result, "RECEIPT_NOT_OBJECT")


def _window(receipt: Mapping[str, Any]) -> dict:
    _require(receipt.get("schema") == INPUT_SCHEMA, "RECEIPT_SCHEMA_MISMATCH")
    _require(receipt.get("source_version") == EXPECTED_VERSION, "SOURCE_VERSION_MISMATCH")
    _require(receipt.get("observation_error_type") is None, "SOURCE_REPORTED_OBSERVATION_ERROR")
    items = receipt.get("observed_samples")
    _require(type(items) is list and 5 <= len(items) <= 12, "OBSERVATION_COUNT_INVALID")
    generated = _ts(receipt.get("generated_utc"))
    samples = []
    for item in items:
        r = _object(item, "SAMPLE_NOT_OBJECT")
        t = _ts(r.get("observed_utc"))
        total = _integer(r.get("total_bytes"), low=1, high=2**64 - 1, code="TOTAL_BYTES_INVALID")
        free = _integer(r.get("free_bytes"), low=0, high=total, code="FREE_BYTES_INVALID")
        used = _integer(r.get("used_bytes"), low=0, high=total, code="USED_BYTES_INVALID")
        _require(free + used == total, "INCONSISTENT_FILESYSTEM_ACCOUNTING")
        samples.append((t, total, free))
    _require(len({s[1] for s in samples}) == 1, "FILESYSTEM_SIZE_CHANGED")
    for left, right in zip(samples, samples[1:]):
        dt = (right[0] - left[0]).total_seconds()
        _require(0.5 <= dt <= 10.0, "SAMPLE_SPACING_INVALID")
    _require(0 <= (generated - samples[-1][0]).total_seconds() <= 30, "RECEIPT_TIME_INCONSISTENT")
    free_list = [s[2] for s in samples]
    minimum, maximum = min(free_list), max(free_list)
    swing = maximum - minimum
    expected_pass = minimum >= RESERVE_GIB * GIB and swing <= SWING_GIB * GIB
    verdict = _object(receipt.get("verdict"), "MISSING_VERDICT")
    policy = _object(verdict.get("policy"), "SOURCE_POLICY_MISSING")
    _require(policy.get("minimum_free_gib") == RESERVE_GIB and type(policy.get("minimum_free_gib")) is int, "SOURCE_POLICY_DRIFT")
    _require(policy.get("maximum_observed_swing_gib") == SWING_GIB and type(policy.get("maximum_observed_swing_gib")) is int, "SOURCE_POLICY_DRIFT")
    for field, v in (
        ("sample_count", len(samples)),
        ("total_capacity_bytes", samples[0][1]),
        ("minimum_observed_free_bytes", minimum),
        ("maximum_observed_free_bytes", maximum),
        ("observed_free_swing_bytes", swing),
        ("last_observed_free_bytes", free_list[-1]),
    ):
        _require(type(verdict.get(field)) is int and verdict[field] == v, "SOURCE_VERDICT_CONTRADICTION")
    _require(verdict.get("storage_only_passed") is expected_pass, "SOURCE_VERDICT_CONTRADICTION")
    _require(verdict.get("other_training_qualification_gates_checked") is False, "SOURCE_GATES_OVERRIDDEN")
    _require(verdict.get("state") == ("CAPACITY_GATE_PASS_ONLY" if expected_pass else "HOLD"), "SOURCE_VERDICT_CONTRADICTION")
    _require(receipt.get("training_job_started") is False, "SOURCE_NOT_READ_ONLY")
    _require(receipt.get("authorizes_flagship_dispatch") is False, "SOURCE_AUTHORITY_CONTRADICTION")
    return {
        "start": samples[0][0],
        "end": samples[-1][0],
        "total": samples[0][1],
        "minimum": minimum,
        "maximum": maximum,
        "swing": swing,
        "samples": len(samples),
        "source_capacity_pass": expected_pass,
    }


def assess(raw_receipts: Sequence[bytes], *, now: datetime | None = None) -> dict:
    """Pure analysis. Returns HOLD or a NON-BINDING storage advisory, never READY."""
    _require(type(raw_receipts) in (list, tuple) and MIN_WINDOWS <= len(raw_receipts) <= MAX_WINDOWS, "INSUFFICIENT_OR_UNBOUNDED_HISTORY")
    if now is None:
        now = datetime.now(timezone.utc)
    _require(isinstance(now, datetime) and now.utcoffset() is not None, "CLOCK_UNVERIFIED")
    now = now.astimezone(timezone.utc)
    windows = [_window(parse_receipt(raw)) for raw in raw_receipts]
    for left, right in zip(windows, windows[1:]):
        _require(left["end"] < right["start"], "WINDOW_OVERLAP_OR_UNSORTED")
        _require((right["start"] - left["end"]).total_seconds() <= MAX_GAP_SECONDS, "HISTORY_GAP_UNOBSERVED")
    _require(len({w["total"] for w in windows}) == 1, "VOLUME_IDENTITY_OR_SIZE_CHANGED")
    span_seconds = (windows[-1]["end"] - windows[0]["start"]).total_seconds()
    _require(span_seconds >= MIN_SPAN_SECONDS, "HISTORY_SPAN_INSUFFICIENT")
    age = (now - windows[-1]["end"]).total_seconds()
    _require(-MAX_FUTURE_SECONDS <= age <= MAX_AGE_SECONDS, "LATEST_WINDOW_STALE_OR_FUTURE")

    # The largest observed depletion rate across ANY prior pair. This is a
    # descriptive sample statistic, NOT a bound on future write demand.
    worst_rate = 0.0
    for i, first in enumerate(windows):
        for last in windows[i + 1:]:
            elapsed = (last["end"] - first["end"]).total_seconds()
            _require(elapsed > 0, "INVALID_ELAPSED_TIME")
            worst_rate = max(worst_rate, (first["minimum"] - last["minimum"]) / elapsed)
    worst_rate = max(0.0, worst_rate)
    recent = windows[-1]
    projected_floor = recent["minimum"] - math.ceil(worst_rate * HORIZON_SECONDS) - BURST_BUFFER_BYTES
    reasons: list[str] = []
    if any(not w["source_capacity_pass"] for w in windows):
        reasons.append("HISTORICAL_CAPACITY_HOLD_PRESENT")
    if recent["minimum"] < RESERVE_GIB * GIB:
        reasons.append("LOCAL_DISK_BELOW_50_GIB")
    if recent["swing"] > SWING_GIB * GIB:
        reasons.append("LOCAL_SWING_ABOVE_2_GIB")
    if projected_floor < RESERVE_GIB * GIB:
        reasons.append("OBSERVED_TREND_EXHAUSTS_4H_PLANNING_RESERVE")
    reasons.append("DISPATCH_OWNER_NOT_ATTESTED_BY_STORAGE_RECEIPTS")

    # Receipt SHA-256s are local content commitments, not trusted signatures.
    digests = [sha256(x).hexdigest() for x in raw_receipts]
    chain = "0" * 64
    for digest in digests:
        chain = sha256(bytes.fromhex(chain) + bytes.fromhex(digest)).hexdigest()
    return {
        "schema": OUTPUT_SCHEMA,
        "state": "HOLD" if reasons else "ADVISORY_ONLY",
        "reasons": sorted(set(reasons)),
        "observed_window_count": len(windows),
        "observed_sample_count": sum(w["samples"] for w in windows),
        "history_span_seconds": round(span_seconds, 3),
        "latest_age_seconds": round(max(0.0, age), 3),
        "total_capacity_bytes": recent["total"],
        "latest_min_free_bytes": recent["minimum"],
        "latest_max_free_bytes": recent["maximum"],
        "latest_swing_bytes": recent["swing"],
        "max_observed_depletion_bytes_per_second": round(worst_rate, 6),
        "planning_horizon_seconds": HORIZON_SECONDS,
        "planning_buffer_bytes": BURST_BUFFER_BYTES,
        "hypothetical_projected_floor_bytes": projected_floor,
        "projection_is_statistical_confidence_bound": False,
        "receipt_chain_sha256": chain,
        "receipt_chain_is_signed_or_origin_verified": False,
        "dispatch_owner_verified": False,
        "training_ready": False,
        "training_launch_authorized": False,
        "salad_spend_or_scale_authorized": False,
        "cloud_provider_liveness_checked": False,
        "flagship54_controls_changed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        raw = []
        _require(MIN_WINDOWS <= len(args.receipt) <= MAX_WINDOWS, "INSUFFICIENT_OR_UNBOUNDED_HISTORY")
        for path in args.receipt:
            _require(path.is_file(), "RECEIPT_UNREADABLE")
            _require(path.stat().st_size <= MAX_SOURCE_BYTES, "RECEIPT_BYTE_BUDGET")
            raw.append(path.read_bytes())
        result = assess(raw)
    except (Hold, OSError) as error:
        result = {
            "schema": OUTPUT_SCHEMA,
            "state": "HOLD_UNOBSERVED",
            "reason": str(error) if isinstance(error, Hold) else "RECEIPT_IO_UNAVAILABLE",
            "training_ready": False,
            "training_launch_authorized": False,
            "salad_spend_or_scale_authorized": False,
        }
    print(json.dumps(result, sort_keys=True))
    return 78 if result["state"].startswith("HOLD") else 0


if __name__ == "__main__":
    sys.exit(main())
