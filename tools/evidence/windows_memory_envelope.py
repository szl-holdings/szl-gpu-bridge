#!/usr/bin/env python3
"""Offline, source-only Windows memory-pressure observation for SZL.

Consumes a bounded, untrusted JSON snapshot of time-separated numeric readings.
Never samples hardware, starts training, cleans caches, changes services/tasks,
opens a network connection, signs evidence, or grants training admission. The
thresholds are conservative *policy assumptions*, not validated safety bounds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Any

SCHEMA = "szl.windows-memory-observation/v1"
OUTPUT_SCHEMA = "szl.windows-memory-envelope-advisory/v1"
GIB = 1024**3
MAX_INPUT_BYTES = 32_768
MIN_SAMPLES = 5
MAX_SAMPLES = 12
MIN_INTERVAL_SEC = 1
MAX_INTERVAL_SEC = 5
FREE_HOLD_BYTES = 2 * GIB
FREE_CAUTION_BYTES = 4 * GIB
COMMIT_HOLD_PPM = 850_000
COMMIT_CAUTION_PPM = 800_000
PAGEFILE_SWING_HOLD_PPM = 250_000
PAGEFILE_SWING_CAUTION_PPM = 100_000
REFUSED_EXIT = 78
SAMPLE_FIELDS = frozenset(
    {
        "observed_utc",
        "physical_total_bytes",
        "physical_free_bytes",
        "committed_bytes",
        "commit_limit_bytes",
        "pagefile_allocated_bytes",
    }
)


class ObservationError(ValueError):
    """Controlled reason codes; never include user-provided data in diagnostics."""


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ObservationError(reason)


def _positive_integer(value: Any, *, allow_zero: bool = True) -> bool:
    return type(value) is int and (value >= 0 if allow_zero else value > 0)


def _utc(value: Any) -> datetime:
    _need(type(value) is str and 20 <= len(value) <= 48, "INVALID_OBSERVATION_TIME")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError):
        raise ObservationError("INVALID_OBSERVATION_TIME") from None
    _need(
        stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0),
        "NON_UTC_OBSERVATION_TIME",
    )
    return stamp


def _ppm_ceil(part: int, whole: int) -> int:
    return (part * 1_000_000 + whole - 1) // whole


def analyze(document: Any) -> dict[str, Any]:
    """Pure classifier; success is *observation only*, never dispatch authority."""
    _need(
        type(document) is dict and set(document) == {"schema", "samples"},
        "INVALID_ROOT_FIELDS",
    )
    _need(document["schema"] == SCHEMA, "INVALID_SCHEMA")
    samples = document["samples"]
    _need(
        type(samples) is list and MIN_SAMPLES <= len(samples) <= MAX_SAMPLES,
        "SAMPLE_COVERAGE_INVALID",
    )
    timestamps: list[datetime] = []
    frees: list[int] = []
    totals: list[int] = []
    commitments: list[int] = []
    committed_ratios: list[int] = []
    pagefiles: list[int] = []
    for row in samples:
        _need(type(row) is dict and set(row) == SAMPLE_FIELDS, "INVALID_SAMPLE_FIELDS")
        point = _utc(row["observed_utc"])
        if timestamps:
            interval = (point - timestamps[-1]).total_seconds()
            _need(
                MIN_INTERVAL_SEC <= interval <= MAX_INTERVAL_SEC,
                "INVALID_SAMPLE_SEPARATION",
            )
        timestamps.append(point)
        total = row["physical_total_bytes"]
        free = row["physical_free_bytes"]
        committed = row["committed_bytes"]
        limit = row["commit_limit_bytes"]
        pagefile = row["pagefile_allocated_bytes"]
        _need(
            _positive_integer(total, allow_zero=False)
            and _positive_integer(free)
            and _positive_integer(committed)
            and _positive_integer(limit, allow_zero=False)
            and _positive_integer(pagefile),
            "INVALID_CAPACITY_VALUE",
        )
        _need(free <= total and committed <= limit, "INCONSISTENT_MEMORY_ACCOUNTING")
        frees.append(free)
        totals.append(total)
        commitments.append(committed)
        committed_ratios.append(_ppm_ceil(committed, limit))
        pagefiles.append(pagefile)
    _need(min(totals) == max(totals), "PHYSICAL_MEMORY_IDENTITY_CHANGED")

    free_min = min(frees)
    commit_ratio_peak = max(committed_ratios)
    pagefile_min = min(pagefiles)
    pagefile_max = max(pagefiles)
    swing = pagefile_max - pagefile_min
    # An initially disabled/zero pagefile must not cause division by zero or
    # silently become a stable claim. New allocation from zero is volatile.
    pagefile_swing_ppm = _ppm_ceil(swing, pagefile_min) if pagefile_min else (1_000_000 if swing else 0)
    red: list[str] = []
    amber: list[str] = []
    if free_min < FREE_HOLD_BYTES:
        red.append("LOW_FREE_PHYSICAL_MEMORY")
    elif free_min < FREE_CAUTION_BYTES:
        amber.append("LIMITED_FREE_PHYSICAL_MEMORY")
    if commit_ratio_peak > COMMIT_HOLD_PPM:
        red.append("HIGH_COMMIT_FRACTION")
    elif commit_ratio_peak > COMMIT_CAUTION_PPM:
        amber.append("ELEVATED_COMMIT_FRACTION")
    if pagefile_swing_ppm > PAGEFILE_SWING_HOLD_PPM:
        red.append("VOLATILE_PAGEFILE_ALLOCATION")
    elif pagefile_swing_ppm > PAGEFILE_SWING_CAUTION_PPM:
        amber.append("PAGEFILE_ALLOCATION_CHANGE")
    if pagefile_min == 0 and pagefile_max == 0:
        amber.append("PAGEFILE_DISABLED_OR_UNOBSERVED")

    state = (
        "ADVISORY_HOLD"
        if red
        else "ADVISORY_CAUTION"
        if amber
        else "OBSERVATION_STABLE_ONLY"
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "state": state,
        "reasons": sorted(red + amber),
        "sample_count": len(samples),
        "first_observed_utc": timestamps[0].isoformat(),
        "last_observed_utc": timestamps[-1].isoformat(),
        "physical_total_bytes": totals[0],
        "minimum_free_physical_bytes": free_min,
        "maximum_committed_fraction_ppm": commit_ratio_peak,
        "minimum_pagefile_allocated_bytes": pagefile_min,
        "maximum_pagefile_allocated_bytes": pagefile_max,
        "pagefile_swing_ppm": pagefile_swing_ppm,
        "policy_assumptions": {
            "free_hold_bytes_below": FREE_HOLD_BYTES,
            "free_caution_bytes_below": FREE_CAUTION_BYTES,
            "commit_hold_ppm_above": COMMIT_HOLD_PPM,
            "commit_caution_ppm_above": COMMIT_CAUTION_PPM,
            "pagefile_swing_hold_ppm_above": PAGEFILE_SWING_HOLD_PPM,
            "pagefile_swing_caution_ppm_above": PAGEFILE_SWING_CAUTION_PPM,
        },
        "source_or_signer_authenticated": False,
        "training_ready": False,
        "grants_training_dispatch": False,
        "changes_existing_services_or_tasks": False,
        "proves_memory_stability_outside_window": False,
    }


def strict_document(raw: bytes) -> Any:
    _need(type(raw) is bytes and 0 < len(raw) <= MAX_INPUT_BYTES, "INPUT_SIZE_INVALID")

    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            _need(key not in obj, "DUPLICATE_JSON_KEY")
            obj[key] = value
        return obj

    def no_noninteger(_: str) -> None:
        raise ObservationError("NONINTEGER_NUMBER")

    try:
        return json.loads(
            raw,
            object_pairs_hook=no_duplicate_keys,
            parse_float=no_noninteger,
            parse_constant=no_noninteger,
        )
    except ObservationError:
        raise
    except (UnicodeError, RecursionError, ValueError):
        raise ObservationError("INPUT_JSON_INVALID") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        with args.input.open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
        result = analyze(strict_document(raw))
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0 if result["state"] == "OBSERVATION_STABLE_ONLY" else REFUSED_EXIT
    except (ObservationError, OSError) as exc:
        code = str(exc) if isinstance(exc, ObservationError) else "INPUT_UNAVAILABLE"
        print(json.dumps({"schema": OUTPUT_SCHEMA, "state": "UNOBSERVED", "reasons": [code], "training_ready": False, "grants_training_dispatch": False}, sort_keys=True))
        return REFUSED_EXIT


if __name__ == "__main__":
    sys.exit(main())
