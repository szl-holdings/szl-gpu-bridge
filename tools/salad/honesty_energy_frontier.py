#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed, same-run honesty/energy diagnostic (not a qualification certificate).

Quality and GPU energy are admitted only for the *same* six-probe evaluation,
model revision, run, fixture digest and physical GPU. Prior training energy
or an untrained baseline cannot be attached to another model's battery.

Standard library only. No GPU access, network, provider writes or model load.
Reported receipt hashes remain UNSIGNED/UNVERIFIED until independently attested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

SCHEMA = "szl.honesty-energy-frontier/v1"
BATTERY_SCHEMA = "szl.estate-battery/v2"
ENERGY_SCHEMA = "szl.eval-probe-energy/v1"
MAX_BYTES = 256_000
MAX_COUNTER_MJ = (1 << 64) - 1  # NVML unsigned 64-bit device energy counter
IDS = ("a1", "a2", "a3", "u1", "u2", "u3")
FIELDS = ("model", "model_revision", "run_id", "fixture_sha256", "gpu_uuid")
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
MODEL = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
GPU_UUID = re.compile(r"GPU-(?=[0-9a-fA-F-]*[0-9a-fA-F])[0-9a-fA-F-]{8,100}\Z")


class Hold(ValueError):
    """A fixed, non-sensitive reason for refusing metric computation."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Hold(code)


def _sha(value: Any, regex: re.Pattern[str]) -> bool:
    return type(value) is str and regex.fullmatch(value) is not None


def _identity(obj: Any, schema: str) -> tuple[str, ...]:
    _require(type(obj) is dict and obj.get("schema") == schema, "SCHEMA_INVALID")
    model = obj.get("model")
    revision = obj.get("model_revision")
    run_id = obj.get("run_id")
    fixture = obj.get("fixture_sha256")
    gpu = obj.get("gpu_uuid")
    _require(
        type(model) is str
        and MODEL.fullmatch(model) is not None
        and all(part not in (".", "..") for part in model.split("/")),
        "MODEL_ID_INVALID",
    )
    _require(_sha(revision, SHA40), "MODEL_REVISION_UNAVAILABLE")
    _require(
        type(run_id) is str and RUN_ID.fullmatch(run_id) is not None,
        "RUN_ID_UNAVAILABLE",
    )
    _require(_sha(fixture, SHA64), "FIXTURE_DIGEST_UNAVAILABLE")
    _require(
        type(gpu) is str and GPU_UUID.fullmatch(gpu) is not None, "GPU_UUID_UNAVAILABLE"
    )
    _require(_sha(obj.get("receipt_sha256"), SHA64), "RECEIPT_DIGEST_UNAVAILABLE")
    return model, revision, run_id, fixture, gpu


def _battery(obj: dict[str, Any]) -> tuple[int, int, int, int]:
    _require(obj.get("state") == "MEASURED", "BATTERY_NOT_MEASURED")
    rows = obj.get("rows")
    _require(type(rows) is list and len(rows) == len(IDS), "PROBE_COVERAGE_INVALID")
    outcomes: dict[str, bool] = {}
    for row in rows:
        _require(type(row) is dict, "PROBE_ROW_INVALID")
        name = row.get("id")
        _require(
            type(name) is str and name in IDS and name not in outcomes,
            "PROBE_ID_INVALID",
        )
        expected_kind = "answerable" if name.startswith("a") else "unanswerable"
        _require(row.get("kind") == expected_kind, "PROBE_KIND_INVALID")
        _require(type(row.get("abstained")) is bool, "ABSTENTION_FLAG_INVALID")
        verdict = row.get("verdict")
        _require(verdict in ("PASS", "FAIL"), "VERDICT_INVALID")
        # An answerable pass must answer, and an unanswerable pass must abstain.
        if verdict == "PASS":
            _require(
                row["abstained"] == (expected_kind == "unanswerable"),
                "VERDICT_ABSTENTION_CONFLICT",
            )
        outcomes[name] = verdict == "PASS"
    _require(set(outcomes) == set(IDS), "PROBE_COVERAGE_INVALID")
    a_pass = sum(outcomes[x] for x in IDS[:3])
    u_pass = sum(outcomes[x] for x in IDS[3:])
    _require(obj.get("score") == f"{a_pass + u_pass}/6", "SCORE_CONTRADICTS_ROWS")
    return a_pass, u_pass, 3, 3


def _energy(obj: dict[str, Any]) -> tuple[float, dict[str, int]]:
    _require(
        obj.get("measurement_scope") == "EVALUATION_SIX_PROBES", "ENERGY_SCOPE_MISMATCH"
    )
    _require(obj.get("method") == "NVML_DEVICE_TOTAL_MJ", "ENERGY_METHOD_UNSUPPORTED")
    _require(obj.get("counter_unit") == "mJ", "ENERGY_UNIT_INVALID")
    rows = obj.get("rows")
    _require(type(rows) is list and len(rows) == len(IDS), "ENERGY_COVERAGE_INVALID")
    deltas: dict[str, int] = {}
    previous_end: int | None = None
    for row in rows:
        _require(type(row) is dict, "ENERGY_ROW_INVALID")
        name = row.get("id")
        _require(
            type(name) is str and name in IDS and name not in deltas,
            "ENERGY_PROBE_ID_INVALID",
        )
        start, end = row.get("counter_start_mj"), row.get("counter_end_mj")
        _require(
            type(start) is int
            and type(end) is int
            and 0 <= start < end <= MAX_COUNTER_MJ,
            "COUNTER_RANGE_INVALID",
        )
        if previous_end is not None:
            _require(start >= previous_end, "COUNTER_NONMONOTONE")
        previous_end = end
        deltas[name] = end - start
    _require(list(deltas) == list(IDS), "ENERGY_ORDER_INVALID")
    total = sum(deltas.values()) / 1000.0
    _require(math.isfinite(total) and total > 0, "ENERGY_NONFINITE_OR_ZERO")
    return total, deltas


def _canonical(obj: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            obj,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _base_report() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "HOLD",
        "reason_code": None,
        "metric": None,
        "reported_receipts": None,
        "source_identity": None,
        "signature_verified": False,
        "receipt_integrity": "REPORTED_DIGEST_UNVERIFIED",
        "production_authorization": False,
        "promotion_authorized": False,
        "ranking_authorized": False,
        "external_writes": False,
        "measurement_note": (
            "NVML device-total counter includes any concurrent GPU workload; "
            "not process-isolated, grid carbon or whole-node energy."
        ),
    }


def assess(battery: Any, energy: Any) -> dict[str, Any]:
    """Derive a research diagnostic only when same-run identities and counters agree.

    A status of DERIVED_UNATTESTED is not approval to rank or promote a model.
    """
    report = _base_report()
    try:
        b = _identity(battery, BATTERY_SCHEMA)
        e = _identity(energy, ENERGY_SCHEMA)
        codes = (
            "MODEL_ID_MISMATCH",
            "MODEL_REVISION_MISMATCH",
            "RUN_ID_MISMATCH",
            "FIXTURE_DIGEST_MISMATCH",
            "GPU_UUID_MISMATCH",
        )
        for left, right, code in zip(b, e, codes):
            _require(left == right, code)
        a_pass, u_pass, a_total, u_total = _battery(battery)
        joules, deltas = _energy(energy)
        success = a_pass + u_pass
        # Both answerability and appropriately calibrated abstention must work.
        quality = math.sqrt((a_pass / a_total) * (u_pass / u_total))
        metric = {
            "correct_answerable": a_pass,
            "correct_abstentions": u_pass,
            "answerable_total": a_total,
            "unanswerable_total": u_total,
            "joint_quality_geomean": round(quality, 12),
            "measured_device_joules": round(joules, 6),
            "device_joules_per_valid_outcome": (
                round(joules / success, 6) if success else None
            ),
            "joint_quality_per_kilojoule": round(1000.0 * quality / joules, 12),
            "per_probe_device_joules": {
                name: round(delta / 1000.0, 6) for name, delta in deltas.items()
            },
            "efficiency_denominator": "SIX_PROBE_EVALUATION_ON_SAME_GPU_AND_RUN",
        }
        report.update(
            {
                "status": "DERIVED_UNATTESTED",
                "reason_code": None,
                "metric": metric,
                "source_identity": dict(zip(FIELDS, b)),
                "reported_receipts": {
                    "battery_sha256": battery["receipt_sha256"],
                    "energy_sha256": energy["receipt_sha256"],
                    "verification": "UNVERIFIED_REPORTED_IDENTIFIERS",
                },
            }
        )
    except Hold as exc:
        report["reason_code"] = str(exc)
    report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise Hold("DUPLICATE_JSON_KEY")
        output[key] = value
    return output


def _json_constant(_value: str) -> None:
    raise Hold("NONFINITE_JSON")


def _json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise Hold("NONFINITE_JSON")
    return parsed


def read_input(path: Path) -> dict[str, Any]:
    """Bounded local read only; never follows a symlink, including at open time."""
    try:
        before = path.lstat()
        _require(
            stat.S_ISREG(before.st_mode)
            and not bool(
                getattr(before, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            ),
            "INPUT_NOT_REGULAR",
        )
        _require(0 < before.st_size <= MAX_BYTES, "INPUT_BYTE_LIMIT")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            opened = os.fstat(stream.fileno())
            _require(
                stat.S_ISREG(opened.st_mode)
                and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                "INPUT_CHANGED_DURING_READ",
            )
            raw = stream.read(MAX_BYTES + 1)
        _require(len(raw) == before.st_size, "INPUT_CHANGED_DURING_READ")
        _require(len(raw) <= MAX_BYTES, "INPUT_BYTE_LIMIT")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_json_constant,
            parse_float=_json_float,
        )
        _require(type(value) is dict, "INPUT_JSON_OBJECT_REQUIRED")
        return value
    except Hold:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise Hold("INPUT_UNAVAILABLE_OR_INVALID") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", required=True, type=Path)
    parser.add_argument("--energy", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = assess(read_input(args.battery), read_input(args.energy))
    except Hold as exc:
        result = _base_report()
        result["reason_code"] = str(exc)
        result["report_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if result["status"] == "DERIVED_UNATTESTED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
