#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded same-run probe/energy capture, with no provider, job or upload authority.

An authorized caller supplies six fixed fixtures, a pinned-source declaration,
a callable inference function, an explicit adjudicator, and an NVML-compatible
device-total meter. This module never loads a model or dispatches a job itself.

All evidence is local, source-declared and UNSIGNED. A successful diagnostic is
DERIVED_UNATTESTED, not model qualification or a production energy claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Protocol

from tools.salad.honesty_energy_frontier import (
    BATTERY_SCHEMA,
    ENERGY_SCHEMA,
    assess,
)

PROBE_IDS = ("a1", "a2", "a3", "u1", "u2", "u3")
FIXTURE_SCHEMA = "szl.honesty-probe-fixture/v1"
MAX_PROMPT_CHARS = 8192
MAX_REFERENCE_CHARS = 1024


class NvmlDeviceMeter:
    """Opt-in, local GPU device-total mJ counter; no model or network access.

    An observed UUID and counter do not attest process isolation or ownership.
    Use in a context manager, after the GPU lane has been separately admitted.
    """

    method = "NVML_DEVICE_TOTAL_MJ"

    def __init__(self, index: int = 0):
        if type(index) is not int or not 0 <= index <= 15:
            raise CaptureHold("NVML_DEVICE_INDEX_INVALID")
        self._nvml = None
        self._active = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._active = True
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            raw_uuid = pynvml.nvmlDeviceGetUUID(self._handle)
            self.device_uuid = (
                raw_uuid.decode("ascii", errors="strict")
                if isinstance(raw_uuid, bytes)
                else raw_uuid
            )
            _require(
                type(self.device_uuid) is str
                and re.fullmatch(r"GPU-[a-fA-F0-9-]{8,100}", self.device_uuid)
                is not None,
                "NVML_UUID_UNAVAILABLE",
            )
        except CaptureHold:
            self.close()
            raise
        except Exception:
            self.close()
            raise CaptureHold("NVML_INIT_UNAVAILABLE") from None

    def read_mj(self) -> int:
        _require(self._active and self._nvml is not None, "METER_UNAVAILABLE")
        try:
            counter = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except Exception:
            raise CaptureHold("METER_UNAVAILABLE") from None
        _require(type(counter) is int and 0 <= counter < 2**64, "METER_COUNTER_INVALID")
        return counter

    def close(self) -> None:
        if self._active and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            finally:
                self._active = False

    def __enter__(self):
        return self

    def __exit__(self, _kind, _value, _traceback):
        self.close()
        return False


class Meter(Protocol):
    device_uuid: str
    method: str

    def read_mj(self) -> int | None: ...


@dataclass(frozen=True)
class CaptureIdentity:
    model: str
    model_revision: str
    run_id: str
    fixture_sha256: str
    gpu_uuid: str


class CaptureHold(ValueError):
    """Only fixed reason codes may escape capture; no raw prompts or responses."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise CaptureHold(reason)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def fixture_sha256(rows: list[dict[str, Any]]) -> str:
    """Canonical digest includes exact prompts/reference labels; receipt does not."""
    return hashlib.sha256(
        canonical({"schema": FIXTURE_SCHEMA, "rows": rows})
    ).hexdigest()


def _validate_fixture(rows: Any) -> None:
    _require(type(rows) is list and len(rows) == 6, "FIXTURE_COVERAGE_INVALID")
    for i, row in enumerate(rows):
        _require(
            type(row) is dict and set(row) == {"id", "kind", "prompt", "reference"},
            "FIXTURE_SCHEMA_INVALID",
        )
        name = PROBE_IDS[i]
        _require(row["id"] == name, "FIXTURE_ORDER_INVALID")
        _require(
            row["kind"] == ("answerable" if i < 3 else "unanswerable"),
            "FIXTURE_KIND_INVALID",
        )
        _require(
            type(row["prompt"]) is str and 0 < len(row["prompt"]) <= MAX_PROMPT_CHARS,
            "FIXTURE_PROMPT_INVALID",
        )
        if i < 3:
            _require(
                type(row["reference"]) is str
                and 0 < len(row["reference"]) <= MAX_REFERENCE_CHARS,
                "FIXTURE_REFERENCE_INVALID",
            )
        else:
            _require(row["reference"] is None, "UNANSWERABLE_HAS_REFERENCE")


def _validate_identity(identity: CaptureIdentity) -> None:
    _require(type(identity) is CaptureIdentity, "IDENTITY_INVALID")
    _require(
        type(identity.model) is str
        and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", identity.model) is not None
        and all(p not in {".", ".."} for p in identity.model.split("/")),
        "MODEL_ID_INVALID",
    )
    _require(
        type(identity.model_revision) is str
        and re.fullmatch("[a-f0-9]{40}", identity.model_revision) is not None,
        "MODEL_REVISION_UNAVAILABLE",
    )
    _require(
        type(identity.run_id) is str
        and re.fullmatch("[A-Za-z0-9_-]{1,64}", identity.run_id) is not None,
        "RUN_ID_UNAVAILABLE",
    )
    _require(
        type(identity.fixture_sha256) is str
        and re.fullmatch("[a-f0-9]{64}", identity.fixture_sha256) is not None,
        "FIXTURE_DIGEST_UNAVAILABLE",
    )
    _require(
        type(identity.gpu_uuid) is str
        and re.fullmatch(r"GPU-[a-fA-F0-9-]{8,100}", identity.gpu_uuid) is not None,
        "GPU_UUID_UNAVAILABLE",
    )


def _result_hold(reason: str) -> dict[str, Any]:
    return {
        "schema": "szl.probe-energy-capture/v1",
        "status": "HOLD",
        "reason_code": reason,
        "battery": None,
        "energy": None,
        "diagnostic": None,
        "provenance": "SOURCE_DECLARED_UNATTESTED",
        "producer_external_writes": False,
        "callback_external_effects": "UNVERIFIED",
        "external_writes": None,
        "model_executed": "UNKNOWN",
        "production_authorization": False,
        "promotion_authorized": False,
    }


def _receipt_sha256(receipt: dict[str, Any]) -> str:
    """A local payload digest only: not a signature or proof of GPU ownership."""
    return hashlib.sha256(canonical(receipt)).hexdigest()


def capture(
    identity: CaptureIdentity,
    rows: list[dict[str, Any]],
    invoke: Callable[[str], Any],
    judge: Callable[[dict[str, Any], Any], dict[str, Any]],
    meter: Meter,
) -> dict[str, Any]:
    """Evaluate an *already admitted* model callback under a six-probe budget.

    No invocation may begin until fixture hash, GPU UUID, method and pin syntax
    match. Any error returns HOLD and discards partial result bodies.
    """
    calls = 0
    try:
        _validate_identity(identity)
        _validate_fixture(rows)
        _require(
            fixture_sha256(rows) == identity.fixture_sha256,
            "FIXTURE_DIGEST_MISMATCH",
        )
        _require(callable(invoke) and callable(judge), "CALLBACK_UNAVAILABLE")
        _require(
            getattr(meter, "method", None) == "NVML_DEVICE_TOTAL_MJ",
            "METER_METHOD_UNAVAILABLE",
        )
        _require(
            getattr(meter, "device_uuid", None) == identity.gpu_uuid,
            "METER_DEVICE_MISMATCH",
        )

        outcome_rows: list[dict[str, Any]] = []
        energy_rows: list[dict[str, Any]] = []
        previous_end = None
        for row in rows:
            try:
                start = meter.read_mj()
            except Exception:
                raise CaptureHold("METER_UNAVAILABLE") from None
            _require(type(start) is int and 0 <= start < 2**64, "METER_COUNTER_INVALID")
            if previous_end is not None:
                _require(start >= previous_end, "METER_COUNTER_RESET")
            try:
                calls += 1
                response = invoke(row["prompt"])
            except Exception:
                raise CaptureHold("INFERENCE_UNAVAILABLE") from None
            try:
                end = meter.read_mj()
            except Exception:
                raise CaptureHold("METER_UNAVAILABLE") from None
            _require(
                type(end) is int and start < end < 2**64,
                "METER_COUNTER_RESET_OR_UNMEASURED",
            )
            previous_end = end
            try:
                verdict = judge(row, response)
            except Exception:
                raise CaptureHold("ADJUDICATION_UNAVAILABLE") from None
            _require(
                type(verdict) is dict
                and set(verdict) == {"abstained", "verdict"}
                and type(verdict["abstained"]) is bool
                and type(verdict["verdict"]) is str
                and verdict["verdict"] in ("PASS", "FAIL"),
                "ADJUDICATION_INVALID",
            )
            # Neither model output nor prompt/reference content is persisted.
            outcome_rows.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "abstained": verdict["abstained"],
                    "verdict": verdict["verdict"],
                }
            )
            energy_rows.append(
                {
                    "id": row["id"],
                    "counter_start_mj": start,
                    "counter_end_mj": end,
                }
            )

        pins = {
            "model": identity.model,
            "model_revision": identity.model_revision,
            "run_id": identity.run_id,
            "fixture_sha256": identity.fixture_sha256,
            "gpu_uuid": identity.gpu_uuid,
        }
        battery = {
            "schema": BATTERY_SCHEMA,
            **pins,
            "state": "MEASURED",
            "score": f"{sum(r['verdict'] == 'PASS' for r in outcome_rows)}/6",
            "rows": outcome_rows,
            "adjudication": "CALLBACK_REPORTED_UNVERIFIED",
        }
        battery["receipt_sha256"] = _receipt_sha256(battery)
        energy = {
            "schema": ENERGY_SCHEMA,
            **pins,
            "measurement_scope": "EVALUATION_SIX_PROBES",
            "method": "NVML_DEVICE_TOTAL_MJ",
            "counter_unit": "mJ",
            "rows": energy_rows,
            "energy_scope": "GPU_DEVICE_TOTAL_NOT_PROCESS_ISOLATED",
        }
        energy["receipt_sha256"] = _receipt_sha256(energy)
        diagnostic = assess(battery, energy)
        _require(
            diagnostic["status"] == "DERIVED_UNATTESTED",
            diagnostic.get("reason_code") or "DIAGNOSTIC_HOLD",
        )
        return {
            "schema": "szl.probe-energy-capture/v1",
            "status": "DERIVED_UNATTESTED",
            "reason_code": None,
            "battery": battery,
            "energy": energy,
            "diagnostic": diagnostic,
            "provenance": "SOURCE_DECLARED_UNATTESTED",
            "producer_external_writes": False,
            "callback_external_effects": "UNVERIFIED",
            "external_writes": None,
            "model_executed": "CALLBACK_INVOKED_UNVERIFIED",
            "production_authorization": False,
            "promotion_authorized": False,
        }
    except CaptureHold as exc:
        # No inferred energy, no partial eval score, no leaked callback exception.
        result = _result_hold(str(exc))
        result["model_executed"] = (
            "CALLBACK_INVOKED_UNVERIFIED" if calls else "NOT_INVOKED"
        )
        return result
