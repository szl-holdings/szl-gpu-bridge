"""Offline reconciliation of recorded worker energy; never an estate meter.

This module reads local JSON only. It does not access credentials, upload,
restart workloads, sign receipts, or infer training completion. A matching
receipt hash is integrity, not signer authenticity or physical meter validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "szl.salad-worker/v1"
METHOD = "nvmlDeviceGetTotalEnergyConsumption cumulative-counter delta (device level)"
MAX_BYTES = 2_000_000
MAX_FILES = 10_000


class EvidenceHold(ValueError):
    """A fixed, secret-free explanation of excluded evidence."""


def _pairs(items):
    result = {}
    for name, value in items:
        if name in result:
            raise EvidenceHold("DUPLICATE_JSON_KEY")
        result[name] = value
    return result


def _constant(_value):
    raise EvidenceHold("NONFINITE_JSON")


def _number(value):
    if type(value) not in (int, float):
        raise EvidenceHold("INVALID_ENERGY_NUMBER")
    try:
        finite = math.isfinite(value)
    except (OverflowError, ValueError):
        finite = False
    if not finite or value < 0:
        raise EvidenceHold("INVALID_ENERGY_NUMBER")
    return value


def inspect_receipt(raw: bytes) -> dict:
    """Admit only the named, hash-consistent, counter-backed delta schema."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
        raise EvidenceHold("RECEIPT_BYTE_BUDGET")
    try:
        receipt = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        if isinstance(error, EvidenceHold):
            raise
        raise EvidenceHold("INVALID_RECEIPT_JSON") from error
    if not isinstance(receipt, dict):
        raise EvidenceHold("RECEIPT_NOT_OBJECT")
    if receipt.get("kind") != SCHEMA:
        raise EvidenceHold("UNSUPPORTED_RECEIPT_SCHEMA")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or re.fullmatch(r"[0-9a-f]{64}", claimed) is None:
        raise EvidenceHold("RECEIPT_HASH_UNAVAILABLE")
    payload = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        canonical = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    except (ValueError, RecursionError, OverflowError) as error:
        raise EvidenceHold("INVALID_RECEIPT_JSON") from error
    if hashlib.sha256(canonical).hexdigest() != claimed:
        raise EvidenceHold("RECEIPT_HASH_MISMATCH")
    delta = receipt.get("energy_j")
    start, end = receipt.get("energy_start"), receipt.get("energy_end")
    if not isinstance(delta, dict):
        raise EvidenceHold("EXPLICIT_DELTA_UNAVAILABLE")
    if delta.get("label") != "MEASURED":
        raise EvidenceHold("DELTA_NOT_MEASURED")
    if delta.get("method") != METHOD:
        raise EvidenceHold("DELTA_METHOD_UNSUPPORTED")
    if not isinstance(start, dict) or not isinstance(end, dict):
        raise EvidenceHold("COUNTER_PAIR_UNAVAILABLE")
    if start.get("label") != "MEASURED" or end.get("label") != "MEASURED":
        raise EvidenceHold("COUNTERS_NOT_MEASURED")
    gpu = start.get("gpu")
    if not isinstance(gpu, str) or not gpu.strip() or end.get("gpu") != gpu:
        raise EvidenceHold("COUNTER_DEVICE_LABEL_MISMATCH")
    a, b, joules = (
        _number(start.get("energy_j")),
        _number(end.get("energy_j")),
        _number(delta.get("joules")),
    )
    if b < a:
        raise EvidenceHold("COUNTER_RESET")
    expected = round(b - a, 1)  # Exact documented producer rounding, not a unit guess.
    if not math.isclose(joules, expected, rel_tol=0, abs_tol=1e-9):
        raise EvidenceHold("DELTA_COUNTER_MISMATCH")
    adapter = receipt.get("adapter")
    if (
        not isinstance(adapter, str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", adapter) is None
    ):
        adapter = None
    return {
        "receipt_sha256": claimed,
        "recorded_delta_joules": joules,
        "adapter": adapter,
        "integrity": "SELF_HASH_AND_COUNTER_ARITHMETIC_VERIFIED",
        "device_identity_verified": False,
        "interval_nonoverlap_verified": False,
        "physical_measurement_independently_verified": False,
        "training_completion_verified": False,
    }


def reconcile(directory: Path) -> dict:
    """Summarize bounded local inputs; unknown coverage/attribution remain explicit."""
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise EvidenceHold("RECEIPT_DIRECTORY_UNAVAILABLE")
    base = directory.resolve()
    paths = []
    for path in directory.rglob("salad-*.json"):
        if ".cache" in path.relative_to(directory).parts:
            continue
        paths.append(path)
        if len(paths) > MAX_FILES:
            raise EvidenceHold("RECEIPT_FILE_BUDGET")
    exclusions, admitted, seen, inputs = Counter(), [], set(), []
    duplicates = 0
    for path in sorted(paths):
        record = {"path": path.relative_to(directory).as_posix()}
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(base):
                raise EvidenceHold("RECEIPT_PATH_ESCAPE")
            with path.open("rb") as source:
                raw = source.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise EvidenceHold("RECEIPT_BYTE_BUDGET")
            record["sha256"] = hashlib.sha256(raw).hexdigest()
            evidence = inspect_receipt(raw)
            digest = evidence["receipt_sha256"]
            if digest in seen:
                duplicates += 1
                record["state"] = "DUPLICATE_RECEIPT"
            else:
                seen.add(digest)
                admitted.append(evidence)
                record["state"] = "VALID_RECORDED_DEVICE_DELTA"
        except EvidenceHold as error:
            record["state"] = str(error)
            exclusions[str(error)] += 1
        except OSError:
            record["state"] = "RECEIPT_READ_UNAVAILABLE"
            exclusions[record["state"]] += 1
        inputs.append(record)
    subtotal = None
    if admitted:
        try:
            subtotal = math.fsum(r["recorded_delta_joules"] for r in admitted)
        except OverflowError as error:
            raise EvidenceHold("DELTA_SUBTOTAL_OVERFLOW") from error
        if not math.isfinite(subtotal):
            raise EvidenceHold("DELTA_SUBTOTAL_OVERFLOW")
    manifest = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "szl.worker-energy-reconciliation/v1",
        "scope": "LOCAL_CACHED_RECEIPT_REPLAY_NOT_LIVE_ESTATE",
        "files_examined": len(inputs),
        "valid_unique_delta_receipts": len(admitted),
        "duplicate_receipts": duplicates,
        "excluded_receipts": dict(sorted(exclusions.items())),
        "recorded_device_delta_subtotal_joules": subtotal,
        "subtotal_scope": "arithmetic diagnostic; not deduplicated by physical device or interval",
        "total_measured_joules": None,
        "estate_energy_state": "UNVERIFIED",
        "estate_energy_reasons": [
            "DEVICE_IDENTITY_AND_INTERVAL_NONOVERLAP_UNPROVEN",
            "LOCAL_CACHE_COMPLETENESS_AND_FRESHNESS_UNPROVEN",
        ],
        "adapter_unattributed_delta_receipts": sum(
            r["adapter"] is None for r in admitted
        ),
        "input_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "inputs": inputs,
        "signature": "UNSIGNED",
        "training_completion_verified": False,
        "provider_mutation": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = reconcile(args.receipt_dir)
        encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        # Never replace an old digest or silently rewrite an observation.
        with args.output.open("xb") as target:
            target.write(encoded)
        summary = {k: v for k, v in result.items() if k != "inputs"}
        print(json.dumps(summary, allow_nan=False))
    except (EvidenceHold, OSError) as error:
        print(
            json.dumps(
                {
                    "state": "HOLD",
                    "reason": str(error)
                    if isinstance(error, EvidenceHold)
                    else type(error).__name__,
                }
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
