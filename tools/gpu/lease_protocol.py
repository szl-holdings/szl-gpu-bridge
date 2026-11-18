#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""SZL GPU lease/proof prototype — deterministic host-only, no GPU execution.

Original Python implementation informed by GPUOS/FuseOS (MIT; pinned upstream
9eb66b9c68a3e449fdb3b2bd6edf3ddf6bd6d2ca), OpenInfer and XSched.
NO upstream code is copied. The GPUOS research prototype itself is not tenant
isolation, hardware preemption or proven execution.

This reference ledger models prepare/grant/complete/revoke/ack/retire under one
lock and treats backend return masks, code epochs and logical memory separately.
The "proof" is a structural hash of an untrusted operator simulation, NOT a
signature, provider attestation, GPU completion event or deployment authority.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys
import threading
from typing import Any

SCHEMA = "szl.gpu-lease-replay/v1"
RECEIPT_SCHEMA = "szl.gpu-lease-structural-observation/v1"
MAX_INPUT_BYTES = 131_072
MAX_EVENTS = 96
MAX_TASKS = 16
MAX_CALIBRATIONS = 4096
MIN_CALIBRATIONS = 300
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
BACKENDS = frozenset(("finite_sliced", "cooperative_resident"))
EVENT_OPS = frozenset(
    (
        "prepare",
        "grant",
        "complete",
        "revoke",
        "ack",
        "reader_add",
        "reader_done",
        "retire",
        "publish_epoch",
    )
)
DEFAULT_CONFIDENCE_PPM = 950000


class Hold(ValueError):
    """An allowlisted error code only; never echo an untrusted input."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise Hold(code)


def exact(value: Any, fields: set[str], code: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == fields, code)
    return value


def integer(value: Any, minimum: int, maximum: int, code: str) -> int:
    require(type(value) is int and minimum <= value <= maximum, code)
    return value


def name(value: Any, code: str) -> str:
    require(type(value) is str and NAME.fullmatch(value) is not None, code)
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class LeaseToken:
    task: str
    generation: int
    epoch: int

    def __post_init__(self) -> None:
        name(self.task, "TASK_ID_INVALID")
        integer(self.generation, 1, 2**31 - 1, "GENERATION_INVALID")
        integer(self.epoch, 1, 2**31 - 1, "CODE_EPOCH_INVALID")

    def display(self) -> str:
        return f"{self.task}:{self.generation}:{self.epoch}"


@dataclass
class Lease:
    token: LeaseToken
    backend: str
    memory_mib: int
    trusted: bool
    submitted: int = 0
    committed: int = 0
    last_grant: int = 0
    inflight: int | None = None
    revoke_frontier: int | None = None
    acknowledged: bool = False
    retired: bool = False
    readers: int = 0
    completion_digest: str | None = None


class LeaseLedger:
    """Single-device simulated protocol; one in-flight slice per task.

    The same lock linearizes grant and revoke; no "pause" can falsely claim
    physical CTA/memory return. Epochs cannot be replaced with live callers.
    """

    def __init__(self, *, memory_limit_mib: int, max_tasks: int = MAX_TASKS):
        self.memory_limit_mib = integer(
            memory_limit_mib, 1, 1_048_576, "MEMORY_LIMIT_INVALID"
        )
        self.max_tasks = integer(max_tasks, 1, MAX_TASKS, "TASK_LIMIT_INVALID")
        self.used_mib = 0
        self.epoch = 1
        self.leases: dict[LeaseToken, Lease] = {}
        self.high_generation: dict[str, int] = {}
        self._lock = threading.RLock()

    def _get(self, token: LeaseToken) -> Lease:
        lease = self.leases.get(token)
        require(lease is not None and not lease.retired, "STALE_OR_ABSENT_LEASE")
        return lease

    def prepare(
        self, token: LeaseToken, *, backend: str, memory_mib: int, trusted: bool
    ) -> str:
        require(type(trusted) is bool and trusted, "TRUST_DOMAIN_UNVERIFIED")
        require(type(backend) is str and backend in BACKENDS, "BACKEND_UNSUPPORTED")
        integer(memory_mib, 1, self.memory_limit_mib, "MEMORY_REQUEST_INVALID")
        with self._lock:
            require(token.epoch == self.epoch, "CODE_EPOCH_STALE")
            require(
                token.generation > self.high_generation.get(token.task, 0),
                "GENERATION_REPLAY",
            )
            require(
                not any(
                    t.task == token.task and not x.retired
                    for t, x in self.leases.items()
                ),
                "TASK_SLOT_LIVE",
            )
            require(
                sum(not x.retired for x in self.leases.values()) < self.max_tasks,
                "TASK_CAP_REACHED",
            )
            require(
                self.used_mib + memory_mib <= self.memory_limit_mib,
                "RESIDENT_MEMORY_EXCEEDED",
            )
            self.leases[token] = Lease(
                token=token,
                backend=backend,
                memory_mib=memory_mib,
                trusted=trusted,
            )
            self.high_generation[token.task] = token.generation
            self.used_mib += memory_mib
            return "PREPARED_NO_EXECUTION"

    def grant(self, token: LeaseToken, *, start: int, extent: int) -> int:
        integer(start, 0, 2**31 - 1, "CURSOR_INVALID")
        integer(extent, 1, 1_000_000, "GRANT_EXTENT_INVALID")
        with self._lock:
            lease = self._get(token)
            require(lease.revoke_frontier is None, "GRANTS_CLOSED")
            require(lease.inflight is None, "INFLIGHT_GRANT_EXISTS")
            require(start == lease.submitted, "SUBMITTED_CURSOR_MISMATCH")
            require(start + extent <= 2**31 - 1, "CURSOR_OVERFLOW")
            lease.submitted += extent
            lease.last_grant += 1
            lease.inflight = lease.last_grant
            return lease.inflight

    def complete(
        self,
        token: LeaseToken,
        *,
        grant_id: int,
        committed: int,
        completion_digest: str,
    ) -> str:
        integer(grant_id, 1, 2**31 - 1, "GRANT_ID_INVALID")
        integer(committed, 0, 2**31 - 1, "COMMITTED_CURSOR_INVALID")
        require(
            type(completion_digest) is str
            and SHA64.fullmatch(completion_digest) is not None,
            "COMPLETION_EVENT_DIGEST_INVALID",
        )
        with self._lock:
            lease = self._get(token)
            require(lease.inflight == grant_id, "STALE_OR_ABSENT_GRANT")
            require(
                lease.committed < committed == lease.submitted,
                "COMMIT_BEFORE_DEVICE_COMPLETION_UNVERIFIED",
            )
            lease.committed = committed
            lease.completion_digest = completion_digest
            lease.inflight = None
            return "COMPLETION_REPORTED_NOT_GPU_VERIFIED"

    def revoke(self, token: LeaseToken) -> int:
        with self._lock:
            lease = self._get(token)
            require(lease.revoke_frontier is None, "ALREADY_REVOKED")
            # Same lock as grant: a concurrent grant either precedes this
            # frontier or is refused forever after this point.
            lease.revoke_frontier = lease.last_grant
            return lease.revoke_frontier

    def ack(self, token: LeaseToken) -> dict[str, bool]:
        with self._lock:
            lease = self._get(token)
            require(lease.revoke_frontier is not None, "NOT_REVOKED")
            require(not lease.acknowledged, "ACK_REPLAY")
            require(
                lease.inflight is None and lease.committed == lease.submitted,
                "INFLIGHT_NOT_DRAINED",
            )
            lease.acknowledged = True
            return {
                "future_issue_closed": True,
                # Only a finished finite slice *can* free its CTA slots.
                # This is a logical backend mask, not verified device state.
                "cta_return_reported": lease.backend == "finite_sliced",
                "memory_return_reported": False,
                "code_unload_safe": False,
            }

    def reader_add(self, token: LeaseToken) -> int:
        with self._lock:
            lease = self._get(token)
            require(lease.revoke_frontier is None, "GRANTS_CLOSED")
            require(not lease.acknowledged, "ACK_ALREADY_PUBLISHED")
            require(lease.readers < 64, "READER_LIMIT")
            lease.readers += 1
            return lease.readers

    def reader_done(self, token: LeaseToken) -> int:
        with self._lock:
            lease = self._get(token)
            require(lease.readers > 0, "READER_UNDERFLOW")
            lease.readers -= 1
            return lease.readers

    def retire(self, token: LeaseToken) -> str:
        with self._lock:
            lease = self._get(token)
            require(lease.acknowledged, "NO_COMPLETION_ACK")
            require(lease.readers == 0, "CONSUMER_STILL_LIVE")
            require(lease.inflight is None, "INFLIGHT_NOT_DRAINED")
            self.used_mib -= lease.memory_mib
            require(self.used_mib >= 0, "MEMORY_ACCOUNTING_INVALID")
            lease.retired = True
            return "RETIRED_LOGICAL_ONLY"

    def publish_epoch(self, *, next_epoch: int) -> int:
        integer(next_epoch, 2, 2**31 - 1, "CODE_EPOCH_INVALID")
        with self._lock:
            require(next_epoch > self.epoch, "CODE_EPOCH_REPLAY")
            require(
                not any(not x.retired for x in self.leases.values()),
                "LIVE_CODE_CALLERS",
            )
            self.epoch = next_epoch
            return self.epoch

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = []
            for token, lease in sorted(
                self.leases.items(), key=lambda kv: kv[0].display()
            ):
                rows.append(
                    {
                        "token": token.display(),
                        "backend": lease.backend,
                        "reserved_mib": lease.memory_mib,
                        "submitted": lease.submitted,
                        "committed": lease.committed,
                        "grant_frontier": lease.last_grant,
                        "inflight": lease.inflight,
                        "revoked": lease.revoke_frontier,
                        "acknowledged": lease.acknowledged,
                        "readers": lease.readers,
                        "retired": lease.retired,
                        "completion_digest": lease.completion_digest,
                    }
                )
            return {
                "code_epoch": self.epoch,
                "memory_limit_mib": self.memory_limit_mib,
                "memory_reserved_mib": self.used_mib,
                "leases": rows,
                "gpu_state_observed": False,
            }


def checkpoint_interval(spec: Any) -> dict[str, Any]:
    fields = {
        "slice_overhead_us",
        "overhead_budget_ppm",
        "pause_target_us",
        "publication_us",
        "drain_us",
        "ack_us",
    }
    data = exact(spec, fields, "CHECKPOINT_FIELDS_INVALID")
    h = integer(data["slice_overhead_us"], 1, 1_000_000, "SLICE_OVERHEAD_INVALID")
    eps = integer(data["overhead_budget_ppm"], 1, 999_999, "OVERHEAD_BUDGET_INVALID")
    pause = integer(data["pause_target_us"], 1, 1_000_000, "PAUSE_TARGET_INVALID")
    p = integer(data["publication_us"], 0, 1_000_000, "PUBLICATION_INVALID")
    drain = integer(data["drain_us"], 0, 1_000_000, "DRAIN_INVALID")
    ack = integer(data["ack_us"], 0, 1_000_000, "ACK_LATENCY_INVALID")
    # Pure dimensional math. These are declared bounds, not measured device
    # worst cases. Cooperative residency cannot promise physical preemption.
    lower = (h * (1_000_000 - eps) + eps - 1) // eps
    upper = pause - p - drain - ack
    return {
        "minimum_useful_slice_us": lower,
        "maximum_useful_slice_us": max(0, upper),
        "nonempty_interval": lower <= upper and upper > 0,
        "bounds_authenticity": "DECLARED_UNVERIFIED",
        "actual_preemption_measured": False,
    }


def tail_budget(
    samples: Any,
    *,
    allowance_ppm: int = 60_000,
    risk_ppm: int = 10_000,
    confidence_ppm: int = DEFAULT_CONFIDENCE_PPM,
) -> dict[str, Any]:
    require(type(samples) is list, "TAIL_SAMPLES_INVALID")
    require(
        MIN_CALIBRATIONS <= len(samples) <= MAX_CALIBRATIONS,
        "TAIL_SAMPLE_COVERAGE_INSUFFICIENT",
    )
    integer(allowance_ppm, 0, 500_000, "SLOWDOWN_BUDGET_INVALID")
    integer(risk_ppm, 1, 100_000, "TAIL_RISK_BUDGET_INVALID")
    integer(confidence_ppm, 500_000, 999_999, "CONFIDENCE_INVALID")
    baseline = []
    with_bg = []
    for sample in samples:
        row = exact(sample, {"baseline_us", "shared_us"}, "TAIL_SAMPLE_FIELDS_INVALID")
        baseline.append(integer(row["baseline_us"], 1, 10_000_000, "BASELINE_INVALID"))
        with_bg.append(
            integer(row["shared_us"], 1, 10_000_000, "SHARED_LATENCY_INVALID")
        )
    n = len(samples)
    baseline_p99 = sorted(baseline)[math.ceil(n * 0.99) - 1]
    shared_p99 = sorted(with_bg)[math.ceil(n * 0.99) - 1]
    allowed_latency = baseline_p99 * (1_000_000 + allowance_ppm) // 1_000_000
    excess = sum(x > allowed_latency for x in with_bg)
    alpha = 1 - confidence_ppm / 1_000_000
    # One-sided exact binomial bound for 0 violations. For any nonzero count,
    # a conservative Hoeffding upper bound avoids anti-conservative shortcuts.
    # Independence and distribution stationarity are NOT authenticated here.
    upper = (
        1 - math.exp(math.log(alpha) / n)
        if excess == 0
        else min(1.0, excess / n + math.sqrt(math.log(1 / alpha) / (2 * n)))
    )
    return {
        "pairs": n,
        "baseline_empirical_p99_us": baseline_p99,
        "shared_empirical_p99_us": shared_p99,
        "allowed_latency_us": allowed_latency,
        "violations_observed": excess,
        "upper_tail_violation_fraction_conditional": round(upper, 8),
        "risk_budget_fraction": risk_ppm / 1_000_000,
        "statistically_eligible_if_iid_and_authentic": upper <= risk_ppm / 1_000_000,
        "iid_independence_verified": False,
        "baseline_source_authenticated": False,
        "candidate_is_production_admission": False,
    }


def _event_fields(op: str) -> set[str]:
    token = {"op", "task", "generation", "epoch"}
    shapes = {
        "prepare": token | {"backend", "trusted", "memory_mib"},
        "grant": token | {"start", "extent"},
        "complete": token | {"grant_id", "committed", "completion_digest"},
        "revoke": token,
        "ack": token,
        "reader_add": token,
        "reader_done": token,
        "retire": token,
        "publish_epoch": {"op", "next_epoch"},
    }
    return shapes[op]


def replay(spec: Any) -> dict[str, Any]:
    root = exact(
        spec,
        {"schema", "source", "memory_limit_mib", "checkpoint", "tail", "events"},
        "REPLAY_FIELDS_INVALID",
    )
    require(root["schema"] == SCHEMA, "SCHEMA_UNSUPPORTED")
    src = exact(
        root["source"],
        {"repository", "revision", "upstream_revision", "evidence_kind"},
        "SOURCE_FIELDS_INVALID",
    )
    require(src["repository"] == "szl-holdings/szl-gpu-bridge", "SOURCE_REPO_INVALID")
    require(
        type(src["revision"]) is str and SHA40.fullmatch(src["revision"]) is not None,
        "SOURCE_SHA_INVALID",
    )
    require(
        src["upstream_revision"] == "9eb66b9c68a3e449fdb3b2bd6edf3ddf6bd6d2ca",
        "UPSTREAM_REFERENCE_DRIFT",
    )
    require(src["evidence_kind"] == "SYNTHETIC", "UNAUTHENTICATED_EVIDENCE")
    ledger = LeaseLedger(memory_limit_mib=root["memory_limit_mib"])
    feasibility = checkpoint_interval(root["checkpoint"])
    tail_input = exact(
        root["tail"],
        {"samples", "slowdown_budget_ppm", "risk_budget_ppm", "confidence_ppm"},
        "TAIL_FIELDS_INVALID",
    )
    tail = tail_budget(
        tail_input["samples"],
        allowance_ppm=tail_input["slowdown_budget_ppm"],
        risk_ppm=tail_input["risk_budget_ppm"],
        confidence_ppm=tail_input["confidence_ppm"],
    )
    events = root["events"]
    require(
        type(events) is list and 1 <= len(events) <= MAX_EVENTS, "EVENT_COUNT_INVALID"
    )
    prev = "0" * 64
    records = []
    failed_at = None
    for index, event in enumerate(events):
        try:
            require(type(event) is dict, "EVENT_INVALID")
            op = event.get("op")
            require(type(op) is str and op in EVENT_OPS, "EVENT_OP_UNSUPPORTED")
            exact(event, _event_fields(op), "EVENT_FIELDS_INVALID")
            if op == "publish_epoch":
                value = ledger.publish_epoch(next_epoch=event["next_epoch"])
            else:
                token = LeaseToken(
                    task=event["task"],
                    generation=event["generation"],
                    epoch=event["epoch"],
                )
                if op == "prepare":
                    value = ledger.prepare(
                        token,
                        backend=event["backend"],
                        trusted=event["trusted"],
                        memory_mib=event["memory_mib"],
                    )
                elif op == "grant":
                    value = ledger.grant(
                        token, start=event["start"], extent=event["extent"]
                    )
                elif op == "complete":
                    value = ledger.complete(
                        token,
                        grant_id=event["grant_id"],
                        committed=event["committed"],
                        completion_digest=event["completion_digest"],
                    )
                elif op == "revoke":
                    value = ledger.revoke(token)
                elif op == "ack":
                    value = ledger.ack(token)
                elif op == "reader_add":
                    value = ledger.reader_add(token)
                elif op == "reader_done":
                    value = ledger.reader_done(token)
                else:
                    value = ledger.retire(token)
            outcome = "SIMULATED_EVENT_ACCEPTED"
            code = None
        except Hold as exc:
            value = None
            outcome = "SIMULATION_HOLD"
            code = str(exc)
            failed_at = index
        # Never serialize the caller's raw event, arbitrary strings, errors or
        # secrets. A digest is a structural checksum and not authenticity.
        item = {
            "index": index,
            "operation": (
                event.get("op")
                if type(event) is dict
                and type(event.get("op")) is str
                and event.get("op") in EVENT_OPS
                else "invalid"
            ),
            "outcome": outcome,
            "hold_code": code,
            "acknowledgement": (
                value if type(event) is dict and event.get("op") == "ack" else None
            ),
            "prior_digest": prev,
            "state_sha256": sha256(ledger.snapshot()),
        }
        item["digest"] = sha256(item)
        prev = item["digest"]
        records.append(item)
        if failed_at is not None:
            break
    state = ledger.snapshot()
    mathematically_eligible = (
        feasibility["nonempty_interval"]
        and tail["statistically_eligible_if_iid_and_authentic"]
        and failed_at is None
        and state["memory_reserved_mib"] == 0
        and bool(state["leases"])
        and all(lease["retired"] for lease in state["leases"])
        and any(
            lease["grant_frontier"] > 0 and lease["committed"] > 0
            for lease in state["leases"]
        )
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "source": {
            "repository": src["repository"],
            "declared_revision": src["revision"],
            "upstream_reference_sha": src["upstream_revision"],
            "input_authenticated": False,
            "upstream_code_copied": False,
        },
        "state": "SIMULATED_PLAN_FEASIBLE_NOT_ADMITTED"
        if mathematically_eligible
        else "SIMULATED_PLAN_HOLD",
        "checkpoint": feasibility,
        "tail": tail,
        "ledger": state,
        "events": records,
        "event_count": len(records),
        "failed_at": failed_at,
        "event_chain_sha256": prev,
        "event_chain_signature": "UNSIGNED_STRUCTURAL_HASH_ONLY",
        "authority": {
            "provider_writes": 0,
            "gpu_kernels_launched": False,
            "model_training_executed": False,
            "production_admission": False,
            "hardware_preemption_verified": False,
            "tenant_isolation_verified": False,
            "signature_verified": False,
            "source_sha_verified": False,
            "cross_provider_parity": "UNVERIFIED",
            "monetization_or_settlement": "NOT_PERFORMED",
        },
    }


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record = {}
    for key, value in pairs:
        require(key not in record, "DUPLICATE_JSON_KEY")
        record[key] = value
    return record


def read_input(path: str) -> tuple[Any, str]:
    source = Path(path)
    try:
        require(not source.is_symlink() and source.is_file(), "SOURCE_UNAVAILABLE")
        require(0 < source.stat().st_size <= MAX_INPUT_BYTES, "SOURCE_BYTE_LIMIT")
        blob = source.read_bytes()
        require(0 < len(blob) <= MAX_INPUT_BYTES, "SOURCE_BYTE_LIMIT")
        data = json.loads(
            blob.decode("utf-8"),
            object_pairs_hook=_duplicate_keys,
            parse_constant=lambda _: (_ for _ in ()).throw(Hold("NONFINITE_JSON")),
        )
        return data, hashlib.sha256(blob).hexdigest()
    except (UnicodeError, json.JSONDecodeError):
        raise Hold("SOURCE_JSON_INVALID") from None
    except OSError:
        raise Hold("SOURCE_UNAVAILABLE") from None


def render_html(result: dict[str, Any]) -> str:
    require(
        result.get("schema") == RECEIPT_SCHEMA
        and result.get("authority", {}).get("production_admission") is False
        and result.get("authority", {}).get("provider_writes") == 0,
        "HTML_RENDER_UNTRUSTED",
    )

    # Static, no script, links, external font, fetch, form, GPU calls or
    # credential-driven enrichment. Escape every dynamic value.
    def row(label: str, value: Any) -> str:
        return (
            "<tr><th scope='row'>"
            + html.escape(label, quote=True)
            + "</th><td>"
            + html.escape(str(value), quote=True)
            + "</td></tr>"
        )

    state = result["ledger"]
    tail = result["tail"]
    fields = [
        ("Verdict", result["state"]),
        ("Code epoch", state["code_epoch"]),
        ("Simulated reserved MiB", state["memory_reserved_mib"]),
        ("Logical leases", len(state["leases"])),
        ("Trace chain SHA-256", result["event_chain_sha256"]),
        (
            "Declared checkpoint interval viable",
            result["checkpoint"]["nonempty_interval"],
        ),
        ("Paired calibration observations", tail["pairs"]),
        (
            "Tail bound conditional on iid",
            tail["upper_tail_violation_fraction_conditional"],
        ),
        ("Production authority", "DENIED / NOT PROVIDED"),
    ]
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none';"
        " style-src 'unsafe-inline'; script-src 'none'; img-src 'none';"
        " connect-src 'none'; form-action 'none'; base-uri 'none'\">"
        "<title>SZL simulated GPU lease protocol</title>"
        "<style>body{background:#101c28;color:#e6f4ff;max-width:850px;"
        "margin:auto;padding:2rem;font:1rem/1.5 system-ui,sans-serif}"
        "table{border-collapse:collapse;width:100%}td,th{padding:.65rem;"
        "border-bottom:1px solid #456;text-align:left;overflow-wrap:anywhere}"
        "aside{border:1px solid #eabf61;padding:1rem;border-radius:.6rem}</style>"
        "</head><body><main><h1>Proof-carrying GPU lease — structural study</h1>"
        "<aside role='status'><strong>RESEARCH HOLD:</strong> no real device "
        "grant, tenant isolation, provider readback, source signature or "
        "production release is established.</aside><table><tbody>"
        + "".join(row(label, value) for label, value in fields)
        + "</tbody></table></main></body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline GPU lease protocol replay; NO GPU or provider effects."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--format", choices=("json", "html"), default="json")
    args = parser.parse_args(argv)
    try:
        value, file_digest = read_input(args.input)
        result = replay(value)
        result["source"]["input_sha256"] = file_digest
        output = (
            render_html(result)
            if args.format == "html"
            else json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
        )
        sys.stdout.write(output + "\n")
        # Nonzero means the *simulated* math/protocol is held. Even an exit-0
        # result never authorizes hardware or a production deployment.
        return (
            0
            if result["failed_at"] is None
            and result["state"] == "SIMULATED_PLAN_FEASIBLE_NOT_ADMITTED"
            else 2
        )
    except (Hold, OverflowError) as exc:
        code = str(exc) if isinstance(exc, Hold) else "NUMERIC_OVERFLOW"
        sys.stderr.write("GPU_LEASE_HOLD:" + code + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
