#!/usr/bin/env python3
"""Multi-model swap pool for one H100/H200-class node. Fail-closed; CPU-testable.

Takes a queue of EXISTING reviewed run manifests plus one node manifest and keeps
the node busy: several runs train at once while their estimated VRAM fits; when a
run finishes, plateaus, hits its cap or its time box, it is checkpointed, the
checkpoint is uploaded to R2, a receipt is written, the run is evicted, and the
next admissible manifest loads.

Every run is launched through the existing per-run entrypoint ONLY:

    python laptop/dispatcher.py <signed-envelope>

so DSSE verification, the engine-key pin, the per-kind contract and the
reviewed execution-binding check all still run inside the child exactly as for a
single launch. The pool adds gates in front of that; it never removes one:

* node gate: provider is salad or massed (never a laptop/owner host), deadline
  in the future, caps present, spend ledger readable, R2 configured;
* per-run binding gate: the reviewed jobspec and signed envelope bytes must hash
  to the pins in the queue entry, and the pinned execution bridge revision must
  equal SZL_EXECUTION_BRIDGE_REVISION (the same value dispatcher.py enforces);
* cost gate: worst-case projected cost (device rate x hard time box) must fit the
  stricter-of every configured per-run, per-day (America/New_York) and
  per-month cap, counting ledger spend and the reservations of running runs.
  No cap value is hardcoded here; a missing cap refuses the launch.

Any gate failure means that run is never launched (a refusal receipt is written).
Plan mode (the default) evaluates admission and launches nothing; --execute is
required to spawn processes. This module never contacts a GPU provider, never
publishes to the Hub and never reads or prints a credential value.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
NODE_SCHEMA = "szl.gpu-pool.node/v1"
QUEUE_SCHEMA = "szl.gpu-pool.queue/v1"
SWAP_SCHEMA = "szl.gpu-pool.swap/v1"
RECEIPT_SCHEMA = "szl.gpu-pool.run-receipt/v1"
POOL_RECEIPT_SCHEMA = "szl.gpu-pool.pool-receipt/v1"
DISPATCH_CAPS_SCHEMAS = ("szl.dispatch.org-caps/v1",)
ENTRYPOINT = "laptop/dispatcher.py"
CAP_KEYS = ("per_run_usd", "per_day_usd", "per_month_usd")
ALLOWED_PROVIDERS = ("salad", "massed")
FORBIDDEN_HOST = re.compile(r"betterwithage|laptop|omen|owner[_-]?gpu", re.I)
ISOLATION = ("mig", "device", "shared")
MAX_CHECKPOINT_INTERVAL_S = 4 * 3600
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Reasons a run leaves the pool.
DONE = "done"
PLATEAU = "plateau"
FAILED = "failed"
GATE_REFUSED_IN_CHILD = "child_gate_refused"
RUN_TIMEBOX = "run_timebox"
CAP_RUN = "cap_run"
CAP_DAY = "cap_day"
CAP_MONTH = "cap_month"
DEADLINE = "node_deadline"
CHECKPOINT_FAILED = "checkpoint_failed"
INTERRUPTED = "pool_interrupted"


class GateError(ValueError):
    """Public reason code; the pool refuses rather than guesses."""


def require(condition, reason):
    if not condition:
        raise GateError(reason)


SECRET_ENV_PREFIXES = ("R2_", "AWS_")


def scrub_child_env(environ):
    """Training children never receive object-store credentials."""

    return {k: v for k, v in environ.items() if not k.startswith(SECRET_ENV_PREFIXES)}


def et_iso(ts):
    return datetime.fromtimestamp(ts, ET).isoformat(timespec="seconds")


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_number(value, reason):
    require(
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0,
        reason,
    )
    return float(value)


def load_json(path, reason):
    try:
        with open(path, encoding="utf-8-sig") as stream:
            data = json.load(stream, parse_constant=lambda _: None)
    except (OSError, ValueError) as exc:
        raise GateError(reason) from exc
    require(type(data) is dict, reason)
    return data


# --------------------------------------------------------------------------- caps


def effective_caps(sources):
    """Stricter-of every configured source per dimension; missing dimension refuses.

    ``sources`` is a list of (label, mapping). A mapping may omit a dimension, but
    if no source provides it the launch is refused. Values are never defaulted.
    """

    caps, provenance = {}, {}
    for label, mapping in sources:
        if mapping is None:
            continue
        require(type(mapping) is dict, f"CAPS_INVALID:{label}")
        for key in CAP_KEYS:
            if key in mapping and mapping[key] is not None:
                value = positive_number(mapping[key], f"CAP_INVALID:{label}:{key}")
                if key not in caps or value < caps[key]:
                    caps[key], provenance[key] = value, label
    for key in CAP_KEYS:
        require(key in caps, f"CAP_MISSING:{key}")
    return caps, provenance


def load_dispatch_caps(path):
    data = load_json(path, "DISPATCH_CAPS_UNREADABLE")
    require(data.get("schema") in DISPATCH_CAPS_SCHEMAS, "DISPATCH_CAPS_SCHEMA")
    require(data.get("fail_closed") is not False, "DISPATCH_CAPS_NOT_FAIL_CLOSED")
    return {key: data.get(key) for key in CAP_KEYS}


# ------------------------------------------------------------------- spend ledger


def read_ledger(path):
    """Rows of (timestamp-ET-ISO, usd). JSONL or CSV (ts_et/at_et + usd columns)."""

    path = Path(path)
    require(path.is_file(), "SPEND_LEDGER_MISSING")
    rows = []
    try:
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    rows.append((row.get("at_et") or row.get("ts_et"), row["usd"]))
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows.append((row["at_et"], row["usd"]))
        parsed = []
        for stamp, usd in rows:
            when = datetime.fromisoformat(str(stamp))
            if when.tzinfo is None:
                when = when.replace(tzinfo=ET)
            amount = float(usd)
            require(math.isfinite(amount) and amount >= 0, "SPEND_LEDGER_INVALID")
            parsed.append((when.astimezone(ET), amount))
    except (KeyError, ValueError, TypeError, OSError) as exc:
        raise GateError("SPEND_LEDGER_INVALID") from exc
    return parsed


def committed_spend(path, now_ts):
    now = datetime.fromtimestamp(now_ts, ET)
    day = month = 0.0
    for when, usd in read_ledger(path):
        if (when.year, when.month) == (now.year, now.month):
            month += usd
            if when.date() == now.date():
                day += usd
    return day, month


def append_ledger(path, row):
    """Append in the ledger's own format so a CSV tracker stays a CSV tracker."""

    path = Path(path)
    if path.suffix != ".csv":
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        return
    with path.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream), None)
    require(header and "usd" in header, "SPEND_LEDGER_CSV_HEADER")
    stamp_column = "at_et" if "at_et" in header else "ts_et"
    require(stamp_column in header, "SPEND_LEDGER_CSV_HEADER")
    values = dict(row)
    values[stamp_column] = values.pop("at_et")
    if "note" in header and "note" not in values:
        values["note"] = f"node={values.get('node_id', '')}"
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, header, restval="", extrasaction="ignore").writerow(
            values
        )


# ------------------------------------------------------------------ node manifest


@dataclass
class Device:
    device_id: str
    vram_gb: float
    isolation: str
    usd_per_hour: float
    gpu_fraction: float
    runs: list = field(default_factory=list)

    def used_gb(self):
        return sum(run.vram_gb for run in self.runs)


def validate_node(node, now_ts, cli_caps_sources=()):
    require(node.get("schema") == NODE_SCHEMA, "NODE_SCHEMA")
    provider = node.get("provider")
    require(provider in ALLOWED_PROVIDERS, "NODE_PROVIDER_NOT_ALLOWED")
    host_text = json.dumps(
        {k: node.get(k) for k in ("node_id", "provider", "host", "gpu_type")}
    )
    require(not FORBIDDEN_HOST.search(host_text), "NODE_FORBIDDEN_HOST")
    require(
        type(node.get("node_id")) is str and RUN_ID.fullmatch(node["node_id"]),
        "NODE_ID",
    )
    require(type(node.get("gpu_type")) is str and node["gpu_type"], "NODE_GPU_TYPE")
    try:
        deadline = datetime.fromisoformat(node["node_deadline"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError("NODE_DEADLINE_INVALID") from exc
    require(deadline.tzinfo is not None, "NODE_DEADLINE_NEEDS_OFFSET")
    margin = positive_number(node.get("deadline_margin_s"), "NODE_DEADLINE_MARGIN")
    hard_stop = deadline.timestamp() - margin
    require(hard_stop > now_ts, "NODE_DEADLINE_PASSED")
    interval = positive_number(
        node.get("checkpoint_interval_s", MAX_CHECKPOINT_INTERVAL_S),
        "CHECKPOINT_INTERVAL",
    )
    require(interval <= MAX_CHECKPOINT_INTERVAL_S, "CHECKPOINT_INTERVAL_ABOVE_4H")
    checkpoint_timeout = positive_number(
        node.get("checkpoint_timeout_s"), "CHECKPOINT_TIMEOUT"
    )
    grace = positive_number(node.get("terminate_grace_s"), "TERMINATE_GRACE")
    # The hard stop must leave room to checkpoint and then terminate every run.
    require(
        checkpoint_timeout + grace < margin, "DEADLINE_MARGIN_BELOW_CHECKPOINT_TIMEOUT"
    )
    headroom = node.get("vram_headroom_fraction", 0.1)
    require(
        type(headroom) in (int, float) and 0 <= headroom < 0.5, "VRAM_HEADROOM_INVALID"
    )
    max_concurrent = node.get("max_concurrent")
    require(
        type(max_concurrent) is int and max_concurrent >= 1, "MAX_CONCURRENT_INVALID"
    )
    devices = []
    raw_devices = node.get("devices")
    require(type(raw_devices) is list and raw_devices, "NODE_DEVICES")
    seen = set()
    for raw in raw_devices:
        require(type(raw) is dict, "NODE_DEVICE_INVALID")
        device_id = str(raw.get("id", ""))
        require(
            re.fullmatch(r"[0-9]{1,2}|MIG-[0-9a-fA-F-]{8,64}", device_id),
            "NODE_DEVICE_ID",
        )
        require(device_id not in seen, "NODE_DEVICE_DUPLICATE")
        seen.add(device_id)
        isolation = raw.get("isolation")
        require(isolation in ISOLATION, "NODE_DEVICE_ISOLATION")
        if isolation == "mig":
            require(device_id.startswith("MIG-"), "MIG_DEVICE_NEEDS_MIG_UUID")
        if isolation == "shared":
            # Memory fraction is a request the trainer must honor; nothing on the
            # node enforces it. The reviewer must opt in explicitly.
            require(
                node.get("allow_unenforced_shared_memory") is True,
                "SHARED_GPU_MEMORY_UNENFORCED",
            )
        devices.append(
            Device(
                device_id=device_id,
                vram_gb=positive_number(raw.get("vram_gb"), "NODE_DEVICE_VRAM"),
                isolation=isolation,
                usd_per_hour=positive_number(
                    raw.get("usd_per_hour"), "NODE_DEVICE_RATE"
                ),
                gpu_fraction=positive_number(
                    raw.get("gpu_fraction", 1.0), "NODE_DEVICE_GPU_FRACTION"
                ),
            )
        )
    r2 = node.get("r2")
    require(type(r2) is dict, "NODE_R2_MISSING")
    require(type(r2.get("bucket")) is str and r2["bucket"], "NODE_R2_BUCKET")
    require(type(r2.get("prefix")) is str and r2["prefix"], "NODE_R2_PREFIX")
    caps, provenance = effective_caps(
        [("node", node.get("caps"))] + list(cli_caps_sources)
    )
    read_ledger(node.get("spend_ledger", ""))
    return {
        "hard_stop_ts": hard_stop,
        "deadline_ts": deadline.timestamp(),
        "checkpoint_interval_s": interval,
        "checkpoint_timeout_s": checkpoint_timeout,
        "headroom": float(headroom),
        "max_concurrent": max_concurrent,
        "devices": devices,
        "caps": caps,
        "caps_provenance": provenance,
        "spend_ledger": node["spend_ledger"],
        "r2": r2,
        "grace_s": grace,
    }


# ----------------------------------------------------------------- run manifests


@dataclass
class Run:
    entry: dict
    run_id: str
    model: str
    vram_gb: float
    max_hours: float
    plateau_threshold: float
    plateau_k: int
    plateau_min_evals: int
    envelope: Path
    jobspec: Path
    output_dir: Path | None
    caps_override: dict | None
    # Runtime state.
    device: Device | None = None
    process: object = None
    started_ts: float = 0.0
    timebox_ts: float = 0.0
    reserve_usd: float = 0.0
    caps: dict | None = None
    losses: list = field(default_factory=list)
    metrics_offset: int = 0
    last_checkpoint_ts: float = 0.0
    pending_checkpoint: dict | None = None
    checkpoint_seq: int = 0
    checkpoints: list = field(default_factory=list)
    run_dir: Path | None = None
    caps_provenance: dict | None = None
    final: bool = False


def validate_run_entry(entry, repo_root, execution_revision):
    """Shape + reviewed-binding gate. Never trusts a path it did not hash."""

    require(type(entry) is dict, "RUN_ENTRY_INVALID")
    run_id = entry.get("run_id")
    require(type(run_id) is str and RUN_ID.fullmatch(run_id), "RUN_ID_INVALID")
    require(entry.get("entrypoint") == ENTRYPOINT, "RUN_ENTRYPOINT_NOT_DISPATCHER")
    model = entry.get("model")
    require(type(model) is str and 0 < len(model) <= 200, "RUN_MODEL_INVALID")
    repo_root = Path(repo_root).resolve()

    def inside(relative, reason):
        require(type(relative) is str and relative, reason)
        path = (repo_root / relative).resolve()
        require(repo_root in path.parents and path.is_file(), reason)
        return path

    jobspec = inside(entry.get("reviewed_jobspec"), "RUN_JOBSPEC_MISSING")
    require(
        jobspec.parent == repo_root / "jobspecs"
        and jobspec.name.endswith("-reviewed.json"),
        "RUN_JOBSPEC_NOT_REVIEWED",
    )
    envelope = inside(entry.get("signed_envelope"), "RUN_ENVELOPE_MISSING")
    require(
        envelope.parent == repo_root / "queue" / "pending", "RUN_ENVELOPE_NOT_QUEUED"
    )
    binding = entry.get("reviewed_binding")
    require(type(binding) is dict, "RUN_BINDING_MISSING")
    for key in ("jobspec_sha256", "envelope_sha256"):
        require(
            type(binding.get(key)) is str and HEX64.fullmatch(binding[key]),
            "RUN_BINDING_INVALID",
        )
    require(
        type(binding.get("execution_bridge_revision")) is str
        and HEX40.fullmatch(binding["execution_bridge_revision"]),
        "RUN_BINDING_INVALID",
    )
    require(sha256_path(jobspec) == binding["jobspec_sha256"], "RUN_JOBSPEC_DRIFT")
    require(sha256_path(envelope) == binding["envelope_sha256"], "RUN_ENVELOPE_DRIFT")
    require(
        execution_revision
        and binding["execution_bridge_revision"] == execution_revision,
        "RUN_EXECUTION_REVISION_MISMATCH",
    )
    plateau = entry.get("plateau")
    require(type(plateau) is dict, "RUN_PLATEAU_MISSING")
    k = plateau.get("k")
    require(type(k) is int and 1 <= k <= 1000, "RUN_PLATEAU_K")
    threshold = plateau.get("threshold")
    require(
        type(threshold) in (int, float)
        and not isinstance(threshold, bool)
        and math.isfinite(threshold)
        and threshold >= 0,
        "RUN_PLATEAU_THRESHOLD",
    )
    min_evals = plateau.get("min_evals", k + 1)
    require(type(min_evals) is int and min_evals >= k + 1, "RUN_PLATEAU_MIN_EVALS")
    output_dir = entry.get("output_dir")
    require(output_dir is None or type(output_dir) is str, "RUN_OUTPUT_DIR")
    caps = entry.get("caps")
    require(caps is None or type(caps) is dict, "RUN_CAPS_INVALID")
    return Run(
        entry=entry,
        run_id=run_id,
        model=model,
        vram_gb=positive_number(entry.get("vram_gb"), "RUN_VRAM"),
        max_hours=positive_number(entry.get("max_hours"), "RUN_MAX_HOURS"),
        plateau_threshold=float(threshold),
        plateau_k=k,
        plateau_min_evals=min_evals,
        envelope=envelope,
        jobspec=jobspec,
        output_dir=Path(output_dir) if output_dir else None,
        caps_override=caps,
    )


def plateaued(losses, threshold, k, min_evals):
    """True when the best eval loss of the last K evals improved on the best
    before them by less than ``threshold`` (absolute)."""

    if len(losses) < max(min_evals, k + 1):
        return False
    before = min(losses[:-k])
    recent = min(losses[-k:])
    return (before - recent) < threshold


# ------------------------------------------------------------------------ runner


def default_command(repo_root, run):
    return [sys.executable, str(Path(repo_root) / ENTRYPOINT), str(run.envelope)]


class PoolRunner:
    def __init__(
        self,
        node,
        queue_entries,
        *,
        repo_root,
        work_dir,
        store,
        dispatch_caps=(),
        execution_revision=None,
        command_factory=None,
        clock=time.time,
        sleep=time.sleep,
        poll_s=5.0,
        environ=None,
        log_stream=None,
    ):
        self.repo_root = Path(repo_root)
        self.work_dir = Path(work_dir)
        self.store = store
        self.clock = clock
        self.sleep = sleep
        self.poll_s = poll_s
        self.environ = dict(os.environ if environ is None else environ)
        self.execution_revision = execution_revision
        self.command_factory = command_factory or (
            lambda run: default_command(self.repo_root, run)
        )
        self.log_stream = log_stream or sys.stdout
        self.dispatch_caps = list(dispatch_caps)
        self.cfg = validate_node(node, self.clock(), self.dispatch_caps)
        self.node = node
        self.queue = list(queue_entries)
        self.running = []
        self.finished = []
        self.refused = []
        self.swaps = []
        self.prev_receipt_sha = None
        self.stopping = False
        (self.work_dir / "receipts").mkdir(parents=True, exist_ok=True)
        (self.work_dir / "runs").mkdir(parents=True, exist_ok=True)
        self.log_path = self.work_dir / "swaps.jsonl"

    # ---------------------------------------------------------------- receipts

    def _write_receipt(self, name, body):
        body = dict(body, prev_receipt_sha256=self.prev_receipt_sha)
        raw = json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode()
        path = self.work_dir / "receipts" / f"{name}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw + b"\n")
        os.replace(tmp, path)
        self.prev_receipt_sha = hashlib.sha256(raw + b"\n").hexdigest()
        return path

    def _emit(self, record):
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, file=self.log_stream, flush=True)

    # --------------------------------------------------------------- admission

    def _refuse(self, entry, reason):
        run_id = entry.get("run_id") if type(entry) is dict else None
        record = {
            "schema": RECEIPT_SCHEMA,
            "status": "REFUSED_NOT_LAUNCHED",
            "run_id": run_id,
            "model": entry.get("model") if type(entry) is dict else None,
            "reason": reason,
            "at_et": et_iso(self.clock()),
            "launched": False,
        }
        safe = run_id if type(run_id) is str and RUN_ID.fullmatch(run_id) else "invalid"
        self._write_receipt(f"{safe}.refused", record)
        self.refused.append(record)

    def _reserved(self):
        return sum(run.reserve_usd for run in self.running)

    def _placement(self, run):
        if len(self.running) >= self.cfg["max_concurrent"]:
            return None
        need = run.vram_gb * (1 + self.cfg["headroom"])
        for device in self.cfg["devices"]:
            if device.isolation in ("mig", "device"):
                if not device.runs and need <= device.vram_gb:
                    return device
            elif device.used_gb() * (1 + self.cfg["headroom"]) + need <= device.vram_gb:
                return device
        return None

    def _could_ever_fit(self, run):
        need = run.vram_gb * (1 + self.cfg["headroom"])
        return any(need <= device.vram_gb for device in self.cfg["devices"])

    def _cost_gate(self, run, device, now):
        caps, provenance = effective_caps(
            [("node", self.node.get("caps"))]
            + self.dispatch_caps
            + [(f"run:{run.run_id}", run.caps_override)]
        )
        hours = min(run.max_hours, (self.cfg["hard_stop_ts"] - now) / 3600)
        require(hours > 0, "NO_TIME_BEFORE_NODE_DEADLINE")
        projected = device.usd_per_hour * hours
        require(projected <= caps["per_run_usd"], "PROJECTED_COST_ABOVE_RUN_CAP")
        day, month = committed_spend(self.cfg["spend_ledger"], now)
        reserved = self._reserved()
        require(
            day + reserved + projected <= caps["per_day_usd"],
            "PROJECTED_COST_ABOVE_DAY_CAP",
        )
        require(
            month + reserved + projected <= caps["per_month_usd"],
            "PROJECTED_COST_ABOVE_MONTH_CAP",
        )
        return hours, projected, caps, provenance

    def _child_env(self, run, device):
        env = scrub_child_env(self.environ)
        # A MIG UUID or a whole-device index; the child sees exactly one device.
        env["CUDA_VISIBLE_DEVICES"] = device.device_id
        env["SZL_POOL_RUN_ID"] = run.run_id
        env["SZL_POOL_RUN_DIR"] = str(run.run_dir)
        env["SZL_POOL_METRICS_FILE"] = str(run.run_dir / "metrics.jsonl")
        env["SZL_POOL_CHECKPOINT_REQUEST"] = str(run.run_dir / "checkpoint.request")
        env["SZL_POOL_CHECKPOINT_DIR"] = str(run.run_dir / "checkpoints")
        if device.isolation == "shared":
            env["SZL_POOL_CUDA_MEMORY_FRACTION"] = f"{run.vram_gb / device.vram_gb:.4f}"
        if self.execution_revision:
            env["SZL_EXECUTION_BRIDGE_REVISION"] = self.execution_revision
        return env

    def plan(self):
        """Admission decisions only; launches nothing."""

        now = self.clock()
        decisions = []
        tally = 0.0
        for entry in self.queue:
            try:
                run = validate_run_entry(entry, self.repo_root, self.execution_revision)
                require(self._could_ever_fit(run), "VRAM_NEVER_FITS")
                device = max(self.cfg["devices"], key=lambda d: d.usd_per_hour)
                hours, projected, caps, _ = self._cost_gate(run, device, now)
                day, month = committed_spend(self.cfg["spend_ledger"], now)
                require(
                    day + tally + projected <= caps["per_day_usd"],
                    "PROJECTED_COST_ABOVE_DAY_CAP_CUMULATIVE",
                )
                require(
                    month + tally + projected <= caps["per_month_usd"],
                    "PROJECTED_COST_ABOVE_MONTH_CAP_CUMULATIVE",
                )
                tally += projected
                decisions.append(
                    {
                        "run_id": run.run_id,
                        "admissible": True,
                        "hard_timebox_h": round(hours, 4),
                        "projected_usd_worst_case": round(projected, 4),
                        "effective_caps": caps,
                    }
                )
            except GateError as exc:
                decisions.append(
                    {
                        "run_id": entry.get("run_id") if type(entry) is dict else None,
                        "admissible": False,
                        "reason": str(exc),
                    }
                )
        return decisions

    def _admit_ready(self):
        if self.stopping:
            return []
        admitted = []
        now = self.clock()
        remaining = []
        for entry in self.queue:
            try:
                run = validate_run_entry(entry, self.repo_root, self.execution_revision)
                require(self._could_ever_fit(run), "VRAM_NEVER_FITS")
            except GateError as exc:
                self._refuse(entry, str(exc))
                continue
            device = self._placement(run)
            if device is None:
                remaining.append(entry)
                continue
            try:
                hours, projected, caps, provenance = self._cost_gate(run, device, now)
            except GateError as exc:
                self._refuse(entry, str(exc))
                continue
            self._launch(run, device, now, hours, projected, caps, provenance)
            admitted.append(run)
        self.queue = remaining
        return admitted

    def _launch(self, run, device, now, hours, projected, caps, provenance):
        run.run_dir = self.work_dir / "runs" / run.run_id
        (run.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        run.device = device
        run.started_ts = now
        run.last_checkpoint_ts = now
        run.timebox_ts = now + hours * 3600
        run.reserve_usd = projected
        run.caps = caps
        run.caps_provenance = provenance
        env = self._child_env(run, device)
        log = (run.run_dir / "child.log").open("ab")
        run.process = subprocess.Popen(
            self.command_factory(run),
            shell=False,
            start_new_session=True,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(self.repo_root),
        )
        log.close()
        device.runs.append(run)
        self.running.append(run)
        self._emit(
            {
                "schema": SWAP_SCHEMA,
                "event": "admit",
                "run_id": run.run_id,
                "model": run.model,
                "at_et": et_iso(now),
                "device": device.device_id,
                "isolation": device.isolation,
                "vram_gb": run.vram_gb,
                "hard_timebox_h": round(hours, 4),
                "projected_usd_worst_case": round(projected, 4),
            }
        )

    # ----------------------------------------------------------- run telemetry

    def _read_metrics(self, run):
        path = run.run_dir / "metrics.jsonl"
        if not path.is_file():
            return
        with path.open("rb") as stream:
            stream.seek(run.metrics_offset)
            data = stream.read()
        end = data.rfind(b"\n")
        if end < 0:
            return
        run.metrics_offset += end + 1
        for line in data[: end + 1].splitlines():
            try:
                record = json.loads(line)
                value = float(record["eval_loss"])
            except (ValueError, KeyError, TypeError):
                continue
            if math.isfinite(value):
                run.losses.append(value)

    def _gpu_hours(self, run, now):
        share = 1.0
        if run.device.isolation == "shared":
            share = min(1.0, run.vram_gb / run.device.vram_gb)
        return (now - run.started_ts) / 3600 * run.device.gpu_fraction * share

    def _spent(self, run, now):
        # Conservative: a run is charged its device's full hourly rate even when
        # it shares the device, so the cap watchdog never under-counts.
        return (now - run.started_ts) / 3600 * run.device.usd_per_hour

    # ------------------------------------------------------------- checkpoints

    def _request_checkpoint(self, run, reason, now):
        run.checkpoint_seq += 1
        label = f"{run.checkpoint_seq:04d}-{reason}"
        target = run.run_dir / "checkpoints" / label
        request = {"seq": run.checkpoint_seq, "label": label, "dir": str(target)}
        tmp = run.run_dir / "checkpoint.request.tmp"
        tmp.write_text(json.dumps(request), encoding="utf-8")
        os.replace(tmp, run.run_dir / "checkpoint.request")
        run.pending_checkpoint = dict(
            request,
            reason=reason,
            deadline=now + self.cfg["checkpoint_timeout_s"],
        )

    def _fallback_checkpoint(self, run):
        roots = [run.run_dir / "checkpoints"]
        if run.output_dir is not None:
            roots.append(run.output_dir)
        candidates = []
        for root in roots:
            if root.is_dir():
                for path in root.iterdir():
                    if path.is_dir() and (
                        (path / "COMPLETE").is_file()
                        or path.name.startswith("checkpoint-")
                    ):
                        candidates.append(path)
        return max(candidates, key=lambda p: p.stat().st_mtime, default=None)

    def _finish_checkpoint(self, run, *, allow_wait):
        """Return (checkpoint record | None, error | None)."""

        pending = run.pending_checkpoint
        source, mode = None, None
        if pending is not None:
            target = Path(pending["dir"])
            while True:
                if (target / "COMPLETE").is_file():
                    source, mode = target, "acknowledged"
                    break
                exited = run.process.poll() is not None
                if not allow_wait or exited or self.clock() >= pending["deadline"]:
                    break
                self.sleep(min(self.poll_s, 0.5))
        if source is None:
            source = self._fallback_checkpoint(run)
            mode = "latest_on_disk" if source is not None else None
        if source is None and run.final and run.output_dir is not None:
            if run.output_dir.is_dir():
                source, mode = run.output_dir, "final_output_dir"
        label = pending["label"] if pending else f"{run.checkpoint_seq + 1:04d}-final"
        run.pending_checkpoint = None
        try:
            (run.run_dir / "checkpoint.request").unlink()
        except FileNotFoundError:
            # The child or a previous attempt already consumed the request file.
            pass
        if source is None:
            return None, "NO_CHECKPOINT_AVAILABLE"
        try:
            uploaded = self.store.upload_checkpoint(
                run.run_id, label, source, self.work_dir
            )
        except Exception as exc:  # noqa: BLE001 - reason code only
            return None, f"R2_UPLOAD_FAILED:{str(exc)[:80] or type(exc).__name__}"
        record = dict(uploaded, label=label, mode=mode, at_et=et_iso(self.clock()))
        run.checkpoints.append(record)
        run.last_checkpoint_ts = self.clock()
        return record, None

    # ---------------------------------------------------------------- eviction

    def _terminate(self, run):
        process = run.process
        if process.poll() is not None:
            return process.returncode, False
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # The owned child exited between poll() and the signal; wait() reaps it.
            pass
        try:
            return process.wait(timeout=self.cfg["grace_s"]), False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                # It completed during the grace-period race; wait() still reaps it.
                pass
            return process.wait(), True

    def _evict(self, run, reason, *, request=True):
        now = self.clock()
        alive = run.process.poll() is None
        if request and alive and run.pending_checkpoint is None:
            self._request_checkpoint(run, reason, now)
        checkpoint, error = self._finish_checkpoint(run, allow_wait=alive)
        returncode, forced = self._terminate(run)
        end = self.clock()
        gpu_hours = self._gpu_hours(run, end)
        usd = self._spent(run, end)
        run.device.runs.remove(run)
        self.running.remove(run)
        append_ledger(
            self.cfg["spend_ledger"],
            {
                "at_et": et_iso(end),
                "run_id": run.run_id,
                "usd": round(usd, 6),
                "basis": "pool estimate: device usd_per_hour x wall time",
                "node_id": self.node["node_id"],
            },
        )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "EVICTED",
            "run_id": run.run_id,
            "model": run.model,
            "reason": reason,
            "node_id": self.node["node_id"],
            "provider": self.node["provider"],
            "gpu_type": self.node["gpu_type"],
            "device": run.device.device_id,
            "isolation": run.device.isolation,
            "started_at_et": et_iso(run.started_ts),
            "ended_at_et": et_iso(end),
            "gpu_hours": round(gpu_hours, 6),
            "usd_estimate": round(usd, 6),
            "usd_basis": "device usd_per_hour x wall time; not a provider bill",
            "effective_caps": run.caps,
            "caps_provenance": run.caps_provenance,
            "reviewed_binding": run.entry["reviewed_binding"],
            "reviewed_jobspec": run.entry["reviewed_jobspec"],
            "signed_envelope": run.entry["signed_envelope"],
            "entrypoint": ENTRYPOINT,
            "child_returncode": returncode,
            "forced_kill": forced,
            "eval_losses_tail": run.losses[-10:],
            "checkpoint": checkpoint,
            "checkpoint_error": error,
            "checkpoints": run.checkpoints,
            "secrets_recorded": False,
            "launched": True,
        }
        self._write_receipt(f"{run.run_id}.evicted", receipt)
        self.finished.append(receipt)
        return {
            "schema": SWAP_SCHEMA,
            "event": "swap",
            "run_id": run.run_id,
            "model": run.model,
            "at_et": et_iso(end),
            "gpu_hours": round(gpu_hours, 6),
            "reason": reason,
            "checkpoint_uri": checkpoint["uri"] if checkpoint else None,
            "checkpoint_error": error,
            "device": run.device.device_id,
            "usd_estimate": round(usd, 6),
        }

    def _evict_and_backfill(self, run, reason, *, request=True):
        record = self._evict(run, reason, request=request)
        admitted = self._admit_ready()
        record["next_run_id"] = admitted[0].run_id if admitted else None
        record["next_model"] = admitted[0].model if admitted else None
        self.swaps.append(record)
        self._emit(record)

    def _stop_all(self, reason):
        self.stopping = True
        now = self.clock()
        for run in list(self.running):
            if run.process.poll() is None and run.pending_checkpoint is None:
                self._request_checkpoint(run, reason, now)
        for run in list(self.running):
            self._evict_and_backfill(run, reason, request=False)
        for entry in self.queue:
            self._refuse(entry, f"NOT_ADMITTED_POOL_STOPPED:{reason}")
        self.queue = []

    # --------------------------------------------------------------------- loop

    def _tick(self):
        now = self.clock()
        if now >= self.cfg["hard_stop_ts"]:
            self._stop_all(DEADLINE)
            return False
        caps_now, _ = effective_caps(
            [("node", self.node.get("caps"))] + self.dispatch_caps
        )
        day, month = committed_spend(self.cfg["spend_ledger"], now)
        live = sum(self._spent(run, now) for run in self.running)
        if day + live >= caps_now["per_day_usd"]:
            self._stop_all(CAP_DAY)
            return False
        if month + live >= caps_now["per_month_usd"]:
            self._stop_all(CAP_MONTH)
            return False
        for run in list(self.running):
            # An earlier eviction in this tick can block for a checkpoint
            # timeout; never judge later runs against a stale timestamp.
            now = self.clock()
            self._read_metrics(run)
            returncode = run.process.poll()
            if returncode is not None:
                reason = (
                    DONE
                    if returncode == 0
                    else GATE_REFUSED_IN_CHILD
                    if returncode == 3
                    else FAILED
                )
                run.final = returncode == 0
                self._evict_and_backfill(run, reason, request=False)
                continue
            if self._spent(run, now) >= run.caps["per_run_usd"]:
                self._evict_and_backfill(run, CAP_RUN)
                continue
            if now >= run.timebox_ts:
                self._evict_and_backfill(run, RUN_TIMEBOX)
                continue
            if plateaued(
                run.losses,
                run.plateau_threshold,
                run.plateau_k,
                run.plateau_min_evals,
            ):
                self._evict_and_backfill(run, PLATEAU)
                continue
            self._periodic_checkpoint(run, now)
        if not self.running:
            self._admit_ready()
            if not self.running:
                return False
        return True

    def _periodic_checkpoint(self, run, now):
        pending = run.pending_checkpoint
        if pending is None:
            if now - run.last_checkpoint_ts >= self.cfg["checkpoint_interval_s"]:
                self._request_checkpoint(run, "periodic", now)
            return
        target = Path(pending["dir"])
        if (target / "COMPLETE").is_file() or now >= pending["deadline"]:
            _, error = self._finish_checkpoint(run, allow_wait=False)
            if error:
                # A run that cannot be checkpointed is not resumable; stop paying.
                self._evict_and_backfill(run, CHECKPOINT_FAILED, request=False)
            else:
                self._emit(
                    {
                        "schema": SWAP_SCHEMA,
                        "event": "checkpoint",
                        "run_id": run.run_id,
                        "model": run.model,
                        "at_et": et_iso(self.clock()),
                        "gpu_hours": round(self._gpu_hours(run, self.clock()), 6),
                        "reason": "periodic",
                        "checkpoint_uri": run.checkpoints[-1]["uri"],
                    }
                )

    def run(self):
        interrupted = []
        previous = {}
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(
                    sig, lambda number, frame: interrupted.append(number)
                )
        try:
            self._admit_ready()
            while self.running:
                if interrupted:
                    self._stop_all(INTERRUPTED)
                    break
                if not self._tick():
                    break
                self.sleep(self.poll_s)
            if self.running:
                self._stop_all(INTERRUPTED)
            for entry in self.queue:
                self._refuse(entry, "NOT_ADMITTED_QUEUE_UNSERVICEABLE")
            self.queue = []
        except BaseException:
            for run in list(self.running):
                try:
                    self._evict(run, INTERRUPTED)
                except Exception:  # noqa: BLE001 - keep terminating the rest
                    self._terminate(run)
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        summary = {
            "schema": POOL_RECEIPT_SCHEMA,
            "node_id": self.node["node_id"],
            "ended_at_et": et_iso(self.clock()),
            "finished": [
                {k: r[k] for k in ("run_id", "reason", "gpu_hours", "usd_estimate")}
                for r in self.finished
            ],
            "refused": [{k: r[k] for k in ("run_id", "reason")} for r in self.refused],
            "swaps": len(self.swaps),
        }
        self._write_receipt("pool", summary)
        return summary


# ------------------------------------------------------------------------ CLI


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--node", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument(
        "--dispatch-caps",
        action="append",
        default=[],
        type=Path,
        help="szl.dispatch.org-caps/v1 file; repeatable; stricter-of applies",
    )
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually spawn runs; without it only the admission plan is printed",
    )
    return parser.parse_args(argv)


def load_queue(path):
    data = load_json(path, "QUEUE_UNREADABLE")
    require(data.get("schema") == QUEUE_SCHEMA, "QUEUE_SCHEMA")
    runs = data.get("runs")
    require(type(runs) is list and runs, "QUEUE_EMPTY")
    ids = [r.get("run_id") for r in runs if type(r) is dict]
    require(len(ids) == len(set(ids)) == len(runs), "QUEUE_DUPLICATE_OR_INVALID")
    return runs


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        node = load_json(args.node, "NODE_UNREADABLE")
        queue = load_queue(args.queue)
        caps_sources = [
            (f"dispatch:{p.name}", load_dispatch_caps(p)) for p in args.dispatch_caps
        ]
        revision = os.environ.get("SZL_EXECUTION_BRIDGE_REVISION")
        if args.execute:
            from r2_checkpoint_store import R2CheckpointStore

            store = R2CheckpointStore(node["r2"]["bucket"], node["r2"]["prefix"])
        else:
            store = None
        runner = PoolRunner(
            node,
            queue,
            repo_root=args.repo_root,
            work_dir=args.work_dir,
            store=store,
            dispatch_caps=caps_sources,
            execution_revision=revision,
            poll_s=args.poll_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - public reason code only
        print(json.dumps({"status": "REFUSED", "reason": str(exc)[:120]}))
        return 3
    if not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "decisions": runner.plan()}, indent=2))
        return 0
    print(json.dumps(runner.run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
