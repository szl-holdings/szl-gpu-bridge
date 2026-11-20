"""Single-writer admission RUNNER: the executor core for the Salad gate.

Companion to tools/salad/admission.py (merged PR #307). The gate maps one
intent to an ADMIT or DENY receipt; this module is the fail-closed executor
that owns the queue, the execution ledger and, in the one supervised process
the owner runs, the credential seam.

Review provenance, recorded 2026-11-20: this design round was reviewed by
Mistral Large 4 and Mistral Nemo (receipts sac_ops_mistral_out.json on the
device; brief sac_ops_brief.txt). Adopted amendments:
* INTENT OUTBOX ordering - a persisted EXECUTING ledger entry is written
  BEFORE the mutation and an EXECUTED entry only after success, so a crash
  between the two leaves an unambiguous halt state (CRASH_AMBIGUOUS) that a
  human must resolve; the runner never guesses and never auto-repairs.
* LEDGER-SCAN IDEMPOTENCE - replay-exactly-once is derived from the
  hash-chained ledger itself, so it stays verifiable offline with
  gates.verify_chain; duplicate or tampered entries fail closed.
* CREDENTIAL BOUNDARY - exactly one supervised process may construct the
  applier closure that holds the Salad credential. The runner core performs
  no Salad I/O and reads no secrets; spreading the key any wider weakens
  the single-writer moat and is forbidden.

Honesty boundaries kept explicit:
* a dry-run ADMIT only records PENDING_EXECUTION; desired_state and budget
  stay untouched until an EXECUTED entry is written after a real mutation;
* ALREADY_RECORDED is the idempotence verdict: an executed or failed intent
  sha is never re-judged, never re-executed and never re-charged;
* a DENY receipt never advances the intent chain head, so one bad intent
  cannot poison the queue;
* the ledger chain is tamper-evidence, NOT authentication; signing lands
  with szl-evidence-core next.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json

from tools.salad import admission
from tools.salad import controller_gates as gates

LEDGER_SCHEMA = "szl.salad-admission-executor/v1"

KIND_PENDING = "PENDING_EXECUTION"
KIND_EXECUTING = "EXECUTING"
KIND_EXECUTED = "EXECUTED"
KIND_FAILED = "EXECUTION_FAILED"
KIND_RECORDED = "ALREADY_RECORDED"

HALT_REASONS = (
    "CLOCK_UNAVAILABLE",
    "RECEIPT_CHAIN_SEED_INVALID",
    "REGISTRY_INVALID",
    "EXECUTION_FAILED",
    "SINK_FAILED",
)
HALT_AT_EXECUTION = "EXECUTION_FAILED"
MAX_QUEUE_INTENTS = 10000


def fail_closed_evidence(intent_bytes, registry, now):
    """Default evidence provider: nothing attested, nothing admitted."""
    return {
        "lease_ok": False,
        "witness_verdict": None,
        "gates_decision": None,
    }


def _entry(
    now,
    kind,
    *,
    intent_sha=None,
    admit_sha=None,
    before=None,
    after=None,
    required=None,
    observed=None,
    note=None,
):
    return {
        "schema": LEDGER_SCHEMA,
        "kind": kind,
        "intent_sha256": intent_sha,
        "admit_receipt_sha256": admit_sha,
        "desired_state_before": before,
        "desired_state_after": after,
        "required_micro_usd": required,
        "observed": observed,
        "note": note,
        "observed_utc": gates.utc_text(now),
    }


def _chain_append(ledger, entry):
    seed = ledger[-1]["sha256"] if ledger else None
    record = gates.chain_record(entry, seed)
    ledger.append(record)
    return record


def scan_ledger(ledger):
    """Derive replay state from the ledger. Fails closed on any drift.

    Returns intent shas keyed by replay meaning: executed and failed are
    terminal sets, pending is the admitted-but-unexecuted queue in ledger
    order. Raises ValueError on a tampered chain, a duplicated terminal
    execution or an EXECUTING entry never closed by EXECUTED: those states
    are reported, never guessed or repaired.
    """
    prev = None
    executed = set()
    failed = set()
    pending = []
    open_executing = set()
    for entry in ledger:
        if type(entry) is not dict or "sha256" not in entry:
            raise ValueError("LEDGER_TAMPER")
        body = dict(entry)
        claimed = body.pop("sha256", None)
        if body.get("prev_sha256") != prev:
            raise ValueError("LEDGER_TAMPER")
        if hashlib.sha256(gates.canonical(body)).hexdigest() != claimed:
            raise ValueError("LEDGER_TAMPER")
        kind = entry.get("kind")
        sha = entry.get("intent_sha256")
        if kind == KIND_EXECUTING:
            if sha in open_executing or sha in executed or sha in failed:
                raise ValueError("LEDGER_TAMPER")
            open_executing.add(sha)
        elif kind == KIND_EXECUTED:
            if sha not in open_executing or sha in executed:
                raise ValueError("LEDGER_DUPLICATE_EXECUTION")
            open_executing.discard(sha)
            executed.add(sha)
        elif kind == KIND_FAILED:
            if sha not in open_executing:
                raise ValueError("LEDGER_TAMPER")
            open_executing.discard(sha)
            failed.add(sha)
        elif kind == KIND_PENDING:
            if (
                sha in executed
                or sha in failed
                or sha in open_executing
                or sha in pending
            ):
                raise ValueError("LEDGER_TAMPER")
            pending.append(sha)
        elif kind == KIND_RECORDED:
            pass
        else:
            raise ValueError("LEDGER_TAMPER")
        prev = claimed
    if open_executing:
        raise ValueError("LEDGER_EXECUTION_CRASH_AMBIGUOUS")
    return {
        "executed": executed,
        "failed": failed,
        "pending": pending,
    }


def _intent_sha(intent_bytes):
    if type(intent_bytes) is not bytes:
        return None
    if not 0 < len(intent_bytes) <= admission.MAX_INTENT_BYTES:
        return None
    return admission.intent_sha256(intent_bytes)


def run_queue(
    registry,
    intents,
    *,
    now,
    ledger,
    evidence=None,
    applier=None,
    sink=None,
    prev_receipt_sha256=None,
):
    """Judge queued intents through the gate; apply admitted ones exactly once.

    Pure core: no Salad I/O, no secrets, no clock reads (now is injected).
    Mutates and returns a result dict with registry, ledger, receipts, a
    halted reason or None and the receipt-chain head used. Registry is
    advanced in two steps: the intent chain head moves at ADMIT (an
    admission fact, even in dry run); desired_state and budget move only
    at EXECUTED (an execution fact, after a real mutation).

    With no applier (the NO_CREDENTIAL estate of 2026-11-20) admitted
    intents record PENDING_EXECUTION entries. With an applier, pending
    entries execute once via the outbox order EXECUTING then EXECUTED; any
    failure or drift halts the queue fail-closed.
    """
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("CLOCK_REQUIRED")
    if evidence is None:
        evidence = fail_closed_evidence
    if type(ledger) is not list:
        raise ValueError("LEDGER_REQUIRED")
    queue = list(intents)
    if len(queue) > MAX_QUEUE_INTENTS:
        raise ValueError("QUEUE_BUDGET")
    try:
        admission.validate_registry(registry)
    except ValueError:
        raise ValueError("REGISTRY_INVALID")
    state = scan_ledger(ledger)
    pending_set = set(state["pending"])
    receipts = []
    halted = None
    gate_prev = prev_receipt_sha256

    def halt(reason):
        return reason

    def execute(sha, admit_sha, before, after, required, intent_bytes, lane_id):
        nonlocal halted
        try:
            admission.validate_registry(registry)
        except ValueError:
            halted = halt("REGISTRY_INVALID")
            return False
        lane = registry["lanes"].get(lane_id)
        if lane is None:
            halted = halt("REGISTRY_INVALID")
            return False
        if lane["desired_state"] != before:
            halted = halt(HALT_AT_EXECUTION)
            _chain_append(
                ledger,
                _entry(
                    now,
                    KIND_FAILED,
                    intent_sha=sha,
                    admit_sha=admit_sha,
                    before=before,
                    after=after,
                    required=required,
                    note="STATE_DRIFT_AT_EXECUTION",
                ),
            )
            return False
        if type(required) is int and lane["budget"]["remaining"] < required:
            halted = halt(HALT_AT_EXECUTION)
            _chain_append(
                ledger,
                _entry(
                    now,
                    KIND_FAILED,
                    intent_sha=sha,
                    admit_sha=admit_sha,
                    before=before,
                    after=after,
                    required=required,
                    note="BUDGET_SHORTFALL_AT_EXECUTION",
                ),
            )
            return False
        executing = _chain_append(
            ledger,
            _entry(
                now,
                KIND_EXECUTING,
                intent_sha=sha,
                admit_sha=admit_sha,
                before=before,
                after=after,
                required=required,
            ),
        )
        if sink is not None:
            try:
                sink(executing)
            except Exception:
                halted = halt("SINK_FAILED")
                _chain_append(
                    ledger,
                    _entry(
                        now,
                        KIND_FAILED,
                        intent_sha=sha,
                        admit_sha=admit_sha,
                        before=before,
                        after=after,
                        required=required,
                        note="SINK_FAILED",
                    ),
                )
                return False
        try:
            applied = applier(
                intent_bytes,
                {
                    "intent_sha256": sha,
                    "admit_receipt_sha256": admit_sha,
                    "desired_state_before": before,
                    "desired_state_after": after,
                    "required_micro_usd": required,
                },
            )
            ok = type(applied) is dict and applied.get("ok") is True
        except Exception:
            applied = None
            ok = False
        if not ok:
            halted = halt(HALT_AT_EXECUTION)
            _chain_append(
                ledger,
                _entry(
                    now,
                    KIND_FAILED,
                    intent_sha=sha,
                    admit_sha=admit_sha,
                    before=before,
                    after=after,
                    required=required,
                    observed=applied if type(applied) is dict else None,
                    note="APPLIER_NOT_OK",
                ),
            )
            return False
        executed = _chain_append(
            ledger,
            _entry(
                now,
                KIND_EXECUTED,
                intent_sha=sha,
                admit_sha=admit_sha,
                before=before,
                after=after,
                required=required,
                observed=applied.get("observed"),
            ),
        )
        if sink is not None:
            try:
                sink(executed)
            except Exception:
                halted = halt("SINK_FAILED")
                return False
        lane["desired_state"] = after
        if type(required) is int:
            lane["budget"]["remaining"] -= required
        return True

    for intent_bytes in queue:
        if halted is not None:
            break
        sha = _intent_sha(intent_bytes)
        if sha is not None and (sha in state["executed"] or sha in state["failed"]):
            _chain_append(
                ledger,
                _entry(now, KIND_RECORDED, intent_sha=sha),
            )
            continue
        if sha is not None and sha in pending_set:
            if applier is None:
                _chain_append(
                    ledger,
                    _entry(now, KIND_RECORDED, intent_sha=sha),
                )
                continue
            try:
                admission.validate_registry(registry)
            except ValueError:
                halted = halt("REGISTRY_INVALID")
                break
            intent = json.loads(intent_bytes.decode("utf-8"))
            executed_ok = False
            for entry in ledger:
                if (
                    entry.get("kind") == KIND_PENDING
                    and entry.get("intent_sha256") == sha
                ):
                    executed_ok = execute(
                        sha,
                        entry.get("admit_receipt_sha256"),
                        entry.get("desired_state_before"),
                        entry.get("desired_state_after"),
                        entry.get("required_micro_usd"),
                        intent_bytes,
                        intent["lane_id"],
                    )
                    break
            if not executed_ok and halted is None:
                halted = halt(HALT_AT_EXECUTION)
            continue
        ev = evidence(intent_bytes, registry, now)
        receipt = admission.admit(
            intent_bytes=intent_bytes,
            registry=registry,
            now=now,
            lease_ok=ev.get("lease_ok"),
            witness_verdict=ev.get("witness_verdict"),
            gates_decision=ev.get("gates_decision"),
            prev_receipt_sha256=gate_prev,
        )
        receipts.append(receipt)
        if receipt.get("sha256"):
            gate_prev = receipt["sha256"]
        if receipt.get("decision") != "ADMIT":
            if receipt.get("reason") in HALT_REASONS:
                halted = halt(receipt["reason"])
            continue
        registry["last_intent_sha256"] = sha
        lane = registry["lanes"][receipt["lane_id"]]
        before = lane["desired_state"]
        after = receipt["desired_state_after"]
        required = receipt.get("required_micro_usd")
        if applier is None:
            pending_entry = _chain_append(
                ledger,
                _entry(
                    now,
                    KIND_PENDING,
                    intent_sha=sha,
                    admit_sha=receipt.get("sha256"),
                    before=before,
                    after=after,
                    required=required,
                ),
            )
            if sink is not None:
                try:
                    sink(pending_entry)
                except Exception:
                    halted = halt("SINK_FAILED")
                    break
            continue
        execute(
            sha,
            receipt.get("sha256"),
            before,
            after,
            required,
            intent_bytes,
            receipt["lane_id"],
        )
    return {
        "registry": registry,
        "ledger": ledger,
        "receipts": receipts,
        "halted": halted,
        "receipt_chain_head": gate_prev,
    }


def load_registry_file(path):
    """Owner-pinned registry from disk; never a cached copy."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_intent_queue_file(path):
    """NDJSON queue: one raw intent document per non-empty line."""
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(line.encode("utf-8"))
    if len(out) > MAX_QUEUE_INTENTS:
        raise ValueError("QUEUE_BUDGET")
    return out


def ledger_lines(ledger):
    """One JSON line per ledger entry, verifiable with gates.verify_chain."""
    return [json.dumps(entry, sort_keys=True) for entry in ledger]


def save_ledger_file(path, ledger):
    text = chr(10).join(ledger_lines(ledger))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + chr(10))


def load_ledger_file(path):
    ledger = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ledger.append(json.loads(line))
    return ledger
