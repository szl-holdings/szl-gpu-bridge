"""Sole authorized credentialed launch path for Massed Compute.

Squad round-8 single next action (kimi_w8_clean.txt and astra_w8_reply.txt,
both verdicts: CONFIRM policy, DENY enforcement readiness until this exists).
The pure admission gate (tools/massed/admission.py) maps an intent to
ADMIT or DENY; THIS module is the only code allowed to stand between an
ADMIT and the provider port. Callers never receive the provider object,
never pass lease booleans, and never pass price numbers: those are
DERIVED by this adapter from injected, trusted sources.

Provider-spy proven properties (tests/test_massed_launch_adapter.py):

* NO PAID CREATE WITHOUT A ONE-TIME ADMIT: every deny path leaves the
  provider port untouched (spy assertion).
* lease_ok is derived from the injected owner-lease verifier scoped to
  the lane; callers cannot assert it (Kimi: forged lease denied).
* The price snapshot comes from the injected price reader WITH a max
  age; stale or missing snapshots deny via PRICE_DRIFT_OR_UNMEASURED.
* An ADMIT is single-use: bound to the intent sha256 AND the registry
  chain head; replays and post-admit mutations deny; a chain-head
  rollback attack still hits INTENT_ALREADY_SPENT (defense in depth).
* Budget reservation is ATOMIC under the adapter lock: two concurrent
  launches whose combined worst case exceeds a cap serialize to at
  most one ADMIT and one provider call.
* AMBIGUOUS provider failures PARK, never roll back (Kimi round-9: a
  timeout may land AFTER the provider accepted the launch, and a
  blanket rollback would permit a duplicate paid create). The
  reservation and the spent marker are KEPT, the intent parks as
  UNKNOWN, and reconcile() resolves it: PROVIDER-CONFIRMED ACCEPTED
  finalizes the reservation; PROVIDER-CONFIRMED REJECTED (certain
  non-acceptance) rolls everything back and allows one retry;
  anything unproven stays parked forever (fail closed).
* Only an owner-gated reconciliation call can move a parked intent.

Role is owner-bound in the registry lane, never caller-selected.

Known remaining boundaries, honestly NOT enforced here (pre-canary
queue from the reviews): the in-memory store proves single-process
atomicity ONLY - a production durable store must survive independent
processes, crash/restart and enforce a uniqueness constraint on spent
intent hashes; storage_pin is required but not yet folded into the
worst case; hard deadlines are recorded (adapter.overdue) but the
termination watchdog and provider reconciliation are owner-gated ops.
"""

from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timedelta

from tools.massed import admission

LAUNCH_SCHEMA = "szl.mass -LAUNCH/v1"
RECONCILE_OUTCOMES = ("ACCEPTED", "REJECTED")
MAX_SNAPSHOT_AGE_S = 300


class InMemoryStateStore:
    """Single-process state store. NOT durable: a deploying operator must
    replace it with an atomic, durable store (save must be atomic)."""

    def __init__(self, state):
        self._state = state
        self.lock = threading.Lock()

    def load(self):
        with self.lock:
            return copy.deepcopy(self._state)

    def save(self, state):
        with self.lock:
            self._state = copy.deepcopy(state)


def _intent_json(intent_bytes):
    try:
        return json.loads(intent_bytes.decode("utf-8"))
    except Exception:
        return {}


def _doc_bytes(doc):
    return json.dumps(doc).encode()


def intent_deadline_s(intent_bytes):
    value = _intent_json(intent_bytes).get("hard_deadline_s")
    return value if type(value) is int else 0


class MassedLaunchAdapter:
    """The only authorized path from intent to a paid provider mutation.

    provider: an object exposing launch(...)/terminate(...). It is the
    ONLY credentialed surface; the adapter never returns it to callers.
    lease_verifier: callable(lane_id) -> bool; the production
    implementation must check a sealed owner lease (validity, expiry,
    lane scope, pin scope), never a caller-supplied boolean.
    price_reader: callable() -> (price_uusd_hr, observed_utc) or None;
    must re-read the measured provider inventory, not accept numbers.
    state_store: load()/save() with atomic save semantics.
    now_fn: callable() -> tz-aware datetime (None or naive = no clock).
    """

    def __init__(
        self,
        *,
        provider,
        lease_verifier,
        price_reader,
        state_store,
        now_fn,
        max_snapshot_age_s=MAX_SNAPSHOT_AGE_S,
    ):
        self._provider = provider
        self._lease_verifier = lease_verifier
        self._price_reader = price_reader
        self._store = state_store
        self._now = now_fn
        self._max_age = max_snapshot_age_s
        self._lock = threading.Lock()

    def _fresh_price(self, now):
        snap = self._price_reader()
        if not snap:
            return None
        price, observed = snap
        if type(price) is not int or price < 1:
            return None
        if not isinstance(observed, datetime) or observed.tzinfo is None:
            return None
        if (now - observed).total_seconds() > self._max_age:
            return None
        return price

    def _adapter_deny(self, now, reason, intent_bytes):
        info = _intent_json(intent_bytes)
        record = {
            "schema": LAUNCH_SCHEMA,
            "decision": "DENY",
            "reason": reason,
            "intent_id": info.get("intent_id"),
            "lane_id": info.get("lane_id"),
            "op": info.get("op"),
            "lease_verified": False,
            "severity": "HIGH",
            "worst_case_micro_usd": None,
            "massed_mutation_authorized": False,
            "provider_touched": False,
            "observed_utc": admission.gates.utc_text(now) if now is not None else None,
        }
        return record

    def launch(self, intent_bytes):
        return self._execute(intent_bytes, op="launch")

    def terminate(self, intent_bytes):
        return self._execute(intent_bytes, op="terminate")

    def _execute(self, intent_bytes, op):
        with self._lock:
            now = self._now()
            if not isinstance(now, datetime) or now.tzinfo is None:
                return self._adapter_deny(None, "CLOCK_UNAVAILABLE", intent_bytes)
            state = self._store.load()
            state.setdefault("unknown_intents", {})
            registry = state["registry"]
            lane_id = _intent_json(intent_bytes).get("lane_id")
            lease_ok = bool(self._lease_verifier(lane_id))
            price = self._fresh_price(now) if op == "launch" else None
            receipt = admission.admit(
                intent_bytes=intent_bytes,
                registry=registry,
                now=now,
                lease_ok=lease_ok,
                price_snapshot_uusd_hr=price,
                prev_receipt_sha256=state["receipt_head"],
            )
            head = receipt.get("sha256") or state["receipt_head"]
            state["receipt_head"] = head
            if receipt["decision"] != "ADMIT":
                self._store.save(state)
                receipt["provider_touched"] = False
                return receipt
            lane = registry["lanes"][receipt["lane_id"]]
            prev_lane = copy.deepcopy(lane)
            prev_head = registry["last_intent_sha256"]
            intent_sha = admission.intent_sha256(intent_bytes)
            if state["spent_intents"].get(intent_sha):
                self._store.save(state)
                return self._adapter_deny(now, "INTENT_ALREADY_SPENT", intent_bytes)
            worst = receipt.get("worst_case_micro_usd") or 0
            accounts = lane["accounts"]
            accounts["wave_planned_uusd"] += worst
            accounts["campaign_committed_uusd"] += worst
            lane["desired_state"] = receipt["desired_state_after"]
            registry["last_intent_sha256"] = intent_sha
            state["spent_intents"][intent_sha] = receipt["intent_id"]
            if op == "launch":
                deadline = now + timedelta(seconds=intent_deadline_s(intent_bytes))
                state["deadlines"][receipt["intent_id"]] = (
                    admission.gates.utc_text(deadline),
                    receipt["lane_id"],
                )
            self._store.save(state)
            try:
                if op == "launch":
                    provider_result = self._provider.launch(
                        lane_id=receipt["lane_id"],
                        product_name=lane["product_name"],
                        region_name=lane["region_name"],
                        image_id=lane["image_id"],
                        intent_id=receipt["intent_id"],
                        hard_deadline_s=intent_deadline_s(intent_bytes),
                    )
                else:
                    provider_result = self._provider.terminate(
                        lane_id=receipt["lane_id"],
                        intent_id=receipt["intent_id"],
                    )
            except Exception:
                # Kimi round-9 pre-canary fix: the exception may be a
                # timeout AFTER the provider accepted the launch. A
                # blanket rollback would permit a duplicate paid create.
                # PARK instead: keep reservation + spent marker, record
                # everything reconcile() needs to resolve either way.
                state["unknown_intents"][intent_sha] = {
                    "intent_id": receipt["intent_id"],
                    "op": op,
                    "lane_id": receipt["lane_id"],
                    "worst_case_micro_usd": worst,
                    "prev_head": prev_head,
                    "prev_lane": prev_lane,
                }
                self._store.save(state)
                record = self._adapter_deny(
                    now, "PROVIDER_LAUNCH_UNKNOWN_PENDING_RECONCILIATION", intent_bytes
                )
                record.update(
                    op=op,
                    provider_touched=True,
                    massed_mutation_authorized=True,
                    worst_case_micro_usd=worst,
                    reservation_kept=True,
                )
                return record
            receipt["provider_touched"] = True
            receipt["provider_result"] = provider_result
            return receipt

    def reconcile(self, intent_id, outcome, provider_result=None):
        """Owner-gated resolution of a parked UNKNOWN intent.

        outcome "ACCEPTED": the provider CONFIRMED the instance exists.
        The reservation finalizes and stays committed; the intent
        remains spent (no duplicate create is ever possible).
        outcome "REJECTED": the provider CONFIRMED certain non-acceptance
        BEFORE creating anything. The reservation, chain head, spent
        marker and deadline roll back completely and the same intent
        may be retried.
        Any other outcome, or an intent that is not parked, denies and
        the intent stays parked (fail closed).
        """
        probe = {"intent_id": intent_id, "op": "reconcile"}
        with self._lock:
            now = self._now()
            state = self._store.load()
            entry = None
            intent_sha = None
            for sha, parked in state.get("unknown_intents", {}).items():
                if parked["intent_id"] == intent_id:
                    entry = parked
                    intent_sha = sha
                    break
            if entry is None:
                return self._adapter_deny(now, "INTENT_NOT_UNKNOWN", _doc_bytes(probe))
            if outcome not in RECONCILE_OUTCOMES:
                self._store.save(state)
                record = self._adapter_deny(
                    now, "RECONCILE_OUTCOME_INVALID", _doc_bytes(probe)
                )
                record["reservation_kept"] = True
                return record
            if outcome == "ACCEPTED":
                del state["unknown_intents"][intent_sha]
                self._store.save(state)
                record = {
                    "schema": LAUNCH_SCHEMA,
                    "decision": "ADMIT",
                    "reason": "PROVIDER_ACCEPTED_RECONCILED",
                    "intent_id": intent_id,
                    "lane_id": entry["lane_id"],
                    "op": entry["op"],
                    "lease_verified": True,
                    "severity": "INFO",
                    "worst_case_micro_usd": entry["worst_case_micro_usd"],
                    "massed_mutation_authorized": True,
                    "provider_touched": True,
                    "reservation_kept": True,
                    "provider_result": provider_result,
                    "observed_utc": admission.gates.utc_text(now)
                    if isinstance(now, datetime) and now.tzinfo
                    else None,
                }
                return record
            # REJECTED: provider confirmed certain non-acceptance.
            lane = state["registry"]["lanes"][entry["lane_id"]]
            prev_lane = entry["prev_lane"]
            worst = entry["worst_case_micro_usd"]
            accounts = lane["accounts"]
            accounts["wave_planned_uusd"] = max(
                0, accounts["wave_planned_uusd"] - worst
            )
            accounts["campaign_committed_uusd"] = max(
                0, accounts["campaign_committed_uusd"] - worst
            )
            lane["desired_state"] = prev_lane["desired_state"]
            state["registry"]["last_intent_sha256"] = entry["prev_head"]
            state["spent_intents"].pop(intent_sha, None)
            state["deadlines"].pop(intent_id, None)
            del state["unknown_intents"][intent_sha]
            self._store.save(state)
            record = {
                "schema": LAUNCH_SCHEMA,
                "decision": "DENY",
                "reason": "PROVIDER_REJECTED_ROLLED_BACK",
                "intent_id": intent_id,
                "lane_id": entry["lane_id"],
                "op": entry["op"],
                "lease_verified": True,
                "severity": "INFO",
                "worst_case_micro_usd": worst,
                "massed_mutation_authorized": False,
                "provider_touched": True,
                "reservation_kept": False,
                "observed_utc": admission.gates.utc_text(now)
                if isinstance(now, datetime) and now.tzinfo
                else None,
            }
            return record

    def overdue(self, now=None):
        """Hard deadlines recorded by this adapter that are past due.
        The actual provider kill is owner-gated ops (watchdog), NOT this
        method: report only, no mutation authorization implied."""
        now = now or self._now()
        if not isinstance(now, datetime) or now.tzinfo is None:
            return []
        state = self._store.load()
        due = []
        for intent_id, entry in state["deadlines"].items():
            deadline_utc = entry[0]
            lane_id = entry[1]
            try:
                dt = datetime.fromisoformat(deadline_utc)
            except ValueError:
                continue
            if dt.tzinfo is not None and dt <= now:
                due.append(
                    {
                        "intent_id": intent_id,
                        "lane_id": lane_id,
                        "deadline_utc": deadline_utc,
                    },
                )
        return due
