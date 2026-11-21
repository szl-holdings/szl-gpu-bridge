"""Owner-gated Massed dispatch with revision-checked, recoverable grants."""

from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from tools.massed import admission

LAUNCH_SCHEMA = "szl.mass -LAUNCH/v1"
RECONCILE_OUTCOMES = ("ACCEPTED", "REJECTED")
MAX_SNAPSHOT_AGE_S = 300
FUTURE_SKEW_S = 0


class InMemoryStateStore:
    """Revision-checked storage shared by adapters in one process."""

    def __init__(self, state):
        self._state = copy.deepcopy(state)
        self._revision = 0
        self.lock = threading.Lock()

    def read(self):
        with self.lock:
            return self._revision, copy.deepcopy(self._state)

    def load(self):
        return self.read()[1]

    def save(self, state, *, expected_revision):
        with self.lock:
            if expected_revision != self._revision:
                return False
            self._state = copy.deepcopy(state)
            self._revision += 1
            return True


def _intent_json(intent_bytes):
    if type(intent_bytes) is not bytes or len(intent_bytes) > admission.MAX_INTENT_BYTES:
        return {}
    try:
        doc = json.loads(intent_bytes.decode("utf-8"))
    except (ValueError, RecursionError):
        return {}
    return doc if type(doc) is dict else {}


def intent_deadline_s(intent_bytes):
    value = _intent_json(intent_bytes).get("hard_deadline_s")
    return value if type(value) is int else 0


def _valid_time(value):
    return isinstance(value, datetime) and value.utcoffset() is not None


def _verified(verifier, *args):
    try:
        return verifier is not None and verifier(*args) is True
    except Exception:
        return False


class MassedLaunchAdapter:
    """Admit and reserve before invoking the credentialed provider port.

    The lease verifier checks the current owner lease for a lane. The price
    reader supplies a measured (price_uusd_hr, observed_utc) snapshot.
    The store returns (revision, isolated state) from read(). Its conditional
    save() must return exactly True only after commit. Production storage
    must make that commit durable and serialize writers across processes.

    The provider receives attempt_id on each mutation. The status verifier
    checks (attempt, outcome, evidence) and returns exactly True only for
    authenticated evidence bound to that attempt. A confirmed rejection must
    preclude acceptance of the attempt, even for an in-flight or late request.
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
        status_verifier=None,
    ):
        if (
            type(max_snapshot_age_s) not in (int, float)
            or not 0 <= max_snapshot_age_s <= MAX_SNAPSHOT_AGE_S
        ):
            raise ValueError("max_snapshot_age_s must be between 0 and 300")
        self._provider = provider
        self._lease_verifier = lease_verifier
        self._price_reader = price_reader
        self._store = state_store
        self._now = now_fn
        self._max_age = max_snapshot_age_s
        self._status_verifier = status_verifier
        self._lock = threading.Lock()

    def _fresh_price(self, snapshot, now):
        if not isinstance(snapshot, (tuple, list)) or len(snapshot) != 2:
            return None
        price, observed = snapshot
        if type(price) is not int or price < 1 or not _valid_time(observed):
            return None
        age = (now - observed.astimezone(timezone.utc)).total_seconds()
        return price if -FUTURE_SKEW_S <= age <= self._max_age else None

    def _save(self, revision, state):
        try:
            return self._store.save(state, expected_revision=revision) is True
        except Exception:
            return False

    def _adapter_deny(self, now, reason, info):
        return {
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
            "observed_utc": admission.gates.utc_text(now) if _valid_time(now) else None,
        }

    def launch(self, intent_bytes):
        return self._execute(intent_bytes, "launch")

    def terminate(self, intent_bytes):
        return self._execute(intent_bytes, "terminate")

    def _execute(self, intent_bytes, op):
        intent = _intent_json(intent_bytes)
        if intent.get("op") != op:
            return self._adapter_deny(None, "INTENT_OPERATION_MISMATCH", intent)
        with self._lock:
            try:
                revision, state = self._store.read()
            except Exception:
                return self._adapter_deny(None, "STATE_UNAVAILABLE", intent)
            registry = state["registry"]
            lane_id = intent.get("lane_id")
            lease_ok = _verified(self._lease_verifier, lane_id)
            snapshot = self._price_reader() if op == "launch" and lease_ok else None
            if isinstance(snapshot, list):
                snapshot = tuple(snapshot)
            now = self._now()
            if not _valid_time(now):
                return self._adapter_deny(None, "CLOCK_UNAVAILABLE", intent)
            now = now.astimezone(timezone.utc)
            price = self._fresh_price(snapshot, now)
            intent_sha = admission.intent_sha256(intent_bytes)
            rejected = state.get("rejected_intents", {}).get(intent_sha)
            retry = (
                rejected is not None
                and rejected["attempt_number"] == 1
                and registry["last_intent_sha256"] == intent_sha
                and state.get("lane_attempts", {}).get(lane_id) == rejected["attempt_id"]
            )
            admit_registry = registry
            if retry:
                # Validate the exact original intent without changing its committed head.
                admit_registry = dict(registry, last_intent_sha256=rejected["prev_head"])
            receipt = admission.admit(
                intent_bytes=intent_bytes,
                registry=admit_registry,
                now=now,
                lease_ok=lease_ok,
                price_snapshot_uusd_hr=price,
                prev_receipt_sha256=state["receipt_head"],
            )
            state["receipt_head"] = receipt.get("sha256") or state["receipt_head"]
            if receipt["decision"] != "ADMIT":
                self._save(revision, state)
                receipt["provider_touched"] = False
                return receipt

            op = receipt["op"]
            spent = state["spent_intents"]
            if intent_sha in spent and not retry:
                return self._adapter_deny(now, "INTENT_ALREADY_SPENT", intent)
            intent_id = receipt["intent_id"]
            if not retry and intent_id in spent.values():
                return self._adapter_deny(now, "INTENT_ID_ALREADY_USED", intent)
            lane = registry["lanes"][lane_id]
            worst = receipt.get("worst_case_micro_usd") or 0
            attempt_id = uuid4().hex
            entry = {
                "attempt_id": attempt_id,
                "attempt_number": 2 if retry else 1,
                "intent_id": intent_id,
                "intent_sha256": intent_sha,
                "grant_sha256": receipt["sha256"],
                "op": op,
                "lane_id": lane_id,
                "worst_case_micro_usd": worst,
                "prev_head": intent["prev_intent_sha256"],
                "prev_desired_state": lane["desired_state"],
            }
            accounts = lane["accounts"]
            accounts["wave_planned_uusd"] += worst
            accounts["campaign_committed_uusd"] += worst
            lane["desired_state"] = receipt["desired_state_after"]
            registry["last_intent_sha256"] = intent_sha
            spent[intent_sha] = intent_id
            state.setdefault("lane_attempts", {})[lane_id] = attempt_id
            if op == "launch":
                deadline = now + timedelta(seconds=intent["hard_deadline_s"])
                state["deadlines"][intent_id] = (admission.gates.utc_text(deadline), lane_id)
            # Any grant that survives without a terminal result is already parked.
            state.setdefault("unknown_intents", {})[intent_sha] = entry
            if not self._save(revision, state):
                return self._adapter_deny(now, "DISPATCH_GRANT_NOT_COMMITTED", intent)

            lease_ok = _verified(self._lease_verifier, lane_id)
            now = self._now()
            if _valid_time(now):
                now = now.astimezone(timezone.utc)
            reason = None
            if not _valid_time(now):
                reason = "CLOCK_UNAVAILABLE"
            elif not lease_ok:
                reason = "OWNER_LEASE_UNVERIFIED"
            elif op == "launch" and (
                now >= deadline or self._fresh_price(snapshot, now) != price
            ):
                reason = "PRICE_DRIFT_OR_UNMEASURED"
            if reason:
                record = self._adapter_deny(now, reason, intent)
                record.update(attempt_id=attempt_id, reservation_kept=True)
                return record

            try:
                if op == "launch":
                    provider_result = self._provider.launch(
                        lane_id=lane_id,
                        product_name=lane["product_name"],
                        region_name=lane["region_name"],
                        image_id=lane["image_id"],
                        intent_id=intent_id,
                        attempt_id=attempt_id,
                        hard_deadline_s=intent["hard_deadline_s"],
                    )
                else:
                    provider_result = self._provider.terminate(
                        lane_id=lane_id, intent_id=intent_id, attempt_id=attempt_id
                    )
            except Exception:
                finalized = False
            else:
                finalized = self._finalize_dispatch(intent_sha, attempt_id)
            if not finalized:
                record = self._adapter_deny(
                    now, "PROVIDER_LAUNCH_UNKNOWN_PENDING_RECONCILIATION", intent
                )
                record.update(
                    attempt_id=attempt_id,
                    lease_verified=True,
                    provider_touched=True,
                    massed_mutation_authorized=True,
                    worst_case_micro_usd=worst,
                    reservation_kept=True,
                )
                return record
            receipt.update(
                provider_touched=True, provider_result=provider_result, attempt_id=attempt_id
            )
            return receipt

    def _finalize_dispatch(self, intent_sha, attempt_id):
        try:
            revision, state = self._store.read()
        except Exception:
            return False
        entry = state["unknown_intents"].get(intent_sha)
        if entry is None or entry["attempt_id"] != attempt_id:
            return False
        del state["unknown_intents"][intent_sha]
        state.setdefault("completed_attempts", {})[attempt_id] = {
            "grant_sha256": entry["grant_sha256"],
            "outcome": "ACCEPTED",
        }
        return self._save(revision, state)

    def reconcile(self, intent_id, outcome, provider_result=None, *, attempt_id=None):
        """Resolve a parked attempt using owner authorization and trusted evidence."""
        probe = {"intent_id": intent_id, "op": "reconcile"}
        with self._lock:
            try:
                revision, state = self._store.read()
            except Exception:
                return self._adapter_deny(None, "STATE_UNAVAILABLE", probe)
            entry = next(
                (
                    parked for parked in state.get("unknown_intents", {}).values()
                    if parked["intent_id"] == intent_id
                ),
                None,
            )
            if entry is None:
                return self._adapter_deny(None, "INTENT_NOT_UNKNOWN", probe)
            if outcome not in RECONCILE_OUTCOMES:
                return self._adapter_deny(None, "RECONCILE_OUTCOME_INVALID", probe)
            if type(attempt_id) is not str or entry.get("attempt_id") != attempt_id:
                return self._adapter_deny(None, "RECONCILE_ATTEMPT_UNBOUND", probe)
            lane_id = entry["lane_id"]
            if not _verified(self._lease_verifier, lane_id):
                return self._adapter_deny(None, "OWNER_LEASE_UNVERIFIED", probe)
            if provider_result is None or not _verified(
                self._status_verifier, copy.deepcopy(entry), outcome, provider_result
            ):
                return self._adapter_deny(None, "RECONCILE_EVIDENCE_UNVERIFIED", probe)
            if not _verified(self._lease_verifier, lane_id):
                return self._adapter_deny(None, "OWNER_LEASE_UNVERIFIED", probe)
            now = self._now()
            if not _valid_time(now):
                return self._adapter_deny(None, "CLOCK_UNAVAILABLE", probe)
            intent_sha = entry["intent_sha256"]
            worst = entry["worst_case_micro_usd"]
            rejected = outcome == "REJECTED"
            if rejected:
                lane = state["registry"]["lanes"][lane_id]
                accounts = lane["accounts"]
                fields = ("wave_planned_uusd", "campaign_committed_uusd")
                if any(accounts[field] < worst for field in fields):
                    return self._adapter_deny(now, "RESERVATION_ACCOUNT_MISMATCH", probe)
                for field in fields:
                    accounts[field] -= worst
                if state["lane_attempts"].get(lane_id) == attempt_id:
                    lane["desired_state"] = entry["prev_desired_state"]
                if entry["op"] == "launch":
                    state["deadlines"].pop(intent_id, None)
                state.setdefault("rejected_intents", {})[intent_sha] = entry
            del state["unknown_intents"][intent_sha]
            state.setdefault("completed_attempts", {})[attempt_id] = {
                "grant_sha256": entry["grant_sha256"],
                "outcome": outcome,
            }
            if not self._save(revision, state):
                return self._adapter_deny(now, "RECONCILIATION_NOT_COMMITTED", probe)
            return {
                "schema": LAUNCH_SCHEMA,
                "decision": "DENY" if rejected else "ADMIT",
                "reason": "PROVIDER_REJECTED_RECONCILED" if rejected else "PROVIDER_ACCEPTED_RECONCILED",
                "intent_id": intent_id,
                "attempt_id": attempt_id,
                "lane_id": lane_id,
                "op": entry["op"],
                "lease_verified": True,
                "severity": "INFO",
                "worst_case_micro_usd": worst,
                "massed_mutation_authorized": False,
                "provider_touched": False,
                "reservation_kept": not rejected,
                "provider_result": provider_result,
                "observed_utc": admission.gates.utc_text(now),
            }


    def overdue(self, now=None):
        """Hard deadlines recorded by this adapter that are past due.
        The actual provider kill is owner-gated ops (watchdog), NOT this
        method: report only, no mutation authorization implied."""
        now = now or self._now()
        if not isinstance(now, datetime) or now.tzinfo is None:
            return []
        _, state = self._store.read()
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
