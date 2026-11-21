"""Provider-spy conformance suite for the Massed launch adapter.

Squad round-8 single next action, both reviews (kimi_w8_clean.txt,
astra_w8_reply.txt): prove, using mocked provider calls, that no
paid create can occur without a freshly verified, owner-authorized,
one-time admission and an atomic budget reservation. Every unsafe
path must leave the provider spy untouched. No network, no keys.

Kimi round-9 additions: ambiguous provider failures PARK (never
blanket-rollback) and only an owner-gated reconcile() with a
provider-confirmed outcome can resolve a parked intent.
"""

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from tools.massed import admission
from tools.massed.launch_adapter import InMemoryStateStore, MassedLaunchAdapter

NOW = datetime(2026, 11, 20, 23, 10, tzinfo=timezone.utc)
MISSING = object()
PRICE = 570_000
WORST_3H = 1_710_000
PARKED = "PROVIDER_LAUNCH_UNKNOWN_PENDING_RECONCILIATION"

CAPS = {
    "canary_cap": 2_000_000,
    "per_job_cap": 5_000_000,
    "wave_cap": 50_000_000,
    "projected_at_completion_cap": 900_000_000,
    "new_commitment_cap": 423_620_000,
    "lane_allocated": 10_000_000,
    "storage_pin": 500_000,
}
ACCOUNTS = {
    "measured_spent_uusd": 476_380_000,
    "wave_planned_uusd": 0,
    "campaign_committed_uusd": 0,
}


def _lane(**over):
    lane = {
        "group": "massed-canary-1",
        "desired_state": "PENDING_LAUNCH",
        "actors": ["cto-lane"],
        "role": "canary",
        "product_name": "gpu_1x_a6000_high_ram",
        "region_name": "us-central-1",
        "image_id": 18,
        "price_uusd_hr": PRICE,
        "caps": dict(CAPS),
        "accounts": dict(ACCOUNTS),
    }
    lane.update(over)
    return lane


def _registry():
    return {
        "schema": admission.REGISTRY_SCHEMA,
        "last_intent_sha256": None,
        "lanes": {"massed-canary-1": _lane()},
    }


def _intent(op="launch", intent_id="intent-1", prev=None, **over):
    doc = {
        "schema": admission.INTENT_SCHEMA,
        "intent_id": intent_id,
        "actor": "cto-lane",
        "op": op,
        "lane_id": "massed-canary-1",
        "group": "massed-canary-1",
        "product_name": "gpu_1x_a6000_high_ram",
        "region_name": "us-central-1",
        "image_id": 18,
        "hard_deadline_s": 10_800,
        "prev_intent_sha256": prev,
    }
    doc.update(over)
    return json.dumps(doc).encode()


class ProviderSpy:
    """Records every paid-surface call. The adapter must be the only
    code that reaches this object."""

    def __init__(self, fail_launch=False):
        self.launch_calls = []
        self.terminate_calls = []
        self.fail_launch = fail_launch

    def launch(self, **kw):
        self.launch_calls.append(kw)
        if self.fail_launch:
            raise RuntimeError("provider refused")
        return "instance-1"

    def terminate(self, **kw):
        self.terminate_calls.append(kw)
        return "terminated"


def _adapter(
    provider=None,
    registry=None,
    lease_ok=True,
    price=MISSING,
    observed=None,
    now=MISSING,
    fail_launch=False,
    store=None,
    lease_verifier=None,
    status_verifier=None,
    **options,
):
    state = {
        "registry": registry if registry is not None else _registry(),
        "receipt_head": None,
        "spent_intents": {},
        "unknown_intents": {},
        "deadlines": {},
    }
    observed = NOW if observed is None else observed
    price = (PRICE, observed) if price is MISSING else price
    now_fn = options.pop(
        "now_fn", (lambda: now) if now is not MISSING else (lambda: NOW)
    )
    return MassedLaunchAdapter(
        provider=provider
        if provider is not None
        else ProviderSpy(fail_launch=fail_launch),
        lease_verifier=lease_verifier or (lambda lane_id: lease_ok),
        price_reader=lambda: price,
        state_store=store if store is not None else InMemoryStateStore(state),
        now_fn=now_fn,
        status_verifier=status_verifier,
        **options,
    )


def _store(adapter):
    return adapter._store


def test_admit_touches_provider_once_and_reserves_atomically():
    spy = ProviderSpy()
    a = _adapter(provider=spy)
    receipt = a.launch(_intent())
    assert receipt["decision"] == "ADMIT"
    assert receipt["provider_touched"] is True
    assert len(spy.launch_calls) == 1
    call = spy.launch_calls[0]
    assert call["product_name"] == "gpu_1x_a6000_high_ram"
    assert call["hard_deadline_s"] == 10_800
    state = _store(a).load()
    lane = state["registry"]["lanes"]["massed-canary-1"]
    assert lane["accounts"]["wave_planned_uusd"] == WORST_3H
    assert lane["accounts"]["campaign_committed_uusd"] == WORST_3H
    assert lane["desired_state"] == "RUNNING"
    head = admission.intent_sha256(_intent())
    assert state["registry"]["last_intent_sha256"] == head
    assert state["deadlines"]["intent-1"][1] == "massed-canary-1"


def test_forged_owner_lease_cannot_reach_provider():
    """Kimi: lease_ok must be DERIVED, a caller cannot assert True."""
    spy = ProviderSpy()
    a = _adapter(provider=spy, lease_ok=False)
    receipt = a.launch(_intent())
    assert receipt["decision"] == "DENY"
    assert receipt["reason"] == "OWNER_LEASE_UNVERIFIED"
    assert receipt["provider_touched"] is False
    assert spy.launch_calls == []
    state = _store(a).load()
    lane = state["registry"]["lanes"]["massed-canary-1"]
    assert lane["accounts"]["wave_planned_uusd"] == 0


def test_stale_or_missing_price_snapshot_denies_without_provider_call():
    """Astra: freshness/provenance must be established, not claimed."""
    stale = NOW - timedelta(seconds=400)
    for snapshot in (None, (PRICE, stale), (777, NOW)):
        spy = ProviderSpy()
        a = _adapter(provider=spy, price=snapshot)
        receipt = a.launch(_intent())
        assert receipt["decision"] == "DENY", snapshot
        assert receipt["reason"] == "PRICE_DRIFT_OR_UNMEASURED"
        assert spy.launch_calls == []


def test_admit_is_single_use_replay_denies():
    launch = _intent()
    spy = ProviderSpy()
    a = _adapter(provider=spy)
    first = a.launch(launch)
    assert first["decision"] == "ADMIT"
    replay = a.launch(launch)
    assert replay["decision"] == "DENY"
    assert replay["reason"] == "INTENT_NOT_CHAIN_HEAD"
    assert len(spy.launch_calls) == 1

    # registry-head rollback attack: even resetting the chain head
    # cannot spend the same intent twice (defense in depth).
    revision, state = _store(a).read()
    state["registry"]["last_intent_sha256"] = None
    state["registry"]["lanes"]["massed-canary-1"]["desired_state"] = "PENDING_LAUNCH"
    assert _store(a).save(state, expected_revision=revision)
    again = a.launch(launch)
    assert again["decision"] == "DENY"
    assert again["reason"] == "INTENT_ALREADY_SPENT"
    assert len(spy.launch_calls) == 1


def test_mutated_intent_after_admit_denies():
    first = _intent()
    spy = ProviderSpy()
    a = _adapter(provider=spy)
    assert a.launch(first)["decision"] == "ADMIT"
    head = admission.intent_sha256(first)
    revision, state2 = _store(a).read()
    state2["registry"]["lanes"]["massed-canary-1"]["desired_state"] = "PENDING_LAUNCH"
    assert _store(a).save(state2, expected_revision=revision)
    mutated = _intent(
        intent_id="intent-2",
        prev=head,
        product_name="gpu_1x_h100",
    )
    receipt = a.launch(mutated)
    assert receipt["decision"] == "DENY"
    assert receipt["reason"] == "LANE_PIN_MISMATCH"
    assert len(spy.launch_calls) == 1
    assert receipt["provider_touched"] is False


def test_concurrent_launches_serialize_to_at_most_one_admit():
    """Kimi concurrency spec: two racing launches whose combined worst
    case exceeds the lane allocation -> at most one ADMIT, exactly one
    provider call, and the account moves by exactly one worst case."""
    results = []
    barrier = threading.Barrier(2)

    def fire(idx):
        barrier.wait()
        results.append(a.launch(_intent(intent_id="intent-" + str(idx))))

    spy = ProviderSpy()
    a = _adapter(provider=spy)
    a._store._state["registry"]["lanes"]["massed-canary-1"]["caps"][
        "lane_allocated"
    ] = WORST_3H
    threads = [threading.Thread(target=fire, args=(i,)) for i in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    decisions = sorted(r["decision"] for r in results)
    assert decisions == ["ADMIT", "DENY"], decisions
    assert len(spy.launch_calls) == 1
    state = _store(a).load()
    lane = state["registry"]["lanes"]["massed-canary-1"]
    assert lane["accounts"]["wave_planned_uusd"] == WORST_3H
    assert len(state["spent_intents"]) == 1


def test_provider_failure_parks_as_unknown_and_never_rolls_back_blindly():
    """Kimi round-9 pre-canary fix: a provider exception may be a timeout
    AFTER acceptance; a blanket rollback would permit a duplicate paid
    create. The reservation and the spent marker are KEPT and the intent
    parks as UNKNOWN pending owner-gated reconciliation."""
    spy = ProviderSpy(fail_launch=True)
    a = _adapter(provider=spy, fail_launch=True)
    receipt = a.launch(_intent())
    assert receipt["decision"] == "DENY"
    assert receipt["reason"] == PARKED
    assert receipt["reservation_kept"] is True
    assert spy.launch_calls
    state = _store(a).load()
    lane = state["registry"]["lanes"]["massed-canary-1"]
    assert lane["accounts"]["wave_planned_uusd"] == WORST_3H
    assert lane["desired_state"] == "RUNNING"
    assert len(state["spent_intents"]) == 1
    assert len(state["unknown_intents"]) == 1
    # a blind retry of the same intent can NEVER double-create:
    # it denies on the advanced chain head (INTENT_NOT_CHAIN_HEAD)
    # or, under a registry rollback attack, on INTENT_ALREADY_SPENT
    retry = a.launch(_intent())
    assert retry["decision"] == "DENY"
    assert retry["reason"] in ("INTENT_NOT_CHAIN_HEAD", "INTENT_ALREADY_SPENT")
    assert len(spy.launch_calls) == 1


def _verified_status(attempt, outcome, evidence):
    return evidence == (attempt["attempt_id"], outcome)


def _reconcile(adapter, attempt, outcome):
    return adapter.reconcile(
        attempt["intent_id"],
        outcome,
        (attempt["attempt_id"], outcome),
        attempt_id=attempt["attempt_id"],
    )


def test_reconcile_rejected_confirmed_non_acceptance_releases_reservation():
    spy = ProviderSpy(fail_launch=True)
    a = _adapter(provider=spy, status_verifier=_verified_status)
    attempt = a.launch(_intent())
    assert attempt["reason"] == PARKED
    fixed = _reconcile(a, attempt, "REJECTED")
    assert fixed["decision"] == "DENY"
    assert fixed["reason"] == "PROVIDER_REJECTED_RECONCILED"
    assert fixed["reservation_kept"] is False
    state = _store(a).load()
    lane = state["registry"]["lanes"]["massed-canary-1"]
    assert lane["accounts"]["wave_planned_uusd"] == 0
    assert lane["accounts"]["campaign_committed_uusd"] == 0
    assert lane["desired_state"] == "PENDING_LAUNCH"
    head = admission.intent_sha256(_intent())
    assert state["registry"]["last_intent_sha256"] == head
    assert state["spent_intents"] == {head: "intent-1"}
    assert state["unknown_intents"] == {}
    # confirmed non-acceptance opens the retry path
    spy.fail_launch = False
    again = a.launch(_intent())
    assert again["decision"] == "ADMIT"
    assert len(spy.launch_calls) == 2


def test_reconcile_accepted_finalizes_reservation_intent_stays_spent():
    spy = ProviderSpy(fail_launch=True)
    a = _adapter(provider=spy, status_verifier=_verified_status)
    attempt = a.launch(_intent())
    assert attempt["reason"] == PARKED
    ok = _reconcile(a, attempt, "ACCEPTED")
    assert ok["decision"] == "ADMIT"
    assert ok["reason"] == "PROVIDER_ACCEPTED_RECONCILED"
    assert ok["provider_result"] == (attempt["attempt_id"], "ACCEPTED")
    state = _store(a).load()
    lane = state["registry"]["lanes"]["massed-canary-1"]
    assert lane["accounts"]["wave_planned_uusd"] == WORST_3H
    assert state["unknown_intents"] == {}
    assert len(state["spent_intents"]) == 1
    # the intent stays spent: no duplicate create possible
    replay = a.launch(_intent())
    assert replay["decision"] == "DENY"
    assert replay["reason"] in ("INTENT_NOT_CHAIN_HEAD", "INTENT_ALREADY_SPENT")
    assert len(spy.launch_calls) == 1


def test_reconcile_requires_parked_intent_and_known_outcome():
    spy = ProviderSpy(fail_launch=True)
    a = _adapter(provider=spy, fail_launch=True)
    assert a.launch(_intent())["reason"] == PARKED
    bad = a.reconcile("intent-1", "MAYBE")
    assert bad["decision"] == "DENY"
    assert bad["reason"] == "RECONCILE_OUTCOME_INVALID"
    assert len(_store(a).load()["unknown_intents"]) == 1
    missing = a.reconcile("intent-nope", "ACCEPTED")
    assert missing["decision"] == "DENY"
    assert missing["reason"] == "INTENT_NOT_UNKNOWN"


def test_terminate_route_admits_without_price_or_reservation():
    spy = ProviderSpy()
    a = _adapter(provider=spy)
    receipt = a.terminate(_intent(op="terminate"))
    assert receipt["decision"] == "ADMIT"
    assert receipt["provider_touched"] is True
    assert len(spy.terminate_calls) == 1
    assert spy.launch_calls == []
    state = _store(a).load()
    lane = state["registry"]["lanes"]["massed-canary-1"]
    assert lane["desired_state"] == "TERMINATED"
    assert lane["accounts"]["wave_planned_uusd"] == 0


def test_clock_unavailable_denies_without_provider_call():
    spy = ProviderSpy()
    a = _adapter(provider=spy, now=None)
    receipt = a.launch(_intent())
    assert receipt["decision"] == "DENY"
    assert receipt["reason"] == "CLOCK_UNAVAILABLE"
    assert receipt["observed_utc"] is None
    assert spy.launch_calls == []


def test_overdue_reports_recorded_deadlines_only():
    a = _adapter()
    assert a.launch(_intent())["decision"] == "ADMIT"
    assert a.overdue(now=NOW) == []
    late = NOW + timedelta(seconds=10_800 + 60)
    due = a.overdue(now=late)
    assert len(due) == 1
    assert due[0]["intent_id"] == "intent-1"
    assert due[0]["lane_id"] == "massed-canary-1"


def test_operation_mismatch_denies_before_verification():
    spy = ProviderSpy()
    verified = []
    adapter = _adapter(provider=spy, lease_verifier=verified.append)
    assert (
        adapter.launch(_intent(op="terminate"))["reason"] == "INTENT_OPERATION_MISMATCH"
    )
    assert adapter.terminate(_intent())["reason"] == "INTENT_OPERATION_MISMATCH"
    assert verified == spy.launch_calls == spy.terminate_calls == []


@pytest.mark.parametrize("value", [(False, "expired"), "False", 1])
def test_truthy_non_boolean_lease_denies(value):
    spy = ProviderSpy()
    assert _adapter(provider=spy, lease_ok=value).launch(_intent())["reason"] == (
        "OWNER_LEASE_UNVERIFIED"
    )
    assert not spy.launch_calls


def test_unverified_or_unbound_reconciliation_leaves_attempt_parked():
    adapter = _adapter(fail_launch=True, status_verifier=_verified_status)
    attempt = adapter.launch(_intent())
    before = _store(adapter).read()
    assert (
        adapter.reconcile("intent-1", "REJECTED")["reason"]
        == "RECONCILE_ATTEMPT_UNBOUND"
    )
    forged = adapter.reconcile(
        "intent-1",
        "REJECTED",
        ("another-attempt", "REJECTED"),
        attempt_id=attempt["attempt_id"],
    )
    assert forged["reason"] == "RECONCILE_EVIDENCE_UNVERIFIED"
    unauthorized = _adapter(
        store=_store(adapter), lease_ok=False, status_verifier=_verified_status
    )
    denied = _reconcile(unauthorized, attempt, "REJECTED")
    assert denied["reason"] == "OWNER_LEASE_UNVERIFIED"
    assert denied["lease_verified"] is False
    assert _store(adapter).read() == before


def test_rejection_preserves_descendant_termination_and_head():
    adapter = _adapter(fail_launch=True, status_verifier=_verified_status)
    attempt = adapter.launch(_intent())
    terminate = _intent(
        op="terminate", intent_id="intent-2", prev=admission.intent_sha256(_intent())
    )
    assert adapter.terminate(terminate)["decision"] == "ADMIT"
    assert _reconcile(adapter, attempt, "REJECTED")["reservation_kept"] is False
    state = _store(adapter).load()
    assert state["registry"]["last_intent_sha256"] == admission.intent_sha256(terminate)
    lane = state["registry"]["lanes"]["massed-canary-1"]
    assert lane["desired_state"] == "TERMINATED"
    assert lane["accounts"]["wave_planned_uusd"] == 0
    assert adapter.launch(_intent())["decision"] == "DENY"


def test_adapters_racing_on_same_revision_grant_dispatch_once():
    class RacingStore(InMemoryStateStore):
        def read(self):
            result = super().read()
            if result[0] == 0:
                barrier.wait(timeout=5)
            return result

    barrier = threading.Barrier(2)
    store = RacingStore(_store(_adapter()).load())
    spy = ProviderSpy()
    adapters = [_adapter(provider=spy, store=store) for _ in range(2)]
    results = []
    workers = [
        threading.Thread(
            target=lambda a: results.append(a.launch(_intent())), args=(adapter,)
        )
        for adapter in adapters
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert sorted(receipt["reason"] for receipt in results) == [
        "ADMITTED_SINGLE_LAUNCHER_ONLY",
        "DISPATCH_GRANT_NOT_COMMITTED",
    ]
    assert len(spy.launch_calls) == 1
    lane = store.load()["registry"]["lanes"]["massed-canary-1"]
    assert lane["accounts"]["wave_planned_uusd"] == WORST_3H


@pytest.mark.parametrize("max_age", [301, float("inf"), float("nan"), -1, True])
def test_snapshot_age_limit_is_bounded(max_age):
    with pytest.raises(ValueError):
        _adapter(max_snapshot_age_s=max_age)


def test_future_snapshot_denies_dispatch():
    spy = ProviderSpy()
    adapter = _adapter(provider=spy, observed=NOW + timedelta(seconds=1))
    assert adapter.launch(_intent())["reason"] == "PRICE_DRIFT_OR_UNMEASURED"
    assert not spy.launch_calls


def test_snapshot_and_lease_rechecked_after_grant_commit():
    spy = ProviderSpy()
    ticks = iter((NOW, NOW + timedelta(seconds=1)))
    adapter = _adapter(
        provider=spy,
        observed=NOW - timedelta(seconds=300),
        now_fn=lambda: next(ticks),
    )
    denied = adapter.launch(_intent())
    assert denied["reason"] == "PRICE_DRIFT_OR_UNMEASURED"
    assert denied["reservation_kept"] is True
    leases = iter((True, False))
    adapter = _adapter(provider=spy, lease_verifier=lambda lane: next(leases))
    assert adapter.launch(_intent())["reason"] == "OWNER_LEASE_UNVERIFIED"
    assert not spy.launch_calls


@pytest.mark.parametrize("after_contact", [False, True])
def test_unfinalized_dispatch_is_recoverable_by_new_adapter(after_contact):
    class InterruptedProvider(ProviderSpy):
        def launch(self, **kw):
            super().launch(**kw)
            raise SystemExit

    spy = InterruptedProvider()
    leases = iter((True,))

    def lease(lane):
        if after_contact or next(leases, False):
            return True
        raise SystemExit

    adapter = _adapter(provider=spy, lease_verifier=lease)
    with pytest.raises(SystemExit):
        adapter.launch(_intent())
    resumed = _adapter(
        provider=spy, store=_store(adapter), status_verifier=_verified_status
    )
    parked = next(iter(_store(resumed).load()["unknown_intents"].values()))
    assert resumed.launch(_intent())["decision"] == "DENY"
    assert len(spy.launch_calls) == int(after_contact)
    if after_contact:
        assert spy.launch_calls[0]["attempt_id"] == parked["attempt_id"]
    assert _reconcile(resumed, parked, "REJECTED")["reservation_kept"] is False


def test_confirmed_retry_is_attempt_bound_and_capped_across_adapters():
    spy = ProviderSpy(fail_launch=True)
    adapter = _adapter(provider=spy, status_verifier=_verified_status)
    first = adapter.launch(_intent())
    assert _reconcile(adapter, first, "REJECTED")["reservation_kept"] is False
    resumed = _adapter(
        provider=spy, store=_store(adapter), status_verifier=_verified_status
    )
    second = resumed.launch(_intent())
    assert second["reason"] == PARKED
    assert second["attempt_id"] != first["attempt_id"]
    assert (
        _reconcile(resumed, first, "REJECTED")["reason"] == "RECONCILE_ATTEMPT_UNBOUND"
    )
    assert _reconcile(resumed, second, "REJECTED")["reservation_kept"] is False
    assert resumed.launch(_intent())["decision"] == "DENY"
    assert len(spy.launch_calls) == 2
