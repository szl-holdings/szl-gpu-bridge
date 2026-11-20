"""All-DENY/no-credential suite for the Massed launch-ceiling gate.

Doctrine: no real credential exists in or for this gate - it is pure.
Every unsafe path must DENY; the only ADMIT is a fully owner-pinned,
lease-verified, price-re-read, whitelisted launch/terminate. Billing
rounds UP to started hours (one second costs one hour), so an A6000
canary at the 2,000,000 uUSD cap may run at most 3 started hours.
These tests are mocked dry-runs by construction (no network, no keys).
"""

import json
from datetime import datetime, timezone

from tools.massed import admission

NOW = datetime(2026, 11, 20, 22, 30, tzinfo=timezone.utc)

CAPS = {
    "canary_cap": 2_000_000,
    "per_job_cap": 5_000_000,
    "wave_cap": 50_000_000,
    "projected_at_completion_cap": 900_000_000,
    "new_commitment_cap": 423_620_000,
    "lane_allocated": 3_000_000,
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
        "price_uusd_hr": 570_000,
        "caps": dict(CAPS),
        "accounts": dict(ACCOUNTS),
    }
    lane.update(over)
    return lane


def _registry(**over):
    registry = {
        "schema": admission.REGISTRY_SCHEMA,
        "last_intent_sha256": None,
        "lanes": {"massed-canary-1": _lane()},
    }
    registry.update(over)
    return registry


def _intent(op="launch", **over):
    doc = {
        "schema": admission.INTENT_SCHEMA,
        "intent_id": "intent-1",
        "actor": "cto-lane",
        "op": op,
        "lane_id": "massed-canary-1",
        "group": "massed-canary-1",
        "product_name": "gpu_1x_a6000_high_ram",
        "region_name": "us-central-1",
        "image_id": 18,
        "hard_deadline_s": 10_800,
        "prev_intent_sha256": None,
    }
    doc.update(over)
    return json.dumps(doc).encode()


def _admit(intent=None, registry=None, **kw):
    kw.setdefault("now", NOW)
    kw.setdefault("lease_ok", True)
    kw.setdefault("price_snapshot_uusd_hr", 570_000)
    return admission.admit(
        intent_bytes=intent if intent is not None else _intent(),
        registry=registry if registry is not None else _registry(),
        **kw,
    )


def test_billing_rounds_up_to_started_hours():
    """One second costs one whole hour: the deficit direction that
    silently kills naive per-second budget arithmetic."""
    assert admission.worst_case_micro_usd(570_000, 1) == 570_000
    assert admission.worst_case_micro_usd(570_000, 3_600) == 570_000
    assert admission.worst_case_micro_usd(570_000, 3_601) == 1_140_000
    assert admission.worst_case_micro_usd(570_000, 10_800) == 1_710_000
    assert admission.worst_case_micro_usd(570_000, 10_801) == 2_280_000


def test_blackwell_started_hour_exceeds_canary_law():
    """Kimi round-7 blocker as arithmetic: PRO 6000 Blackwell at the
    measured 2,190,000 uUSD/hr cannot fit a 2,000,000 uUSD canary cap
    under ANY deadline, because one started hour already exceeds it."""
    assert admission.worst_case_micro_usd(2_190_000, 1) == 2_190_000
    assert admission.worst_case_micro_usd(2_190_000, 3_600) == 2_190_000


def test_deny_clock_unavailable():
    receipt = admission.admit(
        intent_bytes=_intent(),
        registry=_registry(),
        now=datetime(2026, 11, 20, 22, 30),
        lease_ok=True,
        price_snapshot_uusd_hr=570_000,
    )
    assert receipt["decision"] == "DENY"
    assert receipt["reason"] == "CLOCK_UNAVAILABLE"
    assert receipt["massed_mutation_authorized"] is False


def test_deny_registry_invalid_placeholder_price():
    bad = _registry()
    bad["lanes"]["massed-canary-1"]["price_uusd_hr"] = "MODELED"
    receipt = _admit(registry=bad)
    assert receipt["reason"] == "REGISTRY_INVALID"


def test_deny_registry_string_cap_placeholder():
    bad = _registry()
    bad["lanes"]["massed-canary-1"]["caps"]["canary_cap"] = "MODELED"
    receipt = _admit(registry=bad)
    assert receipt["reason"] == "REGISTRY_INVALID"


def test_deny_intent_duplicate_key_and_bad_json():
    raw = b'{"schema": "szl.massed-admission-intent/v1", "schema": "x"}'
    receipt = _admit(intent=raw)
    assert receipt["reason"] == "INTENT_PARSE_FAILED"
    receipt = _admit(intent=b"{")
    assert receipt["reason"] in ("INTENT_PARSE_FAILED", "INTENT_BYTE_BUDGET")


def test_deny_intent_not_chain_head():
    receipt = _admit(intent=_intent(prev_intent_sha256="0" * 64))
    assert receipt["reason"] == "INTENT_NOT_CHAIN_HEAD"


def test_deny_lane_unknown_group_and_actor():
    assert _admit(intent=_intent(lane_id="ghost"))["reason"] == ("LANE_UNKNOWN")
    registry = _registry()
    registry["lanes"]["massed-canary-1"]["group"] = "other-group"
    assert _admit(registry=registry)["reason"] == "GROUP_MISMATCH"
    assert _admit(intent=_intent(actor="sibling-lane"))["reason"] == (
        "ACTOR_NOT_AUTHORIZED"
    )


def test_deny_terminal_and_state_mismatch():
    registry = _registry()
    registry["lanes"]["massed-canary-1"]["desired_state"] = "TERMINATED"
    assert _admit(registry=registry)["reason"] == "LANE_TERMINAL"
    registry = _registry()
    registry["lanes"]["massed-canary-1"]["desired_state"] = "RUNNING"
    assert _admit(registry=registry)["reason"] == ("OP_INVALID_FOR_DESIRED_STATE")


def test_deny_no_lease_admits_nothing():
    """The no-credential law: without the owner lease even a perfect,
    fully-pinned intent denies. No credential argument exists at all."""
    receipt = admission.admit(
        intent_bytes=_intent(),
        registry=_registry(),
        now=NOW,
        lease_ok=False,
        price_snapshot_uusd_hr=570_000,
    )
    assert receipt["decision"] == "DENY"
    assert receipt["reason"] == "OWNER_LEASE_UNVERIFIED"
    assert receipt["massed_mutation_authorized"] is False


def test_deny_blackwell_product_not_whitelisted():
    """The campaign whitelist denies Blackwell before any arithmetic; the
    started-hour arithmetic above proves the cap would deny it anyway."""
    registry = _registry()
    lane = registry["lanes"]["massed-canary-1"]
    lane["product_name"] = "gpu_1x_pro_6000_blackwell"
    lane["price_uusd_hr"] = 2_190_000
    receipt = _admit(
        registry=registry,
        intent=_intent(product_name="gpu_1x_pro_6000_blackwell"),
        price_snapshot_uusd_hr=2_190_000,
    )
    assert receipt["reason"] == "PRODUCT_NOT_WHITELISTED"
    assert receipt["severity"] == "HIGH"


def test_deny_price_over_owner_gate():
    registry = _registry()
    lane = registry["lanes"]["massed-canary-1"]
    lane["product_name"] = "gpu_1x_h100"
    lane["price_uusd_hr"] = 2_730_000
    admission.WHITELIST = frozenset(
        ("gpu_1x_a6000_high_ram", "gpu_1x_a30", "gpu_1x_h100")
    )
    try:
        receipt = _admit(
            registry=registry,
            intent=_intent(product_name="gpu_1x_h100"),
            price_snapshot_uusd_hr=2_730_000,
        )
        assert receipt["reason"] == "PRODUCT_PRICE_OVER_OWNER_GATE"
    finally:
        admission.WHITELIST = frozenset(("gpu_1x_a6000_high_ram", "gpu_1x_a30"))


def test_deny_lane_pin_mismatch_and_price_drift():
    assert (
        _admit(intent=_intent(product_name="gpu_1x_a30"))["reason"]
        == "LANE_PIN_MISMATCH"
    )
    assert _admit(price_snapshot_uusd_hr=None)["reason"] == (
        "PRICE_DRIFT_OR_UNMEASURED"
    )
    assert _admit(price_snapshot_uusd_hr=350_000)["reason"] == (
        "PRICE_DRIFT_OR_UNMEASURED"
    )


def test_deny_budget_floor_unpinned_placeholder():
    registry = _registry()
    lane = registry["lanes"]["massed-canary-1"]
    lane["caps"] = {field: 0 for field in admission.CAP_FIELDS}
    lane["accounts"] = {field: 0 for field in admission.ACCOUNT_FIELDS}
    receipt = _admit(registry=registry)
    assert receipt["reason"] == "BUDGET_FLOOR_UNPINNED"


def test_deny_storage_unpinned():
    """Storage/egress pricing is UNKNOWN on this provider: no owner pin,
    no launch (never assume zero)."""
    registry = _registry()
    registry["lanes"]["massed-canary-1"]["caps"]["storage_pin"] = 0
    assert _admit(registry=registry)["reason"] == "STORAGE_UNPINNED"


def test_deny_lane_allocation_exceeded():
    registry = _registry()
    registry["lanes"]["massed-canary-1"]["caps"]["lane_allocated"] = 1_000_000
    receipt = _admit(registry=registry)
    assert receipt["reason"] == "LANE_ALLOCATION_EXCEEDED"
    assert receipt["worst_case_micro_usd"] == 1_710_000


def test_deny_canary_ceiling_four_started_hours_too_much():
    """3h fits (1,710,000); the 4th started hour (2,280,000) breaks the
    2,000,000 canary cap. The 3h30m plan from the draft memo is thus
    UNLAWFUL under started-hour billing; the canary deadline pins at 3h."""
    receipt = _admit(intent=_intent(hard_deadline_s=14_400))
    assert receipt["reason"] == "LAUNCH_CEILING_CANARY"
    assert receipt["worst_case_micro_usd"] == 2_280_000


def test_deny_job_ceiling():
    registry = _registry()
    lane = registry["lanes"]["massed-canary-1"]
    lane["role"] = "job"
    lane["caps"]["lane_allocated"] = 4_000_000
    lane["caps"]["per_job_cap"] = 2_000_000
    receipt = _admit(registry=registry, intent=_intent(hard_deadline_s=21_600))
    assert receipt["reason"] == "LAUNCH_CEILING_JOB"
    assert receipt["worst_case_micro_usd"] == 3_420_000


def test_deny_wave_ceiling():
    registry = _registry()
    registry["lanes"]["massed-canary-1"]["accounts"]["wave_planned_uusd"] = 48_500_000
    receipt = _admit(registry=registry)
    assert receipt["reason"] == "LAUNCH_CEILING_WAVE"


def test_deny_cumulative_projected_at_completion():
    registry = _registry()
    registry["lanes"]["massed-canary-1"]["accounts"]["measured_spent_uusd"] = (
        898_500_000
    )
    receipt = _admit(registry=registry)
    assert receipt["reason"] == "CUMULATIVE_PROJECTED_EXCEEDED"


def test_deny_new_commitment_cap():
    registry = _registry()
    registry["lanes"]["massed-canary-1"]["accounts"]["campaign_committed_uusd"] = (
        422_500_000
    )
    receipt = _admit(registry=registry)
    assert receipt["reason"] == "NEW_COMMITMENT_EXCEEDED"


def test_admit_fully_pinned_three_hour_canary():
    """The single lawful ADMIT: one variable change (provider), exact
    r6-canary shape, 3h hard stop on the whitelisted A6000 SKU."""
    receipt = _admit(prev_receipt_sha256=None)
    assert receipt["decision"] == "ADMIT"
    assert receipt["reason"] == "ADMITTED_SINGLE_LAUNCHER_ONLY"
    assert receipt["lease_verified"] is True
    assert receipt["worst_case_micro_usd"] == 1_710_000
    assert receipt["desired_state_after"] == "RUNNING"
    assert receipt["massed_mutation_authorized"] is True
    assert receipt["sha256"]
    registry = _registry()
    registry["lanes"]["massed-canary-1"]["desired_state"] = "RUNNING"
    second = _admit(
        intent=_intent(op="terminate"),
        registry=registry,
        prev_receipt_sha256=receipt["sha256"],
    )
    assert second["decision"] == "ADMIT"
    assert second["desired_state_after"] == "TERMINATED"
    assert second["prev_sha256"] == receipt["sha256"]


def test_intent_sha256_stable():
    raw = _intent()
    assert admission.intent_sha256(raw) == admission.intent_sha256(raw)
    assert len(admission.intent_sha256(raw)) == 64
    assert admission.intent_sha256(raw) != admission.intent_sha256(
        _intent(hard_deadline_s=3_600)
    )


def test_all_deny_without_any_owner_pin():
    """The draft registry state (all MODELED-zero pins, no lease, no
    storage pin) must deny every launch - nothing launches until the
    owner pins the lease and integers."""
    registry = _registry()
    lane = registry["lanes"]["massed-canary-1"]
    lane["caps"] = {field: 0 for field in admission.CAP_FIELDS}
    lane["accounts"] = {field: 0 for field in admission.ACCOUNT_FIELDS}
    receipt = admission.admit(
        intent_bytes=_intent(),
        registry=registry,
        now=NOW,
        lease_ok=False,
        price_snapshot_uusd_hr=None,
    )
    assert receipt["decision"] == "DENY"
    assert receipt["massed_mutation_authorized"] is False
