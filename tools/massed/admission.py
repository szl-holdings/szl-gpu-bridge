"""Fail-closed launch-ceiling admission gate for Massed Compute launches.

Doctrine provenance: squad round 7, 2026-11-20 (kimi_w7_clean2.txt,
mistral_w7_clean2.txt, astra_w7_reply.txt and the campaign plan draft
C:/SZL/rt3714/massed_campaign_plan_v0.1_draft.json). The squad converged
on these laws and this module encodes them:

* ONE variable per canary: provider only, the exact r6-canary shape
  otherwise. A canary is a "launch" op on a lane with role "canary".
* No receipt API exists on this provider, so admission happens BEFORE
  the launch: worst_case_micro_usd = price x hard deadline must fit
  every pin at once (the LAUNCH_CEILING deny). Billing rounds UP.
* Hourly rounding makes any product whose single-hour price exceeds the
  canary cap unlawful for a canary (measured PRO 6000 Blackwell =
  2,190,000 uUSD/hr vs cap 2,000,000).
* Storage/egress pricing is UNKNOWN on this provider: a lane with no
  owner-pinned storage ceiling denies launch (STORAGE_UNPINNED).
* MODELED placeholder budgets are never pins: all-zero caps deny
  (BUDGET_FLOOR_UNPINNED, mirroring the salad RATIFY-B amendment) and
  non-integer values fail registry validation outright.
* Product whitelist: only 1x A6000 and 1x A30 (both under the 97c/hr
  owner gate). Anything else denies. A price re-read that drifted from
  the lane pin denies (PRICE_DRIFT_OR_UNMEASURED).
* The gate owns no credential, performs no I/O and never touches Massed.
  It maps one validated intent to an ADMIT or DENY receipt. An ADMIT
  authorizes the single credentialed launcher process ONLY.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re

from tools.salad import controller_gates as gates

INTENT_SCHEMA = "szl.massed-admission-intent/v1"
REGISTRY_SCHEMA = "szl.massed-lane-registry/v1"
RECEIPT_SCHEMA = "szl.massed-admission/v1"

OPS = ("launch", "terminate")
DESIRED_STATES = ("PENDING_LAUNCH", "RUNNING", "TERMINATED")
OP_STATE_TABLE = {
    "launch": ("PENDING_LAUNCH",),
    "terminate": ("PENDING_LAUNCH", "RUNNING"),
}
DESIRED_AFTER = {"launch": "RUNNING", "terminate": "TERMINATED"}

INTENT_FIELDS = (
    "schema",
    "intent_id",
    "actor",
    "op",
    "lane_id",
    "group",
    "product_name",
    "region_name",
    "image_id",
    "hard_deadline_s",
    "prev_intent_sha256",
)
LANE_FIELDS = (
    "group",
    "desired_state",
    "actors",
    "role",
    "product_name",
    "region_name",
    "image_id",
    "price_uusd_hr",
    "caps",
    "accounts",
)
CAP_FIELDS = (
    "canary_cap",
    "per_job_cap",
    "wave_cap",
    "projected_at_completion_cap",
    "new_commitment_cap",
    "lane_allocated",
    "storage_pin",
)
ACCOUNT_FIELDS = (
    "measured_spent_uusd",
    "wave_planned_uusd",
    "campaign_committed_uusd",
)

WHITELIST = frozenset(("gpu_1x_a6000_high_ram", "gpu_1x_a30"))
OWNER_PRICE_GATE_UUSD_HR = 970_000
MAX_INTENT_BYTES = 32768
MIN_DEADLINE_S = 60
MAX_DEADLINE_S = 21_600

_SHA64 = re.compile("[0-9a-f]{64}")
_LANE_ID = re.compile("[0-9a-z][0-9a-z-]{1,96}")
_ACTOR = re.compile("[0-9a-z][0-9a-z._-]{0,63}")
_NAME = re.compile("[0-9a-z][0-9a-z_.-]{0,127}")
_REGION = re.compile("[0-9a-z][0-9a-z-]{0,63}")


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("INTENT_DUPLICATE_KEY")
        out[key] = value
    return out


def _bad_constant(_):
    raise ValueError("INTENT_INVALID_JSON")


def _validate_text(value, pattern, error):
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(error)
    return value


def _validate_nonneg_int(value, error):
    if type(value) is not int or value < 0:
        raise ValueError(error)
    return value


def validate_registry(registry):
    """Fail closed on any drift in the owner-pinned lane registry."""
    if type(registry) is not dict or set(registry) != {
        "schema",
        "last_intent_sha256",
        "lanes",
    }:
        raise ValueError("REGISTRY_SCHEMA_MISMATCH")
    if registry["schema"] != REGISTRY_SCHEMA:
        raise ValueError("REGISTRY_SCHEMA_MISMATCH")
    head = registry["last_intent_sha256"]
    if head is not None and (type(head) is not str or _SHA64.fullmatch(head) is None):
        raise ValueError("REGISTRY_HEAD_INVALID")
    lanes = registry["lanes"]
    if type(lanes) is not dict or not lanes:
        raise ValueError("REGISTRY_SCHEMA_MISMATCH")
    for lane_id, lane in lanes.items():
        if type(lane_id) is not str or _LANE_ID.fullmatch(lane_id) is None:
            raise ValueError("REGISTRY_LANE_ID_INVALID")
        if type(lane) is not dict or set(lane) != set(LANE_FIELDS):
            raise ValueError("REGISTRY_LANE_SCHEMA_MISMATCH")
        _validate_text(lane["group"], _NAME, "REGISTRY_GROUP_INVALID")
        if lane["desired_state"] not in DESIRED_STATES:
            raise ValueError("REGISTRY_STATE_INVALID")
        if lane["role"] not in ("canary", "job"):
            raise ValueError("REGISTRY_ROLE_INVALID")
        _validate_text(lane["product_name"], _NAME, "REGISTRY_PRODUCT_INVALID")
        _validate_text(lane["region_name"], _REGION, "REGISTRY_REGION_INVALID")
        _validate_nonneg_int(lane["image_id"], "REGISTRY_IMAGE_INVALID")
        if lane["image_id"] < 1:
            raise ValueError("REGISTRY_IMAGE_INVALID")
        _validate_nonneg_int(lane["price_uusd_hr"], "REGISTRY_PRICE_INVALID")
        if lane["price_uusd_hr"] < 1:
            raise ValueError("REGISTRY_PRICE_INVALID")
        actors = lane["actors"]
        if (
            type(actors) is not list
            or not actors
            or any(
                type(actor) is not str or _ACTOR.fullmatch(actor) is None
                for actor in actors
            )
            or len(set(actors)) != len(actors)
        ):
            raise ValueError("REGISTRY_ACTORS_INVALID")
        caps = lane["caps"]
        if type(caps) is not dict or set(caps) != set(CAP_FIELDS):
            raise ValueError("CAPS_SCHEMA_MISMATCH")
        for field in CAP_FIELDS:
            _validate_nonneg_int(caps[field], "CAPS_VALUE_INVALID")
        accounts = lane["accounts"]
        if type(accounts) is not dict or set(accounts) != set(ACCOUNT_FIELDS):
            raise ValueError("ACCOUNTS_SCHEMA_MISMATCH")
        for field in ACCOUNT_FIELDS:
            _validate_nonneg_int(accounts[field], "ACCOUNTS_VALUE_INVALID")
    return registry


def _caps_pinned(caps, accounts):
    """Budget floor mirror (RATIFY-B): an all-zero allocation with zero
    planned spend is an unpinned MODELED placeholder and admits no launch."""
    return (
        caps["lane_allocated"] > 0
        or accounts["wave_planned_uusd"] > 0
        or any(caps[field] > 0 for field in ("canary_cap", "per_job_cap"))
    )


def worst_case_micro_usd(price_uusd_hr, hard_deadline_s):
    """Billing rounds UP to started hours: one second costs one hour.

    Kimi round-7 blocker codified: a product whose single started hour
    already exceeds a cap is denied no matter how short the deadline is."""
    started_hours = (hard_deadline_s + 3599) // 3600
    return started_hours * price_uusd_hr


def _base_receipt(now, intent):
    return {
        "schema": RECEIPT_SCHEMA,
        "decision": "DENY",
        "reason": None,
        "intent_id": intent.get("intent_id") if intent else None,
        "lane_id": intent.get("lane_id") if intent else None,
        "group": intent.get("group") if intent else None,
        "op": intent.get("op") if intent else None,
        "actor": intent.get("actor") if intent else None,
        "lease_verified": False,
        "severity": "INFO",
        "worst_case_micro_usd": None,
        "desired_state_after": None,
        "massed_mutation_authorized": False,
        "observed_utc": gates.utc_text(now),
    }


def admit(
    *,
    intent_bytes,
    registry,
    now,
    lease_ok=False,
    price_snapshot_uusd_hr=None,
    prev_receipt_sha256=None,
):
    """Map one Massed intent to an ADMIT or DENY receipt. Pure, no I/O.

    lease_ok: the boolean returned by the owner-lease verifier for this
    lane (Massed launches stay owner-gated). price_snapshot_uusd_hr:
    the price RE-READ from the measured provider inventory immediately
    before judgement; a drifted or absent snapshot denies (fail closed).
    """
    if not isinstance(now, datetime) or now.tzinfo is None:
        return {
            "schema": RECEIPT_SCHEMA,
            "decision": "DENY",
            "reason": "CLOCK_UNAVAILABLE",
            "massed_mutation_authorized": False,
        }

    def deny(reason, intent=None, severity="INFO", **extra):
        record = _base_receipt(now, intent)
        record["reason"] = reason
        record["severity"] = severity
        record.update(extra)
        try:
            return gates.chain_record(record, prev_receipt_sha256)
        except ValueError:
            record.pop("prev_sha256", None)
            record.pop("sha256", None)
            record["reason"] = "RECEIPT_CHAIN_SEED_INVALID"
            return record

    try:
        registry = validate_registry(registry)
    except ValueError:
        return deny("REGISTRY_INVALID")

    if type(intent_bytes) is not bytes or not 0 < len(intent_bytes) <= MAX_INTENT_BYTES:
        return deny("INTENT_BYTE_BUDGET")
    try:
        intent = json.loads(
            intent_bytes.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_bad_constant,
        )
    except (UnicodeError, ValueError):
        return deny("INTENT_PARSE_FAILED")
    if type(intent) is not dict:
        return deny("INTENT_SCHEMA_MISMATCH")

    if set(intent) != set(INTENT_FIELDS) or intent.get("schema") != INTENT_SCHEMA:
        return deny("INTENT_SCHEMA_MISMATCH", intent)
    if (
        type(intent["intent_id"]) is not str
        or _LANE_ID.fullmatch(intent["intent_id"]) is None
        or type(intent["actor"]) is not str
        or _ACTOR.fullmatch(intent["actor"]) is None
        or type(intent["lane_id"]) is not str
        or _LANE_ID.fullmatch(intent["lane_id"]) is None
        or type(intent["group"]) is not str
        or intent["op"] not in OPS
        or not (
            intent["prev_intent_sha256"] is None
            or (
                type(intent["prev_intent_sha256"]) is str
                and _SHA64.fullmatch(intent["prev_intent_sha256"])
            )
        )
    ):
        return deny("INTENT_VALUE_INVALID", intent)

    if (
        type(intent["product_name"]) is not str
        or _NAME.fullmatch(intent["product_name"]) is None
        or type(intent["region_name"]) is not str
        or _REGION.fullmatch(intent["region_name"]) is None
        or type(intent["image_id"]) is not int
        or intent["image_id"] < 1
        or type(intent["hard_deadline_s"]) is not int
        or not MIN_DEADLINE_S <= intent["hard_deadline_s"] <= MAX_DEADLINE_S
    ):
        return deny("INTENT_VALUE_INVALID", intent)

    if intent["prev_intent_sha256"] != registry["last_intent_sha256"]:
        return deny("INTENT_NOT_CHAIN_HEAD", intent)

    lane = registry["lanes"].get(intent["lane_id"])
    if lane is None:
        return deny("LANE_UNKNOWN", intent)
    if intent["group"] != lane["group"]:
        return deny("GROUP_MISMATCH", intent)
    if intent["actor"] not in lane["actors"]:
        return deny("ACTOR_NOT_AUTHORIZED", intent)

    state = lane["desired_state"]
    if state == "TERMINATED":
        return deny("LANE_TERMINAL", intent)
    if state not in OP_STATE_TABLE[intent["op"]]:
        return deny("OP_INVALID_FOR_DESIRED_STATE", intent)

    if lease_ok is not True:
        return deny("OWNER_LEASE_UNVERIFIED", intent, lease_verified=False)

    if intent["op"] == "terminate":
        record = _base_receipt(now, intent)
        record.update(
            decision="ADMIT",
            reason="ADMITTED_SINGLE_LAUNCHER_ONLY",
            lease_verified=True,
            desired_state_after=DESIRED_AFTER["terminate"],
            massed_mutation_authorized=True,
        )
        try:
            return gates.chain_record(record, prev_receipt_sha256)
        except ValueError:
            record.pop("prev_sha256", None)
            record.pop("sha256", None)
            record["decision"] = "DENY"
            record["reason"] = "RECEIPT_CHAIN_SEED_INVALID"
            record["massed_mutation_authorized"] = False
            return record

    caps = lane["caps"]
    accounts = lane["accounts"]

    if lane["product_name"] not in WHITELIST:
        return deny("PRODUCT_NOT_WHITELISTED", intent, severity="HIGH")
    if lane["price_uusd_hr"] > OWNER_PRICE_GATE_UUSD_HR:
        return deny("PRODUCT_PRICE_OVER_OWNER_GATE", intent, severity="HIGH")
    if (
        intent["product_name"] != lane["product_name"]
        or intent["region_name"] != lane["region_name"]
        or intent["image_id"] != lane["image_id"]
    ):
        return deny("LANE_PIN_MISMATCH", intent, severity="HIGH")
    if (
        type(price_snapshot_uusd_hr) is not int
        or price_snapshot_uusd_hr < 1
        or price_snapshot_uusd_hr != lane["price_uusd_hr"]
    ):
        return deny("PRICE_DRIFT_OR_UNMEASURED", intent, severity="HIGH")

    if not _caps_pinned(caps, accounts):
        return deny("BUDGET_FLOOR_UNPINNED", intent, severity="HIGH")
    if caps["storage_pin"] < 1:
        return deny("STORAGE_UNPINNED", intent, severity="HIGH")

    worst = worst_case_micro_usd(price_snapshot_uusd_hr, intent["hard_deadline_s"])

    if worst > caps["lane_allocated"]:
        return deny(
            "LANE_ALLOCATION_EXCEEDED",
            intent,
            "HIGH",
            worst_case_micro_usd=worst,
        )
    if worst > caps["per_job_cap"]:
        return deny(
            "LAUNCH_CEILING_JOB",
            intent,
            "HIGH",
            worst_case_micro_usd=worst,
        )
    if lane["role"] == "canary" and worst > caps["canary_cap"]:
        return deny(
            "LAUNCH_CEILING_CANARY",
            intent,
            "HIGH",
            worst_case_micro_usd=worst,
        )
    if accounts["wave_planned_uusd"] + worst > caps["wave_cap"]:
        return deny(
            "LAUNCH_CEILING_WAVE",
            intent,
            "HIGH",
            worst_case_micro_usd=worst,
        )
    if (
        accounts["measured_spent_uusd"] + accounts["wave_planned_uusd"] + worst
        > caps["projected_at_completion_cap"]
    ):
        return deny(
            "CUMULATIVE_PROJECTED_EXCEEDED",
            intent,
            "HIGH",
            worst_case_micro_usd=worst,
        )
    if accounts["campaign_committed_uusd"] + worst > caps["new_commitment_cap"]:
        return deny(
            "NEW_COMMITMENT_EXCEEDED",
            intent,
            "HIGH",
            worst_case_micro_usd=worst,
        )

    record = _base_receipt(now, intent)
    record.update(
        decision="ADMIT",
        reason="ADMITTED_SINGLE_LAUNCHER_ONLY",
        lease_verified=True,
        worst_case_micro_usd=worst,
        desired_state_after=DESIRED_AFTER["launch"],
        massed_mutation_authorized=True,
    )
    try:
        return gates.chain_record(record, prev_receipt_sha256)
    except ValueError:
        record.pop("prev_sha256", None)
        record.pop("sha256", None)
        record["decision"] = "DENY"
        record["reason"] = "RECEIPT_CHAIN_SEED_INVALID"
        record["massed_mutation_authorized"] = False
        return record


def intent_sha256(intent_bytes):
    """Stable id for the registry head: sha256 of the exact intent bytes."""
    if type(intent_bytes) is not bytes:
        raise ValueError("INTENT_BYTES_INVALID")
    return hashlib.sha256(intent_bytes).hexdigest()
