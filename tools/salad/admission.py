"""Single-writer admission gate for Salad container mutations (pure, fail-closed).

Doctrine provenance, recorded 2026-11-20: the task-21 dream round (Kimi
kimi-k3.1-pro:cloud 02:39Z, Mistral Large 4 02:53Z, Mistral Nemo; receipts in
the estate wave-logs commit 6660a3a4) converged on one rule for the owner
directive "make SZL training on Salad fully operational, laptop = control
room only":

    Exactly ONE writer to the Salad API. Every other actor - human, fleet
    doctor, watchdog, agent - may only propose a hash-chained intent.

This module makes each proposal legible before any credential is used. It
composes three existing fail-closed components and adds nothing weaker than
any of them:

* controller_gates.verify_owner_lease - only the lease the owner pinned, at
  the pinned generation, naming the group, unexpired (the lease_ok argument);
* controller_gates.decide_adopted_lane - fleet eligibility; at most
  RESTART_PERMITTED. Quarantined, terminal or unreconciled lanes never admit;
* restart_lineage_witness.witness - restart lineage evidence. A HOLD verdict
  is final. REVIEW_CANDIDATE_ONLY is necessary and never sufficient.

The gate owns no credential, performs no I/O and never touches Salad. It maps
one validated intent to an ADMIT or DENY receipt plus the next desired state
for the desired-state registry. Denied intents are receipted too, chained off
the same seed as admits.

Honesty boundaries kept explicit:

* the intent hash chain is tamper-evidence, NOT authentication; v1 pins an
  actor allowlist per lane, signing lands with szl-evidence-core next;
* admission authorizes the single controller replica ONLY. It is not
  execution proof and not a post-mutation readback; both stay with the caller
  that holds the credential and performs the HTTP call;
* an adapter artifact or a final/ filename is never completion evidence here;
  a lane declared DONE is terminal and no intent op may mutate it;
* a restart must clear the restart-lineage witness AND the fleet gates AND
  the owner lease AND the budget formula in the registry, all at once.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re

from tools.salad import controller_gates as gates

INTENT_SCHEMA = "szl.salad-admission-intent/v1"
REGISTRY_SCHEMA = "szl.desired-state-registry/v1"
RECEIPT_SCHEMA = "szl.salad-admission/v1"

OPS = ("create", "restart", "stop", "park")
DESIRED_STATES = ("NEW", "RUNNING", "STOPPED", "PARKED", "DONE")
OP_STATE_TABLE = {
    "create": ("NEW",),
    "restart": ("STOPPED",),
    "stop": ("RUNNING",),
    "park": ("RUNNING", "STOPPED"),
}
DESIRED_AFTER = {
    "create": "RUNNING",
    "restart": "RUNNING",
    "stop": "STOPPED",
    "park": "PARKED",
}
INTENT_FIELDS = (
    "schema",
    "intent_id",
    "actor",
    "op",
    "lane_id",
    "group",
    "prev_intent_sha256",
)
LANE_FIELDS = ("group", "desired_state", "actors", "budget")
BUDGET_FIELDS = (
    "remaining",
    "start",
    "replay_step",
    "next_checkpoint",
    "finalize",
    "reserve",
)
MAX_INTENT_BYTES = 32768
MAX_REPLAY_STEPS = 10**7
_SHA64 = re.compile(r"[0-9a-f]{64}\Z")
_LANE_ID = re.compile(r"[0-9a-z][0-9a-z-]{1,96}\Z")
_ACTOR = re.compile(r"[0-9a-z][0-9a-z._-]{0,63}\Z")


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("INTENT_DUPLICATE_KEY")
        out[key] = value
    return out


def _bad_constant(_):
    raise ValueError("INTENT_INVALID_JSON")


def validate_registry(registry):
    """Fail closed on any drift in the owner-pinned desired-state registry."""
    if type(registry) is not dict or set(registry) != {
        "schema",
        "last_intent_sha256",
        "lanes",
    }:
        raise ValueError("REGISTRY_SCHEMA_MISMATCH")
    if registry["schema"] != REGISTRY_SCHEMA:
        raise ValueError("REGISTRY_SCHEMA_MISMATCH")
    head = registry["last_intent_sha256"]
    if head is not None and (
        type(head) is not str or _SHA64.fullmatch(head) is None
    ):
        raise ValueError("REGISTRY_HEAD_INVALID")
    lanes = registry["lanes"]
    if type(lanes) is not dict or not lanes:
        raise ValueError("REGISTRY_SCHEMA_MISMATCH")
    for lane_id, lane in lanes.items():
        if type(lane_id) is not str or _LANE_ID.fullmatch(lane_id) is None:
            raise ValueError("REGISTRY_LANE_ID_INVALID")
        if type(lane) is not dict or set(lane) != set(LANE_FIELDS):
            raise ValueError("REGISTRY_LANE_SCHEMA_MISMATCH")
        if not gates.valid_group(lane["group"]):
            raise ValueError("REGISTRY_GROUP_INVALID")
        if lane["desired_state"] not in DESIRED_STATES:
            raise ValueError("REGISTRY_STATE_INVALID")
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
        _validate_budget(lane["budget"])
    return registry


def _validate_budget(budget):
    if type(budget) is not dict or set(budget) != set(BUDGET_FIELDS):
        raise ValueError("BUDGET_SCHEMA_MISMATCH")
    for key in BUDGET_FIELDS:
        if type(budget[key]) is not int or budget[key] < 0:
            raise ValueError("BUDGET_VALUE_INVALID")
    return budget


def _budget_pinned(budget):
    """Budget floor (RATIFY-B amendment): an all-zero lane budget admits
    no paid operation (create, restart). The owner must pin at least one
    nonzero verified allocation before paid work can be admitted; an
    unpriced paid op is never treated as a zero-cost op."""
    return any(budget[field] > 0 for field in BUDGET_FIELDS)


def _required_micro_usd(budget, replay_steps):
    """Owner formula: remaining must cover start, replay, next checkpoint,
    finalize and reserve. Replay is charged per conditional replay step the
    witness measured; an unmeasured replay is never admitted (see admit)."""
    return (
        budget["start"]
        + replay_steps * budget["replay_step"]
        + budget["next_checkpoint"]
        + budget["finalize"]
        + budget["reserve"]
    )


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
        "witness_state": None,
        "witness_reason": None,
        "gates_decision": None,
        "replay_steps_conditional": None,
        "required_micro_usd": None,
        "remaining_micro_usd": None,
        "desired_state_after": None,
        "salad_mutation_authorized": False,
        "observed_utc": gates.utc_text(now),
    }


def admit(
    *,
    intent_bytes,
    registry,
    now,
    lease_ok=False,
    witness_verdict=None,
    gates_decision=None,
    prev_receipt_sha256=None,
):
    """Map one intent to an ADMIT or DENY receipt. Pure, no side effects.

    intent_bytes: raw bytes of the actor's proposed intent (schema
    szl.salad-admission-intent/v1). registry: the owner-pinned desired-state
    registry (szl.desired-state-registry/v1). lease_ok: the boolean returned
    by controller_gates.verify_owner_lease for this lane's contract. witness
    verdict: output of restart_lineage_witness.witness (required for create
    and restart). gates_decision: output of controller_gates.decide_adopted_lane
    (required for restart). prev_receipt_sha256: seed for this gate's receipt
    chain; None only while the gate has never emitted a receipt.
    """
    if not isinstance(now, datetime) or now.tzinfo is None:
        return {
            "schema": RECEIPT_SCHEMA,
            "decision": "DENY",
            "reason": "CLOCK_UNAVAILABLE",
            "salad_mutation_authorized": False,
        }

    def deny(reason, intent=None, **extra):
        record = _base_receipt(now, intent)
        record["reason"] = reason
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
        or not _LANE_ID.fullmatch(intent["intent_id"])
        or type(intent["actor"]) is not str
        or not _ACTOR.fullmatch(intent["actor"])
        or type(intent["lane_id"]) is not str
        or not _LANE_ID.fullmatch(intent["lane_id"])
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

    # Single-writer serialization: only the registry's chain head may be judged.
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
    if state == "DONE":
        return deny("LANE_TERMINAL", intent)
    if state == "PARKED":
        return deny("LANE_PARKED", intent)
    if state not in OP_STATE_TABLE[intent["op"]]:
        return deny("OP_INVALID_FOR_DESIRED_STATE", intent)

    if lease_ok is not True:
        return deny("OWNER_LEASE_UNVERIFIED", intent, lease_verified=False)

    budget = lane["budget"]
    required = None
    replay_steps = None

    if intent["op"] == "create":
        if type(witness_verdict) is not dict:
            return deny("WITNESS_UNPROVIDED", intent)
        if witness_verdict.get("state") != "REVIEW_CANDIDATE_ONLY":
            return deny(
                "WITNESS_HOLD",
                intent,
                witness_state=witness_verdict.get("state"),
                witness_reason=witness_verdict.get("reason"),
            )
        if witness_verdict.get("reason") != "FRESH_NO_PRIOR_STEPS_UNATTESTED":
            return deny("WITNESS_VERDICT_MISMATCH", intent)
        replay_steps = witness_verdict.get("replay_steps_conditional")
        if replay_steps != 0:
            return deny("WITNESS_REPLAY_UNMEASURED", intent)
        required = _required_micro_usd(budget, 0)

    if intent["op"] == "restart":
        if type(witness_verdict) is not dict:
            return deny("WITNESS_UNPROVIDED", intent)
        if witness_verdict.get("state") != "REVIEW_CANDIDATE_ONLY":
            return deny(
                "WITNESS_HOLD",
                intent,
                witness_state=witness_verdict.get("state"),
                witness_reason=witness_verdict.get("reason"),
            )
        if witness_verdict.get("reason") != "FULL_STATE_DECLARED_UNATTESTED":
            return deny("WITNESS_VERDICT_MISMATCH", intent)
        replay_steps = witness_verdict.get("replay_steps_conditional")
        if type(replay_steps) is not int or not 0 <= replay_steps <= MAX_REPLAY_STEPS:
            return deny("WITNESS_REPLAY_UNMEASURED", intent)
        if type(gates_decision) is not dict or (
            gates_decision.get("decision") != "RESTART_PERMITTED"
            or gates_decision.get("restart_permitted") is not True
        ):
            return deny(
                "FLEET_GATE_NOT_PERMITTED",
                intent,
                gates_decision=gates_decision.get("decision")
                if type(gates_decision) is dict
                else None,
            )
        required = _required_micro_usd(budget, replay_steps)

    if (
        required is not None
        and intent['op'] in ('create', 'restart')
        and not _budget_pinned(budget)
    ):
        return deny(
            'BUDGET_FLOOR_UNPINNED',
            intent,
            replay_steps_conditional=replay_steps,
            required_micro_usd=required,
            remaining_micro_usd=budget["remaining"],
        )
    if required is not None and budget["remaining"] < required:
        return deny(
            "BUDGET_SHORTFALL",
            intent,
            replay_steps_conditional=replay_steps,
            required_micro_usd=required,
            remaining_micro_usd=budget["remaining"],
        )

    record = _base_receipt(now, intent)
    record.update(
        decision="ADMIT",
        reason="ADMITTED_SINGLE_WRITER_ONLY",
        lease_verified=True,
        witness_state=witness_verdict.get("state")
        if type(witness_verdict) is dict
        else None,
        witness_reason=witness_verdict.get("reason")
        if type(witness_verdict) is dict
        else None,
        gates_decision=gates_decision.get("decision")
        if type(gates_decision) is dict
        else None,
        replay_steps_conditional=replay_steps,
        required_micro_usd=required,
        remaining_micro_usd=budget["remaining"],
        desired_state_after=DESIRED_AFTER[intent["op"]],
        salad_mutation_authorized=True,
    )
    try:
        return gates.chain_record(record, prev_receipt_sha256)
    except ValueError:
        record.pop("prev_sha256", None)
        record.pop("sha256", None)
        record["decision"] = "DENY"
        record["reason"] = "RECEIPT_CHAIN_SEED_INVALID"
        record["salad_mutation_authorized"] = False
        return record


def intent_sha256(intent_bytes):
    """Stable id for the registry head: sha256 of the exact intent bytes."""
    if type(intent_bytes) is not bytes:
        raise ValueError("INTENT_BYTES_REQUIRED")
    return hashlib.sha256(intent_bytes).hexdigest()
