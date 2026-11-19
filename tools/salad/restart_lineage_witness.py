"""Fail-closed, offline lineage witness for preemptible training restarts.

All input is caller-provided and unauthenticated by this module. The witness never
permits GPU dispatch, a controller transition, checkpoint deletion, or promotion.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re

SCHEMA = "szl.restart-lineage-input/v1"
SCHEMA_V2 = "szl.restart-lineage-input/v2"
INPUT_FIELDS = (
    "schema",
    "pins",
    "target_steps",
    "prior_step",
    "observed_at_utc",
    "intent",
    "checkpoint",
)
TOPOLOGY_FIELDS = (
    "saved_dp_size",
    "requested_dp_size",
    "dataloader_checkpoint_step",
    "rank_digests",
)
RANK_FIELDS = ("rank", "checkpoint_step", "state_sha256")
MAX_DP_RANKS = 128
PINS = ("source", "base", "dataset", "command")
STATE_FILES = (
    "adapter",
    "trainer_state",
    "optimizer",
    "rng",
    "scheduler",
    "dataloader",
    "tokenizer",
)
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
MAX_BYTES = 32768
MAX_STEP = 10**10
MAX_AGE_SECONDS = 900


class Hold(ValueError):
    """Fixed non-secret reason code, never an exception with input content."""


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise Hold("DUPLICATE_INPUT_KEY")
        out[key] = value
    return out


def _bad_constant(_):
    raise Hold("NONFINITE_NUMBER")


def _object(value, keys, reason):
    if type(value) is not dict or set(value) != set(keys):
        raise Hold(reason)
    return value


def _step(value, *, allow_none=False):
    if value is None and allow_none:
        return None
    if type(value) is not int or not 0 <= value <= MAX_STEP:
        raise Hold("STEP_UNAVAILABLE_OR_INVALID")
    return value


def _pins(value):
    _object(value, PINS, "IMMUTABLE_PINS_MISSING")
    for key in PINS:
        pattern = SHA64 if key == "command" else SHA40
        if type(value[key]) is not str or pattern.fullmatch(value[key]) is None:
            raise Hold("IMMUTABLE_PIN_INVALID")
    return value


def _time(value):
    if type(value) is not str or len(value) > 40:
        raise Hold("OBSERVATION_TIME_INVALID")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Hold("OBSERVATION_TIME_INVALID") from exc
    if result.tzinfo is None:
        raise Hold("OBSERVATION_TIME_INVALID")
    return result.astimezone(timezone.utc)


def _topology(value, *, checkpoint_step: int) -> None:
    """Check *declared* data-parallel loader continuity, never uploaded bytes.

    Megatron Bridge/Energon can reshard model/optimizer state independently,
    but exact rank-local dataloader continuation requires the saved DP size.
    No hash or rank declaration is authenticated by this offline witness.
    """
    if value is None:
        raise Hold("TOPOLOGY_EVIDENCE_REQUIRED")
    _object(value, TOPOLOGY_FIELDS, "TOPOLOGY_SCHEMA_MISMATCH")
    saved, requested = value["saved_dp_size"], value["requested_dp_size"]
    if (
        type(saved) is not int
        or type(requested) is not int
        or not 1 <= saved <= MAX_DP_RANKS
        or not 1 <= requested <= MAX_DP_RANKS
    ):
        raise Hold("DP_SIZE_INVALID")
    if saved != requested:
        raise Hold("DP_TOPOLOGY_CHANGED")
    step = _step(value["dataloader_checkpoint_step"])
    if step != checkpoint_step:
        raise Hold("DATALOADER_STATE_STEP_MISMATCH")
    digests = value["rank_digests"]
    if type(digests) is not list or len(digests) != saved:
        raise Hold("RANK_STATE_INCOMPLETE")
    seen = set()
    for row in digests:
        _object(row, RANK_FIELDS, "RANK_STATE_MALFORMED")
        rank = row["rank"]
        if type(rank) is not int or not 0 <= rank < saved or rank in seen:
            raise Hold("RANK_STATE_INCOMPLETE")
        seen.add(rank)
        if type(row["checkpoint_step"]) is not int or row["checkpoint_step"] != step:
            raise Hold("RANK_STATE_STEP_MISMATCH")
        digest = row["state_sha256"]
        if type(digest) is not str or SHA64.fullmatch(digest) is None:
            raise Hold("RANK_STATE_HASH_INVALID")
    if seen != set(range(saved)):
        raise Hold("RANK_STATE_INCOMPLETE")


def witness(raw: bytes, *, now: datetime) -> dict:
    """Classify a declared restart as HOLD or owner-review candidate, not admission.

    Exact replay-step arithmetic is conditional on a trustworthy same-run
    trainer step and pinned full-state checkpoint. Such trust is NOT established
    by this function or by a hash string in the input.
    """
    result = {
        "schema": "szl.restart-lineage-witness/v1",
        "state": "HOLD",
        "reason": "INPUT_UNOBSERVED",
        "replay_steps_conditional": None,
        "replay_fraction_conditional": None,
        "resume_source_authenticated": False,
        "checkpoint_bytes_verified_here": False,
        "backup_verified_here": False,
        "owner_generation_authenticated": False,
        "gpu_training_authorized": False,
        "controller_transition_authorized": False,
        "model_completion_qualified": False,
        "topology_bytes_verified_here": False,
        "dataloader_resume_verified_here": False,
    }
    try:
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES:
            raise Hold("INPUT_BYTE_BUDGET")
        data = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_bad_constant
        )
        if type(data) is not dict:
            raise Hold("INPUT_SCHEMA_MISMATCH")
        schema = data.get("schema")
        if schema == SCHEMA:
            fields = INPUT_FIELDS
        elif schema == SCHEMA_V2:
            fields = (*INPUT_FIELDS, "resume_topology")
        else:
            raise Hold("INPUT_SCHEMA_MISMATCH")
        _object(data, fields, "INPUT_SCHEMA_MISMATCH")
        pins = _pins(data["pins"])
        target = _step(data["target_steps"])
        if target < 1:
            raise Hold("TARGET_STEPS_INVALID")
        prior = _step(data["prior_step"], allow_none=True)
        stamp = _time(data["observed_at_utc"])
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise Hold("CLOCK_UNAVAILABLE")
        age = (now.astimezone(timezone.utc) - stamp).total_seconds()
        if not 0 <= age <= MAX_AGE_SECONDS:
            raise Hold("OBSERVATION_STALE_OR_CLOCK_SKEW")
        if prior is None:
            raise Hold("PRIOR_TRAINER_STEP_UNOBSERVED")
        if prior > target:
            raise Hold("PRIOR_STEP_EXCEEDS_TARGET")
        if type(data["intent"]) is not str or data["intent"] not in ("new", "resume"):
            raise Hold("RESTART_INTENT_INVALID")
        if data["intent"] == "new":
            if prior:
                # Intentional conditional rework diagnostic, not approval.
                result["replay_steps_conditional"] = prior
                result["replay_fraction_conditional"] = {
                    "numerator": prior,
                    "denominator": target,
                }
                raise Hold("GENESIS_REPLAYS_RECORDED_WORK")
            if data["checkpoint"] is not None:
                raise Hold("NEW_RUN_WITH_EXISTING_CHECKPOINT")
            if schema == SCHEMA_V2 and data["resume_topology"] is not None:
                raise Hold("NEW_RUN_WITH_TOPOLOGY")
            # Do not calculate even a zero replay estimate for a malformed
            # new-run request with a stale checkpoint or topology declaration.
            result["replay_steps_conditional"] = 0
            result["replay_fraction_conditional"] = {
                "numerator": 0,
                "denominator": target,
            }
            result.update(
                state="REVIEW_CANDIDATE_ONLY",
                reason="FRESH_NO_PRIOR_STEPS_UNATTESTED",
            )
            return result
        ckpt = data["checkpoint"]
        if ckpt is None:
            result["replay_steps_conditional"] = prior
            result["replay_fraction_conditional"] = {
                "numerator": prior,
                "denominator": target,
            }
            raise Hold("RESUME_WITHOUT_CHECKPOINT")
        _object(
            ckpt,
            ("step", "pins", "commit", "manifest_sha256", "state_files"),
            "CHECKPOINT_SCHEMA_MISMATCH",
        )
        cstep = _step(ckpt["step"])
        if cstep > prior:
            raise Hold("CHECKPOINT_AHEAD_OF_TRAINER")
        if cstep > target:
            raise Hold("CHECKPOINT_AHEAD_OF_TARGET")
        result["replay_steps_conditional"] = prior - cstep
        result["replay_fraction_conditional"] = {
            "numerator": prior - cstep,
            "denominator": target,
        }
        if _pins(ckpt["pins"]) != pins:
            raise Hold("CHECKPOINT_RUN_IDENTITY_MISMATCH")
        if type(ckpt["commit"]) is not str or SHA40.fullmatch(ckpt["commit"]) is None:
            raise Hold("CHECKPOINT_REVISION_UNPINNED")
        if (
            type(ckpt["manifest_sha256"]) is not str
            or SHA64.fullmatch(ckpt["manifest_sha256"]) is None
        ):
            raise Hold("CHECKPOINT_MANIFEST_UNPINNED")
        _object(ckpt["state_files"], STATE_FILES, "STATE_COMPONENT_COVERAGE_MISSING")
        if any(ckpt["state_files"][k] is not True for k in STATE_FILES):
            raise Hold("INCOMPLETE_RESUME_STATE")
        if schema == SCHEMA:
            # Historical v1 did not bind DP topology or rank-local cursor state.
            # Keep old JSON parseable, but do not call its resume reviewable.
            raise Hold("TOPOLOGY_EVIDENCE_REQUIRED")
        _topology(data["resume_topology"], checkpoint_step=cstep)
        # All source assertions remain self-reported. An owner must verify bytes,
        # signed workload and generation, budget, and independent backup.
        result.update(
            state="REVIEW_CANDIDATE_ONLY",
            reason="FULL_STATE_DECLARED_UNATTESTED",
        )
    except Hold as exc:
        result["reason"] = str(exc)
    except (
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
        OverflowError,
    ):
        result["reason"] = "INVALID_JSON"
    return result
