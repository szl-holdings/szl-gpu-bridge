"""Fail-closed, offline lineage witness for preemptible training restarts.

All input is caller-provided and unauthenticated by this module. The witness never
permits GPU dispatch, a controller transition, checkpoint deletion, or promotion.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re

SCHEMA = "szl.restart-lineage-input/v1"
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
    }
    try:
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES:
            raise Hold("INPUT_BYTE_BUDGET")
        data = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_bad_constant
        )
        _object(
            data,
            (
                "schema",
                "pins",
                "target_steps",
                "prior_step",
                "observed_at_utc",
                "intent",
                "checkpoint",
            ),
            "INPUT_SCHEMA_MISMATCH",
        )
        if data["schema"] != SCHEMA:
            raise Hold("INPUT_SCHEMA_MISMATCH")
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
            result["replay_steps_conditional"] = prior
            result["replay_fraction_conditional"] = {
                "numerator": prior,
                "denominator": target,
            }
            if prior:
                raise Hold("GENESIS_REPLAYS_RECORDED_WORK")
            if data["checkpoint"] is not None:
                raise Hold("NEW_RUN_WITH_EXISTING_CHECKPOINT")
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
        # All source assertions remain self-reported. An owner must verify bytes,
        # signed workload and generation, budget, and independent backup.
        result.update(
            state="REVIEW_CANDIDATE_ONLY",
            reason="FULL_STATE_DECLARED_UNATTESTED",
        )
    except Hold as exc:
        result["reason"] = str(exc)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        result["reason"] = "INVALID_JSON"
    return result
