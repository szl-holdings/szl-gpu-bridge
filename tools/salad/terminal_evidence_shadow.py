"""Read-only, fail-closed interpretation of reported Salad terminal events.

No provider HTTP, credentials, controller, receipt signing, start/stop/scale,
training inference, or release authority. Every claim is untrusted input.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SCHEMA = "szl.salad-terminal-shadow/v1"
RESULT = "szl.salad-terminal-shadow-result/v1"
MAX_BYTES = 65536
SOURCES = {"EDGE_UI", "APPROVED_SDK", "HISTORICAL_RECEIPT"}
API = {
    "DENIED_401",
    "DENIED_403",
    "RATE_429",
    "AUTHORIZED_READ",
    "NOT_QUERIED",
    "UNOBSERVED",
}
GROUPS = {"RUNNING", "STOPPED", "DEPLOYING", "ALLOCATING", "FAILED", "UNKNOWN"}
COUNT_KEYS = {"allocating", "creating", "running", "stopping"}
EVENTS = {
    "EXIT_0",
    "EXIT_NONZERO",
    "ALLOCATED",
    "STARTING",
    "INSTANCE_RUNNING",
    "PROBE_READY",
    "STOPPING",
}
MARKERS = {
    "ALREADY_DONE",
    "FRONTIER_EXIT",
    "FINAL_FILENAME",
    "CHECKPOINT_FILENAME",
    "STEP_PROGRESS_REPORTED",
}
CLAIMS = {
    "source_pinned",
    "steps_verified",
    "final_hash_verified",
    "backup_readback",
    "signer_verified",
    "fence_verified",
}
REQUIRED = {
    "schema",
    "surface",
    "api_state",
    "group_status",
    "desired_replicas",
    "actual_counts",
    "instance_get_confirmed",
    "events",
    "trainer_markers",
    "claimed_evidence",
}


class ShadowHold(ValueError):
    """An observation cannot be safely interpreted."""


def _keys(row, fields):
    if type(row) is not dict or set(row) != fields:
        raise ShadowHold("INVALID_SCHEMA_FIELDS")


def _int(row, maximum=250):
    return type(row) is int and 0 <= row <= maximum


def _tags(value, allowed, maximum):
    if type(value) is not list or len(value) > maximum:
        raise ShadowHold("INVALID_MARKER_COLLECTION")
    if any(type(t) is not str or t not in allowed for t in value):
        raise ShadowHold("UNKNOWN_MARKER")


def evaluate(observation):
    """Deterministic risk triage, never an authenticated provider assertion."""
    _keys(observation, REQUIRED)
    if observation["schema"] != SCHEMA:
        raise ShadowHold("BAD_SCHEMA")
    if type(observation["surface"]) is not str or observation["surface"] not in SOURCES:
        raise ShadowHold("UNKNOWN_SURFACE")
    if type(observation["api_state"]) is not str or observation["api_state"] not in API:
        raise ShadowHold("UNKNOWN_API_STATE")
    if (
        type(observation["group_status"]) is not str
        or observation["group_status"] not in GROUPS
    ):
        raise ShadowHold("UNKNOWN_GROUP_STATUS")
    desired = observation["desired_replicas"]
    if desired is not None and not _int(desired):
        raise ShadowHold("INVALID_DESIRED")
    actual = observation["actual_counts"]
    if actual is not None:
        _keys(actual, COUNT_KEYS)
        if any(not _int(actual[key]) for key in COUNT_KEYS):
            raise ShadowHold("INVALID_ACTUAL_COUNTS")
    if type(observation["instance_get_confirmed"]) is not bool:
        raise ShadowHold("INVALID_READBACK_CLAIM")
    if observation["instance_get_confirmed"] and (
        observation["surface"] != "APPROVED_SDK"
        or observation["api_state"] != "AUTHORIZED_READ"
        or desired is None
        or actual is None
    ):
        raise ShadowHold("CONTRADICTORY_AUTH_CONTEXT")
    _tags(observation["events"], EVENTS, 64)
    _tags(observation["trainer_markers"], MARKERS, 32)
    claims = observation["claimed_evidence"]
    _keys(claims, CLAIMS)
    if any(type(claims[key]) is not bool for key in CLAIMS):
        raise ShadowHold("INVALID_EXTERNAL_CLAIM")

    counts = Counter(observation["events"])
    tags = set(observation["trainer_markers"])
    reasons = set()
    if observation["api_state"] in {"DENIED_401", "DENIED_403", "RATE_429"}:
        reasons.add("APPROVED_API_ACCESS_DENIED_OR_RATE_LIMITED")
    if observation["surface"] != "APPROVED_SDK":
        reasons.add("UI_OR_HISTORICAL_SOURCE_NOT_APPROVED_API_READBACK")
    if not observation["instance_get_confirmed"]:
        reasons.add("INSTANCE_GET_NOT_RECONCILED")
    if desired is None or actual is None:
        reasons.add("REPLICA_COUNTS_PARTIALLY_UNOBSERVED")
    elif (
        observation["group_status"] == "RUNNING"
        and desired > 0
        and actual["running"] == 0
    ):
        reasons.add("RUNNING_LABEL_ZERO_ACTUAL_REPORTED")
    if actual is not None and observation["group_status"] == "STOPPED" and actual["running"] > 0:
        # Both fields are untrusted: the contradiction requires reconciliation,
        # not an assertion that the provider is truly running or stopped.
        reasons.add("STOPPED_LABEL_RUNNING_ACTUAL_REPORTED")
    if actual is not None and desired is not None and sum(actual.values()) > desired:
        reasons.add("NONATOMIC_OR_CONFLICTING_INSTANCE_COUNTS")
    if "ALREADY_DONE" in tags or "FRONTIER_EXIT" in tags:
        reasons.add("TERMINAL_TEXT_NOT_TRAINING_COMPLETION")
    if {"FINAL_FILENAME", "CHECKPOINT_FILENAME"} & tags:
        reasons.add("FILENAME_NOT_ARTIFACT_INTEGRITY")
    if counts["EXIT_0"] >= 2 and desired is not None and desired > 0:
        reasons.add("REPEATED_EXIT0_SUSPECTED")
    if {"PROBE_READY", "INSTANCE_RUNNING"} & set(observation["events"]):
        reasons.add("PROVIDER_PROBE_NOT_TRAINER_PROGRESS")
    if any(claims.values()):
        reasons.add("SELF_REPORTED_EVIDENCE_NOT_INDEPENDENTLY_VERIFIED")

    if "REPEATED_EXIT0_SUSPECTED" in reasons:
        risk = "RESTART_LOOP_SUSPECTED_NOT_PROVEN"
    elif "STOPPED_LABEL_RUNNING_ACTUAL_REPORTED" in reasons:
        risk = "REPORTED_STATUS_INSTANCE_CONFLICT"
    elif "RUNNING_LABEL_ZERO_ACTUAL_REPORTED" in reasons:
        risk = "REPORTED_REPLICA_GAP"
    else:
        risk = "TRAINING_EVIDENCE_UNVERIFIED"
    return {
        "schema": RESULT,
        "decision": "HOLD_NO_PROVIDER_ACTION",
        "risk": risk,
        "reason_codes": sorted(reasons),
        "reported_exit0_events": counts["EXIT_0"],
        "observation_scope": "UNTRUSTED_CATEGORICAL_INPUT_ONLY",
        "provider_liveness_verified": False,
        "training_steps_verified": False,
        "final_artifact_verified": False,
        "independent_backup_verified": False,
        "training_completed": False,
        "model_evaluated": False,
        "release_qualified": False,
        "start_stop_scale_authorized": False,
        "flagship54_shutdown_authorized": False,
        "training_dispatch_authorized": False,
        "next_gate": "EXISTING_OWNER_RECONCILE_AUTH_LEASE_STEPS_ARTIFACTS_BACKUPS_COST",
    }


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ShadowHold("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ShadowHold("NONFINITE_NUMBER")


def _read_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ShadowHold("INPUT_PATH_UNAVAILABLE")
    with path.open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES:
        raise ShadowHold("INPUT_SIZE_INVALID")
    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_unique,
            parse_constant=_nonfinite,
        )
    except ShadowHold:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ShadowHold("INVALID_JSON") from exc
    return evaluate(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("offline_input_json")
    args = parser.parse_args(argv)
    try:
        result = _read_file(args.offline_input_json)
    except (ShadowHold, OSError):
        result = {
            "schema": RESULT,
            "decision": "HOLD_NO_PROVIDER_ACTION",
            "risk": "INVALID_INPUT",
            "training_completed": False,
            "start_stop_scale_authorized": False,
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 78


if __name__ == "__main__":
    raise SystemExit(main())
