"""Offline, non-authorizing review of evidence before a training group scale-down.

No credentials, cloud SDK, file access, subprocesses, deserialization, or writes.
The caller must independently establish the source of each supplied observation.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

SCHEMA = "szl.training-stop-preflight/v1"
_SHA = re.compile(r"[a-f0-9]{40}\Z")
_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
_GROUP = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
COUNTS = ("allocating_count", "creating_count", "running_count", "stopping_count")
MAX_FRESH_AGE = timedelta(minutes=5)
MAX_FUTURE_SKEW = timedelta(seconds=30)


def _hold(reason: str, *, access_stop_required: bool = False) -> dict:
    """Return a reason code only, never a private model/group identity."""
    return {
        "schema": SCHEMA,
        "state": "HOLD",
        "reason": reason,
        "access_stop_required": access_stop_required,
        "ready_for_owner_review": False,
        "shutdown_authorized": False,
        "training_complete_verified": False,
        "provider_mutation_performed": False,
    }


def _utc(value: object) -> datetime | None:
    # Canonical UTC timestamps prevent naive/clock-unspecified comparisons.
    if type(value) is not str or not re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?Z", value
    ):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _sha(value: object) -> bool:
    return type(value) is str and bool(_SHA.fullmatch(value))


def _digest(value: object) -> bool:
    return type(value) is str and bool(_DIGEST.fullmatch(value))


def _fresh(value: object, now: datetime) -> bool:
    t = _utc(value)
    return t is not None and -MAX_FUTURE_SKEW <= now - t <= MAX_FRESH_AGE


def assess_stop_preflight(*, intent: object, evidence: object, now_utc: str) -> dict:
    """Classify *caller-supplied* pre-stop evidence; never authorize a PATCH.

    A positive result is only READY_FOR_OWNER_REVIEW. An authenticated 200,
    model filename, successful PATCH, complete target count, deadline or a
    boolean supplied by a caller cannot independently establish training,
    artifact integrity, backup independence, or permission to stop a lane.
    """
    now = _utc(now_utc)
    if now is None:
        return _hold("CLOCK_UNOBSERVED")
    if type(intent) is not dict or type(evidence) is not dict:
        return _hold("INPUT_UNOBSERVED")
    if intent.get("schema") != SCHEMA:
        return _hold("INTENT_SCHEMA_INVALID")
    group = intent.get("group")
    if type(group) is not str or not _GROUP.fullmatch(group):
        return _hold("GROUP_INVALID")
    lease = intent.get("owner_lease_digest")
    generation = intent.get("owner_generation")
    if not _digest(lease) or type(generation) is not int or generation <= 0:
        return _hold("OWNER_IDENTITY_UNVERIFIED")
    if (
        evidence.get("group") != group
        or evidence.get("owner_lease_digest") != lease
        or type(evidence.get("owner_generation")) is not int
        or evidence["owner_generation"] != generation
    ):
        return _hold("OWNER_EPOCH_MISMATCH")
    if not _fresh(evidence.get("observed_at_utc"), now):
        return _hold("EVIDENCE_STALE_OR_CLOCK_UNCERTAIN")

    http = evidence.get("model_metadata_http_status")
    if type(http) is not int:
        return _hold("MODEL_METADATA_STATUS_UNOBSERVED")
    if http in (401, 403, 429):
        return _hold(f"MODEL_METADATA_HTTP_{http}", access_stop_required=True)
    if http != 200:
        # Explicitly excludes 404-as-proof, time-based stops and transport errors.
        return _hold("MODEL_METADATA_NOT_VERIFIED")

    for field in (
        "source_revision",
        "base_revision",
        "dataset_revision",
        "model_revision",
    ):
        expected = intent.get(field)
        if not _sha(expected) or evidence.get(field) != expected:
            return _hold("IMMUTABLE_BINDING_UNVERIFIED")
    target = intent.get("target_steps")
    if type(target) is not int or target <= 0:
        return _hold("TARGET_INVALID")
    trainer = evidence.get("trainer_state")
    if type(trainer) is not dict:
        return _hold("TRAINER_STATE_UNVERIFIED")
    if (
        trainer.get("revision") != intent["model_revision"]
        or trainer.get("source_revision") != intent["source_revision"]
        or type(trainer.get("global_step")) is not int
        or type(trainer.get("max_steps")) is not int
        or trainer["global_step"] != target
        or trainer["max_steps"] != target
        or trainer.get("bytes_verified") is not True
        or not _digest(trainer.get("trainer_state_sha256"))
    ):
        return _hold("TRAINER_STATE_NOT_TERMINAL_AND_PINNED")

    artifact = evidence.get("artifact")
    if type(artifact) is not dict:
        return _hold("FINAL_ARTIFACT_UNVERIFIED")
    if (
        artifact.get("revision") != intent["model_revision"]
        or artifact.get("bytes_verified") is not True
        or artifact.get("complete_config_and_tokenizer") is not True
        or not _digest(artifact.get("adapter_sha256"))
        or not _digest(artifact.get("manifest_sha256"))
    ):
        return _hold("FINAL_ARTIFACT_UNVERIFIED")
    backup = evidence.get("backup")
    if type(backup) is not dict or (
        backup.get("independent_readback") is not True
        or backup.get("adapter_sha256") != artifact["adapter_sha256"]
        or not _digest(backup.get("manifest_sha256"))
        or not _fresh(backup.get("verified_at_utc"), now)
    ):
        return _hold("INDEPENDENT_BACKUP_UNVERIFIED")

    provider = evidence.get("provider")
    if type(provider) is not dict:
        return _hold("PROVIDER_READBACK_UNOBSERVED")
    if not _fresh(provider.get("observed_at_utc"), now):
        return _hold("PROVIDER_READBACK_STALE")
    if provider.get("group") != group:
        return _hold("PROVIDER_GROUP_MISMATCH")
    desired = provider.get("desired_replicas")
    if type(desired) is not int or desired < 0:
        return _hold("PROVIDER_REPLICA_COUNT_INVALID")
    if type(provider.get("instance_counts")) is not dict:
        return _hold("PROVIDER_INSTANCE_COUNTS_UNOBSERVED")
    if any(
        type(provider["instance_counts"].get(k)) is not int
        or provider["instance_counts"][k] < 0
        for k in COUNTS
    ):
        return _hold("PROVIDER_INSTANCE_COUNTS_INVALID")
    if provider.get("update_pending") is not False:
        return _hold("PROVIDER_PENDING_CHANGE_UNCERTAIN")
    status = provider.get("status")
    if type(status) is not str or status not in {
        "pending",
        "deploying",
        "running",
        "stopping",
        "stopped",
        "failed",
    }:
        return _hold("PROVIDER_STATUS_UNVERIFIED")
    if status == "stopped" and any(
        provider["instance_counts"][key] for key in COUNTS
    ):
        return _hold("PROVIDER_STOPPED_COUNTS_CONFLICT")

    # Group counters and an independently obtained instance listing are
    # different observations. One cannot silently stand in for the other.
    readback = provider.get("instance_readback")
    if type(readback) is not dict:
        return _hold("INSTANCE_READBACK_UNOBSERVED")
    if not _fresh(readback.get("observed_at_utc"), now):
        return _hold("INSTANCE_READBACK_STALE")
    if readback.get("group") != group:
        return _hold("INSTANCE_READBACK_GROUP_MISMATCH")
    first = _utc(provider["observed_at_utc"])
    second = _utc(readback["observed_at_utc"])
    if abs(first - second) > timedelta(seconds=60):
        return _hold("INSTANCE_READBACK_TIME_DIVERGENCE")
    instances = readback.get("instances")
    if type(instances) is not list or len(instances) > 64:
        return _hold("INSTANCE_READBACK_INVALID")
    seen = set()
    actual = {key: 0 for key in COUNTS}
    for instance in instances:
        if type(instance) is not dict:
            return _hold("INSTANCE_RECORD_INVALID")
        instance_id = instance.get("instance_id")
        if (
            type(instance_id) is not str
            or not 0 < len(instance_id) <= 128
            or instance_id in seen
        ):
            return _hold("INSTANCE_IDENTITY_UNVERIFIED")
        seen.add(instance_id)
        state = instance.get("state")
        if state == "downloading":
            # There is no downloading_count in the group summary.
            return _hold("INSTANCE_COUNTS_NOT_COMPARABLE_DOWNLOADING")
        count_key = state + "_count" if type(state) is str else None
        if count_key not in actual:
            return _hold("INSTANCE_STATE_UNRECOGNIZED")
        actual[count_key] += 1
    if actual != provider["instance_counts"]:
        return _hold("INSTANCE_AND_GROUP_COUNTS_DIFFER")
    if desired == 0 and not instances:
        return _hold("GROUP_ALREADY_DRAINED_STOP_NOT_REQUIRED")

    # This is not an authorization token. It reports an inspectable candidate,
    # not independently authenticated receipt provenance or owner approval.
    result = _hold("OWNER_AND_GUARD_ADMISSION_REQUIRED")
    result["state"] = "READY_FOR_OWNER_REVIEW"
    result["ready_for_owner_review"] = True
    return result
