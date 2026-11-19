"""Offline, fail-closed witness for one SaladCloud group transition.

This module neither sends HTTP requests nor changes controller ownership. A successful
curl exit code is NOT a successful provider response. Even a coherent provider
readback is NOT trainer liveness, artifact integrity, evaluation or launch authority.
"""

from __future__ import annotations

import json
import re

SCHEMA = "szl.salad-transition-witness/v1"
COUNTS = ("allocating_count", "creating_count", "running_count", "stopping_count")
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
MAX_BODY = 16384
MAX_INSTANCES = 64


def _out(code: str, *, acknowledged: bool = False) -> dict:
    return {
        "schema": SCHEMA,
        "state": code,
        "provider_write_acknowledged": acknowledged,
        "access_stop_required": code in {"ACCESS_DENIED_401", "ACCESS_DENIED_403", "RATE_LIMITED_429"},
        "group_state_reconciled": False,
        "trainer_process_verified": False,
        "checkpoint_or_final_verified": False,
        "quality_evaluated": False,
        "training_launch_authorized": False,
        "shutdown_authorized": False,
        "provider_mutation_performed": False,
    }


def _unique_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("DUPLICATE_KEY")
        obj[key] = value
    return obj


def _decode_body(body: bytes) -> dict | None:
    if type(body) is not bytes or len(body) > MAX_BODY:
        raise ValueError("BODY_BYTE_BUDGET")
    if not body.strip():
        return None
    val = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_pairs,
                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE")))
    if type(val) is not dict:
        raise ValueError("BODY_NOT_OBJECT")
    return val


def witness(*, transport_exit: int, http_status: int | None, body: bytes,
            group: object, instances: object, name: str,
            requested_replicas: int) -> dict:
    """Return an unsigned, non-authorizing classification of supplied snapshots.

    A caller must independently obtain the HTTP status and readback, using the
    existing credential source and owner. Inputs are not authenticated here.
    """
    if type(transport_exit) is not int or transport_exit != 0:
        return _out("TRANSPORT_FAILED_OR_UNKNOWN")
    if type(http_status) is not int or not 100 <= http_status <= 599:
        return _out("HTTP_STATUS_UNOBSERVED")
    if http_status in (401, 403):
        return _out(f"ACCESS_DENIED_{http_status}")
    if http_status == 429:
        return _out("RATE_LIMITED_429")
    if http_status not in (200, 202, 204):
        return _out("PROVIDER_NON_SUCCESS")
    try:
        response = _decode_body(body)
    except (ValueError, UnicodeError, json.JSONDecodeError):
        return _out("PROVIDER_BODY_INVALID")
    if response is not None:
        embedded_status = response.get("status")
        if type(embedded_status) is int and embedded_status in (401, 403, 429):
            return _out("RATE_LIMITED_429" if embedded_status == 429 else f"ACCESS_DENIED_{embedded_status}")
        problem_type = response.get("type")
        if (
            (type(embedded_status) is int and embedded_status >= 400)
            or (type(problem_type) is str and "pending_update_in_progress" in problem_type)
            or response.get("title") == "Bad Request"
        ):
            return _out("PROVIDER_ERROR_ENVELOPE")
    ack = True
    if type(name) is not str or not NAME.fullmatch(name):
        return _out("GROUP_ID_INVALID", acknowledged=ack)
    if type(requested_replicas) is not int or not 0 <= requested_replicas <= 250:
        return _out("REPLICA_INTENT_INVALID", acknowledged=ack)
    if type(group) is not dict or type(instances) is not dict:
        return _out("READBACK_UNOBSERVED", acknowledged=ack)
    if group.get("name") != name:
        return _out("READBACK_GROUP_MISMATCH", acknowledged=ack)
    if type(group.get("replicas")) is not int or group["replicas"] != requested_replicas:
        return _out("READBACK_REPLICA_MISMATCH", acknowledged=ack)
    if group.get("pending_change") is True:
        return _out("PROVIDER_UPDATE_PENDING", acknowledged=ack)
    state = group.get("current_state")
    if type(state) is not dict or type(state.get("instance_status_counts")) is not dict:
        return _out("READBACK_COUNTS_UNAVAILABLE", acknowledged=ack)
    counts = state["instance_status_counts"]
    if any(type(counts.get(k)) is not int or counts[k] < 0 for k in COUNTS):
        return _out("READBACK_COUNTS_INVALID", acknowledged=ack)
    items = instances.get("instances")
    if type(items) is not list or len(items) > MAX_INSTANCES:
        return _out("INSTANCE_LIST_UNAVAILABLE", acknowledged=ack)
    if any(type(i) is not dict for i in items):
        return _out("INSTANCE_LIST_INVALID", acknowledged=ack)
    raw_status = state.get("status")
    if type(raw_status) is not str or raw_status not in {
        "pending", "deploying", "running", "stopping", "stopped", "failed"
    }:
        return _out("READBACK_STATUS_INVALID", acknowledged=ack)
    observed = _out("PROVIDER_STATE_OBSERVED_NOT_TRAINING", acknowledged=ack)
    observed["group_state_reconciled"] = True
    observed["provider_status"] = raw_status
    observed["desired_replicas"] = requested_replicas
    observed["actual_counts"] = {k: counts[k] for k in COUNTS}
    observed["instances_returned"] = len(items)
    if requested_replicas == 0 and all(counts[k] == 0 for k in COUNTS) and not items:
        observed["state"] = "ZERO_REPLICAS_PROVIDER_REPORTED_NOT_COMPLETION"
    elif requested_replicas > 0 and counts["running_count"] == 0:
        observed["state"] = "NO_RUNNING_INSTANCE_PROVIDER_REPORTED"
    return observed
