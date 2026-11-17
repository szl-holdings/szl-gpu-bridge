"""Read-only evidence contracts for training observers; no cloud mutation.

Resolve one model metadata snapshot and derive every filename from that same
response. Callers must fetch trainer JSON at its returned immutable revision.
Metadata presence is never artifact integrity, model quality or trainer liveness.
"""

import hashlib
import json
import math
import re
from pathlib import PurePosixPath


class EvidenceHold(ValueError):
    """Fixed diagnostic code suitable for a secret-free observation."""


def _path(name):
    if not isinstance(name, str) or not name:
        raise EvidenceHold("INVALID_REPOSITORY_PATH")
    p = PurePosixPath(name)
    if (
        p.is_absolute()
        or str(p) != name
        or ".." in p.parts
        or any(c in name for c in "\\\x00")
        or any(x in {"", "."} for x in name.split("/"))
    ):
        raise EvidenceHold("INVALID_REPOSITORY_PATH")
    return p


def model_snapshot(api, repo_id):
    """Exactly one metadata request; do not re-list a moving branch."""
    info = api.model_info(repo_id, files_metadata=True, timeout=12)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision
    ):
        raise EvidenceHold("IMMUTABLE_REVISION_UNAVAILABLE")
    if type(getattr(info, "private", None)) is not bool:
        raise EvidenceHold("VISIBILITY_UNOBSERVED")
    siblings = getattr(info, "siblings", None)
    if siblings is None:
        raise EvidenceHold("FILE_LIST_UNOBSERVED")
    entries = {}
    for entry in siblings:
        name = entry.rfilename
        _path(name)
        if name in entries:
            raise EvidenceHold("DUPLICATE_FILE_METADATA")
        entries[name] = entry
    steps = sorted(
        {
            int(m.group(1))
            for name in entries
            if (m := re.match(r"^checkpoint-(\d+)/", name))
        }
    )
    latest = steps[-1] if steps else None
    record = {
        "id": repo_id,
        "status": "OBSERVED",
        "revision": revision,
        "private": info.private,
        "file_count": len(entries),
        "latest_checkpoint": latest,
        "final_adapter": "final/adapter_model.safetensors" in entries,
        "final_config": "final/adapter_config.json" in entries,
        "final_tokenizer": any(
            x in entries
            for x in ("final/tokenizer.json", "final/tokenizer_config.json")
        ),
        "root_adapter": "adapter_model.safetensors" in entries,
        "last_modified": str(getattr(info, "last_modified", None)),
        "file_list_basis": "same immutable metadata response",
        "artifact_integrity": "NOT_VERIFIED_BY_METADATA",
        "training_complete": "UNVERIFIED",
        "process_liveness_verified": False,
        "evaluation": "NOT_ASSESSED",
    }
    if latest is not None:
        record["latest_trainer_state_present"] = (
            f"checkpoint-{latest}/trainer_state.json" in entries
        )
    return info, record, entries


def verified_trainer_state(raw, entry, expected_step, max_bytes=2_000_000):
    """Verify a small, pinned JSON object; never deserialize pickle."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= max_bytes:
        raise EvidenceHold("TRAINER_JSON_BYTE_BUDGET")
    size = getattr(entry, "size", None)
    if type(size) is not int or size != len(raw):
        raise EvidenceHold("TRAINER_JSON_SIZE_MISMATCH")
    lfs = getattr(entry, "lfs", None)
    oid = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    sha = hashlib.sha256(raw).hexdigest()
    if oid:
        if not re.fullmatch(r"[0-9a-f]{64}", str(oid)) or sha != oid:
            raise EvidenceHold("TRAINER_JSON_HASH_MISMATCH")
    else:
        oid = getattr(entry, "blob_id", None)
        git = hashlib.sha1(b"blob " + str(size).encode() + b"\0" + raw).hexdigest()
        if not re.fullmatch(r"[0-9a-f]{40}", str(oid)) or git != oid:
            raise EvidenceHold("TRAINER_JSON_HASH_MISMATCH")

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise EvidenceHold("DUPLICATE_JSON_KEY")
            value[key] = item
        return value

    def constant(_):
        raise EvidenceHold("NONFINITE_JSON")

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise EvidenceHold("NONFINITE_JSON")
        return parsed

    try:
        state = json.loads(
            raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=number
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceHold("INVALID_TRAINER_JSON") from exc
    if not isinstance(state, dict):
        raise EvidenceHold("TRAINER_STATE_NOT_OBJECT")
    step, target = state.get("global_step"), state.get("max_steps")
    if type(expected_step) is not int or type(step) is not int or step != expected_step:
        raise EvidenceHold("CHECKPOINT_STEP_MISMATCH")
    if type(target) is not int or target <= 0:
        raise EvidenceHold("TRAINING_TARGET_UNOBSERVED")
    result = {
        k: state.get(k)
        for k in ("global_step", "max_steps", "epoch", "num_train_epochs")
    }
    result.update(trainer_state_sha256=sha, integrity="PINNED_OBJECT_BYTES_VERIFIED")
    return result


def select_training_repos(catalog_ids, mapped_ids, limit=80):
    """Mapped command targets first, then research candidates; disclose omissions."""
    if type(limit) is not int or limit < 1:
        raise EvidenceHold("INVALID_SELECTION_BUDGET")
    valid = re.compile(r"^SZLHOLDINGS/[A-Za-z0-9_.-]+$")
    mapped = sorted(
        {x for x in mapped_ids if isinstance(x, str) and valid.fullmatch(x)}
    )
    focused = sorted(
        {
            x
            for x in catalog_ids
            if isinstance(x, str)
            and valid.fullmatch(x)
            and re.search(r"flagship|frontier|abst|absrl|grpo|distill|salad", x, re.I)
        }
    )
    priority = [x for x in mapped if "flagship" in x.lower()]
    priority += [x for x in mapped if x not in priority]
    priority += [x for x in focused if x not in priority]
    selected = priority[:limit]
    return selected, {
        "mapped_count": len(mapped),
        "eligible_count": len(priority),
        "selected_count": len(selected),
        "limit": limit,
        "omitted_ids": priority[limit:],
        "basis": "explicit command targets plus named research candidates; not all estate training",
    }


def catalog_delta(previous_ids, current_ids, *, current_complete):
    old, new = set(previous_ids), set(current_ids)
    return {
        "newly_returned": sorted(new - old),
        "not_returned": sorted(old - new) if current_complete else [],
        "prior_ids_unassessed": sorted(old - new) if not current_complete else [],
        "current_complete": current_complete,
        "interpretation": "Listing delta only; not deletion, permanent loss or model completion",
    }


# Provider metadata is observation, not process liveness or training proof.
SALAD_INSTANCE_COUNT_FIELDS = (
    "allocating_count",
    "creating_count",
    "running_count",
    "stopping_count",
)
SALAD_INSTANCE_STATES = ("allocating", "creating", "running", "stopping")


def salad_group_page_evidence(groups, *, capture_limit=256, listing_complete=False):
    """Safely summarize a bounded page and disclose every omitted group.

    Do not mistake the provider's group state 'running' for a running instance.
    All items are validated before any slice is applied, including omitted ones.
    listing_complete is supplied from the transport's independently checked
    pagination, not inferred from the presence of a group name.
    """
    if type(capture_limit) is not int or not 1 <= capture_limit <= 1000:
        raise EvidenceHold("GROUP_CAPTURE_LIMIT_INVALID")
    if type(listing_complete) is not bool or not isinstance(groups, list):
        raise EvidenceHold("GROUP_LIST_UNOBSERVED")
    if len(groups) > 1000:
        raise EvidenceHold("GROUP_LIST_BOUND_EXCEEDED")
    seen = set()
    rows = []
    for group in groups:
        if not isinstance(group, dict):
            raise EvidenceHold("GROUP_ITEM_NOT_OBJECT")
        name = group.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name
        ):
            raise EvidenceHold("GROUP_NAME_INVALID")
        if name in seen:
            raise EvidenceHold("DUPLICATE_GROUP_NAME")
        seen.add(name)
        state = group.get("current_state")
        if state is not None and not isinstance(state, dict):
            raise EvidenceHold("GROUP_STATE_INVALID")
        state = state or {}
        source_counts = state.get("instance_status_counts")
        if source_counts is not None and not isinstance(source_counts, dict):
            raise EvidenceHold("GROUP_COUNTS_INVALID")
        source_counts = source_counts or {}
        counts = {}
        for field in SALAD_INSTANCE_COUNT_FIELDS:
            value = source_counts.get(field)
            counts[field] = value if type(value) is int and value >= 0 else None
        desired = group.get("replicas")
        if type(desired) is not int or desired < 0:
            desired = None
        rows.append(
            {
                "name": name,
                "group_status": state.get("status")
                if isinstance(state.get("status"), str)
                else None,
                "desired_replicas": desired,
                "counts": counts,
                "process_liveness_verified": False,
            }
        )
    captured, omitted = rows[:capture_limit], rows[capture_limit:]
    active_omitted = [
        row["name"]
        for row in omitted
        if any(isinstance(v, int) and v > 0 for v in row["counts"].values())
    ]
    unknown_omitted = [
        row["name"] for row in omitted if any(v is None for v in row["counts"].values())
    ]
    return {
        "returned_count": len(rows),
        "captured_count": len(captured),
        "capture_limit": capture_limit,
        "captured": captured,
        "omitted_count": len(omitted),
        "omitted_names": [row["name"] for row in omitted],
        "omitted_with_nonzero_provider_counts": active_omitted,
        "omitted_with_unknown_counts": unknown_omitted,
        "listing_complete": listing_complete,
        "coverage_complete": listing_complete and not omitted,
        "interpretation": "Provider group metadata only; no trainer-process proof",
    }


def salad_instance_evidence(payload, *, capture_limit=12):
    """Parse the actual Salad instance 'state' string and older nested shape.

    The state field takes precedence only when it agrees with other present
    representations. Missing or contradictory status is UNOBSERVED, never 0.
    """
    if type(capture_limit) is not int or not 1 <= capture_limit <= 100:
        raise EvidenceHold("INSTANCE_CAPTURE_LIMIT_INVALID")
    if isinstance(payload, list):
        instances, pagination = payload, True
    elif isinstance(payload, dict):
        instances = payload.get("instances", payload.get("items"))
        nxt = payload.get("next") or (payload.get("pagination") or {}).get("next")
        pagination = not bool(nxt)
    else:
        raise EvidenceHold("INSTANCE_LIST_UNOBSERVED")
    if not isinstance(instances, list) or len(instances) > 10000:
        raise EvidenceHold("INSTANCE_LIST_INVALID")
    normalized = []
    mismatched = 0
    for inst in instances:
        if not isinstance(inst, dict):
            raise EvidenceHold("INSTANCE_NOT_OBJECT")
        nested = inst.get("current_state")
        if nested is not None and not isinstance(nested, dict):
            raise EvidenceHold("INSTANCE_NESTED_STATE_INVALID")
        states = [
            inst.get("state"),
            inst.get("status"),
            nested.get("status") if isinstance(nested, dict) else None,
        ]
        known = [s.lower() for s in states if isinstance(s, str) and s]
        if len(set(known)) > 1:
            mismatched += 1
        status = (
            known[0]
            if known and len(set(known)) == 1 and known[0] in SALAD_INSTANCE_STATES
            else "UNOBSERVED"
        )
        normalized.append(status)
    captured = normalized[:capture_limit]
    state_counts = {s: captured.count(s) for s in SALAD_INSTANCE_STATES}
    return {
        "returned_count": len(instances),
        "captured_count": len(captured),
        "capture_limit": capture_limit,
        "captured_statuses": captured,
        "state_counts": state_counts,
        "unobserved_status_count": captured.count("UNOBSERVED"),
        "conflicting_status_fields": mismatched,
        "capture_truncated": len(instances) > capture_limit,
        "listing_complete": pagination,
        "coverage_complete": pagination and len(instances) <= capture_limit,
        "trainer_process_liveness_verified": False,
        "interpretation": "Only provider-reported instance state; not a heartbeat",
    }


def reconcile_salad_group_instance_counts(group, instances):
    """Compare provider group counts with independently returned instance rows."""
    if not isinstance(group, dict) or not isinstance(instances, dict):
        raise EvidenceHold("PROVIDER_COUNT_INPUT_INVALID")
    counts = group.get("counts")
    if not isinstance(counts, dict):
        raise EvidenceHold("GROUP_COUNTS_UNOBSERVED")
    if (
        not instances.get("coverage_complete")
        or instances.get("unobserved_status_count")
        or instances.get("conflicting_status_fields")
    ):
        return "UNOBSERVED"
    if any(
        type(counts.get(name)) is not int or counts[name] < 0
        for name in SALAD_INSTANCE_COUNT_FIELDS
    ):
        return "UNOBSERVED"
    if any(
        counts[field] != instances["state_counts"].get(field.removesuffix("_count"))
        for field in SALAD_INSTANCE_COUNT_FIELDS
    ):
        return "PROVIDER_COUNT_DRIFT"
    return "PROVIDER_COUNTS_AGREE_NOT_TRAINER_LIVENESS"
