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
            and re.search(r"flagship|frontier|abst|grpo|distill|salad", x, re.I)
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
