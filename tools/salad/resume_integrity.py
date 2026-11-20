"""Immutable, opaque-byte checkpoint readback for the existing Salad worker.

An HF repository listing, a readable trainer_state.json, or a checkpoint
filename alone cannot authorize resumption. No model code or pickle is loaded,
no cloud group changes, and no local checkpoint is deleted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import time

from tools.salad.training_snapshot import (
    EvidenceHold,
    model_snapshot,
    verified_trainer_state,
)

REQUIRED = (
    "trainer_state.json",
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
)
MAX_FILES = 512
MAX_TOTAL_BYTES = 256 * (1024**3)
MAX_TRAINER_BYTES = 2_000_000


class ResumeHold(ValueError):
    """Constant reason codes only, never provider exception bodies or IDs."""


def _status(error):
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status if type(status) is int else None


def inspect_pinned_checkpoint(api, repo_id, *, expected_steps=None):
    """Get one pinned metadata revision; require complete resume candidates."""
    if expected_steps is not None and (
        type(expected_steps) is not int or expected_steps < 1
    ):
        raise ResumeHold("TARGET_INVALID")
    try:
        _, snapshot, entries = model_snapshot(api, repo_id)
    except EvidenceHold:
        raise ResumeHold("REPOSITORY_METADATA_INVALID") from None
    except Exception as error:
        if _status(error) in (401, 403, 429):
            raise ResumeHold("REPOSITORY_ACCESS_STOP") from None
        raise ResumeHold("REPOSITORY_METADATA_UNOBSERVED") from None
    if snapshot["private"] is not True:
        raise ResumeHold("WORKING_REPOSITORY_NOT_PRIVATE")
    step = snapshot["latest_checkpoint"]
    if step is None:
        return None
    prefix = f"checkpoint-{step}/"
    selected = {name: obj for name, obj in entries.items() if name.startswith(prefix)}
    if not 0 < len(selected) <= MAX_FILES:
        raise ResumeHold("CHECKPOINT_FILE_BUDGET")
    names = set(selected)
    # This resume layout accepts files directly inside checkpoint-N only.
    # Reject nested Hub metadata before broad snapshot_download can fetch it;
    # post-download local coverage checks remain defense in depth.
    if any("/" in name[len(prefix):] for name in selected):
        raise ResumeHold("CHECKPOINT_UNEXPECTED_NESTING")
    if any(prefix + item not in names for item in REQUIRED):
        raise ResumeHold("CHECKPOINT_REQUIRED_STATE_MISSING")
    if not any(
        name.startswith(prefix + "rng_state") and name.endswith(".pth")
        for name in names
    ):
        raise ResumeHold("CHECKPOINT_RNG_STATE_MISSING")
    total = 0
    for obj in selected.values():
        size = getattr(obj, "size", None)
        if type(size) is not int or size < 1:
            raise ResumeHold("CHECKPOINT_FILE_SIZE_UNVERIFIED")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ResumeHold("CHECKPOINT_TOTAL_BYTE_BUDGET")
        lfs = getattr(obj, "lfs", None)
        if isinstance(lfs, dict):
            sha = lfs.get("sha256")
        else:
            sha = getattr(lfs, "sha256", None)
        blob = getattr(obj, "blob_id", None)
        if sha:
            if type(sha) is not str or re.fullmatch(r"[a-f0-9]{64}", sha) is None:
                raise ResumeHold("CHECKPOINT_OBJECT_ID_UNVERIFIED")
        elif type(blob) is not str or re.fullmatch(r"[a-f0-9]{40}", blob) is None:
            raise ResumeHold("CHECKPOINT_OBJECT_ID_UNVERIFIED")
    return {
        "checkpoint": prefix[:-1],
        "revision": snapshot["revision"],
        "step": step,
        "files": selected,
        "total_bytes": total,
        "expected_steps": expected_steps,
    }


def _verify_object(path: Path, entry):
    """Hash opaque bytes against immutable Hub metadata; never deserialize."""
    if path.is_symlink() or not path.is_file():
        raise ResumeHold("CHECKPOINT_FILE_MISSING_OR_SYMLINK")
    size = getattr(entry, "size", None)
    try:
        observed_size = path.stat().st_size
    except OSError:
        raise ResumeHold("CHECKPOINT_LOCAL_IO_UNAVAILABLE") from None
    if type(size) is not int or observed_size != size:
        raise ResumeHold("CHECKPOINT_LOCAL_SIZE_MISMATCH")
    lfs = getattr(entry, "lfs", None)
    oid = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if oid:
        digest = hashlib.sha256()
        expected = oid
    else:
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {size}\0".encode("ascii"))
        expected = getattr(entry, "blob_id", None)
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        if digest.hexdigest() != expected or path.stat().st_size != size:
            raise ResumeHold("CHECKPOINT_OBJECT_HASH_MISMATCH")
    except OSError:
        raise ResumeHold("CHECKPOINT_LOCAL_IO_UNAVAILABLE") from None


def verified_resume_checkpoint(api, repo_id, output_dir, *, expected_steps):
    """Read an immutable complete checkpoint into revision-scoped local storage.

    No fallback to moving main, partial local files, a different repo or genesis
    after a metadata/readback failure. Does not claim independent backup,
    model evaluation, or authorization to restart the provider.
    """
    plan = inspect_pinned_checkpoint(api, repo_id, expected_steps=expected_steps)
    if plan is None:
        return None
    out = Path(output_dir)
    if out.is_symlink():
        raise ResumeHold("CHECKPOINT_OUTPUT_SYMLINK_FORBIDDEN")
    try:
        out.mkdir(parents=True, exist_ok=True)
        cache = out / ".szl-verified-resume"
        destination = cache / plan["revision"]
        for item in (cache, destination):
            if item.is_symlink():
                raise ResumeHold("CHECKPOINT_STAGING_SYMLINK_FORBIDDEN")
            item.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ResumeHold("CHECKPOINT_STAGING_IO_UNAVAILABLE") from None
    # Retry only bounded, idempotent reads of the SAME immutable revision.
    # Never retry a 401/403/429, a 404, or any uncertain provider mutation.
    for attempt in range(3):
        try:
            api.snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                revision=plan["revision"],
                allow_patterns=[plan["checkpoint"] + "/*"],
                local_dir=str(destination),
            )
        except Exception as error:
            code = _status(error)
            if code in (401, 403, 429):
                raise ResumeHold("CHECKPOINT_ACCESS_STOP") from None
            transient = code in (500, 502, 503, 504) or (
                code is None and isinstance(error, (TimeoutError, ConnectionError))
            )
            if not transient or attempt == 2:
                raise ResumeHold("CHECKPOINT_DOWNLOAD_UNVERIFIED") from None
            time.sleep(2 * (attempt + 1))
        else:
            break
    root = destination / plan["checkpoint"]
    if root.is_symlink() or not root.is_dir():
        raise ResumeHold("CHECKPOINT_DIRECTORY_UNVERIFIED")
    allowed = set(plan["files"])
    observed = set()
    for name in sorted(plan["files"]):
        relative = name[len(plan["checkpoint"]) + 1 :]
        if "/" in relative or not relative:
            raise ResumeHold("CHECKPOINT_UNEXPECTED_NESTING")
        local = root / relative
        _verify_object(local, plan["files"][name])
        observed.add(name)
    try:
        for local in root.iterdir():
            if local.is_symlink() or not local.is_file():
                raise ResumeHold("CHECKPOINT_UNEXPECTED_LOCAL_ENTRY")
            if plan["checkpoint"] + "/" + local.name not in allowed:
                raise ResumeHold("CHECKPOINT_UNPINNED_LOCAL_FILE")
    except OSError:
        raise ResumeHold("CHECKPOINT_STAGING_IO_UNAVAILABLE") from None
    if observed != allowed:
        raise ResumeHold("CHECKPOINT_COVERAGE_INCOMPLETE")
    state_name = plan["checkpoint"] + "/trainer_state.json"
    try:
        state = verified_trainer_state(
            (root / "trainer_state.json").read_bytes(),
            plan["files"][state_name],
            plan["step"],
            max_bytes=MAX_TRAINER_BYTES,
        )
    except EvidenceHold:
        raise ResumeHold("CHECKPOINT_TRAINER_STATE_UNVERIFIED") from None
    except OSError:
        raise ResumeHold("CHECKPOINT_LOCAL_IO_UNAVAILABLE") from None
    if state["max_steps"] != expected_steps:
        raise ResumeHold("CHECKPOINT_TARGET_CHANGED")
    # A separate owner/lease and same-run command/dataset binding is still
    # required before provider dispatch; this function only verifies bytes.
    return str(root)
