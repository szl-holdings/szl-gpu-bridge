#!/usr/bin/env python3
"""Archive a pinned Flagship54 checkpoint directly from the Hub, without Gate-1.

Uses the existing Hugging Face login. Never trains, loads model code/pickle,
changes cloud state, rewrites a conflicting archive, or asserts model quality.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import uuid

REPO = "SZLHOLDINGS/salad-flagship54-v1"
TARGET = 37000
MAX_BYTES = 1024**3
LEAVES = frozenset(
    {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
    }
)


class Hold(RuntimeError):
    """Only fixed diagnostic codes are passed to the caller."""


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def relative(name):
    p = PurePosixPath(name)
    if (
        not name
        or p.is_absolute()
        or str(p) != name
        or any(x in {".", "..", ""} for x in name.split("/"))
        or any(c in name for c in "\\:\x00")
        or any(part.endswith((".", " ")) for part in p.parts)
    ):
        raise Hold("UNSAFE_PATH")
    return p


def no_links(path):
    """Reject symlink/junction ancestors as well as the destination."""
    for p in (path, *path.parents):
        if p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()):
            raise Hold("LINKED_DESTINATION")


def hashes(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
            git.update(block)
    return size, sha.hexdigest(), git.hexdigest()


def expected(entry):
    lfs = getattr(entry, "lfs", None)
    digest = (
        lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    )
    algo = "sha256" if digest else "git-blob-sha1"
    digest = digest or getattr(entry, "blob_id", None)
    if (
        type(entry.size) is not int
        or not 0 < entry.size <= MAX_BYTES
        or not isinstance(digest, str)
        or not re.fullmatch(
            r"[0-9a-f]{64}" if algo == "sha256" else r"[0-9a-f]{40}", digest
        )
    ):
        raise Hold("INVALID_OBJECT_METADATA")
    return algo, digest


def verify(path, entry):
    algo, oid = expected(entry)
    size, sha, git = hashes(path)
    if size != entry.size:
        raise Hold("BYTE_SIZE_MISMATCH")
    if (sha if algo == "sha256" else git) != oid:
        raise Hold("BYTE_HASH_MISMATCH")
    return {"size": size, "sha256": sha, "hub_oid": oid, "hub_oid_algorithm": algo}


def select(info, revision, step):
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise Hold("PIN_REQUIRED")
    if info.sha != revision or info.private is not True:
        raise Hold("REVISION_OR_PRIVATE_VISIBILITY_MISMATCH")
    if type(step) is not int or not 0 < step <= TARGET:
        raise Hold("STEP_OUT_OF_BOUNDS")
    prefix = f"checkpoint-{step}/"
    entries = {x.rfilename: x for x in info.siblings if x.rfilename.startswith(prefix)}
    if set(entries) != {prefix + name for name in LEAVES}:
        raise Hold("CHECKPOINT_FILE_SET_INCOMPLETE_OR_UNEXPECTED")
    for name, entry in entries.items():
        relative(name)
        expected(entry)
    if sum(x.size for x in entries.values()) > MAX_BYTES:
        raise Hold("ARCHIVE_BUDGET_EXCEEDED")
    if entries[prefix + "trainer_state.json"].size > 4 * 1024**2:
        raise Hold("TRAINER_STATE_TOO_LARGE")
    return entries


def validate_state(state, step):
    if type(state.get("global_step")) is not int or state["global_step"] != step:
        raise Hold("STEP_MISMATCH")
    if type(state.get("max_steps")) is not int or state["max_steps"] != TARGET:
        raise Hold("TARGET_MISMATCH")


def atomic_new(path, data):
    """Exclusive output: never silently overwrite an earlier manifest/receipt."""
    no_links(path)
    with path.open("xb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())


def copy_checked(source, target, entry):
    no_links(target)
    measure = verify(source, entry)
    if target.exists():
        if verify(target, entry) != measure:
            raise Hold("EXISTING_ARCHIVE_CONFLICT")
        return measure
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".part")
    try:
        with source.open("rb") as src, temp.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        verify(temp, entry)
        if target.exists():
            raise Hold("DESTINATION_CHANGED")
        temp.rename(target)
        verify(target, entry)
        return measure
    finally:
        if temp.exists():
            temp.unlink()


@contextmanager
def exclusive_lock(path):
    no_links(path)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise Hold("EXISTING_ARCHIVE_WRITER_OR_UNCLEARED_LOCK") from None
    owner = uuid.uuid4().hex
    with os.fdopen(fd, "w") as f:
        f.write(owner)
        f.flush()
        os.fsync(f.fileno())
    try:
        yield
    finally:
        if path.exists() and path.read_text() == owner:
            path.unlink()


def run_archive(api, downloader, root, revision, step, deadline_seconds=240):
    start = time.monotonic()
    root = Path(root)
    if not root.is_absolute():
        raise Hold("ABSOLUTE_DESTINATION_REQUIRED")
    no_links(root)
    root.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise Hold("PIN_REQUIRED")
    with exclusive_lock(root / (revision + ".lock")):
        info = api.model_info(REPO, revision=revision, files_metadata=True, timeout=20)
        entries = select(info, revision, step)
        total = sum(x.size for x in entries.values())
        if shutil.disk_usage(root).free < total * 2 + 1024**3:
            raise Hold("INSUFFICIENT_DISK")
        folder = root / revision
        no_links(folder)
        folder.mkdir(exist_ok=True)
        records = {}
        state_name = f"checkpoint-{step}/trainer_state.json"
        names = [state_name] + sorted(n for n in entries if n != state_name)
        for name in names:
            if time.monotonic() - start > deadline_seconds:
                raise Hold("ARCHIVE_DEADLINE_REACHED")
            target = folder.joinpath(*relative(name).parts)
            no_links(target)
            if target.exists():
                records[name] = verify(target, entries[name])
            else:
                source = Path(
                    downloader(
                        REPO,
                        name,
                        revision=revision,
                        cache_dir=str(root.parent / "hub-cache"),
                        etag_timeout=15,
                    )
                )
                records[name] = copy_checked(source, target, entries[name])
            if name == state_name:
                validate_state(json.loads(target.read_bytes()), step)
            print(
                json.dumps(
                    {
                        "event": "FILE_VERIFIED",
                        "file": name,
                        "bytes": records[name]["size"],
                    }
                ),
                flush=True,
            )
        fresh = api.model_info(REPO, timeout=20)
        if fresh.private is not True:
            raise Hold("VISIBILITY_CHANGED")
        manifest = {
            "kind": "szl.direct-owner-checkpoint-archive/v1",
            "repo": REPO,
            "revision": revision,
            "checkpoint": step,
            "target_steps": TARGET,
            "observed_at_utc": utc(),
            "files": records,
            "total_bytes": total,
            "private": True,
            "training_complete": False,
            "final_export_verified": False,
            "gate1_status": "UNOBSERVED",
            "pickle_deserialized": False,
            "model_quality_established": False,
            "cloud_mutations": 0,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Pinned Hub object integrity and owner-device copy only; not Gate-1 recovery",
        }
        data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        path = folder / "manifest.json"
        if path.exists():
            old = json.loads(path.read_bytes())
            if (old.get("repo"), old.get("revision"), old.get("files")) != (
                REPO,
                revision,
                records,
            ):
                raise Hold("EXISTING_MANIFEST_CONFLICT")
        else:
            tmp = folder / ("manifest." + uuid.uuid4().hex + ".tmp")
            atomic_new(tmp, data)
            tmp.rename(path)
        for name, entry in entries.items():
            verify(folder.joinpath(*relative(name).parts), entry)
        return {
            "status": "PINNED_HUB_AND_OWNER_COPY_VERIFIED",
            "checkpoint": step,
            "target_steps": TARGET,
            "revision": revision,
            "files": len(records),
            "bytes": total,
            "archive": str(folder),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "gate1_status": "UNOBSERVED",
            "cloud_mutations": 0,
            "training_complete": False,
            "observed_at_utc": utc(),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from huggingface_hub import HfApi, hf_hub_download

    try:
        result = run_archive(
            HfApi(), hf_hub_download, args.root, args.revision, args.step
        )
    except Exception as exc:
        result = {
            "status": "HOLD",
            "reason": str(exc) if isinstance(exc, Hold) else type(exc).__name__,
            "http_status": getattr(getattr(exc, "response", None), "status_code", None),
            "cloud_mutations": 0,
            "observed_at_utc": utc(),
        }
        print(json.dumps(result), flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
