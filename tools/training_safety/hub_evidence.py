"""Hub evidence collection for decide_stop (stdlib only).

Reads one immutable revision, downloads only the required files (adapter,
adapter_config, trainer_state; or an eval receipt), measures size+sha256 of the
downloaded bytes, copies them to an owner-chosen backup directory, re-hashes the
copy, and returns the evidence dict. The token is read from a file per request
and never logged. Nothing here writes to the Hub.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.parse

from szl_controller_gate import (
    HF_API_HOSTS,
    MAX_BODY,
    REVISION,
    _json_obj,
    _safe_rel_path,
    classify_write,
    find_placeholders,
    measure_file,
    open_following,
    provider_call,
)

HF = "https://huggingface.co"
MAX_DOWNLOAD_SECONDS = 1800


def _tok(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _auth(token_file):
    return {"Authorization": "Bearer " + _tok(token_file)}


def repo_snapshot(repo, token_file, *, repo_type="model", call=provider_call):
    kind = "models" if repo_type == "model" else "datasets"
    r = call(
        "GET",
        "%s/api/%s/%s?blobs=true" % (HF, kind, repo),
        _auth(token_file),
        allowed_hosts=HF_API_HOSTS,
    )
    ok, code = classify_write(r)
    if not ok:
        return None, "HUB_INFO_" + code
    info, err = _json_obj(r["body"])
    if err or not isinstance(info, dict):
        return None, "HUB_INFO_INVALID"
    entries = {}
    for s in info.get("siblings") or []:
        lfs = s.get("lfs") or {}
        entries[s.get("rfilename")] = {
            "size": lfs.get("size", s.get("size")),
            "sha256": lfs.get("sha256"),
            "git_blob_sha1": None if lfs else s.get("blobId"),
        }
    priv = info.get("private")
    return {
        "revision": info.get("sha"),
        "repo_private": priv if isinstance(priv, bool) else None,
        "entries": entries,
    }, None


def latest_checkpoint(entries):
    steps = [
        int(m.group(1))
        for p in entries
        for m in [re.match(r"checkpoint-(\d+)/", p or "")]
        if m
    ]
    return max(steps) if steps else None


def stream_download(
    url,
    headers,
    dest,
    *,
    timeout=600,
    max_bytes=None,
    opener=None,
    max_seconds=MAX_DOWNLOAD_SECONDS,
    clock=time.monotonic,
):
    """Stream a file to `dest`; return HTTP status 200, another status, or an error
    string. Redirects are followed by hand: https only, Hugging Face hosts only,
    and the Authorization header goes to huggingface.co alone (the CDN URL is
    pre-signed and gets none). The body is never held in memory or logged, is
    capped at `max_bytes` (the Hub-reported size) and at `max_seconds`; the
    partial file is removed on every failure."""
    resp, err = open_following(url, headers, timeout=timeout, opener=opener)
    if resp is None:
        return err
    started, done = clock(), False
    try:
        status = int(resp.status)
        if status != 200:
            return status
        written = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    return "SIZE_EXCEEDED"
                if clock() - started > max_seconds:
                    return "DEADLINE_EXCEEDED"
                f.write(chunk)
        done = True
        return 200
    except Exception as e:
        return "TRANSPORT_" + type(e).__name__
    finally:
        resp.close()
        if not done and os.path.exists(dest):
            os.remove(dest)  # our own partial temp file only


def fetch_verified(
    repo,
    revision,
    path,
    entry,
    token_file,
    cache_dir,
    *,
    repo_type="model",
    download=stream_download,
):
    """Download one file pinned to `revision`; return (measured {size, sha256, git_blob_sha1}, dest)
    or (None, error code). Only a 200 counts."""
    if not REVISION.fullmatch(revision or "") or not _safe_rel_path(path):
        return None, "UNSAFE_PATH_OR_REVISION"
    base = HF if repo_type == "model" else HF + "/datasets"
    url = "%s/%s/resolve/%s/%s" % (base, repo, revision, urllib.parse.quote(path))
    root = os.path.realpath(cache_dir)
    dest = os.path.realpath(os.path.join(root, revision, *path.split("/")))
    if not dest.startswith(root + os.sep):
        return None, "UNSAFE_PATH_OR_REVISION"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    size = entry.get("size") if isinstance(entry, dict) else None
    status = download(url, _auth(token_file), tmp, max_bytes=size)
    if status != 200:
        if os.path.exists(tmp):
            os.remove(tmp)  # our own partial temp file only
        return None, "DOWNLOAD_HTTP_%s" % status
    os.replace(tmp, dest)
    return measure_file(dest), dest


def backup_copy(local_path, backup_dir, revision, path):
    root = os.path.realpath(backup_dir)
    dest = os.path.realpath(os.path.join(root, revision, *path.split("/")))
    if (
        not REVISION.fullmatch(revision or "")
        or not _safe_rel_path(path)
        or not dest.startswith(root + os.sep)
    ):
        raise ValueError("UNSAFE_PATH_OR_REVISION")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return measure_file(dest)  # never overwrite an existing backup
    shutil.copy2(local_path, dest + ".part")
    os.replace(dest + ".part", dest)
    return measure_file(dest)


def _separate_dirs(backup_dir, cache_dir, *private_files):
    """The backup must not be the cache, inside it (or contain it), or sit in the
    directory that holds a credential file."""
    b, c = os.path.realpath(backup_dir), os.path.realpath(cache_dir)
    if b == c or b.startswith(c + os.sep) or c.startswith(b + os.sep):
        return False
    return all(
        os.path.realpath(os.path.dirname(os.path.abspath(f))) != b
        for f in private_files
        if f
    )


def _volume(path):
    """st_dev of the nearest existing parent (recorded; 'another node' stays
    owner-attested)."""
    d = os.path.abspath(path)
    while d and not os.path.exists(d) and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    try:
        return os.stat(d).st_dev
    except OSError:
        return None


def collect(
    group_cfg,
    token_file,
    *,
    call=provider_call,
    download=stream_download,
    private_files=(),
):
    """Build the decide_stop evidence for one group config. Returns (ev, err)."""
    if find_placeholders(group_cfg):
        return None, "OWNER_PLACEHOLDER_IN_CONFIG"
    repo = group_cfg["repo"]
    paths_cfg = group_cfg.get("required_paths") or []
    if not all(_safe_rel_path(p) for p in paths_cfg):
        return None, "UNSAFE_REQUIRED_PATH"
    if not _separate_dirs(
        group_cfg["backup_dir"], group_cfg["cache_dir"], token_file, *private_files
    ):
        return None, "BACKUP_DIR_NOT_SEPARATE"
    snap, err = repo_snapshot(
        repo, token_file, repo_type=group_cfg.get("repo_type", "model"), call=call
    )
    if err:
        return None, err
    entries, rev = snap["entries"], snap["revision"]
    if not isinstance(rev, str) or not REVISION.fullmatch(rev):
        return None, "HUB_REVISION_INVALID"
    ev = {
        "reason": group_cfg["reason"],
        "revision": rev,
        "repo_private": snap["repo_private"],
        "allow_public_repo": group_cfg.get("allow_public_repo") is True,
        "hub_entries": entries,
        "target_steps": group_cfg.get("target_steps"),
        "completion_marker": group_cfg.get("completion_marker"),
        "receipt_schema": group_cfg.get("receipt_schema"),
        "receipt_required_fields": group_cfg.get("receipt_required_fields"),
        "receipt_observed": {},
        "local_readback": {},
        "backup": {
            "location": group_cfg["backup_dir"],
            "independent": bool(group_cfg.get("backup_independent")),
            "backup_volume": _volume(group_cfg["backup_dir"]),
            "cache_volume": _volume(group_cfg["cache_dir"]),
            "files": {},
        },
    }
    if group_cfg["reason"] == "eval_receipt":
        paths = list(paths_cfg)
        ev["required_paths"] = paths
    else:
        latest = latest_checkpoint(entries)
        ev["latest_checkpoint"] = latest
        if latest is None:
            return ev, None
        paths = [
            "final/adapter_model.safetensors",
            "final/adapter_config.json",
            "checkpoint-%d/adapter_model.safetensors" % latest,
            "checkpoint-%d/trainer_state.json" % latest,
        ]
    for p in paths:
        if p not in entries or not (
            entries[p].get("sha256") or entries[p].get("git_blob_sha1")
        ):
            continue  # decide_stop will HOLD on the missing/incomplete entry
        m, dest = fetch_verified(
            repo,
            rev,
            p,
            entries[p],
            token_file,
            group_cfg["cache_dir"],
            repo_type=group_cfg.get("repo_type", "model"),
            download=download,
        )
        if m is None:
            return ev, dest
        ev["local_readback"][p] = m
        ev["backup"]["files"][p] = backup_copy(dest, group_cfg["backup_dir"], rev, p)
        if p.endswith("trainer_state.json"):
            ts = _read_json_file(dest)
            if not isinstance(ts, dict):
                return ev, "TRAINER_STATE_INVALID"
            ev["trainer_state"] = {
                "global_step": ts.get("global_step"),
                "max_steps": ts.get("max_steps"),
            }
        if group_cfg["reason"] == "eval_receipt":
            rec = _read_json_file(dest)
            if isinstance(rec, dict):
                ev["receipt_observed"][p] = {
                    "schema": rec.get("schema"),
                    "fields": sorted(k for k in rec if isinstance(k, str)),
                }
    return ev, None


def _read_json_file(path):
    """Parsed JSON of a small local file, or None (never raises on bad content)."""
    try:
        if os.path.getsize(path) > MAX_BODY:
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
