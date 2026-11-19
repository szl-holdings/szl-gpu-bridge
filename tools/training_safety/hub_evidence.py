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
import urllib.parse

from szl_controller_gate import provider_call, classify_write, measure_file, _json_obj

HF = "https://huggingface.co"


def _tok(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _auth(token_file):
    return {"Authorization": "Bearer " + _tok(token_file)}


def repo_snapshot(repo, token_file, *, repo_type="model", call=provider_call):
    kind = "models" if repo_type == "model" else "datasets"
    r = call("GET", "%s/api/%s/%s?blobs=true" % (HF, kind, repo), _auth(token_file))
    ok, code = classify_write(r)
    if not ok:
        return None, "HUB_INFO_" + code
    info, err = _json_obj(r["body"])
    if err or not isinstance(info, dict):
        return None, "HUB_INFO_INVALID"
    entries = {}
    for s in info.get("siblings") or []:
        lfs = s.get("lfs") or {}
        entries[s.get("rfilename")] = {"size": lfs.get("size", s.get("size")),
                                       "sha256": lfs.get("sha256"),
                                       "git_blob_sha1": None if lfs else s.get("blobId")}
    return {"revision": info.get("sha"), "repo_private": info.get("private") is True,
            "entries": entries}, None


def latest_checkpoint(entries):
    steps = [int(m.group(1)) for p in entries for m in [re.match(r"checkpoint-(\d+)/", p or "")] if m]
    return max(steps) if steps else None


def stream_download(url, headers, dest, *, timeout=600):
    """Stream a file to `dest`; return HTTP status (or None on transport failure).
    The body is never held in memory and never logged."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
            return int(resp.status)
    except urllib.error.HTTPError as e:
        return int(e.code)
    except Exception:
        return None


def fetch_verified(repo, revision, path, entry, token_file, cache_dir, *,
                   repo_type="model", download=stream_download):
    """Download one file pinned to `revision`; return (measured {size, sha256, git_blob_sha1}, dest)
    or (None, error code). Only a 200 counts."""
    base = HF if repo_type == "model" else HF + "/datasets"
    url = "%s/%s/resolve/%s/%s" % (base, repo, revision, urllib.parse.quote(path))
    dest = os.path.join(cache_dir, revision, *path.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    status = download(url, _auth(token_file), tmp)
    if status != 200:
        if os.path.exists(tmp):
            os.remove(tmp)  # our own partial temp file only
        return None, "DOWNLOAD_HTTP_%s" % status
    os.replace(tmp, dest)
    return measure_file(dest), dest


def backup_copy(local_path, backup_dir, revision, path):
    dest = os.path.join(backup_dir, revision, *path.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return measure_file(dest)  # never overwrite an existing backup
    shutil.copy2(local_path, dest + ".part")
    os.replace(dest + ".part", dest)
    return measure_file(dest)


def collect(group_cfg, token_file, *, call=provider_call, download=stream_download):
    """Build the decide_stop evidence for one group config. Returns (ev, err)."""
    repo = group_cfg["repo"]
    snap, err = repo_snapshot(repo, token_file, repo_type=group_cfg.get("repo_type", "model"), call=call)
    if err:
        return None, err
    entries, rev = snap["entries"], snap["revision"]
    ev = {"reason": group_cfg["reason"], "revision": rev, "repo_private": snap["repo_private"],
          "hub_entries": entries, "target_steps": group_cfg.get("target_steps"),
          "completion_marker": group_cfg.get("completion_marker"),
          "local_readback": {}, "backup": {"location": group_cfg["backup_dir"],
                                           "independent": bool(group_cfg.get("backup_independent")),
                                           "files": {}}}
    if group_cfg["reason"] == "eval_receipt":
        paths = list(group_cfg["required_paths"])
        ev["required_paths"] = paths
    else:
        latest = latest_checkpoint(entries)
        ev["latest_checkpoint"] = latest
        if latest is None:
            return ev, None
        paths = ["final/adapter_model.safetensors", "final/adapter_config.json",
                 "checkpoint-%d/adapter_model.safetensors" % latest,
                 "checkpoint-%d/trainer_state.json" % latest]
    for p in paths:
        if p not in entries or not (entries[p].get("sha256") or entries[p].get("git_blob_sha1")):
            continue  # decide_stop will HOLD on the missing/incomplete entry
        m, dest = fetch_verified(repo, rev, p, entries[p], token_file, group_cfg["cache_dir"],
                                 repo_type=group_cfg.get("repo_type", "model"), download=download)
        if m is None:
            return ev, dest
        ev["local_readback"][p] = m
        ev["backup"]["files"][p] = backup_copy(dest, group_cfg["backup_dir"], rev, p)
        if p.endswith("trainer_state.json"):
            with open(dest, encoding="utf-8") as f:
                ts = json.load(f)
            ev["trainer_state"] = {"global_step": ts.get("global_step"), "max_steps": ts.get("max_steps")}
    return ev, None
