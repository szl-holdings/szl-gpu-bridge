#!/usr/bin/env python3
"""receipt_upload: upload one proofrun run folder to a private HF receipts repo as ONE commit.

  python3 receipt_upload.py <run_dir> --repo SZLHOLDINGS/szl-run-receipts [--run-id ID]
        [--accept-retroactive] [--trusted-key FP] [--exclude GLOB]... [--large-mb N]
        [--root-file LOCAL:REPO_PATH]... [--extra LOCAL:NAME]... [--dry-run] [--create]

Behaviour
  * runs `proofrun verify` first; verdict + exit code go into the commit message.
    A FAIL verdict is still uploaded (it is evidence); the uploader's own exit code is 0 on upload.
  * everything lands as a single create_commit with one operation list under runs/<run_id>/.
    Heartbeats are local files in the folder; they never trigger a commit by themselves.
  * 429 (and 5xx) -> capped exponential backoff with full jitter, bounded tries, then exit 3 cleanly.
  * files over --large-mb (and --exclude matches) are NOT uploaded; their sha256 is recorded in
    runs/<run_id>/EXCLUDED_FILES.json instead.
  * the repo is never made public / never has visibility changed. --create makes it private only;
    an existing non-private repo aborts the upload (exit 4).
  * token is read from env HF_TOKEN and never printed.
Exit codes: 0 uploaded/skipped/dry-run, 2 usage/verify-could-not-run, 3 gave up on rate limit/server errors,
            4 repo missing or not private, 5 other upload error.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROOFRUN_PATH = os.path.join(HERE, "vendor")
RETRY_STATUS = {429, 500, 502, 503, 504}


class GaveUp(Exception):
    pass


class RepoNotPrivate(Exception):
    pass


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def run_verify(run_dir, proofrun_path=DEFAULT_PROOFRUN_PATH, accept_retroactive=False, trusted_keys=()):
    """Return (verdict, exit_code, output_text). verdict in PASS/FAIL/ERROR."""
    cmd = [sys.executable, "-m", "proofrun", "verify", run_dir]
    if accept_retroactive:
        cmd.append("--accept-retroactive")
    for k in trusted_keys:
        cmd += ["--trusted-key", k]
    env = dict(os.environ)
    env["PYTHONPATH"] = proofrun_path + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
    m = re.search(r"^VERDICT:\s*(\w+)", p.stdout, re.M)
    verdict = m.group(1) if m else "ERROR"
    return verdict, p.returncode, out


def collect_files(run_dir, excludes=(), large_mb=None):
    """Return (included[(abs, rel)], excluded[{path,size,sha256,reason}])."""
    inc, exc = [], []
    for root, dirs, files in os.walk(run_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        for fn in sorted(files):
            if fn.startswith("."):
                continue
            ab = os.path.join(root, fn)
            if os.path.islink(ab):
                continue
            rel = os.path.relpath(ab, run_dir).replace(os.sep, "/")
            size = os.path.getsize(ab)
            reason = None
            if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(fn, g) for g in excludes):
                reason = "excluded by pattern"
            elif large_mb is not None and size > large_mb * 1024 * 1024:
                reason = "larger than %s MB; referenced by sha256 (held elsewhere)" % large_mb
            if reason:
                exc.append({"path": rel, "size": size, "sha256": sha256_file(ab), "reason": reason})
            else:
                inc.append((ab, rel))
    return inc, exc


def tree_sha(entries):
    h = hashlib.sha256()
    for rel, digest in sorted(entries):
        h.update(("%s\0%s\n" % (rel, digest)).encode())
    return h.hexdigest()


def backoff_delay(attempt, base=5.0, cap=300.0, rng=random.random):
    """Full-jitter capped exponential: uniform(0.5, 1.0) * min(cap, base * 2**attempt)."""
    return min(cap, base * (2 ** attempt)) * (0.5 + 0.5 * rng())


def status_of(exc):
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) if resp is not None else getattr(exc, "status_code", None)


def retry_after(exc):
    resp = getattr(exc, "response", None)
    try:
        v = resp.headers.get("Retry-After") if resp is not None else None
        return float(v) if v else None
    except Exception:
        return None


def commit_with_backoff(do_commit, max_tries=6, base=5.0, cap=300.0, sleep=time.sleep, rng=random.random, log=print):
    last = None
    for attempt in range(max_tries):
        try:
            return do_commit()
        except Exception as e:  # noqa
            st = status_of(e)
            if st not in RETRY_STATUS:
                raise
            last = e
            if attempt == max_tries - 1:
                break
            d = backoff_delay(attempt, base, cap, rng)
            ra = retry_after(e)
            if ra:
                d = min(cap, max(d, ra))
            log("HTTP %s; retry %d/%d in %.0fs" % (st, attempt + 1, max_tries - 1, d))
            sleep(d)
    raise GaveUp("gave up after %d tries (last HTTP %s)" % (max_tries, status_of(last)))


def ensure_private_repo(api, repo, create=False):
    """Return 'existing' or 'created'. Never changes visibility."""
    try:
        info = api.model_info(repo)
    except Exception as e:  # noqa
        if status_of(e) == 404 or type(e).__name__ == "RepositoryNotFoundError":
            if not create:
                raise FileNotFoundError("repo %s not found (use --create)" % repo)
            api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=False)
            return "created"
        raise
    if not info.private:
        raise RepoNotPrivate("repo %s is not private; refusing to upload" % repo)
    return "existing"


def build_operations(run_dir, run_id, included, excluded, verify_text, extras, root_files, ops_cls, summary):
    ops = []
    base = "runs/%s/" % run_id
    for ab, rel in included:
        ops.append(ops_cls(path_in_repo=base + rel, path_or_fileobj=ab))
    gen = {
        "VERIFY_LIVE.txt": verify_text.encode(),
        "UPLOAD_SUMMARY.json": (json.dumps(summary, indent=1, sort_keys=True) + "\n").encode(),
    }
    if excluded:
        gen["EXCLUDED_FILES.json"] = (json.dumps(excluded, indent=1, sort_keys=True) + "\n").encode()
    for name, data in gen.items():
        ops.append(ops_cls(path_in_repo=base + name, path_or_fileobj=data))
    for local, name in extras:
        ops.append(ops_cls(path_in_repo=base + name, path_or_fileobj=local))
    for local, repo_path in root_files:
        ops.append(ops_cls(path_in_repo=repo_path, path_or_fileobj=local))
    return ops


def upload_run(run_dir, repo, run_id=None, api=None, ops_cls=None, accept_retroactive=False, trusted_keys=(),
               excludes=(), large_mb=None, extras=(), root_files=(), dry_run=False, create=False,
               max_tries=6, base=5.0, cap=300.0, sleep=time.sleep, rng=random.random,
               verify_fn=run_verify, log=print):
    run_dir = os.path.abspath(run_dir)
    if run_id is None:
        run_id = _guess_run_id(run_dir)
    verdict, code, vout = verify_fn(run_dir, accept_retroactive=accept_retroactive, trusted_keys=trusted_keys)
    log("verify: %s exit=%s" % (verdict, code))
    included, excluded = collect_files(run_dir, excludes, large_mb)
    entries = [(rel, sha256_file(ab)) for ab, rel in included]
    summary = {
        "run_id": run_id, "verify_verdict": verdict, "verify_exit_code": code,
        "accept_retroactive": bool(accept_retroactive), "trusted_keys": list(trusted_keys),
        "files_uploaded": len(included), "files_referenced_not_uploaded": len(excluded),
        "tree_sha256": tree_sha(entries),
        "note": "FAIL verdicts are uploaded as evidence; nothing here is LIVE, MEASURED or publication_eligible.",
    }
    msg = ("receipts %s: proofrun verify=%s exit=%s%s\n\ntree_sha256=%s files=%d referenced_only=%d" % (
        run_id, verdict, code, " (retroactive accepted)" if accept_retroactive else "",
        summary["tree_sha256"], len(included), len(excluded)))
    if dry_run:
        log("DRY RUN: would create ONE commit in %s with %d included files under runs/%s/ (+generated, +%d extras, +%d root files)"
            % (repo, len(included), run_id, len(extras), len(root_files)))
        for e in excluded:
            log("  referenced only: %s %d B sha256=%s" % (e["path"], e["size"], e["sha256"]))
        log("commit message:\n" + msg)
        return {"status": "dry-run", "verdict": verdict, "exit_code": code, "summary": summary, "message": msg}
    state = ensure_private_repo(api, repo, create=create)
    ops = build_operations(run_dir, run_id, included, excluded, vout, extras, root_files, ops_cls, summary)
    info = commit_with_backoff(
        lambda: api.create_commit(repo_id=repo, repo_type="model", operations=ops, commit_message=msg.split("\n")[0],
                                  commit_description="\n".join(msg.split("\n")[2:])),
        max_tries=max_tries, base=base, cap=cap, sleep=sleep, rng=rng, log=log)
    return {"status": "uploaded", "repo": state, "verdict": verdict, "exit_code": code, "summary": summary,
            "commit": info}


def _guess_run_id(run_dir):
    for n in ("claim.json", "run_manifest.json"):
        p = os.path.join(run_dir, n)
        if os.path.exists(p):
            try:
                with open(p) as fh:
                    d = json.load(fh)
                for c in (d.get("payload", {}).get("run_id"), d.get("run_id")):
                    if c:
                        return re.sub(r"[^A-Za-z0-9._-]", "_", c)
            except Exception:
                pass
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(run_dir.rstrip("/")))


def _pair(s):
    if ":" not in s:
        raise argparse.ArgumentTypeError("expected LOCAL:NAME")
    a, b = s.rsplit(":", 1)
    return a, b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--repo", default="SZLHOLDINGS/szl-run-receipts")
    ap.add_argument("--run-id")
    ap.add_argument("--accept-retroactive", action="store_true")
    ap.add_argument("--trusted-key", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--large-mb", type=float, default=None)
    ap.add_argument("--extra", action="append", type=_pair, default=[], help="LOCAL:NAME placed under runs/<id>/")
    ap.add_argument("--root-file", action="append", type=_pair, default=[], help="LOCAL:REPO_PATH at repo root")
    ap.add_argument("--proofrun-path", default=DEFAULT_PROOFRUN_PATH)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--create", action="store_true", help="create the repo as PRIVATE if missing")
    ap.add_argument("--max-tries", type=int, default=6)
    ap.add_argument("--base-seconds", type=float, default=5.0)
    ap.add_argument("--cap-seconds", type=float, default=300.0)
    a = ap.parse_args(argv)
    if not os.path.isdir(a.run_dir):
        print("run_dir not found", file=sys.stderr)
        return 2
    vf = lambda d, **kw: run_verify(d, proofrun_path=a.proofrun_path, **kw)  # noqa
    api = ops_cls = None
    if not a.dry_run:
        tok = os.environ.get("HF_TOKEN")
        if not tok:
            print("HF_TOKEN not set", file=sys.stderr)
            return 2
        from huggingface_hub import CommitOperationAdd, HfApi
        api, ops_cls = HfApi(token=tok), CommitOperationAdd
    try:
        r = upload_run(a.run_dir, a.repo, a.run_id, api, ops_cls, a.accept_retroactive, a.trusted_key, a.exclude,
                       a.large_mb, a.extra, a.root_file, a.dry_run, a.create, a.max_tries, a.base_seconds,
                       a.cap_seconds, verify_fn=vf)
    except GaveUp as e:
        print("GAVE UP: %s. Nothing committed; retry later." % e, file=sys.stderr)
        return 3
    except (RepoNotPrivate, FileNotFoundError) as e:
        print("REPO: %s" % e, file=sys.stderr)
        return 4
    except Exception as e:  # noqa
        print("UPLOAD ERROR: %s: %s" % (type(e).__name__, str(e)[:300]), file=sys.stderr)
        return 5
    if r["status"] == "uploaded":
        c = r["commit"]
        print("commit: %s\nurl: %s" % (getattr(c, "oid", c), getattr(c, "commit_url", "")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
