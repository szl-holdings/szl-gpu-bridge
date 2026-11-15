#!/usr/bin/env python3
"""mesh_daemon.py — the daisy-chain that makes N nodes one computer.

  worker    registers, heartbeats, and runs the jobs signed for it
  brain     tracks which workers are alive
  dispatch  signs one job for one node and queues it

The bus is the receipts chain, a public HF dataset: anyone can read it and many
can write to it, so nothing on it is an instruction by itself. A worker runs a
job only when every line holds:

  signed       HMAC-SHA256 over {id, issued_at, kind, node} with MESH_JOB_KEY
  for me       job.node is this host
  allowlisted  job.kind names a fixed argv in ALLOW; a job carries no command
  fresh        issued within MAX_AGE_S
  once         its id is not yet in this node's ledger

Anything else is REJECTED and never executed. Without MESH_JOB_KEY (32+ bytes,
env only) a worker still heartbeats but runs nothing.

Liveness is not evidence. Heartbeats and the brain's roster are overwritten every
HEARTBEAT_S, so they go to a separate liveness dataset (MESH_LIVENESS_DS), never
to the receipts chain: a commit a minute on the chain means the Hub's dataset
viewer never sees a revision that lasts long enough to index. Registrations,
jobs and results are events and stay on the chain.

Usage:
  python3 mesh_daemon.py --mode worker
  python3 mesh_daemon.py --mode brain
  python3 mesh_daemon.py --mode dispatch --node <hostname> --kind gpu-state
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import hmac
import itertools
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DS = "SZLHOLDINGS/frontier-54-runs"
LIVENESS_DS = os.environ.get("MESH_LIVENESS_DS") or "SZLHOLDINGS/mesh-liveness"
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
NODE_ID = socket.gethostname()
HF = "https://huggingface.co"
MESH_DIR = Path(os.environ.get("MESH_HOME", str(Path.home() / "szl-mesh")))
HEARTBEAT_S = 60
MAX_AGE_S = 900
JOB_TIMEOUT_S = 600
MIN_KEY_BYTES = 32
JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

GPU_QUERY = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"]

# Everything a job can make a worker do. A job picks a key; it never supplies the
# argv. Adding a line here is a reviewed code change.
ALLOW = {
    "gpu-state": GPU_QUERY,
    "disk-state": ["df", "-h", "/"],
    "node-sync": ["bash", str(Path(__file__).with_name("node_sync.sh"))],
}


# ============ THE CHAIN ============
def chain_read(path: str, timeout=20, repo: str = DS):
    req = urllib.request.Request(f"{HF}/datasets/{repo}/resolve/main/{path}",
                                 headers={"User-Agent": "szl-mesh/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chain_write(api, path: str, data: dict, repo: str = DS):
    api.upload_file(path_or_fileobj=json.dumps(data, indent=2).encode(),
                    path_in_repo=path, repo_id=repo, repo_type="dataset")


def liveness_write(api, path: str, data: dict):
    """Overwritten-every-minute state. Never the receipts chain."""
    problem = liveness_problem()
    if problem:
        raise RuntimeError(f"FAIL-CLOSED: {problem}")
    chain_write(api, path, data, repo=LIVENESS_DS)


def liveness_problem() -> str | None:
    """Why LIVENESS_DS cannot be used, or None if it can."""
    if LIVENESS_DS == DS:
        return "MESH_LIVENESS_DS must not be the receipts chain"
    if not REPO_ID.fullmatch(LIVENESS_DS):
        return "MESH_LIVENESS_DS must be an owner/name dataset id"
    return None


def receipt(kind, data):
    r = {"kind": kind, "node": NODE_ID, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **data}
    r["sha256"] = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
    return r


def gpu_state():
    """MEASURED or UNAVAILABLE — never zero, never fabricated."""
    try:
        r = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            return {"gpu": parts[0], "util_pct": parts[1], "mem_used": parts[2],
                    "mem_total": parts[3], "state": "MEASURED"}
    except Exception:
        pass
    return {"state": "UNAVAILABLE"}


def age_s(ts: str, now: float) -> int:
    """Seconds since a chain timestamp, which is UTC whatever this host's timezone."""
    return int(now - calendar.timegm(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")))


# ============ JOB ADMISSION ============
def sign(job: dict, key: bytes) -> str:
    body = json.dumps({f: job[f] for f in ("id", "issued_at", "kind", "node")},
                      sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(key, body, hashlib.sha256).hexdigest()


def rejection(job, key: bytes, node: str, now: float, done) -> str | None:
    """Why this job must not run, or None if it may. Pure: no I/O, no side effects."""
    if len(key) < MIN_KEY_BYTES:
        return "NO_KEY"
    if not (isinstance(job, dict)
            and all(isinstance(job.get(f), str) for f in ("id", "node", "kind", "sig"))
            and type(job.get("issued_at")) is int
            and JOB_ID.fullmatch(job["id"])):
        return "MALFORMED"
    if not hmac.compare_digest(sign(job, key).encode(), job["sig"].encode()):
        return "BAD_SIGNATURE"
    if job["node"] != node:
        return "WRONG_NODE"
    if job["kind"] not in ALLOW:
        return "UNKNOWN_KIND"
    if abs(now - job["issued_at"]) > MAX_AGE_S:
        return "STALE"
    if job["id"] in done:
        return "REPLAY"
    return None


class Ledger:
    """Ids of jobs already admitted, kept on disk so a restart cannot replay them."""

    def __init__(self, path):
        self.path = Path(path)
        self.ids = set(json.loads(self.path.read_text())) if self.path.exists() else set()

    def __contains__(self, job_id):
        return job_id in self.ids

    def add(self, job_id):
        self.ids.add(job_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.ids)))
        os.replace(tmp, self.path)


def execute_job(job, *, key: bytes = b"", node: str | None = None, now: float | None = None, done=None):
    """Admit the job, then run its allowlisted argv. Never a shell, never the job's own text."""
    done = set() if done is None else done
    why = rejection(job, key, node or NODE_ID, time.time() if now is None else now, done)
    if why:
        return receipt("szl.mesh-job-result/v1", {"state": "REJECTED", "reason": why})
    done.add(job["id"])  # before running: at most once, even if this process dies mid-job
    boot = receipt("szl.mesh-job-boot/v1", {"job": job["id"], "kind": job["kind"]})
    t0 = time.time()
    try:
        r = subprocess.run(ALLOW[job["kind"]], capture_output=True, text=True, timeout=JOB_TIMEOUT_S)
        stdout, rc = r.stdout[:2000], r.returncode
    except subprocess.TimeoutExpired:
        stdout, rc = "", -2
    except OSError as e:
        stdout, rc = str(e)[:500], -3
    return receipt("szl.mesh-job-result/v1", {
        "job": job["id"], "kind": job["kind"], "rc": rc, "stdout": stdout,
        "wall_s": round(time.time() - t0, 1),
        "state": "COMPLETED" if rc == 0 else "FAILED", "boot": boot["sha256"]})


def dispatch(api, node: str, kind: str, key: bytes, now: float | None = None) -> dict:
    """Sign one allowlisted job for one node and put it on that node's queue."""
    if kind not in ALLOW:
        raise ValueError(f"not an allowlisted job kind: {kind}")
    if len(key) < MIN_KEY_BYTES:
        raise ValueError(f"MESH_JOB_KEY must be at least {MIN_KEY_BYTES} bytes")
    job = {"id": secrets.token_hex(8), "node": node, "kind": kind,
           "issued_at": int(time.time() if now is None else now)}
    job["sig"] = sign(job, key)
    chain_write(api, f"mesh/queue/{node}/next-job.json", job)
    return job


# ============ WORKER ============
def claim_node(path):
    """Hold this node's worker lock, or None if another worker already does.

    Two workers on one node would each run every job."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def poll(api, key: bytes, done, seen, stats: dict, read=chain_read):
    """Look at this node's queue once. Returns the digest of what was there."""
    try:
        job = read(f"mesh/queue/{NODE_ID}/next-job.json")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return seen  # nothing queued
        raise
    digest = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if digest == seen:
        return seen  # this exact file was already handled
    result = execute_job(job, key=key, done=done)
    if result["state"] == "REJECTED":
        stats["rejected"] += 1
        print(f"job REJECTED: {result['reason']}", flush=True)
    else:
        stats["done"] += 1
        chain_write(api, f"mesh/results/{NODE_ID}/{job['id']}.json", result)
        print(f"job {job['id']} ({job['kind']}): {result['state']}", flush=True)
    return digest


def worker_loop(api, brain_url: str, key: bytes) -> int:
    """Register, then heartbeat and poll the queue until stopped."""
    lock = claim_node(MESH_DIR / "worker.lock")  # held for the life of the process
    if lock is None:
        print("FAIL-CLOSED: another mesh worker already runs on this node", file=sys.stderr)
        return 3
    if len(key) < MIN_KEY_BYTES:
        print("job execution DISABLED: MESH_JOB_KEY not set — heartbeats only", flush=True)
    done, seen, stats, started = Ledger(MESH_DIR / "done.json"), None, {"done": 0, "rejected": 0}, time.time()
    chain_write(api, f"mesh/workers/{NODE_ID}/register.json", receipt("szl.mesh-register/v1", {
        "brain": brain_url, "gpu": gpu_state(),
        "roles": os.environ.get("MESH_ROLES", "train,serve,eval").split(",")}))
    print(f"registered with brain {brain_url}", flush=True)
    for cycle in itertools.count(1):
        try:
            liveness_write(api, f"mesh/workers/{NODE_ID}/heartbeat.json", receipt("szl.mesh-heartbeat/v1", {
                "cycle": cycle, "gpu": gpu_state(), "uptime_s": int(time.time() - started),
                "jobs_done": stats["done"], "jobs_rejected": stats["rejected"]}))
        except Exception as e:  # a failed heartbeat must not stop the node polling for work
            print(f"cycle {cycle} heartbeat error: {type(e).__name__}: {e}", flush=True)
        try:
            seen = poll(api, key, done, seen, stats)
        except Exception as e:  # one bad cycle must not take the node off the mesh
            print(f"cycle {cycle} error: {type(e).__name__}: {e}", flush=True)
        time.sleep(HEARTBEAT_S)
    return 0


# ============ BRAIN ============
def brain_loop(api):
    """Track which workers are alive and publish the roster."""
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"brain mode: jobs via {DS}, liveness via {LIVENESS_DS}", flush=True)
    workers = {}
    while True:
        # Workers are upgraded one node at a time, so a worker that has not been
        # upgraded yet still heartbeats on the chain. Read both and keep the newest
        # per node. Reads cost no commits. Drop DS here once every node is upgraded.
        newest = {}
        for repo in (LIVENESS_DS, DS):
            try:
                tree = json.loads(urllib.request.urlopen(
                    urllib.request.Request(f"{HF}/api/datasets/{repo}/tree/main/mesh/workers?recursive=true",
                                           headers={"User-Agent": "szl-mesh/1.0"}), timeout=20).read())
            except urllib.error.HTTPError as e:
                if e.code != 404:  # 404: no worker has written to this repo yet
                    print(f"brain: cannot list workers on {repo}: HTTP {e.code}", flush=True)
                continue
            except Exception as e:
                print(f"brain: cannot list workers on {repo}: {type(e).__name__}: {e}", flush=True)
                continue
            for f in tree:
                if f.get("path", "").endswith("/heartbeat.json"):
                    wid = f["path"].split("/")[2]
                    try:
                        hb = chain_read(f["path"], repo=repo)
                        age = age_s(hb["ts"], time.time())
                    except Exception as e:
                        print(f"brain: unreadable heartbeat {repo}/{f['path']}: {type(e).__name__}", flush=True)
                        continue
                    if wid not in newest or age < newest[wid]["age_s"]:
                        newest[wid] = {"age_s": age, "gpu": hb.get("gpu"),
                                       "alive": age < HEARTBEAT_S * 3}
        workers.update(newest)

        state = receipt("szl.mesh-brain-state/v1", {
            "workers": {k: {"alive": v["alive"], "gpu": v["gpu"]} for k, v in workers.items()},
            "alive_count": sum(1 for v in workers.values() if v["alive"]),
            "total_known": len(workers)})
        try:
            liveness_write(api, "mesh/brain/state.json", state)
            print(f"brain: {state['alive_count']} of {state['total_known']} alive", flush=True)
        except Exception as e:  # one failed roster write must not stop the brain
            print(f"brain: roster write error: {type(e).__name__}: {e}", flush=True)
        time.sleep(HEARTBEAT_S)


def main():
    ap = argparse.ArgumentParser(description="SZL mesh daemon")
    ap.add_argument("--mode", choices=["worker", "brain", "dispatch"], default="worker")
    ap.add_argument("--brain", default="216.81.248.145:8080")
    ap.add_argument("--node", help="dispatch: the worker's hostname")
    ap.add_argument("--kind", choices=sorted(ALLOW), help="dispatch: which allowlisted job")
    args = ap.parse_args()
    if args.mode == "dispatch" and not (args.node and args.kind):
        ap.error("--mode dispatch needs --node and --kind")

    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        print("FAIL-CLOSED: HF_TOKEN required (env only)", file=sys.stderr)
        return 2
    if args.mode != "dispatch" and liveness_problem():
        print(f"FAIL-CLOSED: {liveness_problem()}", file=sys.stderr)
        return 2
    key = os.environ.get("MESH_JOB_KEY", "").encode()
    if args.mode == "dispatch" and len(key) < MIN_KEY_BYTES:
        print(f"FAIL-CLOSED: MESH_JOB_KEY required ({MIN_KEY_BYTES}+ bytes, env only)", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi
    api = HfApi(token=tok)

    if args.mode == "brain":
        return brain_loop(api)
    if args.mode == "dispatch":
        job = dispatch(api, args.node, args.kind, key)
        print(f"queued {job['kind']} for {job['node']} as {job['id']}")
        return 0
    return worker_loop(api, args.brain, key)


if __name__ == "__main__":
    sys.exit(main())

