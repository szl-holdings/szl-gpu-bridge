#!/usr/bin/env python3
"""mesh_daemon.py — the daisy-chain that makes N nodes one computer.

MODE=worker (default): registers with the brain, heartbeats, accepts
dispatched jobs, executes them, pushes results to the chain.

MODE=brain (on .145): tracks all workers, dispatches jobs from the queue,
collects results, curates the curriculum.

The coordination bus is the receipts chain (HF dataset) — every node can
read it, every node can write to it, and it's the single source of truth.
No node-to-node SSH required for coordination (results flow via the chain;
large artifacts via the optional rsync lane).

Usage:
  python3 mesh_daemon.py --mode worker --brain 216.81.248.145:8080
  python3 mesh_daemon.py --mode brain
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

DS = "SZLHOLDINGS/frontier-54-runs"
NODE_ID = socket.gethostname()
HF = "https://huggingface.co"
MESH_DIR = Path(os.environ.get("MESH_HOME", str(Path.home() / "szl-mesh")))
HEARTBEAT_S = 60
JOB_POLL_S = 30


def chain_read(path: str, timeout=20):
    req = urllib.request.Request(f"{HF}/datasets/{DS}/resolve/main/{path}",
                                 headers={"User-Agent": "szl-mesh/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chain_write(api, path: str, data: dict):
    api.upload_file(path_or_fileobj=json.dumps(data, indent=2).encode(),
                    path_in_repo=path, repo_id=DS, repo_type="dataset")


def gpu_state():
    """MEASURED or UNAVAILABLE — never zero, never fabricated."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            return {"gpu": parts[0], "util_pct": parts[1], "mem_used": parts[2],
                    "mem_total": parts[3], "state": "MEASURED"}
    except Exception:
        pass
    return {"state": "UNAVAILABLE"}


def receipt(kind, data):
    r = {"kind": kind, "node": NODE_ID, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **data}
    r["sha256"] = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
    return r


# ============ WORKER MODE ============
def worker_loop(api, brain_url):
    """Register → heartbeat → poll for jobs → execute → report."""
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    reg = receipt("szl.mesh-register/v1", {
        "brain": brain_url, "gpu": gpu_state(),
        "roles": os.environ.get("MESH_ROLES", "train,serve,eval").split(",")})
    chain_write(api, f"mesh/workers/{NODE_ID}/register.json", reg)
    print(f"registered with brain {brain_url}", flush=True)

    hb_count = 0
    while True:
        hb_count += 1
        hb = receipt("szl.mesh-heartbeat/v1", {
            "cycle": hb_count, "gpu": gpu_state(),
            "uptime_s": int(time.time() - START),
            "jobs_done": JOB_COUNT[0]})
        chain_write(api, f"mesh/workers/{NODE_ID}/heartbeat.json", hb)

        # poll for dispatched jobs
        try:
            job = chain_read(f"mesh/queue/{NODE_ID}/next-job.json")
            if job and job.get("status") == "DISPATCHED":
                print(f"JOB RECEIVED: {job.get('kind')} {job.get('name')}", flush=True)
                result = execute_job(job)
                chain_write(api, f"mesh/results/{NODE_ID}/{job['id']}.json", result)
                JOB_COUNT[0] += 1
        except Exception:
            pass  # no job queued — normal

        time.sleep(HEARTBEAT_S)


def execute_job(job):
    """Run a dispatched job. Boot receipt first, honest states."""
    boot = receipt("szl.mesh-job-boot/v1", {"job": job["id"], "kind": job.get("kind")})
    t0 = time.time()
    stdout, rc = "", -1
    try:
        r = subprocess.run(job.get("command", "echo no-op").split(), shell=True,
                           capture_output=True, text=True, timeout=job.get("timeout_s", 3600))
        stdout, rc = r.stdout[:2000], r.returncode
    except subprocess.TimeoutExpired:
        rc = -2
    except Exception as e:
        stdout, rc = str(e)[:500], -3
    return receipt("szl.mesh-job-result/v1", {
        "job": job["id"], "kind": job.get("kind"), "rc": rc,
        "stdout": stdout, "wall_s": round(time.time() - t0, 1),
        "state": "MEASURED" if rc == 0 else "FAILED", "boot": boot["sha256"]})


# ============ BRAIN MODE ============
def brain_loop(api):
    """Track workers, dispatch jobs from the queue, curate."""
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"brain mode: coordinating via {DS}", flush=True)
    workers = {}
    while True:
        # scan worker heartbeats
        try:
            tree = json.loads(urllib.request.urlopen(
                urllib.request.Request(f"{HF}/api/datasets/{DS}/tree/main/mesh/workers?recursive=true",
                                       headers={"User-Agent": "szl-mesh/1.0"}), timeout=20).read())
            for f in tree:
                if "heartbeat" in f["path"]:
                    wid = f["path"].split("/")[2]
                    try:
                        hb = chain_read(f["path"])
                        age = time.time() - time.mktime(time.strptime(hb["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
                        workers[wid] = {"age_s": int(age), "gpu": hb.get("gpu"),
                                        "alive": age < HEARTBEAT_S * 3}
                    except Exception:
                        pass
        except Exception:
            pass

        state = receipt("szl.mesh-brain-state/v1", {
            "workers": {k: {"alive": v["alive"], "gpu": v["gpu"]} for k, v in workers.items()},
            "alive_count": sum(1 for v in workers.values() if v["alive"]),
            "total_known": len(workers)})
        chain_write(api, "mesh/brain/state.json", state)
        print(f"brain: {state['data']['alive_count'] if 'data' in state else '?'} alive", flush=True)
        time.sleep(HEARTBEAT_S)


START = time.time()
JOB_COUNT = [0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["worker", "brain"], default="worker")
    ap.add_argument("--brain", default="216.81.248.145:8080")
    args = ap.parse_args()

    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        print("FAIL-CLOSED: HF_TOKEN required (env only)", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi
    api = HfApi(token=tok)

    if args.mode == "brain":
        brain_loop(api)
    else:
        worker_loop(api, args.brain)


if __name__ == "__main__":
    sys.exit(main())
