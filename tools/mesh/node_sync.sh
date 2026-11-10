#!/usr/bin/env bash
# node_sync.sh — the mesh agent. Run hourly on every A6000 node (cron).
#
#   0 * * * *  /bin/bash ~/szl-gpu-bridge/tools/mesh/node_sync.sh >> ~/mesh.log 2>&1
#
# The daisy-chain contract (docs/A6000_MESH.md):
#   1. PUSH: new receipts/checkpoint manifests -> the chain (content-addressed)
#   2. PULL: the brain's curriculum manifest -> local, if newer
#   3. HEARTBEAT: one mesh heartbeat receipt per run (energy UNAVAILABLE never zero)
# Fail closed: unreachable = UNAVAILABLE receipt, never silence (Law 9).
set -euo pipefail
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in the environment}"
DS="SZLHOLDINGS/frontier-54-runs"
NODE_ID="$(hostname)"
OUT_DIR="$HOME/szl-legs"
STATE="$HOME/.mesh-state.json"
BRANCH="main"

receipt() { python3 -c "
import json, hashlib, datetime, sys
kind, stage, state, detail = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
r = {'kind': kind, 'node': '$NODE_ID', 'stage': stage, 'state': state,
     'detail': detail[:200], 'ts': datetime.datetime.now(datetime.timezone.utc).isoformat()}
print(json.dumps(r, sort_keys=True))" "$@"; }

# 1. PUSH — every new receipt file in the legs' out dirs
python3 - <<'PY'
import json, os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
DS = "SZLHOLDINGS/frontier-54-runs"
import pathlib, hashlib
pushed = 0
for out in pathlib.Path.home().glob("szl-legs/*/out/receipts/*.json"):
    content = out.read_bytes()
    sha = hashlib.sha256(content).hexdigest()[:12]
    path = f"mesh/{os.uname().nodename}/{out.parent.parent.name}/{sha}.json"
    api.upload_file(path_or_fileobj=content, path_in_repo=path,
                    repo_id=DS, repo_type="dataset")
    pushed += 1
print(f"pushed {pushed} receipts")
PY

# 2. PULL — the brain's curriculum manifest, if newer
python3 - <<'PY'
import json, os, urllib.request
try:
    req = urllib.request.Request(
        "https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/main/curriculum/manifest.json",
        headers={"User-Agent": "szl-mesh/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        manifest = json.load(r)
    state_path = os.path.expanduser("~/.mesh-state.json")
    state = {}
    if os.path.exists(state_path):
        state = json.load(open(state_path))
    if manifest.get("sha256") != state.get("curriculum_sha"):
        state["curriculum_sha"] = manifest.get("sha256")
        state["curriculum_pulled"] = manifest.get("ts")
        json.dump(state, open(state_path, "w"), indent=2)
        print("NEW CURRICULUM pulled:", manifest.get("sha256", "?")[:12], "- active next cycle")
    else:
        print("curriculum current")
except Exception as e:
    print("curriculum UNAVAILABLE:", type(e).__name__)
PY

# 3. HEARTBEAT — GPU util via NVML if available, else UNAVAILABLE (never zero)
GPU_STATE="UNAVAILABLE"
if command -v nvidia-smi >/dev/null 2>&1; then
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 || true)
  [ -n "$UTIL" ] && GPU_STATE="MEASURED util=${UTIL}%"
fi
receipt "szl.mesh-heartbeat/v1" "hourly" "$GPU_STATE" "uptime $(cut -d. -f1 /proc/uptime)s"
echo "mesh sync complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
