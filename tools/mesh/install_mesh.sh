#!/usr/bin/env bash
# install_mesh.sh — one command per node: joins the mesh as worker or brain.
#
# On .118/.191 (workers):  bash install_mesh.sh worker
# On .145 (the brain):     bash install_mesh.sh brain
#
# The daemon registers via the chain and the brain tracks all workers. Jobs
# dispatch via mesh/queue/<node>/ on the chain. Heartbeats (every 60s, with GPU
# state) and the brain's roster go to the liveness dataset, MESH_LIVENESS_DS
# (default SZLHOLDINGS/mesh-liveness), never to the chain. The token needs write
# access to both datasets.
#
# Re-running this on a node replaces the daemon that is already running there.
set -euo pipefail
MODE="${1:-worker}"
cd "$(dirname "$0")"

read -s -p "HF token: " HFTOK && export HF_TOKEN="$HFTOK" && echo

# boot receipt: prove the mesh node started
python3 -c "
import json, datetime, socket
r = {'kind': 'szl.mesh-boot/v1', 'node': socket.gethostname(), 'mode': '$MODE',
     'ts': datetime.datetime.now(datetime.timezone.utc).isoformat()}
print(json.dumps(r, sort_keys=True))"

# launch in screen (survives disconnect, logs to disk)
SCREEN_NAME="mesh-$MODE"
mkdir -p ~/szl-mesh

# Stop the daemon this node already runs, however it was started. Without this the
# new worker loses the node lock and exits while the old one keeps running.
screen -S "$SCREEN_NAME" -X quit >/dev/null 2>&1 || true
pkill -f "mesh_daemon.py --mode $MODE" >/dev/null 2>&1 || true
for _ in 1 2 3 4 5 6 7 8 9 10; do
  pgrep -f "mesh_daemon.py --mode $MODE" >/dev/null 2>&1 || break
  sleep 1
done
if pgrep -f "mesh_daemon.py --mode $MODE" >/dev/null 2>&1; then
  echo "MESH FAILED TO START — the old $MODE daemon would not stop"; exit 3
fi
screen -dmS "$SCREEN_NAME" bash -c "
  export HF_TOKEN='$HFTOK'
  python3 mesh_daemon.py --mode $MODE >> $HOME/szl-mesh/daemon.log 2>&1"
sleep 3
if pgrep -f "mesh_daemon.py --mode $MODE" >/dev/null 2>&1; then
  echo "MESH $MODE LAUNCHED (screen $SCREEN_NAME, logs ~/szl-mesh/daemon.log)"
  echo "verify: screen -ls && tail -5 ~/szl-mesh/daemon.log"
else
  echo "MESH FAILED TO START — check ~/szl-mesh/daemon.log"; exit 3
fi
