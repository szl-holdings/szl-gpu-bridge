#!/usr/bin/env bash
# install_mesh.sh — one command per node: joins the mesh as worker or brain.
#
# On .118/.191 (workers):  bash install_mesh.sh worker
# On .145 (the brain):     bash install_mesh.sh brain
#
# The daemon registers via the chain, heartbeats every 60s with GPU state,
# and the brain tracks all workers. Jobs dispatch via mesh/queue/<node>/.
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
screen -dmS "$SCREEN_NAME" bash -c "
  export HF_TOKEN='$HFTOK'
  python3 mesh_daemon.py --mode $MODE >> $HOME/szl-mesh/daemon.log 2>&1"
mkdir -p ~/szl-mesh
sleep 3
if screen -ls | grep -q "$SCREEN_NAME"; then
  echo "MESH $MODE LAUNCHED (screen $SCREEN_NAME, logs ~/szl-mesh/daemon.log)"
  echo "verify: screen -ls && tail -5 ~/szl-mesh/daemon.log"
else
  echo "MESH FAILED TO START — check ~/szl-mesh/daemon.log"; exit 3
fi
