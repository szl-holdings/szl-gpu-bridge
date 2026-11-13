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
# The HF token is written to a mode-0600 env file and exported from that file
# inside the screen session. It is never placed on a command line (`ps` shows argv).
set -euo pipefail
MODE="${1:-worker}"
cd "$(dirname "$0")"
MESH_SRC="$(pwd)"

read -r -s -p "HF token: " HFTOK && echo

# Keep the token out of argv. `screen ... bash -c "export HF_TOKEN=..."` is
# visible to `ps`. The session sources a mode-0600 file instead, which exports
# HF_TOKEN into the daemon's environment.
MESH_DIR="${HOME}/szl-mesh"
mkdir -p "$MESH_DIR"
chmod 700 "$MESH_DIR"
ENV_FILE="${MESH_DIR}/hf-token.env"
umask 077
: > "$ENV_FILE"
chmod 0600 "$ENV_FILE"
printf 'export HF_TOKEN=%q\n' "$HFTOK" > "$ENV_FILE"
chmod 0600 "$ENV_FILE"
unset HFTOK
env_mode="$(stat -c %a "$ENV_FILE")"
if [[ "$env_mode" != "600" ]]; then
  echo "MESH FAILED TO START — token file mode is ${env_mode}, want 600" >&2
  exit 3
fi
# shellcheck disable=SC1090
. "$ENV_FILE"

# boot receipt: prove the mesh node started
python3 -c "
import json, datetime, socket
r = {'kind': 'szl.mesh-boot/v1', 'node': socket.gethostname(), 'mode': '$MODE',
     'ts': datetime.datetime.now(datetime.timezone.utc).isoformat()}
print(json.dumps(r, sort_keys=True))"

# launch in screen (survives disconnect, logs to disk)
SCREEN_NAME="mesh-$MODE"

# Stop the daemon this node already runs, however it was started. Without this the
# new worker loses the node lock and exits while the old one keeps running.
screen -S "$SCREEN_NAME" -X quit >/dev/null 2>&1 || true
pkill -f "mesh_daemon.py --mode $MODE" >/dev/null 2>&1 || true
for _ in 1 2 3 4 5 6 7 8 9 10; do
  pgrep -f "mesh_daemon.py --mode $MODE" >/dev/null 2>&1 || break
  sleep 1
done
if pgrep -f "mesh_daemon.py --mode $MODE" >/dev/null 2>&1; then
  echo "MESH FAILED TO START — the old $MODE daemon would not stop"
  exit 3
fi
# Single-quoted on purpose: the session sources the token file itself.
# shellcheck disable=SC2016
screen -dmS "$SCREEN_NAME" bash -c '
  set -euo pipefail
  cd "$1"
  # shellcheck disable=SC1091
  . "${HOME}/szl-mesh/hf-token.env"
  exec python3 mesh_daemon.py --mode "$2" >> "${HOME}/szl-mesh/daemon.log" 2>&1
' _ "$MESH_SRC" "$MODE"
sleep 3
if pgrep -f "mesh_daemon.py --mode $MODE" >/dev/null 2>&1; then
  echo "MESH $MODE LAUNCHED (screen $SCREEN_NAME, logs ~/szl-mesh/daemon.log)"
  echo "token file: ${ENV_FILE} (mode 0600; not on the command line)"
  echo "verify: screen -ls && tail -5 ~/szl-mesh/daemon.log"
else
  echo "MESH FAILED TO START — check ~/szl-mesh/daemon.log"
  exit 3
fi
