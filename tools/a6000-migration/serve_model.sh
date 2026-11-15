#!/usr/bin/env bash
# Start a GGUF or HF serving process with literal arguments and bounded health
# checks. A successful /health response is readiness, not model-quality proof.
set -euo pipefail
MODEL="${1:-}"; SRC="${2:-}"
[[ "$MODEL" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || { echo 'invalid model name' >&2; exit 2; }
[ -n "$SRC" ] || { echo 'a GGUF path or HF repository is required' >&2; exit 2; }
for command in screen python3 curl; do
  command -v "$command" >/dev/null || { echo "$command is not installed" >&2; exit 2; }
done
session_exists() {
  screen -ls | awk -v n="serve-$MODEL" '$1 ~ /^[0-9]+\./ {sub(/^[0-9]+\./,"",$1); if ($1==n) found=1} END {exit !found}'
}
session_exists && { echo "session already exists: serve-$MODEL" >&2; exit 2; }
# An already-listening service must not be mistaken for the newly launched one.
python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('0.0.0.0', 8080))
PY
OUT="$HOME/szl-serve/$MODEL"; mkdir -p -- "$OUT"
case "$SRC" in
  *.gguf)
    [ -f "$SRC" ] || { echo 'GGUF file does not exist' >&2; exit 2; }
    command -v llama-server >/dev/null || { echo 'llama-server is not installed' >&2; exit 2; }
    COMMAND=(llama-server -m "$SRC" --host 0.0.0.0 --port 8080)
    ;;
  *) COMMAND=(python3 -m vllm.entrypoints.openai.api_server --model "$SRC" --host 0.0.0.0 --port 8080) ;;
esac
python3 - "$MODEL" "$SRC" "$OUT/boot-receipt.json" <<'PY'
import datetime, json, sys
receipt = {'kind': 'szl.serve-boot-receipt/v1', 'model': sys.argv[1],
           'source': sys.argv[2], 'status': 'LAUNCH_REQUESTED',
           'ts': datetime.datetime.now(datetime.timezone.utc).isoformat()}
with open(sys.argv[3], 'w') as out:
    json.dump(receipt, out, sort_keys=True)
PY
screen -dmS "serve-$MODEL" bash -c 'log=$1; shift; exec "$@" >> "$log" 2>&1' bash "$OUT/serve.log" "${COMMAND[@]}"
for attempt in {1..30}; do
  if ! session_exists; then
    echo "FAILED TO START; check $OUT/serve.log" >&2; exit 3
  fi
  if curl --fail --silent --max-time 2 http://127.0.0.1:8080/health >/dev/null; then
    echo "READY: $MODEL; /health passed. Model quality and throughput remain unverified."
    exit 0
  fi
  sleep 1
done
echo "READINESS UNVERIFIED: $MODEL; startup exceeded the health window. Session retained; log: $OUT/serve.log" >&2
exit 4
