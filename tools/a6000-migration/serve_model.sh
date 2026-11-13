#!/usr/bin/env bash
# serve_model.sh — deploy one estate model natively on an A6000 node.
#
# Usage:  bash serve_model.sh <model-name> [gguf-path-or-hf-repo]
#
# Bind: HOST defaults to 127.0.0.1. Set HOST to listen somewhere else.
# Loopback is localhost, ::1, or any address in 127.0.0.0/8. Any other HOST
# is refused unless SZL_API_KEY is set. When that key is set it is passed
# to both llama-server and vLLM as --api-key.
#
# The Massed way: no containers, no restart_policy masquerade —
# boot receipt FIRST, screen session, health endpoint, mesh heartbeat.
set -euo pipefail
MODEL="${1:?usage: bash serve_model.sh <model-name> [gguf-path-or-hf-repo]}"
SRC="${2:-}"
HOST="${HOST:-127.0.0.1}"

# True for names and addresses that cannot accept remote connections.
loopback_bind() {
  local host="$1"
  local octet
  case "$host" in
    localhost | ::1 | "[::1]")
      return 0
      ;;
  esac
  [[ "$host" =~ ^127\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]] || return 1
  for octet in "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"; do
    if ((10#$octet > 255)); then
      return 1
    fi
  done
  return 0
}

bind_args=(--host "$HOST" --port 8080)
if [[ -n "${SZL_API_KEY:-}" ]]; then
  bind_args+=(--api-key "$SZL_API_KEY")
elif ! loopback_bind "$HOST"; then
  echo "Refusing to listen on ${HOST}: set SZL_API_KEY when HOST is not loopback (127.0.0.0/8, localhost, or ::1)." >&2
  exit 2
fi

OUT="$HOME/szl-serve/$MODEL"
mkdir -p "$OUT"

# 1. BOOT RECEIPT — prove the intent before the server starts. A refused bind
#    (non-loopback HOST without SZL_API_KEY) exits above and writes nothing.
python3 - "$MODEL" "$SRC" "$OUT/boot-receipt.json" <<'PY_RECEIPT'
import json
import datetime
import sys

r = {'kind': 'szl.serve-boot-receipt/v1', 'model': sys.argv[1], 'source': sys.argv[2],
     'ts': datetime.datetime.now(datetime.timezone.utc).isoformat()}
with open(sys.argv[3], 'w') as output:
    output.write(json.dumps(r, sort_keys=True))
print(json.dumps(r, sort_keys=True))
PY_RECEIPT

# 2. Pick the serving stack by artifact type
case "$SRC" in
  *.gguf)
    # llama.cpp server (the szl-serve honest path)
    command -v llama-server >/dev/null || { echo "llama-server not installed — see szl-serve recipe" >&2; exit 2; }
    cmd=(llama-server -m "$SRC" "${bind_args[@]}")
    ;;
  *)
    # HF checkpoint → vLLM (frontier-bench measures it)
    command -v python3 >/dev/null || exit 2
    cmd=(python3 -m vllm.entrypoints.openai.api_server --model "$SRC" "${bind_args[@]}")
    ;;
esac

# 3. Screen session: survives disconnect; logs to disk.
# The server command is a separate argument list so the model path and
# --api-key value are not word-split. The bash -c body is single-quoted so
# $1 and $@ expand in the session, not in this process.
# shellcheck disable=SC2016
screen -dmS "serve-$MODEL" bash -c 'log="$1"; shift; exec "$@" >> "$log" 2>&1' _ "$OUT/serve.log" "${cmd[@]}"
sleep 5
if screen -ls | grep -q "serve-$MODEL"; then
  echo "SERVING: $MODEL (screen serve-$MODEL, logs $OUT/serve.log)"
  echo "boot receipt: $OUT/boot-receipt.json"
  case "$HOST" in
    \[*\]) health_host="$HOST" ;;
    *:*) health_host="[$HOST]" ;;
    *) health_host="$HOST" ;;
  esac
  echo "bind: ${health_host}:8080"
  echo "health: curl -s http://${health_host}:8080/health"
  echo "liveness truth: the mesh agent's hourly heartbeat — a dead server shows dead"
else
  echo "FAILED TO START — check $OUT/serve.log"
  exit 3
fi
