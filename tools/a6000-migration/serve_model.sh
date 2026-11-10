#!/usr/bin/env bash
# serve_model.sh — deploy one estate model natively on an A6000 node.
#
# Usage:  bash serve_model.sh <model-name> [gguf-path-or-hf-repo]
# The Massed way: no containers, no restart_policy masquerade —
# boot receipt FIRST, screen session, health endpoint, mesh heartbeat.
set -euo pipefail
MODEL="$1"; SRC="${2:-}"
OUT="$HOME/szl-serve/$MODEL"; mkdir -p "$OUT"

# 1. BOOT RECEIPT FIRST — prove the intent before anything else
python3 -c "
import json, datetime
r = {'kind': 'szl.serve-boot-receipt/v1', 'model': '$MODEL', 'source': '$SRC',
     'ts': datetime.datetime.now(datetime.timezone.utc).isoformat()}
open('$OUT/boot-receipt.json', 'w').write(json.dumps(r, sort_keys=True))
print(json.dumps(r, sort_keys=True))"

# 2. Pick the serving stack by artifact type
case "$SRC" in
  *.gguf)
    # llama.cpp server (the szl-serve honest path)
    command -v llama-server >/dev/null || { echo "llama-server not installed — see szl-serve recipe"; exit 2; }
    SCREEN_CMD="llama-server -m $SRC --host 0.0.0.0 --port 8080"
    ;;
  *)
    # HF checkpoint → vLLM (frontier-bench measures it)
    command -v python3 >/dev/null || exit 2
    SCREEN_CMD="python3 -m vllm.entrypoints.openai.api_server --model $SRC --host 0.0.0.0 --port 8080"
    ;;
esac

# 3. Screen session: survives disconnect; logs to disk
screen -dmS "serve-$MODEL" bash -c "$SCREEN_CMD >> $OUT/serve.log 2>&1"
sleep 5
if screen -ls | grep -q "serve-$MODEL"; then
  echo "SERVING: $MODEL (screen serve-$MODEL, logs $OUT/serve.log)"
  echo "boot receipt: $OUT/boot-receipt.json"
  echo "health: curl -s localhost:8080/health"
  echo "liveness truth: the mesh agent's hourly heartbeat — a dead server shows dead"
else
  echo "FAILED TO START — check $OUT/serve.log"; exit 3
fi
