#!/usr/bin/env bash
# Launch an existing leg from any working directory. No GPU work is started
# until the bounded BF16 preflight passes; a session is not a training result.
set -euo pipefail
LEG="${1:-}"
[[ "$LEG" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || { echo 'invalid leg name' >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LEG_SCRIPT="$SCRIPT_DIR/legs/$LEG.sh"
[ -f "$LEG_SCRIPT" ] || { echo "no such leg: $LEG" >&2; exit 2; }
[ -n "${HF_TOKEN:-}" ] || { echo 'HF_TOKEN must be supplied in the environment' >&2; exit 2; }
command -v screen >/dev/null || { echo 'screen is not installed' >&2; exit 2; }
if screen -ls | awk -v n="$LEG" '$1 ~ /^[0-9]+\./ {sub(/^[0-9]+\./,"",$1); if ($1==n) found=1} END {exit !found}'; then
  echo "session already exists: $LEG" >&2; exit 2
fi
OUT="$HOME/szl-legs/$LEG"
mkdir -p -- "$OUT"
python3 "$SCRIPT_DIR/../../laptop/gpu_preflight.py" --require-bf16 > "$OUT/preflight.json"
python3 - "$LEG" "$OUT/boot-receipt.json" <<'PY'
import datetime, json, sys
receipt = {'kind': 'szl.boot-receipt/v1', 'leg': sys.argv[1],
           'status': 'LAUNCH_REQUESTED', 'training_verified': False,
           'ts': datetime.datetime.now(datetime.timezone.utc).isoformat()}
with open(sys.argv[2], 'w') as out:
    json.dump(receipt, out, sort_keys=True)
print(json.dumps(receipt, sort_keys=True))
PY
# Credentials are inherited environment values, never interpolated into argv.
HF_TOKEN_INJECT="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
HF_HOME="$OUT/hf" JOB="$LEG" PIP_BREAK_SYSTEM_PACKAGES=1 \
  screen -dmS "$LEG" bash -c 'exec bash "$1" >> "$2" 2>&1' bash "$LEG_SCRIPT" "$OUT/leg.log"
sleep 3
if screen -ls | awk -v n="$LEG" '$1 ~ /^[0-9]+\./ {sub(/^[0-9]+\./,"",$1); if ($1==n) found=1} END {exit !found}'; then
  echo "LEG $LEG LAUNCHED; training success remains unverified. Log: $OUT/leg.log"
else
  echo "LEG $LEG FAILED TO START; check $OUT/leg.log" >&2; exit 3
fi
