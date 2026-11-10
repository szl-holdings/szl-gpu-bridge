#!/usr/bin/env bash
# launch_leg.sh — start one migrated training leg on an A6000 node, the non-stop way.
#
# Usage:  HF_TOKEN=<rotated-token> bash launch_leg.sh <leg-name>
#   e.g.  HF_TOKEN=hf_xxx bash launch_leg.sh szl-brain-v35
#
# The Salad death was heredoc-in-JSON packaging (bash parse death before Python).
# Here every leg is a REAL FILE — no heredocs, no JSON wrapping — and the first
# thing it does is emit a BOOT RECEIPT. No boot receipt within the sweep window
# = stopped = restart (runbook rule, docs/NONSTOP_TRAINING_RUNBOOK.md).
set -euo pipefail
LEG="$1"
[ -f "legs/${LEG}.sh" ] || { echo "no such leg: $LEG"; ls legs/; exit 2; }
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set — the token comes from env, never a file"; exit 2; }
OUT="$HOME/szl-legs/$LEG"
mkdir -p "$OUT"
# BOOT RECEIPT FIRST — prove the leg started before doing anything else
python3 -c "
import json, datetime
r = {'kind': 'szl.boot-receipt/v1', 'leg': '$LEG', 'host': 'a6000',
     'ts': datetime.datetime.now(datetime.timezone.utc).isoformat()}
open('$OUT/boot-receipt.json','w').write(json.dumps(r, sort_keys=True))
print(json.dumps(r, sort_keys=True))"
# one screen session per leg; logs appended; survives disconnect
screen -dmS "$LEG" bash -c "
  export HF_TOKEN JOB=$LEG HF_HOME=$OUT/hf PIP_BREAK_SYSTEM_PACKAGES=1
  bash legs/${LEG}.sh >> $OUT/leg.log 2>&1"
sleep 3
screen -ls | grep "$LEG" && echo "LEG $LEG LAUNCHED — boot receipt at $OUT/boot-receipt.json" \
  || { echo "LEG $LEG FAILED TO START — check legs/${LEG}.sh syntax"; exit 3; }
