#!/usr/bin/env bash
# train_and_gate.sh - run a training job, then battery-gate its candidates.
# Every training run ends in a MEASURED receipt on-chain, or fails loudly.
#
# Usage: HF_TOKEN=... ./train_and_gate.sh <train command...> -- <battery_spec.json>
# Example:
#   HF_TOKEN=${HF_TOKEN} ./train_and_gate.sh bash train_all.sh -- battery_spec.json
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
SPEC=""
HAVE_CMD=0
CMD=()
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; SPEC="${1:-}"; break ;;
    *) CMD+=("$1"); HAVE_CMD=1; shift ;;
  esac
done
if [ "$HAVE_CMD" -eq 0 ] || [ -z "$SPEC" ]; then
  echo "usage: train_and_gate.sh <train command...> -- <battery_spec.json>" >&2
  exit 2
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN missing: fail-closed, refusing to run unreceipted" >&2
  exit 2
fi
"${CMD[@]}"
python3 "$here/battery_gate.py" --spec "$SPEC"
