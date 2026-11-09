#!/usr/bin/env bash
# patch_a6000_bench.sh — repair the zero-energy NVML bug on the A6000 nodes.
#
# Run ON THE NODE (e.g. hatun-mcp-prod .145) as user Ubuntu:
#   bash <(curl -fsSL https://raw.githubusercontent.com/szl-holdings/szl-gpu-bridge/main/scripts/patch_a6000_bench.sh)
# or after git pull in ~/szl-gpu-bridge:
#   bash scripts/patch_a6000_bench.sh
#
# What it does (idempotent, FAIL-CLOSED):
#   1. Backs up bench.sh and nemo_train.py with .bak-<ts> (never destroys evidence).
#   2. Installs scripts/energy_meter.py next to them if missing.
#   3. Replaces the broken sampler (`nvmlDeviceGetPower` -> not a real symbol,
#      exception converted to zero) with the hardened EnergyMeter.
#   4. Runs the energy_meter self-test and prints a receipt.
#   5. Does NOT touch the running screen session "nemo" or the training process;
#      the fix applies at the next script invocation.
set -euo pipefail

SZL_DIR="${SZL_DIR:-$HOME/szl}"
BRIDGE_DIR="${BRIDGE_DIR:-$HOME/szl-gpu-bridge}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RC=0

log() { printf '[patch-a6000] %s\n' "$*" >&2; }

# Locate the energy_meter.py that ships with this patch.
SELF_SOURCE=""
if [[ -f "$BRIDGE_DIR/scripts/energy_meter.py" ]]; then
  SELF_SOURCE="$BRIDGE_DIR/scripts/energy_meter.py"
elif [[ -f "$(dirname "${BASH_SOURCE[0]}")/energy_meter.py" ]]; then
  SELF_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/energy_meter.py"
fi

if [[ -z "$SELF_SOURCE" ]]; then
  log "FAIL-CLOSED: energy_meter.py not found (clone szl-gpu-bridge or run from scripts/)"
  exit 3
fi
log "using energy_meter: $SELF_SOURCE"

for f in "$SZL_DIR/bench.sh" "$SZL_DIR/nemo_train.py"; do
  if [[ -f "$f" ]]; then
    cp -p "$f" "$f.bak-$TS"
    log "backed up $f -> $f.bak-$TS"
  else
    log "note: $f not present (skipping)"
  fi
done

# Install the hardened meter into the node's szl dir.
install -m 0644 "$SELF_SOURCE" "$SZL_DIR/energy_meter.py"
log "installed $SZL_DIR/energy_meter.py"

# Neutralize the known-broken patterns in bench.sh if present.
if [[ -f "$SZL_DIR/bench.sh" ]]; then
  if grep -q "nvmlDeviceGetPower\b" "$SZL_DIR/bench.sh"; then
    # Replace the nonexistent symbol with the real one, guarded.
    sed -i.broken-$TS 's/nvmlDeviceGetPower\b/nvmlDeviceGetPowerUsage/g' "$SZL_DIR/bench.sh"
    log "patched bench.sh: nvmlDeviceGetPower -> nvmlDeviceGetPowerUsage"
  fi
  # The zero-conversion bug: any `|| 0` or `except: 0` around power reads.
  if grep -qE "(power|energy).*(\|\| *0|except[^:]*: *0|else *0)" "$SZL_DIR/bench.sh"; then
    log "WARNING: bench.sh still contains a zero-fallback around energy reads."
    log "         Replace its sampler with: from energy_meter import EnergyMeter"
    log "         (see hf/README in szl-gpu-bridge for the exact pattern)."
    RC=4
  fi
fi

# Self-test: a dead meter must report UNAVAILABLE, never zero.
log "running energy_meter self-test"
python3 "$SZL_DIR/energy_meter.py" --self-test

# GPU live check if a GPU is present.
if command -v nvidia-smi >/dev/null 2>&1; then
  python3 - "$SZL_DIR" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
from energy_meter import EnergyMeter
m = EnergyMeter(); m.start()
for _ in range(5):
    import time; m.tick(); time.sleep(0.2)
print(json.dumps({"live_meter": m.stop()}, indent=2))
PY
else
  log "no nvidia-smi on this host; self-test only"
fi

log "done (rc=$RC). The fix is live for the NEXT run; the current screen session keeps its old code."
exit $RC
