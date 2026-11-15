#!/usr/bin/env bash
# aeroff_proving_ground.sh — 24-hour proving ground for the Aero-FF claims.
#
# Run on a node:  bash aeroff_proving_ground.sh 2>&1 | tee aeroff-pg.log
# Emits a boot receipt FIRST, then one khipu-style receipt per cycle.
# Fail closed: any stage that cannot verify reports UNAVAILABLE, never zero.
set -euo pipefail
OUT="$HOME/aeroff-pg"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERDICT="$HERE/lean_axiom_verdict.py"
mkdir -p "$OUT/logs"
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }
RECEIPT() { python3 -c "
import json,sys,hashlib,datetime
kind=sys.argv[1]; stage=sys.argv[2]; state=sys.argv[3]; detail=sys.argv[4]
r={'kind':'szl.proving-ground/v1','subject':'aero-ff','stage':stage,'state':state,
   'detail':detail[:200],'ts':datetime.datetime.now(datetime.timezone.utc).isoformat()}
line=json.dumps(r,sort_keys=True)
sha=hashlib.sha256(line.encode()).hexdigest()
r['sha256']=sha
final=json.dumps(r,sort_keys=True)
print(final); open('$OUT/receipts.jsonl','a').write(final+'\n')" "$@"; }

RECEIPT boot "proving-ground" "BOOTED" "24h Aero-FF proving ground started"

# ---- Stage 1: clone the canonical sources (receipted) ----
cd "$OUT"
[ -d lutar-lean ] && git -C lutar-lean pull -q || git clone -q https://github.com/szl-holdings/lutar-lean.git
[ -d acoustic-validation ] && git -C acoustic-validation pull -q || git clone -q https://github.com/szl-holdings/acoustic-validation.git
LEANSHA=$(git -C lutar-lean rev-parse HEAD)
ACSHA=$(git -C acoustic-validation rev-parse HEAD)
RECEIPT sources "clone" "MEASURED" "lutar-lean@$LEANSHA acoustic-validation@$ACSHA"

# ---- Stage 2: the numerics battery (deterministic, CPU) ----
# Every Chad-claim that is a formula gets computed and printed with its inputs.
python3 - <<'PY' 2>&1 | tee "$OUT/logs/numerics.log"
import math, json
# Inputs from the disclosure (M1 system): PVC cone E=2.9GPa, nu=0.38
E, nu = 2.9e9, 0.38
rho = 1400.0  # PVC kg/m^3 (nominal; recorded as INPUT_ASSUMED)
# Rayleigh speed: the physical root of the Rayleigh equation
#   xi^3 - 8 xi^2 + (24 - 16 k) xi - 16 (1 - k) = 0,
#   xi = (cR/cS)^2,  k = (cS/cP)^2 = (1 - 2 nu) / (2 (1 - nu)),
# found by bisection on (0, 1). The Viktorov closed form
# (0.87 + 1.12 nu) / (1 + nu) is printed as a cross-check; it has no square root.
cS = math.sqrt(E/(2*rho*(1+nu)))          # shear speed from assumed modulus
k = (1 - 2*nu)/(2*(1 - nu))
f = lambda x: x**3 - 8*x**2 + (24 - 16*k)*x - 16*(1 - k)
lo, hi = 0.0, 1.0
for _ in range(200):
    mid = (lo + hi)/2
    if f(mid) > 0: hi = mid
    else: lo = mid
ratio = math.sqrt((lo + hi)/2)
cR = cS * ratio
print(json.dumps({"stage":"numerics","rayleigh_cR_mps":cR,"cS_mps":cS,
                  "cR_over_cS_exact":ratio,"cR_over_cS_viktorov":(0.87 + 1.12*nu)/(1 + nu),
                  "inputs":{"E_Pa":E,"nu":nu,"rho_assumed":rho},
                  "state":"COMPUTED","note":"rho is INPUT_ASSUMED, not measured"}))
# EP gain G = 2/(1-|R|^2) for |R| in [0.3,0.7]
for R in (0.3, 0.5, 0.7):
    G = 2/(1-R*R)
    print(json.dumps({"stage":"numerics","ep_gain_G":round(G,4),"absR":R,
                      "power_gain_dB":round(10*math.log10(G),2),
                      "amplitude_dB":round(20*math.log10(G),2),"state":"COMPUTED"}))
# TOF separation d(1/cR - 1/c0) per meter, c0=343 m/s
c0 = 343.0
print(json.dumps({"stage":"numerics","tof_separation_per_m_us":
                  round(1e6*(1/cR-1/c0),2),"c0_mps":c0,"state":"COMPUTED"}))
PY
RECEIPT numerics "formulas" "COMPUTED" "Rayleigh/EP-gain/TOF battery done; see logs/numerics.log"

# ---- Stage 3: the falsifiable acoustic-validation suite ----
cd "$OUT/acoustic-validation"
if ! python3 -m pytest --version >/dev/null 2>&1; then
  RECEIPT acoustic "suite" "UNAVAILABLE" "pytest is not installed on this node; suite not run at $ACSHA"
else
  set +e
  python3 -m pytest -q 2>&1 | tee "$OUT/logs/acoustic.log"; AVRC=${PIPESTATUS[0]}
  set -e
  AVSUM=$(tail -1 "$OUT/logs/acoustic.log" | tr -d '=' | cut -c1-100)
  if [ "$AVRC" -eq 0 ]; then
    RECEIPT acoustic "suite" "PASS" "pytest exit 0 at $ACSHA:$AVSUM"
  elif [ "$AVRC" -eq 5 ]; then
    RECEIPT acoustic "suite" "UNAVAILABLE" "pytest collected no tests at $ACSHA"
  else
    RECEIPT acoustic "suite" "FAIL" "pytest exit $AVRC at $ACSHA:$AVSUM"
  fi
fi

# ---- Stage 4: Lean — prove the base case, explore the open n-axis ----
cd "$OUT/lutar-lean"
RECEIPT lean "build" "STARTED" "lake build beginning (mathlib compile is the long pole)"
if lake build 2>&1 | tee "$OUT/logs/lean-build.log"; then
  RECEIPT lean "build" "MEASURED" "lake build exit 0 at $LEANSHA"
  # Verdicts come from the kernel (#print axioms), not from the theorem's text:
  # a sorry proof or an added axiom also builds and also matches a grep.
  for pair in "axiom3-2axis:Lutar.Lambda.lambda_two_axis_schur_concave" \
              "axiom3-naxis:Lutar.Lambda.lambda_schur_concave_n_axis"; do
    STAGE="${pair%%:*}"; DECL="${pair#*:}"
    printf 'import Lutar.Lambda.SchurConcave\n#print axioms %s\n' "$DECL" > "$OUT/logs/audit-$STAGE.lean"
    set +e
    lake env lean "$OUT/logs/audit-$STAGE.lean" > "$OUT/logs/audit-$STAGE.log" 2>&1; ARC=$?
    V=$(python3 "$VERDICT" 0 "$ARC" "$OUT/logs/audit-$STAGE.log" "$DECL")
    set -e
    VSTATE=$(printf '%s' "$V" | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["state"])
except Exception: print("UNAVAILABLE")')
    VDETAIL=$(printf '%s' "$V" | python3 -c 'import json,sys
try:
    v=json.load(sys.stdin); print("axioms=%s; %s" % (v["axioms"], v["reason"]))
except Exception: print("verdict tool produced no result")')
    RECEIPT lean "$STAGE" "$VSTATE" "$VDETAIL"
  done
else
  RECEIPT lean "build" "UNAVAILABLE" "build failed — verdicts WITHHELD (grep on a failed build proves nothing)"
fi

# ---- Stage 5: 24h loop — re-verify every hour, receipt each cycle ----
RECEIPT loop "24h" "STARTED" "hourly Hub reachability probe for 24 cycles"
for i in $(seq 1 24); do
  sleep 3600
  # Reachability probe only: the stages above are NOT re-run and commit growth is not compared.
  N=$(python3 -c "
import urllib.request, json
try:
    with urllib.request.urlopen('https://huggingface.co/api/datasets/SZLHOLDINGS/frontier-54-runs/commits/main', timeout=20) as r:
        print(len(json.load(r)))
except Exception: print(-1)")
  if [ "$N" -gt 0 ]; then STATE="REACHABLE first_page_commits=$N"; else STATE="UNAVAILABLE"; fi
  RECEIPT cycle "$i/24" "$STATE" "hourly Hub reachability probe; stages not re-run, growth not compared"
done
RECEIPT loop "24h" "COMPLETE" "24-hour proving ground complete; receipts in $OUT/receipts.jsonl"
echo "DONE. Receipts: $OUT/receipts.jsonl"
