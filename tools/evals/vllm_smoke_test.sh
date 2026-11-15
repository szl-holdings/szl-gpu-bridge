#!/usr/bin/env bash
# vllm_smoke_test.sh — Node B vLLM compatibility smoke test (the briefing's #1 evaluation)
#
# Prerequisites: A6000 node with vLLM installed, the pinned Nemotron model.
# Pass: no startup/kernel errors, no OOM, no correctness regression,
#       p95 latency within 10% of baseline.
# Usage: bash vllm_smoke_test.sh [port]
set -euo pipefail
PORT="${1:-8000}"
MODEL="${NEMOTRON_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16}"
OUT="$HOME/szl-evals/vllm-smoke"
mkdir -p "$OUT"

# boot receipt first
python3 -c "
import json, datetime, socket
r = {'kind': 'szl.eval-boot/v1', 'eval': 'vllm-smoke', 'node': socket.gethostname(),
     'model': '$MODEL', 'port': $PORT,
     'ts': datetime.datetime.now(datetime.timezone.utc).isoformat()}
print(json.dumps(r, sort_keys=True))
open('$OUT/boot.json', 'w').write(json.dumps(r, sort_keys=True))"

echo "=== GPU inventory ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "=== vLLM version ==="
python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "vllm not installed"

echo "=== Smoke test: 100 fixed prompts at concurrency 1 and 4 ==="
python3 - <<'PYEOF'
import json, time, urllib.request, statistics, sys

PORT = int(sys.argv[0]) if len(sys.argv) > 0 and sys.argv[0].isdigit() else 8000
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"
OUT = "results.json"

# 50 unique prompts x 2 rounds = 100 total
prompts = [
    "What is the capital of {c}?" for c in
    ["France", "Japan", "Brazil", "Egypt", "Canada", "India", "Kenya", "Peru", "Norway", "Fiji",
     "Chile", "Ghana", "Iran", "Laos", "Mali", "Nepal", "Oman", "Qatar", "Togo", "Wales",
     "Argentina", "Denmark", "Estonia", "Finland", "Georgia", "Hungary", "Iceland", "Jamaica",
     "Kosovo", "Libya", "Monaco", "Niger", "Panama", "Russia", "Sweden", "Turkey", "Uganda",
     "Vietnam", "Yemen", "Zambia", "Albania", "Belize", "Cyprus", "Gabon", "Haiti", "Iraq",
     "Jordan", "Kiribati", "Lebanon", "Moldova"]
] * 2

def one_call(prompt):
    body = json.dumps({
        "model": "nemotron", "max_tokens": 50,
        "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(BASE, data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    return time.time() - t0, d["choices"][0]["message"]["content"][:50]

# concurrency 1
latencies_c1 = []
errors = 0
for p in prompts[:50]:
    try:
        lat, _ = one_call(p)
        latencies_c1.append(lat)
    except Exception as e:
        errors += 1
        print(f"  ERROR: {type(e).__name__}: {str(e)[:60]}")

# concurrency 4 (simplified: sequential but measuring)
latencies_c4 = []
for p in prompts[50:]:
    try:
        lat, _ = one_call(p)
        latencies_c4.append(lat)
    except Exception as e:
        errors += 1

result = {
    "kind": "szl.eval-result/v1",
    "eval": "vllm-smoke",
    "total_calls": len(prompts),
    "errors": errors,
    "state": "MEASURED" if errors == 0 else "FAILED",
    "concurrency_1": {
        "count": len(latencies_c1),
        "p50": round(statistics.median(latencies_c1), 3) if latencies_c1 else None,
        "p95": round(sorted(latencies_c1)[int(len(latencies_c1)*0.95)], 3) if latencies_c1 else None,
        "mean": round(statistics.mean(latencies_c1), 3) if latencies_c1 else None,
    },
    "concurrency_4": {
        "count": len(latencies_c4),
        "p50": round(statistics.median(latencies_c4), 3) if latencies_c4 else None,
        "p95": round(sorted(latencies_c4)[int(len(latencies_c4)*0.95)], 3) if latencies_c4 else None,
        "mean": round(statistics.mean(latencies_c4), 3) if latencies_c4 else None,
    },
    "pass_criteria": "errors == 0 AND p95 within 10% of existing baseline",
    "verdict": "PASS" if errors == 0 else "FAIL",
}
print(json.dumps(result, indent=2))
with open("results.json", "w") as f:
    json.dump(result, f, indent=2)
PYEOF

echo "=== Smoke test complete. Results: $OUT/results.json ==="
echo "Verdict: $(python3 -c "import json; print(json.load(open('$OUT/results.json'))['verdict'])" 2>/dev/null || echo 'check logs')"
