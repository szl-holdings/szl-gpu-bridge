#!/usr/bin/env bash
# vllm_smoke_test.sh — Node B vLLM compatibility smoke test (the briefing's #1 evaluation)
#
# Prerequisites: A6000 node with vLLM installed, the pinned Nemotron model.
# Pass: no startup/kernel errors, no OOM, no correctness regression,
#       p95 latency within 10% of baseline.
# Usage: bash vllm_smoke_test.sh [port]
#   [port] selects http://127.0.0.1:<port>/v1/chat/completions (default 8000).
#   SZL_EVAL_OUT overrides the output directory (default $HOME/szl-evals/vllm-smoke).
#   SERVED_MODEL overrides the request model name (default nemotron).
set -euo pipefail
PORT="${1:-8000}"
case "$PORT" in
  ''|*[!0-9]*) echo "port must be a decimal integer, got: $PORT" >&2; exit 2 ;;
esac
MODEL="${NEMOTRON_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16}"
OUT="${SZL_EVAL_OUT:-$HOME/szl-evals/vllm-smoke}"
mkdir -p "$OUT"
# Values reach Python through the environment, never by interpolation into code.
export SMOKE_PORT="$PORT" SMOKE_MODEL="$MODEL" SMOKE_OUT="$OUT" SMOKE_SERVED_MODEL="${SERVED_MODEL:-nemotron}"

# boot receipt first
python3 -c "
import json, datetime, os, socket
r = {'kind': 'szl.eval-boot/v1', 'eval': 'vllm-smoke', 'node': socket.gethostname(),
     'model': os.environ['SMOKE_MODEL'], 'port': int(os.environ['SMOKE_PORT']),
     'ts': datetime.datetime.now(datetime.timezone.utc).isoformat()}
print(json.dumps(r, sort_keys=True))
open(os.path.join(os.environ['SMOKE_OUT'], 'boot.json'), 'w').write(json.dumps(r, sort_keys=True))"

echo "=== GPU inventory ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "=== vLLM version ==="
python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "vllm not installed"

echo "=== Smoke test: 100 fixed prompts at concurrency 1 and 4 ==="
python3 - <<'PYEOF'
import json, os, time, urllib.request, statistics, threading, unicodedata
from concurrent.futures import ThreadPoolExecutor

PORT = int(os.environ["SMOKE_PORT"])
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"
OUT = os.path.join(os.environ["SMOKE_OUT"], "results.json")
SERVED_MODEL = os.environ["SMOKE_SERVED_MODEL"]
# Loopback traffic must not be routed through an HTTP proxy from the environment.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 50 unique prompts x 2 rounds = 100 total
prompts = [
    (c, f"What is the capital of {c}?") for c in
    ["France", "Japan", "Brazil", "Egypt", "Canada", "India", "Kenya", "Peru", "Norway", "Fiji",
     "Chile", "Ghana", "Iran", "Laos", "Mali", "Nepal", "Oman", "Qatar", "Togo", "Wales",
     "Argentina", "Denmark", "Estonia", "Finland", "Georgia", "Hungary", "Iceland", "Jamaica",
     "Kosovo", "Libya", "Monaco", "Niger", "Panama", "Russia", "Sweden", "Turkey", "Uganda",
     "Vietnam", "Yemen", "Zambia", "Albania", "Belize", "Cyprus", "Gabon", "Haiti", "Iraq",
     "Jordan", "Kiribati", "Lebanon", "Moldova"]
] * 2

# Accepted answers, compared after folding case, accents and punctuation.
# Known limit: an answer that only echoes "Monaco" counts as correct.
CAPITALS = {
    "France": ["Paris"], "Japan": ["Tokyo"], "Brazil": ["Brasilia"], "Egypt": ["Cairo"],
    "Canada": ["Ottawa"], "India": ["New Delhi", "Delhi"], "Kenya": ["Nairobi"],
    "Peru": ["Lima"], "Norway": ["Oslo"], "Fiji": ["Suva"], "Chile": ["Santiago"],
    "Ghana": ["Accra"], "Iran": ["Tehran", "Teheran"], "Laos": ["Vientiane"],
    "Mali": ["Bamako"], "Nepal": ["Kathmandu"], "Oman": ["Muscat"], "Qatar": ["Doha"],
    "Togo": ["Lome"], "Wales": ["Cardiff"], "Argentina": ["Buenos Aires"],
    "Denmark": ["Copenhagen"], "Estonia": ["Tallinn"], "Finland": ["Helsinki"],
    "Georgia": ["Tbilisi", "Atlanta"],  # the prompt does not exclude the US state
    "Hungary": ["Budapest"], "Iceland": ["Reykjavik"], "Jamaica": ["Kingston"],
    "Kosovo": ["Pristina", "Prishtina"], "Libya": ["Tripoli"], "Monaco": ["Monaco"],
    "Niger": ["Niamey"], "Panama": ["Panama City"], "Russia": ["Moscow"],
    "Sweden": ["Stockholm"], "Turkey": ["Ankara"], "Uganda": ["Kampala"],
    "Vietnam": ["Hanoi", "Ha Noi"], "Yemen": ["Sanaa", "Sana"], "Zambia": ["Lusaka"],
    "Albania": ["Tirana", "Tirane"], "Belize": ["Belmopan"], "Cyprus": ["Nicosia", "Lefkosia"],
    "Gabon": ["Libreville"], "Haiti": ["Port au Prince"], "Iraq": ["Baghdad"],
    "Jordan": ["Amman"], "Kiribati": ["Tarawa"], "Lebanon": ["Beirut"],
    "Moldova": ["Chisinau", "Kishinev"],
}
assert {c for c, _ in prompts} == set(CAPITALS), "every prompt needs an answer key"


def fold(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return " " + " ".join("".join(ch if ch.isalnum() else " " for ch in text).split()) + " "


def is_correct(country, answer):
    return any(fold(a) in fold(answer) for a in CAPITALS[country])

def one_call(prompt):
    body = json.dumps({
        "model": SERVED_MODEL, "max_tokens": 50, "temperature": 0,
        "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(BASE, data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with OPENER.open(req, timeout=30) as r:
        d = json.load(r)
    return time.monotonic() - t0, d["choices"][0]["message"]["content"] or "", d.get("model")

def run_lane(batch, workers):
    # At most `workers` requests in flight; records the maximum actually observed.
    lock = threading.Lock()
    live = {"now": 0, "max": 0}

    def call(item):
        country, prompt = item
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        try:
            return (country,) + one_call(prompt) + (None,)
        except Exception as e:
            return country, None, None, None, e
        finally:
            with lock:
                live["now"] -= 1

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(call, batch))
    return rows, live["max"], round(time.monotonic() - t0, 3)


def summary(lat):
    return {
        "count": len(lat),
        "p50": round(statistics.median(lat), 3) if lat else None,
        "p95": round(sorted(lat)[int(len(lat)*0.95)], 3) if lat else None,
        "mean": round(statistics.mean(lat), 3) if lat else None,
    }


errors, correct = 0, 0
lanes, served, incorrect = {}, set(), []
# concurrency 1 is sequential; concurrency 4 keeps four requests in flight
for name, batch, workers in (("concurrency_1", prompts[:50], 1),
                             ("concurrency_4", prompts[50:], 4)):
    rows, max_in_flight, wall_s = run_lane(batch, workers)
    latencies = []
    for country, lat, content, model, err in rows:
        if err is not None:
            errors += 1
            print(f"  ERROR ({name}): {type(err).__name__}: {str(err)[:60]}")
            continue
        latencies.append(lat)
        served.add(str(model))
        if is_correct(country, content):
            correct += 1
        else:
            incorrect.append({"lane": name, "country": country, "answer": content[:50]})
    lanes[name] = dict(summary(latencies), workers=workers,
                       max_in_flight=max_in_flight, wall_s=wall_s)

result = {
    "kind": "szl.eval-result/v1",
    "eval": "vllm-smoke",
    "endpoint": BASE,
    "served_model_requested": SERVED_MODEL,
    "served_models_reported": sorted(served),
    "total_calls": len(prompts),
    "unique_prompts": len({p for _, p in prompts}),
    "errors": errors,
    "state": "MEASURED" if errors == 0 else "FAILED",
    "concurrency_1": lanes["concurrency_1"],
    "concurrency_4": lanes["concurrency_4"],
    "correctness": {
        "method": "expected capital appears in the folded answer",
        "checked": correct + len(incorrect),
        "correct": correct,
        "incorrect_count": len(incorrect),
        "incorrect": incorrect[:20],
    },
    "pass_criteria": "errors == 0 AND p95 within 10% of existing baseline",
    "criteria_evaluated": ["errors == 0"],
    "criteria_not_evaluated": [
        "correctness (recorded above; not used by state or verdict)",
        "p95 within 10% of existing baseline (no baseline is loaded)",
        "startup/kernel/OOM errors (server logs are not read)",
    ],
    "verdict": "PASS" if errors == 0 else "FAIL",
}
print(json.dumps(result, indent=2))
with open(OUT + ".tmp", "w") as f:
    json.dump(result, f, indent=2)
os.replace(OUT + ".tmp", OUT)
PYEOF

echo "=== Smoke test complete. Results: $OUT/results.json ==="
echo "Verdict: $(python3 -c "import json, os; print(json.load(open(os.path.join(os.environ['SMOKE_OUT'], 'results.json')))['verdict'])" 2>/dev/null || echo 'check logs')"
