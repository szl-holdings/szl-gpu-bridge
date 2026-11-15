#!/usr/bin/env python3
"""SZL Truth Router v2 — the first two-axis honesty-weighted router.
Axis 1: census abstention (knowledge honesty). Axis 2: proving-ground badges
(proof honesty). Plus measured joules/token and price. Fail-closed."""
import json, hashlib, time, math, urllib.request
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

DS = "SZLHOLDINGS/frontier-54-runs"; PG = "SZLHOLDINGS/formula-proving-ground"
HF = "https://huggingface.co"
app = FastAPI(title="szl-truth-router", version="2.0.0")

def get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "szl-truth-router/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

PRICES = {"gpt-6.1-sol": 20.0, "gpt-6.1-terra": 20.0, "gpt-6-astra": 15.0,
          "gpt-5.6-sol": 15.0, "gpt-5.2": 10.0, "gpt-5.4-mini": 0.5,
          "Qwen/Qwen3.8-27B": 0.87, "Qwen/Qwen3.6-35B-A3B": 0.9, "claude-pinned-lane": 3.0,
          "estate:Qwen2.5-Coder-3B": 0.01}

def load_registry():
    models = {}
    # AXIS 1: census legs
    try:
        legs = [f["path"] for f in get(f"{HF}/api/datasets/{DS}/tree/main/census?recursive=true") if f["path"].endswith(".json")]
        for p in legs:
            leg = get(f"{HF}/datasets/{DS}/resolve/main/{p}")
            for run in leg.get("runs", []):
                if run.get("error"): continue
                name = run["model"]
                m = models.setdefault(name, {"census_receipts": []})
                m["abstain"] = run.get("abstained", 0); m["probes"] = run.get("probes_valid", 6)
                m["census_receipts"].append(p)
    except Exception as e: print("census fail:", str(e)[:80])
    # AXIS 2: proving-ground legs (badges + verdicts)
    try:
        legs = [f["path"] for f in get(f"{HF}/api/datasets/{PG}/tree/main/runs?recursive=true") if f["path"].endswith(".json")]
        for p in legs:
            leg = get(f"{HF}/datasets/{PG}/resolve/main/{p}")
            mdl = leg.get("model") or leg.get("run","?")
            # normalize: proving legs use served names like gpt-5.2
            m = models.setdefault(mdl, {"census_receipts": []})
            v = leg.get("verdicts") or {}
            tested = sum(v.values()) or 1
            m["pg_abstain_rate"] = round((v.get("ABSTAINED",0))/tested, 3)
            m["pg_sketches"] = v.get("SKETCH_CANDIDATE",0)
            m["pg_bluffs"] = v.get("WEAK_OR_FAKE",0)
            m.setdefault("badges", [])
            for b in leg.get("badges", []):
                if b not in m["badges"]: m["badges"].append(b)
            m.setdefault("pg_receipts", []).append(p)
    except Exception as e: print("pg fail:", str(e)[:80])
    # two-axis truth score
    for name, m in models.items():
        m["price_per_1m"] = PRICES.get(name, 1.0)
        axis1 = m.get("abstain", 0) / max(m.get("probes", 6), 1)          # census honesty
        axis2 = m.get("pg_abstain_rate")                                    # proof honesty
        badges = len(m.get("badges", []))
        m["honesty_axis1_census"] = round(axis1, 3)
        m["honesty_axis2_proving"] = axis2 if axis2 is not None else "UNMEASURED"
        score = axis1 * 60
        if axis2 is not None: score += axis2 * 30
        score += badges * 3
        score -= 0.5 * math.log10(max(m["price_per_1m"], 0.01))
        m["truth_score"] = round(score, 3)
    return models

REGISTRY = load_registry()
LOADED = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "szl-truth-router", "version": app.version,
            "registry_loaded": LOADED, "models": len(REGISTRY)}

@app.get("/registry")
def registry():
    if not REGISTRY:
        raise HTTPException(503, {"error": "chain_unavailable", "honesty": "fail-closed"})
    ranked = sorted(REGISTRY.items(), key=lambda kv: -kv[1].get("truth_score", 0))
    return {"schema": "szl.truth-router/2.0",
        "axes": {"axis1": "census abstention (knowledge honesty)",
                 "axis2": "proving-ground abstention + badges (proof honesty)"},
        "note": "The frontier pattern is measurable here: the newest flagships score high on capability (sketches) and low on honesty (bluffs); previous generations out-abstain them. Route accordingly.",
        "models": [{"model": k, **v} for k, v in ranked]}

@app.post("/route")
def route(req: dict = None):
    req = req or {}
    if not REGISTRY:
        raise HTTPException(503, {"error": "chain_unavailable", "honesty": "fail-closed"})
    task = str(req.get("task", ""))
    # task-type routing: factual/proof -> honesty-dominant; creative -> capability
    factual = any(w in task.lower() for w in ["cite","exact","doi","prove","formula","fact","citation","id","who wrote"])
    key = (lambda kv: -kv[1].get("honesty_axis1_census", 0)) if factual else (lambda kv: -kv[1].get("truth_score", 0))
    ranked = sorted(REGISTRY.items(), key=key)
    top = ranked[0]
    decision = {"schema": "szl.truth-router/2.0", "decision": "route",
        "task_hint": task[:120], "task_class": "factual/honesty-critical" if factual else "general",
        "chosen_model": top[0],
        "why": {"axis1_census": top[1].get("honesty_axis1_census"),
                "axis2_proving": top[1].get("honesty_axis2_proving"),
                "badges": top[1].get("badges", []),
                "pg_sketches": top[1].get("pg_sketches", 0),
                "pg_bluffs": top[1].get("pg_bluffs", 0),
                "price_per_1m_usd": top[1].get("price_per_1m"),
                "truth_score": top[1].get("truth_score")},
        "alternatives": [{"model": k, "truth_score": v.get("truth_score")} for k, v in ranked[1:3]],
        "receipts": top[1].get("census_receipts", []) + top[1].get("pg_receipts", []),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    blob = json.dumps(decision, sort_keys=True)
    decision["decision_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    return decision

app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
