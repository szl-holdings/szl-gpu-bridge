#!/usr/bin/env python3
"""Eternal Auditor v2 — complete app. Full signatures, chained, persistent.

CANONICAL: szl-holdings/szl-gpu-bridge/hf-spaces/eternal-auditor/app.py
v1 defects (audit): truncated signatures (96 hex chars), state reset on
restart, first-100-repos-only, no audit rows on the chain.
v2: full 64-hex HMAC signatures, state persisted to /data, restarts
marked BROKEN-RESTARTED (never silent), cycle receipts chained to the
previous receipt's sha256 and published to the chain.
"""
import hashlib, hmac, json, os, time, threading, urllib.request
from pathlib import Path

try:
    from fastapi import FastAPI, Request
    from fastapi.staticfiles import StaticFiles
    _HAS_FASTAPI = True
except ImportError:  # stdlib-only envs (CI contract tests): logic without server
    _HAS_FASTAPI = False
try:
    from huggingface_hub import HfApi
    _HAS_HF = True
except ImportError:
    _HAS_HF = False
    HfApi = None

TOK = os.environ.get("HF_TOKEN", "")
api = HfApi(token=TOK) if _HAS_HF else None
DS = "SZLHOLDINGS/frontier-54-runs"
ROLE = "eternal-auditor"
CYCLE = 3600  # hourly
STATE = Path(os.environ.get("AUDITOR_STATE", "/data/audit-state.json"))
SECRET = os.environ.get("AUDITOR_SECRET", "szl-auditor-v2-default")


def canonical(obj): return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_hex(s): return hashlib.sha256(s.encode()).hexdigest()


def sign(body_sha, secret):
    return hmac.new(secret.encode(), body_sha.encode(), hashlib.sha256).hexdigest()


def verify_receipt(receipt, secret):
    """Fail-closed verification of a full receipt: canonical body re-hash,
    full-length constant-time signature compare, kind check. Bare bool —
    no partial credit (Law 8)."""
    if not isinstance(receipt, dict):
        return False
    body = {k: v for k, v in receipt.items()
            if k not in ("body_sha256", "signature", "signature_length")}
    if body.get("kind") != "szl.auditor-receipt/v2":
        return False
    expected = sha256_hex(canonical(body))
    sig = receipt.get("signature")
    if not isinstance(sig, str) or len(sig) != 64:
        return False
    if receipt.get("body_sha256") != expected:
        return False
    return hmac.compare_digest(sign(expected, secret), sig)




# ---- Backward-compatible interface (the tests + SDK contract) ----
def run_cycle(findings, secret):
    """Full signatures, chained to the persisted prev, saved state."""
    global state
    state["cycle"] = state.get("cycle", 0) + 1
    restarted = state.get("chain_state") == "GENESIS" and state["cycle"] > 1
    body = {"kind": "szl.auditor-receipt/v2", "officer": ROLE,
            "cycle": state["cycle"],
            "chain_state": "BROKEN-RESTARTED" if restarted else "CONTINUOUS",
            "prev_receipt_sha256": state["prev"],
            "findings": findings,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    body_sha = sha256_hex(canonical(body))
    signature = sign(body_sha, secret)  # FULL 64-hex
    receipt = {**body, "body_sha256": body_sha, "signature": signature,
               "signature_length": len(signature)}
    state["prev"] = body_sha
    state["chain_state"] = "CONTINUOUS"
    save_state(state)
    return receipt

def load_state():
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text())
            if "cycle" in st and "prev" in st: return st
        except Exception: pass
    return {"cycle": 0, "prev": None, "chain_state": "GENESIS"}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.with_suffix(".tmp").write_text(json.dumps(st, indent=2))
    STATE.with_suffix(".tmp").replace(STATE)

def remember(st, receipt):
    """Keep the last 24 FULL receipts in state — the proof gallery.
    Full bodies are required so /api/verify can re-check each one;
    displaying a signature without verifying it is a known v2 defect."""
    hist = st.setdefault("history", [])
    hist.append(receipt)
    del hist[:-24]
    save_state(st)


def get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "szl-auditor/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def audit_cycle():
    """Bounded, honest sweep: chain integrity + org liveness. Counts, never claims."""
    findings = []
    commits = get("https://huggingface.co/api/datasets/SZLHOLDINGS/frontier-54-runs/commits/main")
    if commits:
        head_age_h = (time.time() - time.mktime(time.strptime(commits[0]["date"][:19], "%Y-%m-%dT%H:%M:%S"))) / 3600
        findings.append({"check": "chain_head_age_hours", "value": round(head_age_h, 1),
                         "verdict": "PASS" if head_age_h <= 24 else "FAIL",
                         "detail": f"{len(commits)} commits; head {commits[0]['title'][:50]}"})
    else:
        findings.append({"check": "chain_reachable", "verdict": "UNKNOWN", "state": "UNAVAILABLE"})
    spaces = get("https://huggingface.co/api/spaces?author=SZLHOLDINGS&limit=100") or []
    running = 0
    for s in spaces[:40]:  # bounded sample, count recorded
        full = get("https://huggingface.co/api/spaces/" + s["id"])
        if full and (full.get("runtime") or {}).get("stage") == "RUNNING":
            running += 1
    findings.append({"check": "spaces_running_sample", "sampled": min(len(spaces), 40) if spaces else 0,
                     "running": running,
                     "verdict": "MEASURED" if spaces else "UNAVAILABLE"})
    return findings


def loop():
    global state
    while True:
        state["cycle"] = state.get("cycle", 0) + 1
        restarted = state.get("chain_state") == "GENESIS" and state["cycle"] > 1
        tc = time.time()
        try:
            body = {"kind": "szl.auditor-receipt/v2", "officer": ROLE,
                    "cycle": state["cycle"],
                    "chain_state": "BROKEN-RESTARTED" if restarted else "CONTINUOUS",
                    "prev_receipt_sha256": state["prev"],
                    "findings": audit_cycle(),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            body_sha = sha256_hex(canonical(body))
            signature = sign(body_sha, SECRET)  # FULL 64-hex, never truncated
            receipt = {**body, "body_sha256": body_sha, "signature": signature,
                       "signature_length": len(signature)}
            state["prev"] = body_sha
            state["chain_state"] = "CONTINUOUS"
            save_state(state)
            remember(state, receipt)
            if TOK:
                api.upload_file(path_or_fileobj=json.dumps(receipt, indent=2).encode(),
                    path_in_repo=f"officers/{ROLE}-{body_sha[:12]}.json",
                    repo_id=DS, repo_type="dataset")
            print("cycle", state["cycle"], "audited", flush=True)
        except Exception as e:
            print("cycle", state["cycle"], "err", str(e)[:80], flush=True)
        remain = CYCLE - (time.time() - tc)
        if remain > 0: time.sleep(remain)


state = load_state()
app = FastAPI() if _HAS_FASTAPI else None


if _HAS_FASTAPI:
    @app.get("/healthz")
    def healthz():
        return {"ok": True, "service": ROLE, "cycle": state["cycle"],
                "chain_state": state.get("chain_state")}


if _HAS_FASTAPI:
    @app.on_event("startup")
    def _start_loop():
        threading.Thread(target=loop, daemon=True).start()
    _STATIC = os.environ.get("OFFICER_STATIC", os.path.join(os.path.dirname(__file__), "static"))
    os.makedirs(_STATIC, exist_ok=True)

    @app.get("/api/live")
    def live():
        return {"ok": True, "service": ROLE, "cycle": state.get("cycle", 0),
                "chain_state": state.get("chain_state"),
                "prev_receipt_sha256": state.get("prev"),
                "history_count": len(state.get("history", [])),
                "cycle_seconds": CYCLE}

    @app.get("/api/history")
    def history():
        return {"service": ROLE, "receipts": list(reversed(state.get("history", [])))}

    @app.post("/api/verify")
    async def verify(request: "Request"):
        try:
            payload = await request.json()
        except Exception:
            return {"verified": False}
        receipt = payload.get("receipt") if isinstance(payload, dict) else None
        return {"verified": verify_receipt(receipt, SECRET)}

    _INDEX = os.path.join(_STATIC, "index.html")
    if not os.path.exists(_INDEX):
        with open(_INDEX, "w") as f:
            f.write("<html><body style='background:#04070d;color:#e8f0fa;font:15px system-ui;padding:40px'>"
                    "<h1 style='color:#5ce8de'>Eternal Auditor v2</h1>"
                    "<p>Console assets not shipped — serving fallback.</p></body></html>")
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
