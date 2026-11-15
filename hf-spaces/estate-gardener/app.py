#!/usr/bin/env python3
"""Estate Gardener v2 — complete app. Append-only, content-preserving.

CANONICAL: szl-holdings/szl-gpu-bridge/hf-spaces/estate-gardener/app.py
v1 defect (audit): could REPLACE a dataset README with a generic card.
v2: NEVER writes READMEs directly. Each cycle inventories the org's
public datasets via the public API, and for any without a license tag,
prepares an APPEND-ONLY proposal (existing content + license block),
verifies the proposal is a strict superset, and emits a receipt with
before/after sha256. The actual write stays OWNER-GATED (Law 6/7).
"""
import hashlib, json, os, time, threading, urllib.request
from pathlib import Path

try:
    from fastapi import FastAPI
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
ROLE = "estate-gardener"
CYCLE = 86400
STATE = Path(os.environ.get("GARDENER_STATE", "/data/state.json"))
LICENSE_BLOCK = ("\n\n---\n*License: Apache-2.0. Copyright 2026 SZL Holdings · "
                 "Stephen P. Lutar Jr. · ORCID 0009-0001-0110-4173*\n")


def sha256_text(s): return hashlib.sha256(s.encode()).hexdigest()




# ---- Backward-compatible interface (the tests + SDK contract) ----
def append_license_block(readme, has_license_tag):
    """v2 semantics: APPEND-ONLY, verified superset, or skip. Never replace."""
    if has_license_tag:
        return readme, "SKIP_ALREADY_TAGGED"
    if not readme.strip():
        return LICENSE_BLOCK.strip() + "\n", "WRITE_EMPTY"
    new = readme.rstrip("\n") + LICENSE_BLOCK
    if not new.startswith(readme.rstrip("\n")):
        return readme, "REFUSED_NOT_A_PREFIX"
    return new, "APPEND"


def garden_readme(name, readme, has_license_tag=False, dry_run=True):
    """v2: proposes, never writes. Receipt with before/after sha256."""
    before = sha256_text(readme)
    new, action = append_license_block(readme, has_license_tag)
    return {"target": name, "action": action, "dry_run": dry_run,
            "before_sha256": before, "after_sha256": sha256_text(new),
            "preserved_prefix": new.startswith(readme.rstrip("\n"))}, new

def load_state():
    if STATE.exists():
        try: return json.loads(STATE.read_text())
        except Exception: pass
    return {"cycle": 0, "proposals": {}}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.with_suffix(".tmp").write_text(json.dumps(st, indent=2))
    STATE.with_suffix(".tmp").replace(STATE)


def get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "szl-gardener/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def garden_cycle(state):
    """Inventory public datasets; propose (never write) append-only license fixes."""
    datasets = get("https://huggingface.co/api/datasets?author=SZLHOLDINGS&limit=100") or []
    proposals = []
    for d in datasets[:30]:  # bounded pass
        did = d.get("id", "")
        tags = d.get("tags") or []
        has_license = any(t.startswith("license:") for t in tags)
        if has_license:
            continue
        # fetch the current README (public resolve)
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/datasets/{did}/raw/main/README.md",
                headers={"User-Agent": "szl-gardener/2.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                readme = r.read().decode(errors="replace")
        except Exception:
            proposals.append({"dataset": did, "action": "SKIP_NO_READLINE",
                              "state": "UNAVAILABLE"})
            continue
        proposed = readme.rstrip("\n") + LICENSE_BLOCK
        preserved = proposed.startswith(readme.rstrip("\n"))
        proposals.append({
            "dataset": did, "action": "PROPOSE_APPEND_LICENSE",
            "before_sha256": sha256_text(readme),
            "after_sha256": sha256_text(proposed) if preserved else None,
            "preserved_prefix": preserved,
            "write": "OWNER_GATED — the gardener proposes; the owner writes (Law 6/7)",
        })
    receipt = {
        "kind": "szl.gardener-receipt/v2", "officer": ROLE,
        "cycle": state["cycle"], "datasets_scanned": len(datasets),
        "proposals": proposals[:20],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Append-only proposals, verified supersets, owner-gated writes. "
                 "A proposal is not a change.",
    }
    return receipt


def loop():
    global state
    while True:
        state["cycle"] = state.get("cycle", 0) + 1
        tc = time.time()
        try:
            receipt = garden_cycle(state)
            line = json.dumps(receipt, sort_keys=True)
            receipt["row_sha256"] = hashlib.sha256(line.encode()).hexdigest()
            state["proposals"][str(state["cycle"])] = receipt["row_sha256"]
            hist = state.setdefault("history", [])
            hist.append({k: receipt.get(k) for k in ("cycle", "ts", "row_sha256", "datasets_scanned", "proposals_made")})
            del hist[:-24]
            save_state(state)
            if TOK:
                api.upload_file(path_or_fileobj=json.dumps(receipt, indent=2).encode(),
                    path_in_repo=f"officers/{ROLE}-{receipt['row_sha256'][:12]}.json",
                    repo_id=DS, repo_type="dataset")
            print("cycle", state["cycle"], "gardened", flush=True)
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
                "proposals_recorded": len(state.get("proposals", {}))}


if _HAS_FASTAPI:
    @app.on_event("startup")
    def _start_loop():
        threading.Thread(target=loop, daemon=True).start()
    _STATIC = os.environ.get("OFFICER_STATIC", os.path.join(os.path.dirname(__file__), "static"))
    os.makedirs(_STATIC, exist_ok=True)

    @app.get("/api/live")
    def live():
        return {"ok": True, "service": ROLE, "cycle": state.get("cycle", 0),
                "proposals_recorded": len(state.get("proposals", {})),
                "history_count": len(state.get("history", [])), "cycle_seconds": CYCLE}

    @app.get("/api/history")
    def history():
        return {"service": ROLE, "receipts": list(reversed(state.get("history", [])))}

    _INDEX = os.path.join(_STATIC, "index.html")
    if not os.path.exists(_INDEX):
        with open(_INDEX, "w") as f:
            f.write("<html><body style='background:#04070d;color:#e8f0fa;font:15px system-ui;padding:40px'>"
                    "<h1 style='color:#5ce8de'>Estate Gardener v2</h1>"
                    "<p>Console assets not shipped — serving fallback.</p></body></html>")
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
