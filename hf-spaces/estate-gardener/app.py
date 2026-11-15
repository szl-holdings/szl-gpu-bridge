#!/usr/bin/env python3
"""Estate Gardener v2 — complete app. Append-only, content-preserving.

CANONICAL: szl-holdings/szl-gpu-bridge/hf-spaces/estate-gardener/app.py
v1 defect (audit): could REPLACE a dataset README with a generic card.
v2: NEVER writes READMEs directly. Each cycle inventories the org's
public datasets via the public API, and for any without a license tag,
prepares an idempotent APPEND-ONLY review notice, never a license grant,
preserves every original byte, and emits bounded observation evidence with
before/after sha256. The actual write stays OWNER-GATED (Law 6/7).
"""
import hashlib, json, os, re, time, threading, urllib.request
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
# A missing Hub tag is not evidence of rights or authority to select a license.
LICENSE_BLOCK = (
    "\n\n<!-- szl-license-review:v1 -->\n"
    "> License metadata requires owner review. No license or redistribution "
    "rights are granted by this notice; preserve existing terms and attribution.\n"
)


def sha256_text(s): return hashlib.sha256(s.encode()).hexdigest()


def append_license_block(readme, has_license_tag):
    """Compatibility helper: append a neutral review notice, never a license grant."""
    if has_license_tag:
        return readme, "SKIP_ALREADY_TAGGED"
    if LICENSE_BLOCK in readme:
        return readme, "SKIP_REVIEW_ALREADY_PRESENT"
    # Do not strip whitespace: byte preservation includes trailing blank lines.
    return readme + LICENSE_BLOCK, "WRITE_EMPTY" if not readme else "APPEND"


def garden_readme(name, readme, has_license_tag=False, dry_run=True):
    """Propose only. Neither a missing tag nor a proposal establishes rights."""
    before = sha256_text(readme)
    new, action = append_license_block(readme, has_license_tag)
    return {"target": name, "action": action, "dry_run": dry_run,
            "before_sha256": before, "after_sha256": sha256_text(new),
            "preserved_prefix": new.startswith(readme),
            "proposal_only": True, "write_performed": False,
            "license_state": "TAG_PRESENT_NOT_REVERIFIED" if has_license_tag
                             else "UNKNOWN_REVIEW_REQUIRED"}, new


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
    """Observe one bounded public page; missing evidence grants no license."""
    datasets = get("https://huggingface.co/api/datasets?author=SZLHOLDINGS&limit=100")
    available = isinstance(datasets, list)
    observed = len(datasets) if available else None
    scanned = min(observed, 30) if available else None
    proposals = []
    for d in datasets[:30] if available else []:
        did = d.get("id") if isinstance(d, dict) else None
        tags = d.get("tags", []) if isinstance(d, dict) else None
        if (not isinstance(did, str)
                or not re.fullmatch(r"SZLHOLDINGS/[A-Za-z0-9_.-]+", did)
                or not isinstance(tags, list)
                or not all(isinstance(tag, str) for tag in tags)):
            proposals.append({"dataset": None, "action": "SKIP_INVALID_METADATA",
                              "state": "UNAVAILABLE"})
            continue
        if any(tag.startswith("license:") for tag in tags):
            continue
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/datasets/{did}/raw/main/README.md",
                headers={"User-Agent": "szl-gardener/2.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                readme = r.read().decode("utf-8")
        except Exception:
            proposals.append({"dataset": did, "action": "SKIP_README_UNAVAILABLE",
                              "state": "UNAVAILABLE"})
            continue
        review, _ = garden_readme(did, readme)
        proposals.append({
            "dataset": did,
            "action": "PROPOSE_LICENSE_REVIEW" if review["action"] in ("APPEND", "WRITE_EMPTY")
                      else review["action"],
            "before_sha256": review["before_sha256"],
            "after_sha256": review["after_sha256"],
            "preserved_prefix": review["preserved_prefix"],
            "license_state": review["license_state"],
            "write_performed": False,
            "write": "OWNER_GATED — license evidence and exact-source review required",
        })
    return {
        "kind": "szl.gardener-receipt/v2", "officer": ROLE,
        "cycle": state["cycle"], "datasets_scanned": scanned,
        "datasets_observed": observed,
        "datasets_deferred": observed - scanned if available else None,
        "inventory_state": "BOUNDED_FIRST_PAGE" if available else "UNAVAILABLE",
        "inventory_complete": False,
        "proposals_made": len(proposals),
        "proposals": proposals,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "One bounded public page, at most 30 rows inspected; not an estate census. "
                "Review proposals grant no license and perform no README writes.",
    }


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
            print("cycle", state["cycle"], receipt["inventory_state"], flush=True)
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
                    "<p>Append-only license review. No license is inferred; writes are owner-gated.</p>"
                    "<p>Proposals require owner review. A proposal is not a change.</p></body></html>")
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
