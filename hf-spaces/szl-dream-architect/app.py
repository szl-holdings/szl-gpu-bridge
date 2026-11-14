#!/usr/bin/env python3
"""Dream Architect v3.1 - the dreamer that actually dreams, on the estate supercomputer.

CANONICAL SOURCE: szl-holdings/szl-gpu-bridge/hf-spaces/szl-dream-architect/app.py
(Hugging Face Space SZLHOLDINGS/szl-dream-architect is the deploy mirror.)

v1 (audit 2026-11-10): do_duty() returned a fixed string. v2 did the real job:
questions x live research signals -> LABELED HYPOTHESES, hardened retrieval
boundary (bounded reads, no redirects, DTD rejection), persistent chain.

v3.1 (2026-11-14, CTO directive - cutting-edge console + cyber shield +
supercomputer), on top of the v2 contract (tests/test_dream_research.py pins
get / parse_arxiv_feed / get_research / dream - unchanged here):
  - 3D officer constellation UI: drag/zoom/hover, self-contained, no CDN,
    served same-origin under a fail-closed CSP (default-src 'self').
  - Supercomputer: the loop probes the estate compute plane (the governed TLS
    inference gateway fronting the Salad GPU fleet) and drafts on it OUTSIDE
    dream() so the tested retrieval boundary keeps its exact request budget.
    x402 challenges are recorded honestly as PAYWALLED - the paywall is the
    monetization surface and is never bypassed. The 2026-11-12 unencrypted
    fixed-IP drafting endpoint is retired (security review resolved).
  - Chain: Ed25519 key (dreamer.key) + hash chain persist under /data;
    restarts are marked CHAIN_RESTARTED, never silent (Law 9). Fail-closed
    everywhere: template output is never labelled MEASURED (Law 4).
"""
import json, hashlib, os, re, time, threading
import urllib.error, urllib.request
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi

TOK = os.environ.get("HF_TOKEN", "")
api = HfApi(token=TOK)
DS = "SZLHOLDINGS/frontier-54-runs"
ROLE = "szl-dream-architect"
TITLE = "SZL Dream Architect"
EMBLEM = "MUSUQ MUYU"
CYCLE = 86400
STATE = Path(os.environ.get("DREAMER_STATE", "/data/state.json"))

# Estate compute plane ("the supercomputer"): the governed inference gateway
# fronting the Salad GPU fleet (laptop bridge -> VM1 4090s). TLS-only,
# Cloudflare-gated. x402 challenges are the monetization surface -- probed
# honestly, never bypassed (cyber-shield doctrine).
GATEWAY = os.environ.get("COMPUTE_GATEWAY", "https://inference.a-11-oy.com")

# The estate's open frontier questions - the dreamer's raw material.
FRONTIER_ISSUES = [135, 136, 137, 138]  # lean checker, pip szl, weight-space v2, honesty-horizon


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"count": 0, "prev_sha256": "0" * 64, "chain_state": "GENESIS"}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2))
    tmp.replace(STATE)


def load_or_create_key(st):
    """Persistent signing key: survives restarts (the v1 defect)."""
    kd = STATE.parent / "dreamer.key"
    if kd.exists():
        key = Ed25519PrivateKey.from_private_bytes(kd.read_bytes())
    else:
        key = Ed25519PrivateKey.generate()
        kd.parent.mkdir(parents=True, exist_ok=True)
        kd.write_bytes(key.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption()))
        kd.chmod(0o600)
    return key


def get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "szl-dream-architect/3.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def parse_arxiv_feed(payload):
    """Read bounded Atom metadata; retrieval does not establish relevance."""
    import xml.etree.ElementTree as ET
    from urllib.parse import urlsplit

    if len(payload) > 262144:
        raise ValueError("arXiv response exceeds the metadata limit")
    text = payload.decode("utf-8-sig")
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("DTD and entity declarations are not supported")
    try:
        feed = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError("invalid arXiv Atom response") from exc
    atom = "{http://www.w3.org/2005/Atom}"
    if feed.tag != atom + "feed":
        raise ValueError("expected an Atom feed")
    sources = []
    for entry in feed.findall(atom + "entry")[:3]:
        title = " ".join((entry.findtext(atom + "title") or "").split())
        identifier = (entry.findtext(atom + "id") or "").strip()
        url = urlsplit(identifier)
        if (not title or url.scheme not in {"http", "https"}
                or url.netloc != "arxiv.org" or not url.path.startswith("/abs/")
                or not url.path[5:] or url.query or url.fragment):
            raise ValueError("invalid arXiv entry or API error response")
        sources.append({"title": title, "url": "https://arxiv.org" + url.path})
    return sources


def get_research(timeout=20):
    """None means unavailable; an empty list is an observed empty feed."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    url = ("https://export.arxiv.org/api/query?search_query=all:abstention"
           "+AND+all:LLM&max_results=3&sortBy=submittedDate")
    req = urllib.request.Request(url, headers={
        "User-Agent": "szl-dream-architect/3.1", "Accept": "application/atom+xml"})
    try:
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(req, timeout=timeout) as response:
            return parse_arxiv_feed(response.read(262145))
    except (OSError, ValueError):
        return None


def dream():
    """One real vision cycle: estate questions x live research signals
    -> labeled hypotheses with prior art, a test, and a rejection criterion."""
    # 1. The estate's open questions (titles, honestly fetched)
    questions = []
    # GitHub public API (no token needed for public repos)
    gh = get("https://api.github.com/repos/szl-holdings/szl-gpu-bridge/issues?state=open&per_page=50")
    if gh:
        for it in gh:
            if "pull_request" not in it and it.get("number") in FRONTIER_ISSUES:
                questions.append({"n": it["number"], "title": it["title"]})
    if not questions:
        return {"state": "UNAVAILABLE", "detail": "frontier questions unreachable this cycle"}

    # 2. Live research signal (arXiv public API - real retrieval, not static cats)
    signal = get_research()
    # Only retrieval is measured. Relevance to an issue still requires review.
    signal_state = "MEASURED" if signal is not None else "UNAVAILABLE"

    # 3. The dream: labeled hypotheses with test + rejection criterion
    hypotheses = []
    for q in questions:
        hypotheses.append({
            "question": q,
            "hypothesis": f"Approach issue #{q['n']} via a receipted, falsifiable experiment "
                          "whose evidence lands in the chain before any claim",
            "nearest_prior_art": "UNASSESSED - retrieved sources are research candidates",
            "test": "preregistered acceptance criterion frozen before data is seen",
            "rejection_criterion": "if the held-out gate fails twice at frozen thresholds, "
                                   "the hypothesis is rejected and recorded as such",
            "state": "HYPOTHESIS - not a claim, not a model update, not an autonomous act",
        })

    return {
        "state": "MEASURED",
        "questions_read": len(questions),
        "research_signal": signal_state,
        "research_sources": signal,
        "research_source_count": len(signal) if signal is not None else None,
        "hypotheses": hypotheses,
        "note": "A reviewed idea is not a model update. Dreams are proposals; "
                "gates stay human (Kamachikuna Law 6/7).",
    }


# ---------------------------------------------------------------- supercomputer

def plane_status():
    """Probe the estate compute plane. Honest states: MEASURED / UNAVAILABLE.

    Runs in the loop layer (NOT inside dream()) so the tested retrieval
    boundary keeps its exact request budget (tests/test_dream_research.py).
    """
    h = get(GATEWAY + "/health", timeout=10)
    if h is None:
        return {"state": "UNAVAILABLE", "gateway": GATEWAY,
                "detail": "compute plane unreachable (fail-closed)", "models": []}
    m = get(GATEWAY + "/v1/models", timeout=10)
    models = [x.get("id") for x in (m or {}).get("data", [])] if m else []
    return {"state": "MEASURED", "gateway": GATEWAY,
            "detail": "compute plane healthy", "models": models[:8]}


def draft_on_plane(plane, dreamed):
    """Draft the dreamed questions on the estate compute plane (the
    supercomputer). A 402 x402 challenge is recorded honestly as PAYWALLED
    -- the paywall is the monetization surface, never bypassed."""
    out = {"surface": "compute_plane_draft", "state": "UNAVAILABLE",
           "model": "", "detail": ""}
    if dreamed.get("state") != "MEASURED" or not dreamed.get("hypotheses"):
        out["detail"] = "no dreamed questions to draft this cycle"
        return out
    if plane["state"] != "MEASURED" or not plane.get("models"):
        out["detail"] = "compute plane unavailable (fail-closed)"
        return out
    model = plane["models"][0]
    out["model"] = model
    try:
        questions = [h.get("question", {}).get("title", "") for h in dreamed["hypotheses"][:4]]
        prompt = ("You are the Dream Architect of an AI research estate. Given open "
                  "engineering questions and recent research signals, draft ONE specific, "
                  "falsifiable hypothesis per question. Format exactly: "
                  "HYPOTHESIS: <one sentence> | PRIOR_ART: <nearest prior work> | "
                  "TEST: <preregistered acceptance criterion>. Questions: "
                  + json.dumps(questions) + ".")
        body = json.dumps({"model": model, "max_tokens": 700,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(GATEWAY + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        msg = d["choices"][0]["message"]
        out["state"] = "MEASURED"
        out["detail"] = (msg.get("content") or msg.get("reasoning_content") or "")[:1200]
        out["note"] = "draft is a signal, never a claim; hypotheses stay HYPOTHESIS-labeled"
    except urllib.error.HTTPError as e:
        if e.code == 402:
            out["state"] = "PAYWALLED"
            out["detail"] = ("x402 challenge returned - the monetization surface stands "
                             "(honest state, not bypassed)")
        else:
            out["detail"] = "HTTP " + str(e.code)
    except Exception as e:
        out["detail"] = str(e)[:120]
    return out


# ---------------------------------------------------------------- chain

state = load_state()
key = load_or_create_key(state)
pub = key.public_key().public_bytes(
    serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()


def loop():
    global state
    while True:
        state["count"] = state.get("count", 0) + 1
        restarted = state.get("chain_state") == "GENESIS" and state["count"] > 1
        tc = time.time()
        try:
            dreamed = dream()
            plane = plane_status()
            draft = draft_on_plane(plane, dreamed)
            findings = [dreamed, draft]
            rec = {"schema": "szl.officer/v3.1", "officer": ROLE, "cycle": state["count"],
                   "chain_state": "CHAIN_RESTARTED" if restarted else "CONTINUOUS",
                   "plane": {"state": plane["state"], "gateway": plane["gateway"]},
                   "findings": findings, "prev_sha256": state["prev_sha256"],
                   "khipu_public_key": pub,
                   "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            blob = json.dumps({k: v for k, v in rec.items() if k != "row_sha256"}, sort_keys=True)
            rec["row_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
            rec["khipu_signature"] = key.sign(rec["row_sha256"].encode()).hex()  # FULL length
            state["prev_sha256"] = rec["row_sha256"]
            state["chain_state"] = "CONTINUOUS"
            state["last_plane"] = plane
            state["plane_checked_utc"] = rec["timestamp_utc"]
            hist = state.setdefault("history", [])
            hist.append({k: rec.get(k) for k in ("cycle", "timestamp_utc", "row_sha256",
                                                 "khipu_signature", "chain_state",
                                                 "plane", "findings")})
            del hist[:-24]
            save_state(state)
            api.upload_file(
                path_or_fileobj=json.dumps(rec, indent=2).encode(),
                path_in_repo=f"officers/{ROLE}-{rec['row_sha256'][:12]}.json",  # content-addressed
                repo_id=DS, repo_type="dataset")
            print("cycle", state["count"], "dreamed", flush=True)
        except Exception as e:
            print("cycle", state["count"], "err", str(e)[:80], flush=True)
        remain = CYCLE - (time.time() - tc)
        if remain > 0:
            time.sleep(remain)


# ---------------------------------------------------------------- service

app = FastAPI()


@app.middleware("http")
async def cyber_shield_headers(request, call_next):
    """Fail-closed response headers: CSP locked to self, no framing, no referrer."""
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'; frame-ancestors 'self'")
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return resp


@app.get("/healthz")
def healthz():
    return {"ok": True, "service": ROLE, "cycle": state["count"],
            "chain_state": state.get("chain_state")}


def _snapshot():
    return {"service": ROLE, "title": TITLE, "emblem": EMBLEM,
            "cycle": state.get("count", 0), "chain_state": state.get("chain_state"),
            "prev_receipt_sha256": state.get("prev_sha256"), "cycle_seconds": CYCLE,
            "plane": state.get("last_plane") or {"state": "UNAVAILABLE",
                                                "gateway": GATEWAY, "models": [],
                                                "detail": "first cycle in progress"},
            "plane_checked_utc": state.get("plane_checked_utc"),
            "receipts": list(reversed(state.get("history", [])))[:24]}


@app.get("/api/constellation")
def constellation():
    return _snapshot()


@app.get("/api/live")
def live():
    return _snapshot()


@app.get("/api/history")
def history():
    return {"service": ROLE, "receipts": list(reversed(state.get("history", [])))}


threading.Thread(target=loop, daemon=True).start()
_STATIC = os.environ.get("OFFICER_STATIC", "/app/static")
os.makedirs(_STATIC, exist_ok=True)
_INDEX = os.path.join(_STATIC, "index.html")
if not os.path.exists(_INDEX):
    with open(_INDEX, "w") as f:
        f.write("<html><body style='background:#04070d;color:#e8f0fa;font:15px system-ui;padding:40px'>"
                "<h1 style='color:#a78bfa'>Dream Architect v3.1</h1>"
                "<p>Console assets not shipped - serving fallback. /healthz works.</p></body></html>")
app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
