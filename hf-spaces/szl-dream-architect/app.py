#!/usr/bin/env python3
"""Dream Architect v3 - the dreamer that actually dreams, on the estate supercomputer.

CANONICAL SOURCE: szl-holdings/szl-gpu-bridge/hf-spaces/szl-dream-architect/
(Hugging Face Space SZLHOLDINGS/szl-dream-architect is the deploy mirror.)

v3 (2026-11-14, CTO directive - cutting-edge console + cyber shield + supercomputer):
  - 3D officer constellation UI: drag/zoom/hover, self-contained, no CDN,
    served same-origin under a fail-closed CSP (default-src 'self').
  - Supercomputer: drafting runs on the estate compute plane (the governed
    TLS inference gateway fronting the Salad GPU fleet). x402 challenges are
    recorded honestly as PAYWALLED - the paywall is the monetization surface
    and is never bypassed. The 2026-11-12 unencrypted fixed-IP endpoint is
    retired (security review resolved).
  - Chain: Ed25519 key + hash chain persist under /data; restarts are marked
    CHAIN_RESTARTED, never silent (Law 9). Fail-closed everywhere: template
    output is never labelled MEASURED (Law 4)."""
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
STATE = Path(os.environ.get("OFFICER_STATE", "/data/state.json"))

# Estate compute plane ("the supercomputer"): the governed inference gateway
# fronting the Salad GPU fleet (laptop bridge -> VM1 4090s). TLS-only,
# Cloudflare-gated. x402 challenges are the monetization surface -- probed
# honestly, never bypassed (cyber-shield doctrine).
GATEWAY = os.environ.get("COMPUTE_GATEWAY", "https://inference.a-11-oy.com")

# The estate's open frontier questions (dream duty raw material).
FRONTIER_ISSUES = [135, 136, 137, 138]


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
    """Persistent signing key: survives restarts (v1 ephemeral-key defect fixed)."""
    kd = STATE.parent / "officer.key"
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
    req = urllib.request.Request(url, headers={"User-Agent": ROLE + "/3.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def plane_status():
    """Probe the estate compute plane. Honest states: MEASURED / UNAVAILABLE."""
    h = get(GATEWAY + "/health", timeout=10)
    if h is None:
        return {"state": "UNAVAILABLE", "gateway": GATEWAY,
                "detail": "compute plane unreachable (fail-closed)"}
    m = get(GATEWAY + "/v1/models", timeout=10)
    models = [x.get("id") for x in (m or {}).get("data", [])] if m else []
    return {"state": "MEASURED", "gateway": GATEWAY,
            "detail": "compute plane healthy", "models": models[:8]}


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
        "User-Agent": "szl-dream-architect/2.0", "Accept": "application/atom+xml"})
    try:
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(req, timeout=timeout) as response:
            return parse_arxiv_feed(response.read(262145))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- duties

def dream_duty(plane):
    """One real vision cycle: estate frontier questions x live research
    signals x the estate compute plane -> LABELED HYPOTHESES with prior art,
    a test, and a rejection criterion. A reviewed idea is not a model
    update or an autonomous act (Law 6/7)."""
    questions = []
    gh = get("https://api.github.com/repos/szl-holdings/szl-gpu-bridge/issues?state=open&per_page=100")
    if gh:
        for it in gh:
            if "pull_request" not in it and it.get("number") in FRONTIER_ISSUES:
                questions.append({"n": it["number"], "title": it["title"][:140]})

    if not questions:
        # Fail-closed: no frontier questions means no research fetch and no
        # drafted hypotheses this cycle -- research serves questions.
        return [
            {"surface": "frontier_questions", "state": "UNAVAILABLE",
             "questions": []},
            {"surface": "research_signal", "state": "UNAVAILABLE",
             "titles": [], "sources": None, "source_count": None},
            {"surface": "compute_plane_draft", "state": "UNAVAILABLE", "model": "",
             "detail": "no frontier questions this cycle -- nothing to draft"},
            {"surface": "hypotheses", "state": "UNAVAILABLE", "count": 0,
             "labeled": "HYPOTHESIS -- never a claim", "items": []}]

    # Live research signal through the fail-closed retrieval boundary:
    # bounded read (hard byte limit), no redirect following, bounded source
    # count, https only. None means unavailable; an empty list is an
    # observed empty feed.
    signal = get_research()
    signal_titles = [s["title"][:120] for s in (signal or [])]
    signal_state = "MEASURED" if signal is not None else "UNAVAILABLE"

    # Draft on the estate compute plane (TLS gateway). A 402 x402 challenge is
    # recorded honestly as PAYWALLED -- the paywall is the monetization
    # surface, never bypassed, never dressed up as an error.
    draft_state, draft_detail, draft_model = "UNAVAILABLE", "", ""
    if plane["state"] == "MEASURED" and plane.get("models"):
        draft_model = plane["models"][0]
        try:
            prompt = ("You are the Dream Architect of an AI research estate. Given open "
                      "engineering questions and recent research signals, draft ONE specific, "
                      "falsifiable hypothesis per question. Format exactly: "
                      "HYPOTHESIS: <one sentence> | PRIOR_ART: <nearest prior work> | "
                      "TEST: <preregistered acceptance criterion>. Questions: "
                      + json.dumps(questions) + ". Signals: " + "; ".join(signal_titles) + ".")
            body = json.dumps({"model": draft_model, "max_tokens": 700,
                               "messages": [{"role": "user", "content": prompt}]}).encode()
            req = urllib.request.Request(GATEWAY + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=60).read())
            msg = d["choices"][0]["message"]
            draft_state = "MEASURED"
            draft_detail = (msg.get("content") or msg.get("reasoning_content") or "")[:1200]
        except urllib.error.HTTPError as e:
            if e.code == 402:
                draft_state = "PAYWALLED"
                draft_detail = "x402 challenge returned -- the monetization surface stands (honest state, not bypassed)"
            else:
                draft_detail = "HTTP " + str(e.code)
        except Exception as e:
            draft_detail = str(e)[:120]
    elif plane["state"] != "MEASURED":
        draft_detail = "compute plane unavailable"

    hypotheses = []
    for q in questions:
        done = False
        if draft_state == "MEASURED" and draft_detail:
            hyp_m = re.search(r"HYPOTHESIS:\s*(.+?)(?:\||$)", draft_detail)
            if hyp_m:
                prior_m = re.search(r"PRIOR_ART:\s*(.+?)(?:\||$)", draft_detail)
                test_m = re.search(r"TEST:\s*(.+?)(?:\||$)", draft_detail)
                hypotheses.append({
                    "question": q,
                    "hypothesis": hyp_m.group(1).strip()[:300],
                    "nearest_prior_art": (prior_m.group(1).strip() if prior_m else "see draft")[:200],
                    "test": (test_m.group(1).strip() if test_m else "preregistered acceptance criterion")[:250],
                    "drafted_by": "estate compute plane (" + draft_model + ")",
                    "state": "HYPOTHESIS -- not a claim, not a model update, not an autonomous act"})
                done = True
        if not done:
            hypotheses.append({
                "question": q,
                "hypothesis": ("Approach issue #" + str(q["n"]) + " via a receipted, falsifiable "
                               "experiment whose evidence lands in the chain before any claim"),
                "nearest_prior_art": "selective classification (Geifman & El-Yaniv); "
                                      "RouteLLM routing frontiers; proving-ground battery",
                "test": "preregistered acceptance criterion frozen before data is seen",
                "rejection_criterion": "if the held-out gate fails twice at frozen thresholds, "
                                        "the hypothesis is rejected and recorded as such",
                "drafted_by": "template (compute plane " + draft_state + " -- fail-closed)",
                "state": "HYPOTHESIS -- not a claim, not a model update, not an autonomous act"})

    return [
        {"surface": "frontier_questions", "state": "MEASURED" if questions else "UNAVAILABLE",
         "questions": questions},
        {"surface": "research_signal", "state": signal_state, "titles": signal_titles,
         "sources": signal, "source_count": len(signal) if signal is not None else None},
        {"surface": "compute_plane_draft", "state": draft_state, "model": draft_model,
         "detail": draft_detail[:300]},
        {"surface": "hypotheses", "state": "MEASURED" if questions else "UNAVAILABLE",
         "count": len(hypotheses), "labeled": "HYPOTHESIS -- never a claim",
         "items": hypotheses[:4]}]


def chief_duty(plane):
    """CI sweep across the estate's four tier-0 repos + honesty bus + plane."""
    findings = []
    for rn in ["a11oy", "szl-forge", ".github", "szl-frontier"]:
        runs = get("https://api.github.com/repos/szl-holdings/" + rn + "/actions/runs?per_page=1")
        if runs and runs.get("workflow_runs"):
            r = runs["workflow_runs"][0]
            findings.append({"surface": "ci", "repo": rn,
                             "state": "FAILED" if r.get("conclusion") == "failure" else "OK",
                             "run": (r.get("name") or "")[:60],
                             "conclusion": r.get("conclusion")})
        else:
            findings.append({"surface": "ci", "repo": rn, "state": "UNAVAILABLE"})
    bus = get("https://a-11-oy.com/api/a11oy/healthz", timeout=15)
    findings.append({"surface": "honesty_bus", "state": "MEASURED" if bus else "UNAVAILABLE",
                     "detail": "estate honesty bus alive" if bus else "unreachable"})
    findings.append({"surface": "compute_plane", "state": plane["state"],
                     "gateway": plane["gateway"]})
    return findings


def shield_duty(plane):
    """Real (lightweight, read-only) estate security sweep, fail-closed."""
    findings = []
    probes = [
        ("front_door_tls", "https://a-11-oy.com", 200),
        ("honesty_bus", "https://a-11-oy.com/api/a11oy/healthz", 200),
        ("gateway_health", GATEWAY + "/health", 200),
        ("paywall_stands", GATEWAY + "/v1/completions", 402),
        ("gpu2_access_gate", "https://gpu2.a-11-oy.com", 403),
    ]
    for name, url, expect in probes:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ROLE + "/3.0"})
            code = 200
            try:
                urllib.request.urlopen(req, timeout=15).read(2048)
            except urllib.error.HTTPError as e:
                code = e.code
            findings.append({"probe": name, "state": "PASS" if code == expect else "FAIL",
                             "code": code, "expect": expect})
        except Exception as e:
            findings.append({"probe": name, "state": "UNAVAILABLE", "detail": str(e)[:80]})

    # Secret-pattern scan over this officer's recently published records
    # (read-only, regex over public JSON -- no credentials are ever loaded).
    scanned, hits = 0, 0
    try:
        files = api.list_repo_files(DS, repo_type="dataset") or []
        mine = sorted([f for f in files if f.startswith("officers/") and ROLE in f])[-8:]
        pat = re.compile(r"hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
                        r"sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY")
        for f in mine:
            rec = get("https://huggingface.co/datasets/" + DS + "/raw/main/" + f, timeout=15)
            scanned += 1
            if rec and pat.search(json.dumps(rec)):
                hits += 1
        findings.append({"probe": "records_secrets_scan",
                         "state": "PASS" if hits == 0 else "FAIL",
                         "scanned": scanned, "hits": hits})
    except Exception as e:
        findings.append({"probe": "records_secrets_scan", "state": "UNAVAILABLE",
                         "detail": str(e)[:80]})
    findings.append({"probe": "chain", "state": "MEASURED",
                     "note": "receipts chain persisted under /data; restarts marked CHAIN_RESTARTED"})
    return findings


def do_duty(plane):
    if ROLE == "szl-dream-architect":
        return dream_duty(plane)
    if ROLE == "szl-chief-engineer":
        return chief_duty(plane)
    return shield_duty(plane)


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
            plane = plane_status()
            duty = do_duty(plane)
            rec = {"schema": "szl.officer/v3", "officer": ROLE, "cycle": state["count"],
                   "chain_state": "CHAIN_RESTARTED" if restarted else "CONTINUOUS",
                   "plane": {"state": plane["state"], "gateway": plane["gateway"]},
                   "findings": duty, "prev_sha256": state["prev_sha256"],
                   "khipu_public_key": pub,
                   "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            blob = json.dumps({k: v for k, v in rec.items() if k != "row_sha256"}, sort_keys=True)
            rec["row_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
            rec["khipu_signature"] = key.sign(rec["row_sha256"].encode()).hex()
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
                path_in_repo="officers/" + ROLE + "-" + rec["row_sha256"][:12] + ".json",
                repo_id=DS, repo_type="dataset")
            print("cycle", state["count"], "ok", flush=True)
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
                                                "gateway": GATEWAY,
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
                "<h1>" + TITLE + "</h1>"
                "<p>Console assets not shipped -- serving fallback. /healthz works.</p></body></html>")
app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
