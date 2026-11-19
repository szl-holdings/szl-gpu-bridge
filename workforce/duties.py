"""Officer duties, ported from the paused SZLHOLDINGS Spaces.

Each duty takes a ``net`` (see :class:`workforce.core.Net`) and returns a list
of findings. Duties are read-only: no duty writes to GitHub, Hugging Face, or
any estate host. Sources ported (Hub main SHAs, 2026-11-19):

  eternal-auditor       1d819017 (v2)    estate-gardener     842638f7 (v2)
  szl-chief-engineer    c61bdf80 (v3)    szl-infra-warden    73c2a8df (v1)
  szl-cyber-shield      efeaf229 (v3)    szl-research-council bd714e38 (v1)
  szl-dream-architect   c7a5d91d (v3.1)

Tightening applied per /workspace/estate/rvo_item4_workforce_2026-11-13.md is
marked "TIGHTENED" below. Nothing here claims more than it measures.
"""

import time
import urllib.parse
import xml.etree.ElementTree as ET

from . import core

HF = "https://huggingface.co"
GH = "https://api.github.com"
ORG_HF = "SZLHOLDINGS"
ORG_GH = "szl-holdings"

# Bounded, documented samples (recorded in every finding).
SPACE_CENSUS_CAP = 100
GARDEN_LIST_LIMIT = 100
GARDEN_SCAN_CAP = 30

# The seven workforce Spaces are paused on purpose (2026-11-12 security hold)
# and replaced by this workflow. A PAUSED stage for them is expected, not an
# incident. TIGHTENED (infra-warden): reason codes, never "not RUNNING = bad".
EXPECTED_PAUSED = frozenset(
    {
        "eternal-auditor",
        "estate-gardener",
        "szl-chief-engineer",
        "szl-infra-warden",
        "szl-cyber-shield",
        "szl-research-council",
        "szl-dream-architect",
    }
)

CHIEF_REPOS = ("a11oy", "szl-forge", ".github", "szl-frontier")
WARDEN_HOSTS = ("anatomy.a-11-oy.com", "router.a-11-oy.com")
FRONTIER_ISSUES = (135, 136, 137, 138)
DEFAULT_GATEWAY = "https://inference.a-11-oy.com"


def _space_census(net, cap=SPACE_CENSUS_CAP):
    """Public SZLHOLDINGS Spaces by runtime stage (unauthenticated: public only)."""
    spaces = net.get_json(f"{HF}/api/spaces?author={ORG_HF}&limit=1000")
    if not isinstance(spaces, list):
        return None
    stages, not_running = {}, []
    sampled = spaces[:cap]
    for s in sampled:
        sid = s.get("id", "")
        full = net.get_json(f"{HF}/api/spaces/{sid}")
        stage = ((full or {}).get("runtime") or {}).get("stage") or "UNKNOWN"
        if full is None:
            stage = "UNAVAILABLE"
        stages[stage] = stages.get(stage, 0) + 1
        if stage != "RUNNING":
            name = sid.split("/")[-1]
            not_running.append(
                {"space": name, "stage": stage, "expected": name in EXPECTED_PAUSED}
            )
    return {
        "listed_public": len(spaces),
        "sampled": len(sampled),
        "sample_rule": f"first {cap} of the public listing (private Spaces not visible)",
        "stages": stages,
        "not_running": not_running,
    }


# ---------------------------------------------------------------- auditor


def eternal_auditor(net, now=None):
    """Chain-head age of the receipts dataset + Space liveness census."""
    now = time.time() if now is None else now
    findings = []
    commits = net.get_json(f"{HF}/api/datasets/{core.DEFAULT_DATASET}/commits/main")
    if isinstance(commits, list) and commits:
        try:
            age_h = (now - core.parse_utc(commits[0]["date"])) / 3600  # TIGHTENED: UTC
            findings.append(
                {
                    "check": "chain_head_age_hours",
                    "value": round(age_h, 1),
                    "verdict": "PASS" if age_h <= 24 else "FAIL",
                    "detail": f"{len(commits)} commits; head {str(commits[0].get('title', ''))[:50]}",
                }
            )
        except (KeyError, ValueError, TypeError):
            findings.append({"check": "chain_head_age_hours", "verdict": "UNKNOWN"})
    else:
        findings.append(
            {"check": "chain_reachable", "verdict": "UNKNOWN", "state": "UNAVAILABLE"}
        )
    census = _space_census(net)
    if census is None:
        findings.append({"check": "spaces_running_census", "verdict": "UNAVAILABLE"})
    else:
        findings.append(
            {
                "check": "spaces_running_census",
                "verdict": "MEASURED",
                "listed_public": census["listed_public"],
                "sampled": census["sampled"],
                "sample_rule": census["sample_rule"],
                "running": census["stages"].get("RUNNING", 0),
            }
        )
    return findings


# ---------------------------------------------------------------- gardener

LICENSE_BLOCK = (
    "\n\n---\n*License: Apache-2.0. Copyright 2026 SZL Holdings · "
    "Stephen P. Lutar Jr. · ORCID 0009-0001-0110-4173*\n"
)


def estate_gardener(net, now=None):
    """Propose (never write) append-only license blocks for untagged datasets."""
    datasets = net.get_json(
        f"{HF}/api/datasets?author={ORG_HF}&limit={GARDEN_LIST_LIMIT}"
    )
    if not isinstance(datasets, list):
        return [{"surface": "datasets", "state": "UNAVAILABLE"}]
    proposals = []
    for d in datasets[:GARDEN_SCAN_CAP]:
        did = d.get("id", "")
        if any(str(t).startswith("license:") for t in (d.get("tags") or [])):
            continue
        raw = net.get_bytes(f"{HF}/datasets/{did}/raw/main/README.md", timeout=15)
        if raw is None:
            proposals.append(
                {"dataset": did, "action": "SKIP_NO_README", "state": "UNAVAILABLE"}
            )
            continue
        readme = raw.decode(errors="replace")
        proposed = readme.rstrip("\n") + LICENSE_BLOCK
        preserved = proposed.startswith(readme.rstrip("\n"))
        proposals.append(
            {
                "dataset": did,
                "action": "PROPOSE_APPEND_LICENSE",
                "before_sha256": core.sha256_hex(readme),
                "after_sha256": core.sha256_hex(proposed) if preserved else None,
                "preserved_prefix": preserved,
                "write": "OWNER_GATED: the gardener proposes; the owner writes",
            }
        )
    return [
        {
            "surface": "datasets",
            "state": "MEASURED",
            "datasets_listed": len(datasets),
            "scan_rule": f"first {GARDEN_SCAN_CAP} of up to {GARDEN_LIST_LIMIT} public datasets",
            "proposals": proposals[:20],
            "note": "A proposal is not a change. No README is ever written.",
        }
    ]


# ---------------------------------------------------------------- chief engineer


def plane_status(net, gateway):
    h = net.get_json(gateway + "/health", timeout=10)
    if h is None:
        return {"state": "UNAVAILABLE", "gateway": gateway, "models": []}
    m = net.get_json(gateway + "/v1/models", timeout=10)
    models = [x.get("id") for x in (m or {}).get("data", []) if isinstance(x, dict)]
    return {"state": "MEASURED", "gateway": gateway, "models": models[:8]}


def chief_engineer(net, now=None, gateway=DEFAULT_GATEWAY):
    """Latest default-branch CI conclusion for a named allowlist + liveness."""
    findings = []
    for rn in CHIEF_REPOS:
        repo = net.get_json(f"{GH}/repos/{ORG_GH}/{rn}")
        branch = (repo or {}).get("default_branch")
        if not branch:
            findings.append({"surface": "ci", "repo": rn, "state": "UNAVAILABLE"})
            continue
        q = urllib.parse.quote(branch, safe="")
        runs = net.get_json(
            f"{GH}/repos/{ORG_GH}/{rn}/actions/runs?branch={q}&per_page=1"
        )
        wr = (runs or {}).get("workflow_runs") or []
        if not wr:
            findings.append(
                {"surface": "ci", "repo": rn, "branch": branch, "state": "UNAVAILABLE"}
            )
            continue
        r = wr[0]
        concl = r.get("conclusion")
        # TIGHTENED: v3 called anything that was not "failure" OK.
        state = {"success": "OK", "failure": "FAILED"}.get(concl, "NOT_CONCLUSIVE")
        findings.append(
            {
                "surface": "ci",
                "repo": rn,
                "branch": branch,
                "state": state,
                "run": str(r.get("name") or "")[:60],
                "conclusion": concl,
                "status": r.get("status"),
            }
        )
    bus = net.get_json("https://a-11-oy.com/api/a11oy/healthz", timeout=15)
    findings.append(
        {"surface": "honesty_bus", "state": "MEASURED" if bus else "UNAVAILABLE"}
    )
    plane = plane_status(net, gateway)
    findings.append(
        {"surface": "compute_plane", "state": plane["state"], "gateway": gateway}
    )
    return findings


# ---------------------------------------------------------------- infra warden


def infra_warden(net, now=None):
    """Space stages with reason codes + DNS/HTTPS for a named host allowlist."""
    findings = []
    census = _space_census(net)
    if census is None:
        findings.append({"surface": "spaces", "state": "UNAVAILABLE"})
    else:
        unexpected = [x for x in census["not_running"] if not x["expected"]]
        findings.append(
            {
                "surface": "spaces",
                "state": "MEASURED",
                "listed_public": census["listed_public"],
                "sampled": census["sampled"],
                "sample_rule": census["sample_rule"],
                "stages": census["stages"],  # TIGHTENED: PAUSED/SLEEPING kept apart
                "not_running": census["not_running"][:25],
                "unexpected_not_running": len(unexpected),
            }
        )
    hosts = []
    for host in WARDEN_HOSTS:
        resolved = net.resolve(host)
        code = net.status(f"https://{host}/") if resolved else None
        hosts.append({"host": host, "dns": resolved, "https_status": code})
    findings.append({"surface": "hosts", "state": "MEASURED", "hosts": hosts})
    return findings


# ---------------------------------------------------------------- cyber shield

SHIELD_PROBES = (
    ("front_door_tls", "https://a-11-oy.com", 200),
    ("honesty_bus", "https://a-11-oy.com/api/a11oy/healthz", 200),
    ("gateway_health", "{gw}/health", 200),
    ("paywall_stands", "{gw}/v1/completions", 402),
    ("gpu2_access_gate", "https://gpu2.a-11-oy.com", 403),
)


def cyber_shield(net, now=None, gateway=DEFAULT_GATEWAY):
    """Fail-closed posture probes + secret-pattern scan of own recent records."""
    findings = []
    for name, url, expect in SHIELD_PROBES:
        code = net.status(url.format(gw=gateway))
        if code is None:
            findings.append({"probe": name, "state": "UNAVAILABLE", "expect": expect})
        else:
            findings.append(
                {
                    "probe": name,
                    "state": "PASS" if code == expect else "FAIL",
                    "code": code,
                    "expect": expect,
                }
            )
    tree = net.get_json(f"{HF}/api/datasets/{core.DEFAULT_DATASET}/tree/main/officers")
    if not isinstance(tree, list):
        findings.append({"probe": "records_secrets_scan", "state": "UNAVAILABLE"})
        return findings
    mine = sorted(
        e.get("path", "")
        for e in tree
        if "szl-cyber-shield" in e.get("path", "")
        and e.get("path", "").endswith(".json")
    )[-8:]
    scanned = hits = 0
    for path in mine:
        raw = net.get_bytes(f"{HF}/datasets/{core.DEFAULT_DATASET}/raw/main/{path}")
        if raw is None:
            continue
        scanned += 1
        if core.SECRET_PATTERN.search(raw.decode(errors="replace")):
            hits += 1
    findings.append(
        {
            "probe": "records_secrets_scan",
            "state": "PASS" if hits == 0 else "FAIL",
            "scanned": scanned,
            "hits": hits,
            "scope": "this officer's own last 8 public records only",
        }
    )
    return findings


# ---------------------------------------------------------------- arXiv (bounded)


def parse_arxiv_feed(payload, max_entries=3):
    """Bounded Atom metadata; DTD/entities refused. Ported from dream v3.1."""
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
    for entry in feed.findall(atom + "entry")[:max_entries]:
        title = " ".join((entry.findtext(atom + "title") or "").split())
        url = urllib.parse.urlsplit((entry.findtext(atom + "id") or "").strip())
        if (
            not title
            or url.scheme not in {"http", "https"}
            or url.netloc != "arxiv.org"
            or not url.path.startswith("/abs/")
            or not url.path[5:]
            or url.query
            or url.fragment
        ):
            raise ValueError("invalid arXiv entry or API error response")
        sources.append({"id": url.path[5:], "title": title[:200]})
    return sources


def get_arxiv(net, query, max_entries=3):
    url = (
        "https://export.arxiv.org/api/query?search_query="
        + query
        + f"&max_results={max_entries}&sortBy=submittedDate"
    )
    raw = net.get_bytes(url, timeout=20, limit=262144)
    if raw is None:
        return None
    try:
        return parse_arxiv_feed(raw, max_entries)
    except ValueError:
        return None


# ---------------------------------------------------------------- research council

COUNCIL_QUERY = "cat:cs.AI+OR+cat:cs.CL+OR+cat:stat.ML"


def research_council(net, now=None):
    """TIGHTENED: v1 wrote a constant category list. This fetches one pinned
    arXiv query and records ids and titles only. Retrieval is not review."""
    sources = get_arxiv(net, COUNCIL_QUERY, max_entries=5)
    return [
        {
            "surface": "arxiv",
            "state": "MEASURED" if sources is not None else "UNAVAILABLE",
            "query": COUNCIL_QUERY,
            "items": sources or [],
            "note": "ids and titles only; relevance is not assessed",
        }
    ]


# ---------------------------------------------------------------- dream architect


def dream(net):
    """Frontier questions x live research signal -> template HYPOTHESES."""
    gh = net.get_json(
        f"{GH}/repos/{ORG_GH}/szl-gpu-bridge/issues?state=open&per_page=50"
    )
    questions = []
    for it in gh if isinstance(gh, list) else []:
        if "pull_request" not in it and it.get("number") in FRONTIER_ISSUES:
            questions.append(
                {"n": it["number"], "title": str(it.get("title", ""))[:140]}
            )
    if not questions:
        return {"state": "UNAVAILABLE", "detail": "frontier questions unreachable"}
    signal = get_arxiv(net, "all:abstention+AND+all:LLM", max_entries=3)
    hypotheses = [
        {
            "question": q,
            "hypothesis": f"Approach issue #{q['n']} via a receipted, falsifiable "
            "experiment whose evidence lands in the chain before any claim",
            "nearest_prior_art": "UNASSESSED - retrieved sources are research candidates",
            "test": "preregistered acceptance criterion frozen before data is seen",
            "rejection_criterion": "if the held-out gate fails twice at frozen "
            "thresholds, the hypothesis is rejected and recorded as such",
            "drafted_by": "template",
            "state": "HYPOTHESIS - not a claim, not a model update, not an autonomous act",
        }
        for q in questions
    ]
    return {
        "state": "MEASURED",
        "questions_read": len(questions),
        "research_signal": "MEASURED" if signal is not None else "UNAVAILABLE",
        "research_sources": signal,
        "hypotheses": hypotheses,
    }


def dream_architect(net, now=None, backend=None):
    from . import dream_backend

    dreamed = dream(net)
    backend = backend or dream_backend.Backend.from_env()
    draft = backend.draft(net, dreamed)
    return [dreamed, draft]


ROLES = {
    "eternal-auditor": eternal_auditor,
    "estate-gardener": estate_gardener,
    "szl-chief-engineer": chief_engineer,
    "szl-infra-warden": infra_warden,
    "szl-cyber-shield": cyber_shield,
    "szl-research-council": research_council,
    "szl-dream-architect": dream_architect,
}
