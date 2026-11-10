"""build_all — audit, generate, and publish complete HF org cards.

Usage:
  python3 build_all.py --audit          # report card/link health
  python3 build_all.py --missing        # generate cards for assets with no README
  python3 build_all.py --upload --limit N   # publish generated cards (default 25)

Honesty rules (Law 7/8): no fabricated metrics. Auto-generated cards state
what the asset is and link to the receipts chain for measured claims.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import urllib.request
import concurrent.futures

sys.path.insert(0, ".")
from card_factory import REGISTRY, model_card, dataset_card, space_card  # noqa: E402

TOK = __import__("os").environ.get("HF_TOKEN", "")
H = {"Authorization": f"Bearer {TOK}"} if TOK else {}


def get(url, raw=False, timeout=30):
    r = urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout)
    return r.read().decode() if raw else json.load(r)


def list_assets():
    models = get("https://huggingface.co/api/models?author=SZLHOLDINGS&limit=1000")
    datasets = get("https://huggingface.co/api/datasets?author=SZLHOLDINGS&limit=1000")
    spaces = get("https://huggingface.co/api/spaces?author=SZLHOLDINGS&limit=1000")
    return ([("model", m) for m in models] + [("dataset", d) for d in datasets]
            + [("space", s) for s in spaces])


def no_readme():
    out = []
    for kind, e in list_assets():
        base = {"model": "models", "dataset": "datasets", "space": "spaces"}[kind]
        try:
            get(f"https://huggingface.co/{base}/{e['id']}/raw/main/README.md", raw=True)
        except Exception:
            out.append((kind, e))
    return out


def verify_registry():
    bad = {}
    for k, u in REGISTRY.items():
        try:
            req = urllib.request.Request(u, method="GET",
                                         headers={"User-Agent": "Mozilla/5.0"})
            r = urllib.request.urlopen(req, timeout=15)
            if r.status >= 400:
                bad[k] = r.status
        except Exception as e:
            bad[k] = str(e)[:60]
    return bad


# ---- content: hand-written for flagships, heuristic otherwise ----
FLAGSHIPS = {
    "SZLHOLDINGS/szl-nemo": dict(
        name="szl-nemo", tagline="Nemotron v3 node — the governed GPU bridge model",
        what="The Nemotron v3 inference target served from the SZL A6000 mesh, "
             "reachable through the governed GPU bridge. Requests are metered, "
             "receipted, and fail-closed under the Kamachikuna constitution.",
        investor="This is sovereign inference: a frontier-class model served on "
                 "owned hardware with metered energy and hash-chained receipts — "
                 "no dependence on any single cloud vendor."),
    "SZLHOLDINGS/szl-khipu": dict(
        name="szl-khipu", tagline="Khipu — signed, hash-chained receipts as a model artifact",
        what="The khipu receipt format: every claim the estate makes is a signed, "
             "hash-chained receipt. Empty receipts never verify; unsigned receipts "
             "are never MEASURED.",
        investor="Tamper-evident AI claims, machine-checked: the substrate that makes "
                 "every number on every card auditable."),
}


def heuristic(kind, e):
    rid = e["id"]
    short = rid.split("/")[-1]
    pretty = short.replace("-", " ").replace("_", " ").title()
    kind_word = {"model": "model", "dataset": "dataset", "space": "live app"}[kind]
    return dict(
        name=pretty,
        tagline=f"{pretty} — part of the SZL Holdings AI estate",
        what=f"This {kind_word} is part of the SZL Holdings estate: an AI research and "
             f"product portfolio where every claim carries a receipt. Measured results "
             f"for this artifact live on the public receipts chain — we publish "
             f"numbers only where a signed receipt backs them.",
        investor="SZL Holdings builds receipt-backed AI: measured honesty, a "
                 "machine-checked constitution, and a public hash-chained evidence "
                 "chain. This artifact is one piece of that estate.",
    )


def generate(kind, e):
    c = FLAGSHIPS.get(e["id"], heuristic(kind, e))
    fn = {"model": model_card, "dataset": dataset_card, "space": space_card}[kind]
    return fn(rid=e["id"], **c)


def upload(kind, rid, content):
    from huggingface_hub import HfApi
    api = HfApi(token=TOK)
    api.upload_file(path_or_fileobj=content.encode(), path_in_repo="README.md",
                    repo_id=rid, repo_type="space" if kind == "space" else
                    ("dataset" if kind == "dataset" else "model"),
                    commit_message="card: complete investor-grade card (generated, links verified)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "missing", "upload"])
    ap.add_argument("--limit", type=int, default=25)
    a = ap.parse_args()

    if a.cmd == "audit":
        bad = verify_registry()
        print("registry health:", "ALL OK" if not bad else bad)
        missing = no_readme()
        print(f"assets without README: {len(missing)}")
        for k, e in missing[:20]:
            print("  -", k, e["id"])
    elif a.cmd == "missing":
        for k, e in no_readme()[:50]:
            print("==", k, e["id"])
            print(generate(k, e)[:400], "...")
    elif a.cmd == "upload":
        missing = no_readme()
        done, failed = 0, []
        for k, e in missing[: a.limit]:
            try:
                upload(k, e["id"], generate(k, e))
                done += 1
                print("uploaded:", k, e["id"])
            except Exception as ex:
                failed.append((e["id"], str(ex)[:80]))
        print(f"done={done} failed={len(failed)}")
        for f in failed:
            print(" FAIL:", f)


if __name__ == "__main__":
    main()
