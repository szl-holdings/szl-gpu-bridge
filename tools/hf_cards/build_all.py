"""build_all — audit, generate, and publish complete HF org cards.

Usage:
  python3 build_all.py --audit          # report card/link health
  python3 build_all.py --missing        # generate cards for assets with no README
  python3 build_all.py --upload --limit N   # publish generated cards (default 25)

Honesty rules (Law 7/8): no fabricated metrics. Auto-generated cards state
the limits of inventory evidence and preserve supplied metadata.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import urllib.request
import concurrent.futures

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    """Only a successful file inventory can establish that README.md is absent."""
    import concurrent.futures
    out = []
    def check(item):
        kind, e = item
        base = {"model": "models", "dataset": "datasets", "space": "spaces"}[kind]
        try:
            d = get(f"https://huggingface.co/api/{base}/{e['id']}?full=true")
            siblings = d.get("siblings")
            if not isinstance(siblings, list):
                return None  # missing inventory is unknown, never absence
            if any(not isinstance(f, dict) or not isinstance(f.get("rfilename"), str) for f in siblings):
                return None
            if not any(f["rfilename"].casefold() == "readme.md" for f in siblings):
                return (kind, d)
        except Exception:
            return None  # unreadable != missing; never count errors as missing
        return None
    with concurrent.futures.ThreadPoolExecutor(12) as ex:
        for r in ex.map(check, list_assets()):
            if r: out.append(r)
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


def heuristic(kind, e):
    rid = e["id"]
    short = rid.split("/")[-1]
    pretty = short.replace("-", " ").replace("_", " ").title()
    kind_word = {"model": "model", "dataset": "dataset", "space": "Space"}[kind]
    return dict(
        name=pretty,
        tagline=f"{pretty} — {kind_word} repository; qualification pending",
        what=f"This {kind_word} repository is listed under SZLHOLDINGS. Its name alone does not establish the artifact type, training status, deployment, or capabilities.",
        investor="An inventory entry is not evidence of performance or production readiness.",
    )


def generate(kind, e):
    c = heuristic(kind, e)
    c["metadata"] = e.get("cardData") or {}
    fn = {"model": model_card, "dataset": dataset_card, "space": space_card}[kind]
    return fn(rid=e["id"], **c)


def upload(kind, rid, content):
    from huggingface_hub import HfApi
    api = HfApi(token=TOK)
    repo_type = {"space": "space", "dataset": "dataset", "model": "model"}[kind]
    info = api.repo_info(rid, repo_type=repo_type)
    if info.siblings is None or not info.sha:
        raise RuntimeError("Cannot establish current repository contents and revision")
    if any(f.rfilename.casefold() == "readme.md" for f in info.siblings):
        raise RuntimeError("README already exists; preserve it for a reviewed edit")
    api.upload_file(path_or_fileobj=content.encode(), path_in_repo="README.md",
                    repo_id=rid, repo_type=repo_type, parent_commit=info.sha,
                    commit_message="docs: add unverified inventory card with supplied metadata")
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
