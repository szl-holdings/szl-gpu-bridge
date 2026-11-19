#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Detect (and, only when explicitly asked, publish) a stale receipts index.

Pins ONE dataset revision, snapshots only its *.json/*.jsonl files, rebuilds the
index with build_receipts_index.py, and compares it with the index published at
that same revision.

  check   (default) read-only. Writes stale=true|false to $GITHUB_OUTPUT and a
          markdown report to --report. Never writes to the Hub.
  publish commits ONLY data/receipts-index.jsonl (a single-file commit that
          cannot overwrite other writers' files). Refuses when any published
          row would be removed or any row is INVALID_JSON, because that needs a
          human decision. Needs HF_TOKEN with write access to the dataset.

Honesty: hash_status values come from the builder untouched. Nothing here marks
a receipt VERIFIED.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "build_receipts_index", HERE / "build_receipts_index.py"
)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)

DEFAULT_REPO = "SZLHOLDINGS/frontier-54-runs"
DEFAULT_MAX_ADDED = 50  # the dataset takes new receipts every few seconds


def is_stale(report: dict, max_added: int = DEFAULT_MAX_ADDED) -> bool:
    """Stale = a published row is gone or changed, or too many receipts are unindexed."""
    return bool(
        report["removed"] > 0 or report["changed_fields"] or report["added"] > max_added
    )


def publish_blockers(report: dict, new_rows: list[dict]) -> list[str]:
    out = []
    if report["removed"] > 0:
        out.append(f"{report['removed']} published rows would be removed")
    invalid = report["hash_status"].get(builder.HASH_STATUS_INVALID, 0)
    if invalid or any(
        r["hash_status"] == builder.HASH_STATUS_INVALID for r in new_rows
    ):
        out.append("INVALID_JSON rows present")
    return out


def commit_message(report: dict, revision: str, source_sha: str) -> tuple[str, str]:
    hs = report["hash_status"]
    title = (
        "index: rebuild data/receipts-index.jsonl with szl-gpu-bridge "
        f"builder {source_sha[:8]}"
    )
    body = (
        f"Rebuilt offline from dataset snapshot {revision} with "
        "tools/receipts_index/build_receipts_index.py. "
        f"Rows {report['old_rows']} -> {report['new_rows']}, removed {report['removed']}. "
        f"hash_status: VERIFIED {hs.get('VERIFIED', 0)}, UNVERIFIED {hs.get('UNVERIFIED', 0)}, "
        f"NO_HASH {hs.get('NO_HASH', 0)}, INVALID_JSON {hs.get('INVALID_JSON', 0)}. "
        "Hashes are recomputed under the builder's fixed rules; a self-hash shows "
        "internal consistency only, not authorship or ordering. Receipts uploaded "
        "after the snapshot revision are not in this index. Only this file is changed."
    )
    return title, body


def markdown_report(report: dict, revision: str, stale: bool, max_added: int) -> str:
    hs = report["hash_status"]
    return "\n".join(
        [
            "## Receipts index freshness",
            "",
            f"- Dataset revision checked: `{revision}`",
            f"- Published index rows: {report['old_rows']}; rebuilt rows: {report['new_rows']}",
            f"- Unindexed receipts: {report['added']} (stale threshold {max_added}); "
            f"removed: {report['removed']}; changed fields: {report['changed_fields'] or 'none'}",
            f"- hash_status of the rebuilt index: VERIFIED {hs.get('VERIFIED', 0)}, "
            f"UNVERIFIED {hs.get('UNVERIFIED', 0)}, NO_HASH {hs.get('NO_HASH', 0)}, "
            f"INVALID_JSON {hs.get('INVALID_JSON', 0)}",
            f"- Stale: **{'yes' if stale else 'no'}**",
            "",
            "Republishing is a manual, approval-gated dispatch of this workflow "
            "with `publish=true`. This check never writes to the Hub.",
        ]
    )


def snapshot(repo: str, revision: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        allow_patterns=["*.json", "*.jsonl"],
        token=os.environ.get("HF_TOKEN") or None,
    )


def resolve_revision(repo: str) -> str:
    from huggingface_hub import HfApi

    refs = HfApi().list_repo_refs(repo, repo_type="dataset")
    return next(b.target_commit for b in refs.branches if b.name == "main")


def publish(repo: str, index_path: Path, revision: str, report: dict, source_sha: str):
    from huggingface_hub import CommitOperationAdd, HfApi

    title, body = commit_message(report, revision, source_sha)
    return HfApi().create_commit(
        repo,
        repo_type="dataset",
        operations=[CommitOperationAdd(builder.INDEX_RELPATH, str(index_path))],
        commit_message=title,
        commit_description=body,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--mode", choices=["check", "publish"], default="check")
    ap.add_argument("--max-added", type=int, default=DEFAULT_MAX_ADDED)
    ap.add_argument("--report", type=Path, help="write the markdown report here")
    ap.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA", "unknown"))
    args = ap.parse_args(argv)

    revision = resolve_revision(args.repo)
    snap = args.workdir / "snapshot"
    snapshot(args.repo, revision, snap)
    published = snap / builder.INDEX_RELPATH
    old_rows = (
        [
            json.loads(x)
            for x in published.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        if published.exists()
        else []
    )
    out = args.workdir / "new-index.jsonl"
    rows = builder.build_rows(snap, [out])
    out.write_text(builder.serialise(rows), encoding="utf-8")
    report = builder.compare(old_rows, rows)
    stale = is_stale(report, args.max_added)

    md = markdown_report(report, revision, stale, args.max_added)
    print(md)
    if args.report:
        args.report.write_text(md + "\n", encoding="utf-8")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            fh.write(f"stale={'true' if stale else 'false'}\nrevision={revision}\n")

    if args.mode == "publish":
        blockers = publish_blockers(report, rows)
        if blockers:
            print("REFUSING TO PUBLISH: " + "; ".join(blockers), file=sys.stderr)
            return 2
        info = publish(args.repo, out, revision, report, args.source_sha)
        print(f"published {info.oid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
