#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Detect (and, only when explicitly asked, publish) a stale receipts index.

Pins ONE dataset revision, snapshots only its *.json/*.jsonl files, rebuilds the
index with build_receipts_index.py, and compares it with the index published at
that same revision.

  check   (default) read-only. Reports FRESH or STALE after a complete check.
          If Hub rate limiting prevents completion, reports CHECK_UNAVAILABLE
          and returns nonzero. Never writes to the Hub.
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


def snapshot(repo: str, revision: str, dest: Path, *, read_only: bool = False) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        allow_patterns=["*.json", "*.jsonl"],
        token=False if read_only else os.environ.get("HF_TOKEN") or None,
        **({"max_workers": 2} if read_only else {}),
    )


def provably_stale_json_paths(repo: str, revision: str, max_added: int) -> int | None:
    """Prove staleness cheaply, or decline to make a freshness determination.

    The pinned index and a paginated, metadata-only Hub tree can prove that
    >max_added new .json *files* exist: each produces at least one index row,
    including invalid JSON. JSONL files may be empty, so they cannot support
    the same inference. This NEVER returns FRESH; inconclusive results must
    undergo the full byte-for-byte snapshot comparison.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    try:
        indexed = Path(
            hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=builder.INDEX_RELPATH,
                revision=revision,
                token=False,
            )
        )
    except Exception as exc:
        from huggingface_hub.errors import EntryNotFoundError

        if isinstance(exc, EntryNotFoundError):
            # Missing published index: the full pinned comparison treats it as
            # zero old rows and reports STALE, rather than losing the alert.
            return None
        raise
    old_paths: set[str] = set()
    for line in indexed.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError("Published receipts index has a non-string path")
            old_paths.add(row["path"])

    missing = 0
    for entry in HfApi().list_repo_tree(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        token=False,
    ):
        if not isinstance(entry, RepoFile):
            continue
        rel = Path(entry.path)
        if (
            entry.path == builder.INDEX_RELPATH
            or rel.suffix != ".json"
            or any(part.startswith(".") for part in rel.parts)
        ):
            continue
        if entry.path not in old_paths:
            missing += 1
            if missing > max_added:
                return missing
    return None


def provable_staleness_report(revision: str, missing: int, max_added: int) -> str:
    return "\n".join(
        [
            "## Receipts index is stale (path-only proof)",
            "",
            f"- Dataset revision checked: `{revision}`",
            f"- At least {missing} unindexed .json files (threshold {max_added})",
            "- Finding: **STALE**, from a pinned index and pinned file tree.",
            "",
            " ".join(
                (
                    "This is a lower bound, not a full rebuild. File bodies, JSONL rows,",
                    "changed/removed rows, hash status, and signatures were NOT checked.",
                    "A complete rebuild and owner-approved publication remain required.",
                    "No Hub publication was performed.",
                )
            ),
        ]
    )


def resolve_revision(repo: str, *, read_only: bool = False) -> str:
    from huggingface_hub import HfApi

    refs = HfApi().list_repo_refs(
        repo, repo_type="dataset", **({"token": False} if read_only else {})
    )
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


def is_rate_limited(error: BaseException) -> bool:
    """Find a real 429 response, including one wrapped by the Hub client."""
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        if getattr(getattr(error, "response", None), "status_code", None) == 429:
            return True
        next_error = error.__cause__ or error.__context__
        if next_error is None:
            break
        error = next_error
    return False


def write_report(markdown: str, path: Path | None, **outputs: str) -> None:
    print(markdown)
    if path:
        path.write_text(markdown + "\n", encoding="utf-8")
    if gh_out := os.environ.get("GITHUB_OUTPUT"):
        with open(gh_out, "a", encoding="utf-8") as fh:
            for name, value in outputs.items():
                fh.write(f"{name}={value}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--mode", choices=["check", "publish"], default="check")
    ap.add_argument("--max-added", type=int, default=DEFAULT_MAX_ADDED)
    ap.add_argument("--report", type=Path, help="write the markdown report here")
    ap.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA", "unknown"))
    args = ap.parse_args(argv)

    snap = args.workdir / "snapshot"
    read_only = args.mode == "check"
    try:
        revision = resolve_revision(args.repo, read_only=read_only)
        if read_only:
            missing = provably_stale_json_paths(args.repo, revision, args.max_added)
            if missing is not None:
                write_report(
                    provable_staleness_report(revision, missing, args.max_added),
                    args.report,
                    check_status="STALE",
                    stale="true",
                    revision=revision,
                )
                return 1  # An actionable stale index is not a green check.
        snapshot(args.repo, revision, snap, read_only=read_only)
    except Exception as exc:
        if args.mode != "check" or not is_rate_limited(exc):
            raise
        write_report(
            "## Receipts index check unavailable\n\n"
            "Hugging Face returned HTTP 429 before the comparison completed. "
            "Index freshness was not determined. No index was published.",
            args.report,
            check_status="CHECK_UNAVAILABLE",
        )
        return 3
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

    write_report(
        markdown_report(report, revision, stale, args.max_added),
        args.report,
        check_status="STALE" if stale else "FRESH",
        stale="true" if stale else "false",
        revision=revision,
    )

    if read_only and stale:
        return 1

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
