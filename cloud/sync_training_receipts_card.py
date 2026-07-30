#!/usr/bin/env python3
"""Publish the source-bound private training-receipt dataset card."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "hf" / "szl-training-receipts" / "README.md"
TARGET = "SZLHOLDINGS/szl-training-receipts"


def render_card(source_sha: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source_sha must be an exact lowercase 40-character Git SHA")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__SOURCE_REVISION__") != 1:
        raise RuntimeError("dataset card must contain exactly one source placeholder")
    return template.replace("__SOURCE_REVISION__", source_sha).encode("utf-8")


def main() -> int:
    from huggingface_hub import CommitOperationAdd, HfApi

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required in the approved secret store")

    api = HfApi(token=token)
    before = api.dataset_info(TARGET, token=token)
    card = render_card(args.source_sha)
    commit = api.create_commit(
        repo_id=TARGET,
        repo_type="dataset",
        token=token,
        parent_commit=before.sha,
        commit_message=f"Bind receipt card to GitHub {args.source_sha[:12]}",
        commit_description=(
            "Source: https://github.com/szl-holdings/szl-gpu-bridge/commit/"
            f"{args.source_sha}\nNo receipt payloads were added, changed, or deleted."
        ),
        operations=[
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(card))
        ],
    )
    print(
        json.dumps(
            {
                "status": "PUBLISHED",
                "target": TARGET,
                "source_revision": args.source_sha,
                "previous_hf_revision": before.sha,
                "hf_revision": commit.oid,
                "receipt_payloads_mutated": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
