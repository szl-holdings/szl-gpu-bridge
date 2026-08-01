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
HF_LICENSE_NAME_PATTERN = re.compile(r"[a-z0-9-.]+")


def validate_card_metadata(card: str) -> None:
    """Fail locally when the card would violate Hugging Face metadata rules."""
    if not card.startswith("---\n") or "\n---\n" not in card[4:]:
        raise ValueError("dataset card must contain YAML front matter")
    front_matter = card.split("\n---\n", 1)[0][4:]
    fields = {}
    for line in front_matter.splitlines():
        if ":" not in line or line.startswith((" ", "-")):
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()

    license_name = fields.get("license_name", "")
    if not HF_LICENSE_NAME_PATTERN.fullmatch(license_name):
        raise ValueError(
            "license_name must satisfy the Hugging Face pattern /^[a-z0-9-.]+$/"
        )
    if fields.get("license") != "other":
        raise ValueError("governed receipt evidence must retain license: other")
    if "no-blanket-reuse" not in license_name:
        raise ValueError("license_name must preserve the no-blanket-reuse boundary")
    if "do not receive a blanket data-reuse grant" not in card:
        raise ValueError("card body must preserve the no-blanket-reuse terms")


def render_card(source_sha: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source_sha must be an exact lowercase 40-character Git SHA")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__SOURCE_REVISION__") != 1:
        raise RuntimeError("dataset card must contain exactly one source placeholder")
    card = template.replace("__SOURCE_REVISION__", source_sha)
    validate_card_metadata(card)
    return card.encode("utf-8")


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
