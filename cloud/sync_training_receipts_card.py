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


FULL_SHA = re.compile(r"[0-9a-f]{40}")


def publish(*, api, downloader, operation, source_sha: str, token: str) -> dict:
    """Commit the card on top of the observed Hub head, then read it back.

    The commit is optimistic (``parent_commit``): a concurrent Hub writer makes
    it fail instead of being overwritten. Success is only reported after the
    created revision is resolved again and its README.md bytes equal the
    rendered card (HF plan D4). Any mismatch raises; nothing is retried.
    """
    card = render_card(source_sha)
    before = api.dataset_info(TARGET, token=token)
    if not FULL_SHA.fullmatch(str(getattr(before, "sha", ""))):
        raise RuntimeError("Hub predecessor revision is not an exact 40-hex SHA")
    commit = api.create_commit(
        repo_id=TARGET,
        repo_type="dataset",
        token=token,
        parent_commit=before.sha,
        commit_message=f"Bind receipt card to GitHub {source_sha[:12]}",
        commit_description=(
            "Source: https://github.com/szl-holdings/szl-gpu-bridge/commit/"
            f"{source_sha}\nNo receipt payloads were added, changed, or deleted."
        ),
        operations=[
            operation(path_in_repo="README.md", path_or_fileobj=io.BytesIO(card))
        ],
    )
    revision = str(getattr(commit, "oid", ""))
    if not FULL_SHA.fullmatch(revision):
        raise RuntimeError("Hub did not return an exact 40-hex commit id")
    if revision == before.sha:
        raise RuntimeError("Hub reported no new revision for the card commit")
    after = api.dataset_info(TARGET, revision=revision, token=token)
    if getattr(after, "sha", None) != revision:
        raise RuntimeError("Hub revision readback does not match the created commit")
    remote = Path(
        downloader(
            repo_id=TARGET,
            repo_type="dataset",
            filename="README.md",
            revision=revision,
            token=token,
            force_download=True,
        )
    ).read_bytes()
    if remote != card:
        raise RuntimeError(
            "Hub README.md at the created revision differs from the rendered card"
        )
    return {
        "status": "PUBLISHED_AND_READ_BACK",
        "target": TARGET,
        "source_revision": source_sha,
        "previous_hf_revision": before.sha,
        "hf_revision": revision,
        "readback": {
            "revision": revision,
            "readme_bytes": len(remote),
            "matches_rendered_card": True,
        },
        "receipt_payloads_mutated": False,
    }


def main() -> int:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required in the approved secret store")

    result = publish(
        api=HfApi(token=token),
        downloader=hf_hub_download,
        operation=CommitOperationAdd,
        source_sha=args.source_sha,
        token=token,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
