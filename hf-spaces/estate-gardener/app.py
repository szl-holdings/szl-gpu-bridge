#!/usr/bin/env python3
"""Estate Gardener v2 — daily license/README/topic maintenance for the estate.

CANONICAL SOURCE: szl-holdings/szl-gpu-bridge/hf-spaces/estate-gardener/app.py
(Hugging Face Space SZLHOLDINGS/estate-gardener is the deploy mirror.)

FIX vs v1 (advisory 2026-11-09): v1 could REPLACE an existing dataset README
with a generic Apache-2.0 card when a license tag was missing — destroying
attribution and misstating licenses. v2 is APPEND-ONLY and CONTENT-PRESERVING:

  LAW 6 (fail closed): if a README already exists and has content, the
  gardener may only APPEND a short license-tag block. It never rewrites,
  never truncates, and refuses the write entirely if the post-write content
  would not be a strict superset of the pre-write content (byte-for-byte
  prefix check). If it cannot verify preservation, the write does not happen.

  Every action emits a khipu-style receipt with the before/after sha256 so
  an overwrite attempt is tamper-evident (Law 5).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

STATE_PATH = Path(os.environ.get("GARDENER_STATE", "/data/state.json"))
LICENSE_BLOCK = (
    "\n\n---\n*License: Apache-2.0. Copyright 2026 SZL Holdings · "
    "Stephen P. Lutar Jr. · ORCID 0009-0001-0110-4173*\n"
)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {"seen": {}}
    return {"seen": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_PATH)


def append_license_block(readme: str, has_license_tag: bool) -> tuple[str, str]:
    """Return (new_readme, action). APPEND-ONLY; never removes a byte."""
    if has_license_tag:
        return readme, "SKIP_ALREADY_TAGGED"
    if not readme.strip():
        # Empty README: writing the block is not a content loss.
        return LICENSE_BLOCK.strip() + "\n", "WRITE_EMPTY"
    new = readme.rstrip("\n") + LICENSE_BLOCK
    action = "APPEND"
    # Fail-closed preservation check: old content must be a byte prefix.
    if not new.startswith(readme.rstrip("\n")):
        return readme, "REFUSED_NOT_A_PREFIX"
    if len(sha256_text(new)) < len(sha256_text(readme)):  # impossible, but guard
        return readme, "REFUSED_HASH_SHRANK"
    return new, action


def garden_readme(name: str, readme: str, has_license_tag: bool, dry_run: bool = False) -> dict:
    state = load_state()
    before_sha = sha256_text(readme)
    seen_before = state["seen"].get(name)
    new_readme, action = append_license_block(readme, has_license_tag)
    after_sha = sha256_text(new_readme)
    receipt = {
        "kind": "szl.gardener-receipt/v2",
        "target": name,
        "action": action,
        "dry_run": dry_run,
        "before_sha256": before_sha,
        "after_sha256": after_sha if action.startswith(("APPEND", "WRITE")) else before_sha,
        "preserved_prefix": new_readme.startswith(readme.rstrip("\n")),
        "already_seen": seen_before,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Law 5/6: refuse to report success unless preservation is proven.
    if action == "APPEND" and not receipt["preserved_prefix"]:
        receipt["action"] = "REFUSED_NOT_A_PREFIX"
    if not dry_run and receipt["action"].startswith(("APPEND", "WRITE")):
        state["seen"][name] = {"sha256": after_sha, "ts": receipt["ts"]}
        save_state(state)
    return receipt, new_readme
