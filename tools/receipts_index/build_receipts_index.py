#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministically rebuild data/receipts-index.jsonl for SZLHOLDINGS/frontier-54-runs.

Reads a local checkout/snapshot of the dataset (it never downloads or uploads),
emits one row per receipt (one per line for *.jsonl files), and re-computes each
receipt's self-hash under a fixed list of documented canonical forms.

Honesty rules:
  * Fields are copied from the receipt, never inferred. A field the receipt does
    not carry is written as "UNKNOWN".
  * A claimed hash that none of the documented rules reproduces is UNVERIFIED,
    not MISMATCH: the builder cannot tell a tampered file from an unknown rule.
    Only a receipt whose own rule is known AND fails would be a mismatch, and no
    rule is assumed per family, so the builder never reports one.
  * A self-hash proves internal consistency only, not authorship or ordering.
  * The output depends only on file bytes: paths are sorted, rows are serialised
    with sorted keys, and no clock or environment value is read.

Usage:
  build_receipts_index.py --root SNAPSHOT [--output PATH] [--compare EXISTING] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

UNKNOWN = "UNKNOWN"
INDEX_RELPATH = "data/receipts-index.jsonl"

# Field lookup order. The first key present with a non-empty scalar wins.
KIND_KEYS = ("kind", "schema")
DETAIL_KEYS = ("detail", "state")
TS_KEYS = ("ts", "timestamp_utc", "started_utc")
# Same order as the 2026-11-12 rebuild (a9c5630).
HASH_KEYS = ("receipt_sha256", "body_sha256", "sha256", "row_sha256", "package_sha256")
SIGNATURE_KEYS = frozenset({"signature", "signature_length"})

HASH_STATUS_VERIFIED = "VERIFIED"
HASH_STATUS_UNVERIFIED = "UNVERIFIED"
HASH_STATUS_NO_HASH = "NO_HASH"
HASH_STATUS_INVALID = "INVALID_JSON"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dump_default(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True).encode("utf-8")


def _dump_compact(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _without(obj: dict, drop: Iterable[str]) -> dict:
    drop = set(drop)
    return {k: v for k, v in obj.items() if k not in drop}


# (rule name, function(receipt, hash_field) -> bytes or None). Order is fixed.
RULES: tuple[tuple[str, Callable[[dict, str], bytes | None]], ...] = (
    ("sorted-json-without-hash-field", lambda r, f: _dump_default(_without(r, [f]))),
    (
        "sorted-compact-json-without-hash-field",
        lambda r, f: _dump_compact(_without(r, [f])),
    ),
    (
        "sorted-compact-json-without-hash-and-signature",
        lambda r, f: _dump_compact(_without(r, {f} | SIGNATURE_KEYS)),
    ),
    (
        "sorted-json-without-hash-and-signature",
        lambda r, f: _dump_default(_without(r, {f} | SIGNATURE_KEYS)),
    ),
    (
        "sorted-json-of-files-member",
        lambda r, f: (
            _dump_default(r["files"])
            if f == "package_sha256" and isinstance(r.get("files"), dict)
            else None
        ),
    ),
)


def _scalar(value: Any) -> bool:
    return (
        isinstance(value, (str, int, float))
        and not isinstance(value, bool)
        and str(value) != ""
    )


def _first(receipt: dict, keys: Iterable[str]) -> tuple[str, str]:
    for key in keys:
        if key in receipt and _scalar(receipt[key]):
            return str(receipt[key]), key
    return UNKNOWN, UNKNOWN


def check_hash(receipt: dict) -> dict:
    """Return hash fields for one receipt object. Never raises on content."""
    claimed, field = _first(receipt, HASH_KEYS)
    if field == UNKNOWN:
        return {
            "sha256": UNKNOWN,
            "hash_field": UNKNOWN,
            "hash_status": HASH_STATUS_NO_HASH,
            "hash_rule": UNKNOWN,
        }
    for name, rule in RULES:
        try:
            payload = rule(receipt, field)
        except (TypeError, ValueError):
            payload = None
        if payload is not None and _sha(payload) == claimed:
            return {
                "sha256": claimed,
                "hash_field": field,
                "hash_status": HASH_STATUS_VERIFIED,
                "hash_rule": name,
            }
    return {
        "sha256": claimed,
        "hash_field": field,
        "hash_status": HASH_STATUS_UNVERIFIED,
        "hash_rule": UNKNOWN,
    }


def row_for(path: str, receipt: Any, raw: bytes, line: int | None) -> dict:
    row = {
        "path": path,
        "line": line if line is not None else UNKNOWN,
        "file_sha256": _sha(raw),
    }
    if not isinstance(receipt, dict):
        row.update(
            kind=UNKNOWN,
            detail=UNKNOWN,
            ts=UNKNOWN,
            sha256=UNKNOWN,
            hash_field=UNKNOWN,
            hash_status=HASH_STATUS_NO_HASH,
            hash_rule=UNKNOWN,
        )
        return row
    row["kind"], _ = _first(receipt, KIND_KEYS)
    row["detail"], _ = _first(receipt, DETAIL_KEYS)
    row["ts"], _ = _first(receipt, TS_KEYS)
    row.update(check_hash(receipt))
    return row


def _invalid(path: str, raw: bytes, line: int | None) -> dict:
    return {
        "path": path,
        "line": line if line is not None else UNKNOWN,
        "file_sha256": _sha(raw),
        "kind": UNKNOWN,
        "detail": UNKNOWN,
        "ts": UNKNOWN,
        "sha256": UNKNOWN,
        "hash_field": UNKNOWN,
        "hash_status": HASH_STATUS_INVALID,
        "hash_rule": UNKNOWN,
    }


def receipt_files(root: Path, exclude: Iterable[Path] = ()) -> list[Path]:
    skip = {Path(e).resolve() for e in exclude}
    out = []
    for p in root.rglob("*"):
        # A snapshot index must not read through symlinks into other files.
        if p.is_symlink():
            continue
        if p.resolve() in skip:
            continue  # never index the index being written
        rel = p.relative_to(root).as_posix()
        if not p.is_file() or p.suffix not in (".json", ".jsonl"):
            continue
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue  # .cache/, .git/, .huggingface/
        if rel == INDEX_RELPATH:
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def build_rows(root: Path, exclude: Iterable[Path] = ()) -> list[dict]:
    rows = []
    for p in receipt_files(root, exclude):
        rel = p.relative_to(root).as_posix()
        data = p.read_bytes()
        if p.suffix == ".jsonl":
            for n, raw_line in enumerate(data.splitlines(), start=1):
                if not raw_line.strip():
                    continue
                try:
                    rows.append(row_for(rel, json.loads(raw_line), raw_line, n))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    rows.append(_invalid(rel, raw_line, n))
        else:
            try:
                rows.append(row_for(rel, json.loads(data), data, None))
            except (json.JSONDecodeError, UnicodeDecodeError):
                rows.append(_invalid(rel, data, None))
    return rows


def serialise(rows: list[dict]) -> str:
    return "".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows
    )


def _key(row: dict) -> tuple:
    # The v1 index has no line numbers, so JSONL rows match on (path, sha256).
    if str(row.get("path", "")).endswith(".jsonl"):
        return (row["path"], row.get("sha256"))
    return (row["path"],)


def compare(old_rows: list[dict], new_rows: list[dict]) -> dict:
    old = {_key(r): r for r in old_rows}
    new = {_key(r): r for r in new_rows}
    added = sorted(k for k in new if k not in old)
    removed = sorted(k for k in old if k not in new)
    changed: dict[str, int] = {}
    for k in sorted(set(old) & set(new)):
        for field in ("kind", "detail", "ts", "sha256"):
            o, n = old[k].get(field), new[k].get(field)
            o_norm = UNKNOWN if o in (None, "", "?") else o
            if o_norm != n:
                changed[field] = changed.get(field, 0) + 1
    by_dir: dict[str, int] = {}
    for k in added:
        top = k[0].split("/", 1)[0]
        by_dir[top] = by_dir.get(top, 0) + 1
    status: dict[str, int] = {}
    for r in new_rows:
        status[r["hash_status"]] = status.get(r["hash_status"], 0) + 1
    return {
        "old_rows": len(old_rows),
        "new_rows": len(new_rows),
        "added": len(added),
        "added_by_dir": dict(sorted(by_dir.items())),
        "added_paths": [k[0] for k in added],
        "removed": len(removed),
        "removed_paths": [k[0] for k in removed],
        "changed_fields": dict(sorted(changed.items())),
        "hash_status": dict(sorted(status.items())),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--root", type=Path, required=True, help="local snapshot of the dataset"
    )
    ap.add_argument(
        "--output", type=Path, help="write the index here (default: stdout)"
    )
    ap.add_argument(
        "--compare",
        type=Path,
        help="existing index to diff against; prints a JSON report to stderr",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if --output differs from what would be written",
    )
    args = ap.parse_args(argv)
    rows = build_rows(args.root, [args.output] if args.output else [])
    text = serialise(rows)
    if args.compare:
        old = [
            json.loads(line)
            for line in args.compare.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(json.dumps(compare(old, rows), indent=2, sort_keys=True), file=sys.stderr)
    if args.check:
        current = (
            args.output.read_text(encoding="utf-8")
            if args.output and args.output.exists()
            else ""
        )
        return 0 if current == text else 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
