"""Bound source bytes in the signed image to one immutable Git revision.

Runs only during OCI build (create) or container start (verify). No network,
tokens, provider SDK, model loading, private data or cloud mutations. A manifest
inside an image is not itself a signature or proof of build authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SCHEMA = "szl.training-image-source/v1"
REPOSITORY = "szl-holdings/szl-gpu-bridge"
SOURCE_FILES = (
    "salad_worker.py",
    "nemotron_h.py",
    "tools/salad/resume_integrity.py",
    "tools/salad/training_snapshot.py",
    "step_heartbeat.py",
    "readiness_probe.py",
    "receipts/receipt_upload.py",
)
MAX_SOURCE_BYTES = 2_000_000
MAX_MANIFEST_BYTES = 32_768
COMMIT_RE = re.compile(r"[a-f0-9]{40}\Z")
DIGEST_RE = re.compile(r"[a-f0-9]{64}\Z")


class SourceHold(ValueError):
    pass


def _pairs_unique(pairs):
    item = {}
    for key, value in pairs:
        if key in item:
            raise SourceHold("MANIFEST_DUPLICATE_KEY")
        item[key] = value
    return item


def _nonfinite(_):
    raise SourceHold("MANIFEST_NONFINITE")


def _path(root: Path, rel: str) -> Path:
    if rel not in SOURCE_FILES:
        raise SourceHold("UNAPPROVED_FILE")
    current = root
    if root.is_symlink():
        raise SourceHold("SOURCE_SYMLINK")
    for part in Path(rel).parts:
        current = current / part
        if current.is_symlink():
            raise SourceHold("SOURCE_SYMLINK")
    if not current.is_file():
        raise SourceHold("SOURCE_FILE_MISSING")
    return current


def _file_evidence(root: Path, rel: str) -> dict:
    path = _path(root, rel)
    size = path.stat().st_size
    if size < 1 or size > MAX_SOURCE_BYTES:
        raise SourceHold("SOURCE_SIZE_INVALID")
    sha = hashlib.sha256()
    received = 0
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            received += len(chunk)
            if received > MAX_SOURCE_BYTES:
                raise SourceHold("SOURCE_BYTE_BUDGET")
            sha.update(chunk)
    if received != size:
        raise SourceHold("SOURCE_CHANGED_DURING_READ")
    return {"bytes": received, "sha256": sha.hexdigest()}


def create_manifest(root: Path, commit: str, manifest: Path) -> dict:
    if type(commit) is not str or not COMMIT_RE.fullmatch(commit):
        raise SourceHold("SOURCE_REVISION_INVALID")
    if manifest.is_symlink() or manifest.exists():
        raise SourceHold("MANIFEST_ALREADY_EXISTS")
    files = {rel: _file_evidence(root, rel) for rel in SOURCE_FILES}
    evidence = {
        "schema": SCHEMA,
        "repository": REPOSITORY,
        "source_revision": commit,
        "files": files,
        "proof_status": "CONTENT_DIGESTS_ONLY_NOT_SIGNATURES",
        "runtime_qualified": False,
    }
    raw = (json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise SourceHold("MANIFEST_BYTE_BUDGET")
    with manifest.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    verify_manifest(root, commit, manifest)
    return evidence


def verify_manifest(root: Path, commit: str, manifest: Path) -> dict:
    if type(commit) is not str or not COMMIT_RE.fullmatch(commit):
        raise SourceHold("SOURCE_REVISION_INVALID")
    if manifest.is_symlink() or not manifest.is_file():
        raise SourceHold("MANIFEST_MISSING_OR_SYMLINK")
    with manifest.open("rb") as handle:
        raw = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise SourceHold("MANIFEST_BYTE_BUDGET")
    try:
        data = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_unique,
            parse_constant=_nonfinite,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SourceHold("MANIFEST_INVALID_JSON") from exc
    if type(data) is not dict or set(data) != {
        "schema",
        "repository",
        "source_revision",
        "files",
        "proof_status",
        "runtime_qualified",
    }:
        raise SourceHold("MANIFEST_SCHEMA_INVALID")
    if (
        data["schema"] != SCHEMA
        or data["repository"] != REPOSITORY
        or data["source_revision"] != commit
        or data["proof_status"] != "CONTENT_DIGESTS_ONLY_NOT_SIGNATURES"
        or data["runtime_qualified"] is not False
    ):
        raise SourceHold("SOURCE_REVISION_OR_SCOPE_MISMATCH")
    files = data["files"]
    if type(files) is not dict or set(files) != set(SOURCE_FILES):
        raise SourceHold("MANIFEST_MEMBER_SET_INVALID")
    for rel in SOURCE_FILES:
        record = files[rel]
        if (
            type(record) is not dict
            or set(record) != {"bytes", "sha256"}
            or type(record["bytes"]) is not int
            or record["bytes"] < 1
            or type(record["sha256"]) is not str
            or not DIGEST_RE.fullmatch(record["sha256"])
        ):
            raise SourceHold("MANIFEST_MEMBER_INVALID")
        if _file_evidence(root, rel) != record:
            raise SourceHold("SOURCE_BYTES_DRIFT")
    return {
        "schema": SCHEMA,
        "state": "SOURCE_BYTES_MATCH_UNATTESTED",
        "source_revision": commit,
        "release_authorized": False,
        "training_qualified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("create", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    try:
        if args.operation == "create":
            create_manifest(args.root, args.source_revision, args.manifest)
        result = verify_manifest(args.root, args.source_revision, args.manifest)
    except (SourceHold, OSError) as exc:
        reason = str(exc) if isinstance(exc, SourceHold) else "SOURCE_IO_UNAVAILABLE"
        print(json.dumps({"state": "HOLD", "reason": reason}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
