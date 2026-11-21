"""Offline readers only. Verify a manifest committed last against independent inventory."""

import hashlib
import json
import math
import re


class Hold(ValueError):
    """Public reason code, never the offending value."""


def require(condition, reason):
    if not condition:
        raise Hold(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def document(raw, *, allow_float=False):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def reject(_):
        raise Hold("INVALID_NUMBER")

    def number(value):
        n = float(value)
        require(allow_float and math.isfinite(n), "INVALID_NUMBER")
        return n

    require(type(raw) is bytes and 0 < len(raw) <= 2_000_000, "DOCUMENT_SIZE")
    try:
        result = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject,
            parse_float=number,
        )
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, Hold):
            raise
        raise Hold("INVALID_JSON") from exc
    require(type(result) is dict, "DOCUMENT_NOT_OBJECT")
    return result


def inventory(paths):
    """Portable, case-insensitively unique regular-file namespace, no normalization."""
    require(type(paths) is list and 0 < len(paths) <= 4096, "INVALID_INVENTORY")
    names = set()
    for path in paths:
        require(type(path) is str and 0 < len(path) <= 256, "INVALID_SHARD_PATH")
        for part in path.split("/"):
            require(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
                and not part.endswith(".")
                and not re.fullmatch(
                    r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])", part.split(".")[0]
                ),
                "INVALID_SHARD_PATH",
            )
        require(path.lower() not in names, "DUPLICATE_SHARD_PATH")
        names.add(path.lower())
    for name in names:
        parts = name.split("/")
        require(
            not any("/".join(parts[:i]) in names for i in range(1, len(parts))),
            "INVALID_SHARD_PATH",
        )
    return set(paths)


class CheckpointContract:
    @staticmethod
    def verify(manifest_bytes, pin, run_id, attempt, site, expected_paths, readbacks):
        """Consume (path, iterable of byte chunks) pairs, never open a path or URI."""
        manifest = document(manifest_bytes)
        require(digest(manifest_bytes) == pin, "MANIFEST_DIGEST_MISMATCH")
        require(
            set(manifest)
            == {"schema", "runId", "attempt", "site", "step", "totalBytes", "shards"}
            and manifest["schema"] == "szl.checkpoint-shards/v1"
            and manifest["runId"] == run_id
            and type(manifest["attempt"]) is int
            and manifest["attempt"] == attempt
            and manifest["site"] == site
            and type(manifest["step"]) is int
            and manifest["step"] >= 0,
            "INVALID_CHECKPOINT",
        )
        expected = inventory(expected_paths)
        rows = manifest["shards"]
        require(type(rows) is list and len(rows) <= 4096, "INVALID_SHARDS")
        for row in rows:
            require(
                type(row) is dict
                and set(row) == {"path", "bytes", "sha256"}
                and type(row["bytes"]) is int
                and 0 < row["bytes"] < 2**63
                and type(row["sha256"]) is str
                and re.fullmatch("[0-9a-f]{64}", row["sha256"]),
                "INVALID_SHARD_RECORD",
            )
        # Never construct a dictionary from shard rows before detecting collisions.
        require(inventory([r["path"] for r in rows]) == expected, "PARTIAL_MANIFEST")
        sizes = {r["path"]: (r["bytes"], r["sha256"]) for r in rows}
        require(
            type(manifest["totalBytes"]) is int
            and manifest["totalBytes"] == sum(n for n, _ in sizes.values()),
            "CHECKPOINT_TOTAL_MISMATCH",
        )
        seen = set()
        try:
            for path, chunks in readbacks:
                require(
                    type(path) is str and path in sizes and path not in seen,
                    "READBACK_INVENTORY",
                )
                seen.add(path)
                limit, sha = sizes[path]
                h = hashlib.sha256()
                n = 0
                for chunk in chunks:
                    require(type(chunk) is bytes and chunk, "INVALID_READBACK_CHUNK")
                    n += len(chunk)
                    require(n <= limit, "SHARD_SIZE_MISMATCH")
                    h.update(chunk)
                require(n == limit, "SHARD_SIZE_MISMATCH")
                require(h.hexdigest() == sha, "SHARD_HASH_MISMATCH")
        except Hold:
            raise
        except Exception as exc:
            raise Hold("READBACK_UNAVAILABLE") from exc
        require(seen == expected, "READBACK_INVENTORY")
