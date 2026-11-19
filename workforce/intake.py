"""Admit exactly one officer record into the privileged attest job.

    python -B -m workforce.intake --dir DIR --role ROLE

The duty job runs code that reads untrusted network data and holds no signing
rights; the attest job holds ``id-token: write`` / ``attestations: write``.
The record crosses between them as a run artifact, so the attest job treats
it as untrusted input and signs it only if it is exactly what a duty job could
legitimately have produced:

  * the directory holds exactly one regular file, no symlinks, no extras;
  * its name is ``<role>-<first 12 hex of sha256(content)>.json`` for the
    expected role (so a swapped or edited file no longer matches its name);
  * it is no larger than core.MAX_BODY and is a JSON object carrying the
    officer schema, the same role and ``hf_space_required: false``;
  * it carries no credential-shaped string.

Prints ``record=<path>`` (also to $GITHUB_OUTPUT) and exits 0; any violation
exits 3 and writes no output, so nothing downstream runs.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from . import core, duties

NAME_RE = re.compile(r"^(?P<role>[a-z][a-z-]*)-(?P<digest>[0-9a-f]{12})\.json$")


def admit(directory, role, env=None):
    """Return the admitted record Path or raise ValueError(reason)."""
    if role not in duties.ROLES:
        raise ValueError(f"unknown role {role!r}")
    d = Path(directory)
    if not d.is_dir() or d.is_symlink():
        raise ValueError("artifact directory missing")
    entries = sorted(d.iterdir())
    if len(entries) != 1:
        raise ValueError(f"expected exactly one record file, found {len(entries)}")
    path = entries[0]
    if path.is_symlink() or not path.is_file():
        raise ValueError("record is not a regular file")
    m = NAME_RE.match(path.name)
    if not m or m.group("role") != role:
        raise ValueError("record name does not match the expected role")
    if path.stat().st_size > core.MAX_BODY:
        raise ValueError("record exceeds the size limit")
    blob = path.read_bytes()
    if core.sha256_hex(blob)[:12] != m.group("digest"):
        raise ValueError("record content does not match its name")
    try:
        rec = json.loads(blob)
    except ValueError:
        raise ValueError("record is not JSON") from None
    if (
        not isinstance(rec, dict)
        or rec.get("schema") != core.SCHEMA
        or rec.get("officer") != role
        or rec.get("hf_space_required") is not False
    ):
        raise ValueError("record does not carry the officer schema")
    core.refuse_secrets(blob, env=env)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--role", required=True)
    a = ap.parse_args(argv)
    try:
        path = admit(a.dir, a.role)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    print(f"record={path}")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"record={path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
