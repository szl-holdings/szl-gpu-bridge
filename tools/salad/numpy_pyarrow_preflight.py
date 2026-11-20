"""Fail-closed static NumPy/PyArrow preflight for source-owned training recipes.

PyArrow 26 uses NumPy >=2 at runtime, but some wheels omit a resolver-visible
NumPy runtime requirement. 'pip install' exit 0 is not proof that 'import
pyarrow' succeeds. This checks *only* exact claimed pins, never imports model
libraries or calls cloud providers. Approved runtime import and GPU evaluation
are separate owner gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

SCHEMA = "szl.arrow-numpy-preflight/v1"
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\Z")
_PYTHON = re.compile(r"3\.(10|11|12|13|14|15)\Z")
_PIN = re.compile(r"\s*(numpy|pyarrow)==([0-9]+\.[0-9]+\.[0-9]+)\s*\\?\s*\Z")
MAX_LOCK_BYTES = 256_000
# Specific older wheels with NumPy 1.26 documented in upstream Apache Arrow
# issue #53954, plus the owner-reproduced 19.0.1 lane. Other pairs stay HOLD.
_NUMPY1_ARROW = {"19.0.1", "21.0.0", "22.0.0", "23.0.1", "24.0.0", "25.0.1"}
_NUMPY2_ARROW = {"26.0.0", "26.0.1"}


def _hold(reason: str) -> dict:
    return {
        "schema": SCHEMA,
        "verdict": "HOLD",
        "reason": reason,
        "runtime_import_verified": False,
        "dependency_install_verified": False,
        "training_complete_verified": False,
        "model_qualified": False,
        "cloud_mutation_authorized": False,
    }


def review_pair(*, numpy: object, pyarrow: object, python: object) -> dict:
    """Check a strictly pinned candidate; never establish successful import."""
    if type(numpy) is not str or not _VERSION.fullmatch(numpy):
        return _hold("NUMPY_EXACT_PIN_MISSING")
    if type(pyarrow) is not str or not _VERSION.fullmatch(pyarrow):
        return _hold("PYARROW_EXACT_PIN_MISSING")
    if type(python) is not str or not _PYTHON.fullmatch(python):
        return _hold("PYTHON_RUNTIME_UNVERIFIED")
    npv = tuple(map(int, numpy.split(".")))
    pyv = int(python.split(".")[1])

    if pyarrow in _NUMPY2_ARROW:
        if pyv < 11:
            return _hold("PYARROW26_PYTHON_TOO_OLD")
        if not 2 <= npv[0] < 3:
            return _hold("PYARROW26_REQUIRES_NUMPY2")
    elif pyarrow in _NUMPY1_ARROW:
        # Do not infer NumPy 2 binary interoperability of older wheels.
        if npv != (1, 26, 4):
            return _hold("HISTORICAL_NUMPY1_PAIR_UNVERIFIED")
    else:
        return _hold("PYARROW_VERSION_NOT_IN_QUALIFIED_SOURCE_SET")

    result = _hold("SOURCE_PAIR_ONLY_RUNTIME_IMPORT_REQUIRED")
    result["verdict"] = "SOURCE_PAIR_FOR_OWNER_REVIEW"
    return result


def review_lock_contents(contents: object, *, python: str) -> dict:
    """Parse only NumPy and PyArrow exact pins in a bounded pip hash-lock."""
    if (
        type(contents) is not str
        or not contents
        or len(contents.encode("utf-8")) > MAX_LOCK_BYTES
    ):
        return _hold("LOCK_BYTES_INVALID")
    pinned: dict[str, str] = {}
    for line in contents.splitlines():
        stripped = line.strip().lower()
        if not stripped or stripped.startswith("#"):
            continue
        package = None
        for name in ("numpy", "pyarrow"):
            if stripped.startswith(name) and (
                len(stripped) == len(name)
                or stripped[len(name)] in "<=>!~["
            ):
                package = name
                break
        if package is None:
            continue
        match = _PIN.fullmatch(stripped)
        if not match or match[1] != package:
            return _hold("LOCK_EXACT_PIN_UNVERIFIED")
        if package in pinned:
            return _hold("LOCK_DUPLICATE_PIN")
        pinned[package] = match[2]
    if set(pinned) != {"numpy", "pyarrow"}:
        return _hold("LOCK_NUMPY_PYARROW_PIN_MISSING")
    return review_pair(
        numpy=pinned["numpy"],
        pyarrow=pinned["pyarrow"],
        python=python,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requirements_file", type=Path)
    parser.add_argument("--python-version", required=True)
    args = parser.parse_args()
    try:
        if args.requirements_file.is_symlink():
            result = _hold("LOCK_SYMLINK_UNVERIFIED")
        elif not args.requirements_file.is_file():
            result = _hold("LOCK_FILE_MISSING")
        elif args.requirements_file.stat().st_size > MAX_LOCK_BYTES:
            result = _hold("LOCK_BYTES_INVALID")
        else:
            result = review_lock_contents(
                args.requirements_file.read_text(encoding="utf-8"),
                python=args.python_version,
            )
    except (OSError, UnicodeError):
        result = _hold("LOCK_IO_UNAVAILABLE")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "SOURCE_PAIR_FOR_OWNER_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
