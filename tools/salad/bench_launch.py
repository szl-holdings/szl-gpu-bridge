#!/usr/bin/env python3
# Copyright 2026 SZL Holdings -- SPDX-License-Identifier: Apache-2.0
"""One-shot, plan-first benchmark entrypoint. No scheduler or resource-control API.

The controller must admit the source/contract and authorize compute separately.
A matching hash proves bytes, NOT approval, execution history, or model quality.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SCHEMA = "szl.salad-bench-launch/v1"
MAX_BYTES = 1024 * 1024


def read_bytes(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("expected a regular non-symlink file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("launch input exceeds byte limit")
    return raw


def load_contract(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate contract key")
            result[key] = value
        return result

    def reject(_):
        raise ValueError("non-finite contract number")

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("contract must be an object")
    return value


def exact_keys(obj, names):
    if not isinstance(obj, dict) or set(obj) != set(names):
        raise ValueError("missing or unexpected contract fields")


def pattern(value, expression):
    if not isinstance(value, str) or re.fullmatch(expression, value) is None:
        raise ValueError("invalid contract identity")
    return value


def relative_path(value):
    pattern(value, r"[A-Za-z0-9_./-]+")
    if any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError("unsafe repository path")
    return value


def positive(value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer bound")
    return value


def plan(contract, harness, work_root):
    exact_keys(
        contract,
        ("schema", "attempt_id", "source", "inputs", "destination", "measurement"),
    )
    if contract["schema"] != SCHEMA:
        raise ValueError("unsupported launch contract")
    attempt = pattern(contract["attempt_id"], r"[0-9a-f]{32}")
    source = contract["source"]
    exact_keys(source, ("repository", "revision", "sha256"))
    if source["repository"] != "szl-holdings/szl-gpu-bridge":
        raise ValueError("unexpected source authority")
    revision = pattern(source["revision"], r"[0-9a-f]{40}")
    expected = pattern(source["sha256"], r"[0-9a-f]{64}")
    raw = read_bytes(harness)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("harness bytes do not match the admitted contract")
    tree = ast.parse(raw, filename="bench_eval.py")
    schema = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SCHEMA" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if (
        schema != ["szl.salad-bench/v2"]
        or not {"--run-id", "--launch-contract-sha256", "--source-revision"}
        <= constants
    ):
        raise ValueError("harness lacks the required v2 launch-binding interface")
    inputs, destination, measurement = (
        contract["inputs"],
        contract["destination"],
        contract["measurement"],
    )
    exact_keys(inputs, ("base", "adapter", "dataset"))
    exact_keys(inputs["base"], ("repo_id", "revision"))
    exact_keys(inputs["adapter"], ("repo_id", "revision", "directory"))
    exact_keys(
        inputs["dataset"],
        (
            "repo_id",
            "revision",
            "eval_file",
            "eval_sha256",
            "train_file",
            "train_sha256",
        ),
    )
    exact_keys(destination, ("repo_id", "repo_type", "prefix"))
    exact_keys(measurement, ("max_length", "train_ref_rows", "seed"))
    repo_pattern = r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*"
    for item in inputs.values():
        pattern(item["repo_id"], repo_pattern)
        pattern(item["revision"], r"[0-9a-f]{40}")
    pattern(destination["repo_id"], repo_pattern)
    if destination["repo_type"] not in ("dataset", "model", "space"):
        raise ValueError("receipt repository type must be explicit")
    relative_path(destination["prefix"])
    relative_path(inputs["adapter"]["directory"])
    for split in ("eval", "train"):
        relative_path(inputs["dataset"][split + "_file"])
        pattern(inputs["dataset"][split + "_sha256"], r"[0-9a-f]{64}")
    positive(measurement["max_length"], 2)
    positive(measurement["train_ref_rows"])
    positive(measurement["seed"], 0)
    root = Path(work_root).absolute()
    if root.is_symlink():
        raise ValueError("work root cannot be a symlink")
    attempt_dir = root / attempt
    if attempt_dir.exists() or attempt_dir.is_symlink():
        raise FileExistsError("attempt already exists; reconcile, do not remeasure")
    canonical = (
        json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    contract_hash = hashlib.sha256(canonical).hexdigest()
    flags = {
        "base": inputs["base"]["repo_id"],
        "base-revision": inputs["base"]["revision"],
        "adapter-repo": inputs["adapter"]["repo_id"],
        "adapter-revision": inputs["adapter"]["revision"],
        "adapter-dir": inputs["adapter"]["directory"],
        "dataset": inputs["dataset"]["repo_id"],
        "dataset-revision": inputs["dataset"]["revision"],
        "eval-file": inputs["dataset"]["eval_file"],
        "eval-sha256": inputs["dataset"]["eval_sha256"],
        "train-file": inputs["dataset"]["train_file"],
        "train-sha256": inputs["dataset"]["train_sha256"],
        "receipts-repo": destination["repo_id"],
        "receipts-repo-type": destination["repo_type"],
        "receipt-dir": destination["prefix"],
        "run-id": attempt,
        "source-revision": revision,
        "launch-contract-sha256": contract_hash,
        "max-length": measurement["max_length"],
        "train-ref-rows": measurement["train_ref_rows"],
        "seed": measurement["seed"],
        "output-dir": str(attempt_dir / "bundle"),
    }
    argv = [sys.executable, "-B", "-u", str(attempt_dir / "bench_eval.py")]
    for key, value in flags.items():
        argv.extend(("--" + key, str(value)))
    return (
        {
            "schema": SCHEMA,
            "status": "PLANNED_NOT_EXECUTED",
            "attempt_id": attempt,
            "source_revision": revision,
            "source_sha256": expected,
            "contract_sha256": contract_hash,
            "source_admission_verified": False,
            "argv": argv,
            "attempt_directory": str(attempt_dir),
        },
        raw,
        canonical,
    )


def stage(plan_record, harness_bytes, contract_bytes):
    directory = Path(plan_record["attempt_directory"])
    directory.mkdir(parents=True, exist_ok=False)
    for name, raw in (
        ("bench_eval.py", harness_bytes),
        ("launch-contract.json", contract_bytes),
        ("launch-plan.json", (json.dumps(plan_record, indent=2) + "\n").encode()),
    ):
        with (directory / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    # The directory is a once-only local claim, not a distributed controller lease.
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument(
        "--harness", default=str(Path(__file__).with_name("bench_eval.py"))
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="requires separate controller authorization; may incur compute/upload effects",
    )
    args = parser.parse_args(argv)
    try:
        record, raw, canonical = plan(
            load_contract(read_bytes(args.contract)), args.harness, args.work_root
        )
        if not args.execute:
            print(json.dumps(record, sort_keys=True), flush=True)
            return 0
        stage(record, raw, canonical)
        print(
            "BENCH_LAUNCH "
            + json.dumps(
                {
                    "attempt_id": record["attempt_id"],
                    "contract_sha256": record["contract_sha256"],
                }
            ),
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        # Windows execv can report the replaced wrapper's status instead of the
        # child's failure. Wait for the exact child there; production Linux uses
        # process replacement so container signals reach the benchmark directly.
        if sys.platform == "win32":
            return subprocess.run(record["argv"], check=False).returncode
        os.execv(sys.executable, record["argv"])
    except (OSError, ValueError, TypeError, SyntaxError, KeyError) as exc:
        print("LAUNCH_UNAVAILABLE: " + type(exc).__name__, file=sys.stderr, flush=True)
        return 7
    raise RuntimeError("exec unexpectedly returned")


if __name__ == "__main__":
    sys.exit(main())
