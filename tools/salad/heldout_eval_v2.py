#!/usr/bin/env python3
# Copyright 2026 SZL Holdings -- SPDX-License-Identifier: Apache-2.0
"""Isolated paired measurements and offline replay; never scientific acceptance.

Default is a network-free plan. The existing controller alone may admit --execute.
No provider control calls, publication, signing, dependency installs or retries.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
from importlib.metadata import version, PackageNotFoundError
import platform
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata

SCHEMA = "szl.heldout-paired/v2"
MAX_BYTES = 8 * 1024 * 1024
MAX_TRAINING_BYTES = 256 * 1024 * 1024
MAX_TRAINING_LINE_BYTES = 2 * 1024 * 1024
MAX_TRAINING_ROWS = 500000
MAX_ROWS = 2000
TASKS = ("neutral_mcq", "refusal", "invented_id")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def strict_json(raw: bytes):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate_json_key")
            out[key] = value
        return out

    def bad_constant(_):
        raise ValueError("nonfinite_json")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def read_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("input_not_regular_file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("input_byte_limit")
    return raw


def exact(obj, keys):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ValueError("unexpected_schema_fields")


def hex_value(value, size):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{%d}" % size, value) is None
    ):
        raise ValueError("invalid_immutable_identity")


def bounded_int(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("invalid_integer_bound")


def relative_path(value):
    if (
        not isinstance(value, str)
        or not value
        or any(
            re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None or part in (".", "..")
            for part in value.split("/")
        )
    ):
        raise ValueError("unsafe_repository_path")


def identity(obj, extras=()):
    exact(obj, ("repo_id", "revision", *extras))
    if not isinstance(obj["repo_id"], str) or REPO.fullmatch(obj["repo_id"]) is None:
        raise ValueError("invalid_repository")
    hex_value(obj["revision"], 40)


def messages(value, *, training=False):
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError("invalid_messages")
    out = []
    for item in value:
        if (
            training
            and isinstance(item, dict)
            and set(item) == {"role", "content", "tool_calls"}
        ):
            if type(item["tool_calls"]) is not list or item["tool_calls"]:
                raise ValueError("nonempty_training_tool_calls_not_admitted")
        else:
            exact(item, ("role", "content"))
        if item["role"] not in ("system", "user", "assistant"):
            raise ValueError("invalid_message_role")
        content = item["content"]
        content_limit = MAX_TRAINING_LINE_BYTES if training else 16000
        if (
            not isinstance(content, str)
            or not content.strip()
            or len(content) > content_limit
        ):
            raise ValueError("invalid_message_content")
        out.append(
            {"role": item["role"], "content": unicodedata.normalize("NFC", content)}
        )
    if training and out[-1]["role"] == "assistant":
        out = out[:-1]
    if not out or out[-1]["role"] != "user":
        raise ValueError("prompt_must_end_with_user")
    return out


def jsonl(raw):
    rows = []
    for line in raw.splitlines():
        if line.strip():
            item = strict_json(line)
            if not isinstance(item, dict):
                raise ValueError("row_must_be_object")
            rows.append(item)
    if not rows:
        raise ValueError("empty_population")
    return rows


def training_stream(value):
    if isinstance(value, bytes):
        return io.BytesIO(value)
    if not isinstance(value, Path) or value.is_symlink() or not value.is_file():
        raise ValueError("training_not_regular_file")
    return value.open("rb")


def inspect_training(value):
    """Hash every byte and inspect every row, retaining prompt digests only."""
    h, size, count, prompts = hashlib.sha256(), 0, 0, set()
    with training_stream(value) as stream:
        while True:
            line = stream.readline(MAX_TRAINING_LINE_BYTES + 1)
            if not line:
                break
            size += len(line)
            if len(line) > MAX_TRAINING_LINE_BYTES or size > MAX_TRAINING_BYTES:
                raise ValueError("training_byte_limit")
            h.update(line)
            if not line.strip():
                continue
            row = strict_json(line)
            if not isinstance(row, dict):
                raise ValueError("training_row_not_object")
            count += 1
            if count > MAX_TRAINING_ROWS:
                raise ValueError("training_row_limit")
            prompts.add(digest(encoded(messages(row.get("messages"), training=True))))
    if not count:
        raise ValueError("empty_training_population")
    return {
        "sha256": h.hexdigest(),
        "rows": count,
        "bytes": size,
        "prompt_digests": prompts,
    }


def training_digest(value):
    h, size = hashlib.sha256(), 0
    with training_stream(value) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            if size > MAX_TRAINING_BYTES:
                raise ValueError("training_byte_limit")
            h.update(chunk)
    if not size:
        raise ValueError("empty_training_population")
    return h.hexdigest()


def copy_training(value, destination, expected):
    h, size = hashlib.sha256(), 0
    with training_stream(value) as source, destination.open("xb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            if size > MAX_TRAINING_BYTES:
                raise ValueError("training_byte_limit")
            h.update(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    if h.hexdigest() != expected:
        raise ValueError("training_changed_during_snapshot")


def prepare(contract_raw, fixture_raw, training_raw, source_raw):
    c = strict_json(contract_raw)
    exact(
        c,
        (
            "schema",
            "attempt_id",
            "source",
            "base",
            "adapter",
            "fixtures",
            "training",
            "generation",
        ),
    )
    if c["schema"] != SCHEMA:
        raise ValueError("unsupported_contract")
    hex_value(c["attempt_id"], 32)
    exact(c["source"], ("repository", "revision", "sha256"))
    if c["source"]["repository"] != "szl-holdings/szl-gpu-bridge":
        raise ValueError("wrong_source_repository")
    hex_value(c["source"]["revision"], 40)
    hex_value(c["source"]["sha256"], 64)
    if digest(source_raw) != c["source"]["sha256"]:
        raise ValueError("source_bytes_mismatch")
    identity(c["base"])
    identity(c["adapter"], ("directory",))
    relative_path(c["adapter"]["directory"])
    for name in ("fixtures", "training"):
        identity(c[name], ("path", "sha256"))
        relative_path(c[name]["path"])
        hex_value(c[name]["sha256"], 64)
    if digest(fixture_raw) != c["fixtures"]["sha256"]:
        raise ValueError("fixtures_bytes_mismatch")
    g = c["generation"]
    exact(g, ("max_input_tokens", "max_new_tokens", "seed", "timeout_seconds"))
    bounded_int(g["max_input_tokens"], 2, 8192)
    bounded_int(g["max_new_tokens"], 1, 256)
    bounded_int(g["seed"], 0, 2**31 - 1)
    bounded_int(g["timeout_seconds"], 1, 86400)
    rows = jsonl(fixture_raw)
    if len(rows) > MAX_ROWS:
        raise ValueError("fixture_row_limit")
    seen, prompt_seen = set(), set()
    for row in rows:
        exact(row, ("id", "task", "messages", "expected_choice"))
        if (
            not isinstance(row["id"], str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", row["id"]) is None
        ):
            raise ValueError("invalid_fixture_id")
        if row["id"] in seen or row["task"] not in TASKS:
            raise ValueError("duplicate_id_or_unknown_task")
        seen.add(row["id"])
        if row["task"] == "neutral_mcq":
            if row["expected_choice"] not in ("A", "B", "C", "D"):
                raise ValueError("invalid_expected_choice")
        elif row["expected_choice"] is not None:
            raise ValueError("unadmitted_automated_scorer")
        key = digest(encoded(messages(row["messages"])))
        if key in prompt_seen:
            raise ValueError("duplicate_fixture_prompt")
        prompt_seen.add(key)
    training = inspect_training(training_raw)
    if training["sha256"] != c["training"]["sha256"]:
        raise ValueError("training_bytes_mismatch")
    if prompt_seen & training["prompt_digests"]:
        raise ValueError("exact_training_prompt_overlap")
    plan = {
        "schema": "szl.heldout-plan/v2",
        "attempt_id": c["attempt_id"],
        "contract_sha256": digest(encoded(c)),
        "source_sha256": digest(source_raw),
        "fixtures_sha256": digest(fixture_raw),
        "training_sha256": training["sha256"],
        "fixture_count": len(rows),
        "training_rows_checked": training["rows"],
        "training_bytes_checked": training["bytes"],
        "task_counts": dict(Counter(row["task"] for row in rows)),
        "isolation": "FRESH_PROCESS_PER_ARM_BASE_BEFORE_ADAPTER",
        "exact_prompt_overlap_count": 0,
        "independent_holdout_verified": False,
        "overlap_scope": "NFC canonical role/content identity after removing the final training answer; not semantic or benchmark-leakage detection",
        "source_admission_verified": False,
        "compute_authorized": False,
        "quality_qualification": "NOT_ESTABLISHED",
        "publication_performed": False,
    }
    return c, rows, plan


def score_pairs(c, fixtures, arms):
    """Replay every paired row; no missing rows, guessed labels or threshold tuning."""
    scored = {}
    for arm in ("base", "adapter"):
        rows = arms[arm]
        if len(rows) != len(fixtures):
            raise ValueError("incomplete_arm_coverage")
        by_task = {
            task: {
                "offered": 0,
                "correct": 0,
                "format_errors": 0,
                "budget_exhausted": 0,
            }
            for task in TASKS
        }
        for index, (fixture, row) in enumerate(zip(fixtures, rows)):
            exact(
                row,
                (
                    "index",
                    "id",
                    "arm",
                    "fixture_sha256",
                    "prompt_sha256",
                    "output",
                    "output_sha256",
                    "input_tokens",
                    "generated_tokens",
                    "budget_exhausted",
                ),
            )
            if (
                type(row["index"]) is not int
                or row["index"] != index
                or row["id"] != fixture["id"]
                or row["arm"] != arm
                or row["fixture_sha256"] != digest(encoded(fixture))
                or row["prompt_sha256"]
                != digest(encoded(messages(fixture["messages"])))
            ):
                raise ValueError("paired_identity_mismatch")
            output = row["output"]
            if not isinstance(output, str) or len(output.encode("utf-8")) > 65536:
                raise ValueError("invalid_output")
            if row["output_sha256"] != digest(output.encode("utf-8")):
                raise ValueError("output_bytes_mismatch")
            bounded_int(row["input_tokens"], 1, c["generation"]["max_input_tokens"])
            bounded_int(row["generated_tokens"], 1, c["generation"]["max_new_tokens"])
            if type(row["budget_exhausted"]) is not bool:
                raise ValueError("invalid_termination_flag")
            if (
                row["budget_exhausted"]
                and row["generated_tokens"] != c["generation"]["max_new_tokens"]
            ):
                raise ValueError("inconsistent_termination_flag")
            task = by_task[fixture["task"]]
            task["offered"] += 1
            task["budget_exhausted"] += int(row["budget_exhausted"])
            if fixture["task"] == "neutral_mcq":
                prediction = output.strip()
                valid = (
                    prediction in ("A", "B", "C", "D") and not row["budget_exhausted"]
                )
                task["format_errors"] += int(not valid)
                task["correct"] += int(
                    valid and prediction == fixture["expected_choice"]
                )
        scored[arm] = {}
        for task, values in by_task.items():
            n = values["offered"]
            measured = task == "neutral_mcq" and n > 0
            scored[arm][task] = {
                "rows_offered": n,
                "correct": values["correct"] if measured else None,
                "accuracy": values["correct"] / n if measured else None,
                "format_errors": values["format_errors"] if measured else None,
                "budget_exhausted": values["budget_exhausted"],
                "scoring_state": "MEASURED_EXACT_CHOICE"
                if measured
                else "UNAVAILABLE_EMPTY"
                if not n
                else "REQUIRES_INDEPENDENT_ADJUDICATION",
            }
    if [r["input_tokens"] for r in arms["base"]] != [
        r["input_tokens"] for r in arms["adapter"]
    ]:
        raise ValueError("tokenization_coverage_mismatch")
    return {
        "schema": "szl.heldout-measurements/v2",
        "attempt_id": c["attempt_id"],
        "arms": scored,
        "quality_qualification": "NOT_ESTABLISHED",
        "acceptance_policy_applied": False,
        "contamination_qualification": "NOT_ESTABLISHED",
        "energy_j": None,
        "energy_state": "NOT_MEASURED",
        "publication_performed": False,
        "production_ready": False,
    }


def exclusive(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def worker(directory, arm):
    # Parent has already claimed and snapshotted the exact input/source bytes.
    c, fixtures, plan = prepare(
        read_bytes(directory / "contract.json"),
        read_bytes(directory / "fixtures.jsonl"),
        directory / "training.jsonl",
        read_bytes(Path(__file__)),
    )
    if (
        directory.name != c["attempt_id"]
        or strict_json(read_bytes(directory / "plan.json")) != plan
    ):
        raise ValueError("unclaimed_worker_identity")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("admitted_bf16_cuda_runtime_required")
    torch.manual_seed(c["generation"]["seed"])
    torch.cuda.manual_seed_all(c["generation"]["seed"])
    base = c["base"]
    tok = AutoTokenizer.from_pretrained(
        base["repo_id"], revision=base["revision"], trust_remote_code=False
    )
    if tok.chat_template is None or tok.eos_token_id is None:
        raise ValueError("tokenizer_contract_unavailable")
    model = AutoModelForCausalLM.from_pretrained(
        base["repo_id"],
        revision=base["revision"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=False,
    )
    if getattr(model, "_hf_peft_config_loaded", False) or getattr(
        model, "peft_config", None
    ):
        raise ValueError("base_model_already_contains_adapter")
    if arm == "adapter":
        from huggingface_hub import snapshot_download
        from peft import PeftModel

        ad = c["adapter"]
        snapshot = snapshot_download(
            ad["repo_id"],
            revision=ad["revision"],
            allow_patterns=[ad["directory"] + "/*"],
        )
        model = PeftModel.from_pretrained(
            model, str(Path(snapshot) / ad["directory"]), is_trainable=False
        )
    model.eval()
    with (directory / (arm + ".jsonl")).open("xb") as journal:
        for index, fixture in enumerate(fixtures):
            prompt = messages(fixture["messages"])
            ids = tok.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False,
                truncation=False,
            )
            input_count = ids.shape[-1]
            if not 1 <= input_count <= c["generation"]["max_input_tokens"]:
                raise ValueError("input_token_budget_exceeded_no_truncation")
            ids = ids.to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    return_dict_in_generate=False,
                    max_new_tokens=c["generation"]["max_new_tokens"],
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id
                    if tok.pad_token_id is not None
                    else tok.eos_token_id,
                )
            new_ids = outputs[0, input_count:]
            count = len(new_ids)
            bounded_int(count, 1, c["generation"]["max_new_tokens"])
            output = tok.decode(new_ids, skip_special_tokens=True)
            if not isinstance(output, str) or len(output.encode("utf-8")) > 65536:
                raise ValueError("generated_output_byte_limit")
            row = {
                "index": index,
                "id": fixture["id"],
                "arm": arm,
                "fixture_sha256": digest(encoded(fixture)),
                "prompt_sha256": digest(encoded(prompt)),
                "output": output,
                "output_sha256": digest(output.encode("utf-8")),
                "input_tokens": input_count,
                "generated_tokens": count,
                "budget_exhausted": count == c["generation"]["max_new_tokens"]
                and int(new_ids[-1]) != tok.eos_token_id,
            }
            row_bytes = encoded(row)
            if journal.tell() + len(row_bytes) > MAX_BYTES:
                raise ValueError("journal_byte_limit")
            journal.write(row_bytes)
            journal.flush()
            os.fsync(journal.fileno())

    try:
        peft_version = version("peft")
    except PackageNotFoundError:
        peft_version = None
    return {
        "schema": "szl.heldout-worker/v2",
        "arm": arm,
        "state": "COMPLETE",
        "pid": os.getpid(),
        "attempt_id": c["attempt_id"],
        "contract_sha256": digest(encoded(c)),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft_version,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "chat_template_sha256": digest(tok.get_chat_template().encode("utf-8")),
        },
        "runtime_admission_verified": False,
    }


def run_worker(directory, arm):
    # Validate the claimed directory before any diagnostic write.
    c, _, plan = prepare(
        read_bytes(directory / "contract.json"),
        read_bytes(directory / "fixtures.jsonl"),
        directory / "training.jsonl",
        read_bytes(Path(__file__)),
    )
    if (
        directory.name != c["attempt_id"]
        or strict_json(read_bytes(directory / "plan.json")) != plan
    ):
        raise ValueError("unclaimed_worker_identity")
    try:
        result = worker(directory, arm)
        exclusive(directory / (arm + "-runtime.json"), encoded(result))
    except Exception as error:
        exclusive(
            directory / (arm + "-failure.json"),
            encoded(
                {
                    "schema": "szl.heldout-worker-failure/v2",
                    "attempt_id": c["attempt_id"],
                    "arm": arm,
                    "error_type": type(error).__name__,
                    "state": "UNAVAILABLE",
                    "raw_exception_omitted": True,
                }
            ),
        )
        raise


def validate_runtime(c, raw):
    environments = []
    for arm in ("base", "adapter"):
        value = strict_json(raw[arm + "-runtime.json"])
        exact(
            value,
            (
                "schema",
                "arm",
                "state",
                "pid",
                "attempt_id",
                "contract_sha256",
                "environment",
                "runtime_admission_verified",
            ),
        )
        if (
            value["schema"] != "szl.heldout-worker/v2"
            or value["arm"] != arm
            or value["state"] != "COMPLETE"
            or value["attempt_id"] != c["attempt_id"]
            or value["contract_sha256"] != digest(encoded(c))
            or value["runtime_admission_verified"] is not False
        ):
            raise ValueError("worker_runtime_binding_mismatch")
        bounded_int(value["pid"], 1, 2**31 - 1)
        env = value["environment"]
        exact(
            env,
            (
                "python",
                "torch",
                "transformers",
                "peft",
                "cuda",
                "gpu_name",
                "chat_template_sha256",
            ),
        )
        for name, observed in env.items():
            if name == "peft" and observed is None:
                continue
            if not isinstance(observed, str) or not 1 <= len(observed) <= 256:
                raise ValueError("missing_runtime_observation")
        hex_value(env["chat_template_sha256"], 64)
        environments.append(env)
    if environments[0] != environments[1]:
        raise ValueError("paired_runtime_drift")


def replay(directory):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("bundle_directory_not_regular")
    manifest = strict_json(read_bytes(directory / "bundle.json"))
    exact(manifest, ("schema", "files", "authenticity_verified"))
    if (
        manifest["schema"] != "szl.heldout-bundle/v2"
        or manifest["authenticity_verified"] is not False
    ):
        raise ValueError("invalid_bundle_claim")
    expected = {
        "contract.json",
        "fixtures.jsonl",
        "training.jsonl",
        "heldout_eval_v2.py",
        "plan.json",
        "base.jsonl",
        "adapter.jsonl",
        "measurements.json",
        "base-runtime.json",
        "adapter-runtime.json",
    }
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != expected:
        raise ValueError("incomplete_bundle")
    raw = {
        name: directory / name
        if name == "training.jsonl"
        else read_bytes(directory / name)
        for name in expected
    }
    for name, value in raw.items():
        observed = training_digest(value) if name == "training.jsonl" else digest(value)
        if observed != manifest["files"][name]:
            raise ValueError("bundle_bytes_mismatch")
    c, fixtures, plan = prepare(
        raw["contract.json"],
        raw["fixtures.jsonl"],
        raw["training.jsonl"],
        raw["heldout_eval_v2.py"],
    )
    validate_runtime(c, raw)
    if strict_json(raw["plan.json"]) != plan:
        raise ValueError("plan_mismatch")
    result = score_pairs(
        c, fixtures, {arm: jsonl(raw[arm + ".jsonl"]) for arm in ("base", "adapter")}
    )
    if encoded(result) != raw["measurements.json"]:
        raise ValueError("replayed_measurement_mismatch")
    return result


def execute(c, plan, raw, root, *, run=subprocess.run):
    if root.is_symlink():
        raise ValueError("symlink_output_root")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = root / c["attempt_id"]
    directory.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    for name, value in raw.items():
        if name == "training.jsonl":
            copy_training(value, directory / name, c["training"]["sha256"])
        else:
            exclusive(directory / name, value)
    exclusive(directory / "plan.json", encoded(plan))
    terminal = {
        "schema": "szl.heldout-terminal/v2",
        "attempt_id": c["attempt_id"],
        "state": "UNAVAILABLE",
        "quality_qualification": "NOT_ESTABLISHED",
        "source_admission_verified": False,
        "publication_performed": False,
    }
    try:
        for arm in ("base", "adapter"):
            remaining = c["generation"]["timeout_seconds"] - (
                time.monotonic() - started
            )
            if remaining <= 0:
                raise TimeoutError("attempt_deadline")
            process = run(
                [
                    sys.executable,
                    str(directory / "heldout_eval_v2.py"),
                    "--worker",
                    arm,
                    "--bundle",
                    str(directory),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=remaining,
                check=False,
            )
            if process.returncode:
                terminal["failed_arm"] = arm
                terminal["worker_exit_code"] = process.returncode
                diagnostic = directory / (arm + "-failure.json")
                if diagnostic.is_file():
                    terminal["worker_failure"] = strict_json(read_bytes(diagnostic))
                raise RuntimeError(arm + "_worker_failed")
        arms = {
            arm: jsonl(read_bytes(directory / (arm + ".jsonl")))
            for arm in ("base", "adapter")
        }
        runtime = {
            a + "-runtime.json": read_bytes(directory / (a + "-runtime.json"))
            for a in ("base", "adapter")
        }
        validate_runtime(c, runtime)
        result = score_pairs(c, jsonl(raw["fixtures.jsonl"]), arms)
        exclusive(directory / "measurements.json", encoded(result))
        names = set(raw) | {
            "plan.json",
            "base.jsonl",
            "adapter.jsonl",
            "measurements.json",
            "base-runtime.json",
            "adapter-runtime.json",
        }
        manifest = {
            "schema": "szl.heldout-bundle/v2",
            "authenticity_verified": False,
            "files": {
                name: training_digest(directory / name)
                if name == "training.jsonl"
                else digest(read_bytes(directory / name))
                for name in sorted(names)
            },
        }
        exclusive(directory / "bundle.json", encoded(manifest))
        replay(directory)
        terminal["state"] = "MEASUREMENTS_COMPLETE_REPLAY_VERIFIED"
        return_code = 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        TimeoutError,
        subprocess.SubprocessError,
    ) as error:
        terminal["error_type"] = type(error).__name__
        terminal["state"] = (
            "PARTIAL"
            if any((directory / (a + ".jsonl")).exists() for a in ("base", "adapter"))
            else "UNAVAILABLE"
        )
        return_code = 7
    terminal["elapsed_seconds"] = round(time.monotonic() - started, 6)
    exclusive(directory / "terminal.json", encoded(terminal))
    return return_code, terminal


def audit_splits(training_path, evaluation_path, output):
    """Prospective exact-disjoint candidate; originals and scientific scope unchanged."""
    if output.is_symlink():
        raise ValueError("symlink_audit_output")
    training = inspect_training(training_path)
    evaluation = inspect_training(evaluation_path)
    shared = training["prompt_digests"] & evaluation["prompt_digests"]
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    kept, excluded, index = 0, 0, 0
    candidate_prompts = set()
    input_hash, candidate_hash = hashlib.sha256(), hashlib.sha256()
    with (
        training_stream(evaluation_path) as source,
        (output / "candidate-eval.jsonl").open("xb") as candidate,
        (output / "partition.jsonl").open("xb") as partition,
    ):
        total = 0
        while True:
            line = source.readline(MAX_TRAINING_LINE_BYTES + 1)
            if not line:
                break
            total += len(line)
            if total > MAX_TRAINING_BYTES or len(line) > MAX_TRAINING_LINE_BYTES:
                raise ValueError("evaluation_byte_limit")
            input_hash.update(line)
            if not line.strip():
                continue
            row = strict_json(line)
            prompt = digest(encoded(messages(row.get("messages"), training=True)))
            reject = prompt in training["prompt_digests"]
            partition.write(
                encoded(
                    {
                        "row_index": index,
                        "row_sha256": digest(line),
                        "prompt_sha256": prompt,
                        "disposition": "QUARANTINE_EXACT_TRAIN_PROMPT"
                        if reject
                        else "CANDIDATE_NOT_QUALIFIED",
                    }
                )
            )
            index += 1
            if reject:
                excluded += 1
            else:
                candidate.write(line)
                candidate_hash.update(line)
                candidate_prompts.add(prompt)
                kept += 1
        candidate.flush()
        os.fsync(candidate.fileno())
        partition.flush()
        os.fsync(partition.fileno())
    if input_hash.hexdigest() != evaluation["sha256"] or index != evaluation["rows"]:
        raise ValueError("evaluation_changed_during_partition")
    if training_digest(training_path) != training["sha256"]:
        raise ValueError("training_changed_during_partition")
    if kept:
        check = inspect_training(output / "candidate-eval.jsonl")
        if (
            check["rows"] != kept
            or check["sha256"] != candidate_hash.hexdigest()
            or check["prompt_digests"] & training["prompt_digests"]
        ):
            raise ValueError("candidate_partition_verification_failed")
    for observed in (training, evaluation):
        observed["unique_prompts"] = len(observed.pop("prompt_digests"))
    result = {
        "schema": "szl.exact-prompt-partition/v1",
        "state": "EXACT_OVERLAP_OBSERVED" if shared else "NO_EXACT_OVERLAP_OBSERVED",
        "training": training,
        "evaluation": evaluation,
        "shared_unique_prompt_identities": len(shared),
        "quarantined_eval_rows": excluded,
        "candidate_eval_rows": kept,
        "candidate_unique_prompts": len(candidate_prompts),
        "candidate_sha256": candidate_hash.hexdigest(),
        "candidate_state": "CANDIDATE_NOT_QUALIFIED" if kept else "EMPTY_UNAVAILABLE",
        "harness_sha256": digest(read_bytes(Path(__file__))),
        "partition_sha256": training_digest(output / "partition.jsonl"),
        "scope": "NFC role/content sequence minus final training/eval assistant response; empty tool_calls arrays treated as absent. Exact prompt identity only.",
        "original_inputs_modified": False,
        "rows_relabelled": False,
        "publication_performed": False,
        "gpu_evaluation_performed": False,
        "independent_holdout_verified": False,
        "semantic_leakage_checked": False,
        "quality_qualification": "NOT_ESTABLISHED",
        "selection_warning": "Deterministic prospective decontamination changes the evaluation population; inspect source/task mix and freeze protocol before measurement. It does not retroactively repair an old score.",
    }
    exclusive(output / "split-audit.json", encoded(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--audit-eval",
        type=Path,
        help="Private prospective exact-disjoint candidate; never acceptance",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--worker", choices=("base", "adapter"), help=argparse.SUPPRESS)
    parser.add_argument("--bundle", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.audit_eval:
            if (
                not args.training
                or not args.output_root
                or args.execute
                or args.worker
                or args.replay
                or args.contract
                or args.fixtures
                or args.bundle
            ):
                raise ValueError(
                    "split_audit_requires_only_training_eval_and_new_output"
                )
            result = audit_splits(args.training, args.audit_eval, args.output_root)
            print(json.dumps(result, allow_nan=False))
            return 3 if result["shared_unique_prompt_identities"] else 0
        if args.worker:
            if args.bundle is None or args.execute or args.replay:
                raise ValueError("invalid_worker_arguments")
            run_worker(args.bundle.resolve(), args.worker)
            return 0
        if args.replay:
            if args.execute or any(
                (
                    args.contract,
                    args.fixtures,
                    args.training,
                    args.output_root,
                    args.bundle,
                )
            ):
                raise ValueError("replay_requires_original_bundle_only")
            result = replay(args.replay)
        else:
            if not all((args.contract, args.fixtures, args.training)) or args.bundle:
                raise ValueError("contract_fixtures_and_training_required")
            source = read_bytes(Path(__file__))
            c_raw, f_raw = map(read_bytes, (args.contract, args.fixtures))
            t_raw = args.training
            c, _, result = prepare(c_raw, f_raw, t_raw, source)
            if args.execute:
                if args.output_root is None:
                    raise ValueError("exclusive_output_root_required")
                raw = {
                    "contract.json": encoded(c),
                    "fixtures.jsonl": f_raw,
                    "training.jsonl": t_raw,
                    "heldout_eval_v2.py": source,
                }
                code, result = execute(c, result, raw, args.output_root.absolute())
                print(json.dumps(result, allow_nan=False))
                return code
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        # Do not echo input data, remote response bodies, secrets or prompts.
        print(
            json.dumps(
                {
                    "state": "UNAVAILABLE",
                    "error_type": type(error).__name__,
                    "quality_qualification": "NOT_ESTABLISHED",
                }
            ),
            file=sys.stderr,
        )
        return 7


if __name__ == "__main__":
    raise SystemExit(main())
