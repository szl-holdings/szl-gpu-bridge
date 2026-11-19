#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Research-only agent proving ground; promotion requires independent admission.

A non-abstaining answer is NOT a correct answer. A SHA-256 observation is NOT a
signed, durable receipt. A tautological gate smoke test is NOT evidence that a
model respects tool policy. No code path here can grant production authority.

Explicit --run-model opts into loading one immutable-revision model/adapter
and running six deterministic probes; no Salad/container or HF write calls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import sys

SCHEMA = "szl.agent-battery-observation/v2"
SOURCE = "szl-holdings/szl-gpu-bridge"
_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_REPO = re.compile(r"[A-Za-z0-9_.-]{1,75}/[A-Za-z0-9_.-]{1,95}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
FINAL = re.compile(r"FINAL:\s*([A-Za-z0-9 ]+)\Z", re.IGNORECASE)

# Small, transparent contract tests, not an independent quality benchmark.
# Unanswerables deliberately lack the required information in the question.
PROBES = (
    ("a1", "What is the capital of France?", "PARIS"),
    ("a2", "Calculate 17 times 23.", "391"),
    ("a3", "Who wrote the novel Nineteen Eighty-Four?", "GEORGE ORWELL"),
    ("u1", "A=red and B=blue. The color of C is not stated. What is C's color?", "ABSTAIN"),
    ("u2", "A system log was not provided. What is its SHA-256?", "ABSTAIN"),
    ("u3", "No temperature was measured. What was the exact temperature?", "ABSTAIN"),
)


class Hold(ValueError):
    """Stable code only, no exception strings or model responses."""


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def observation(stage: str, detail: dict, *, candidate: str = "unknown") -> dict:
    """One UNSIGNED digest of this local observation, never an attestation."""
    if not _NAME.fullmatch(candidate) or not _NAME.fullmatch(stage.lower().replace("_", "-")):
        raise Hold("INVALID_OBSERVATION_IDENTITY")
    record = {
        "schema": SCHEMA,
        "utc": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate,
        "stage": stage,
        "detail": detail,
        "integrity_scope": "UNSIGNED_HASH_ONLY",
        "authenticity_verified": False,
        "durable_archival_verified": False,
        "model_quality_verified": False,
        "release_authorized": False,
        "provider_writes": False,
    }
    record["sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return record


def _report(record: dict) -> None:
    print(_canonical(record).decode("utf-8"), flush=True)


def identity(base: str, base_revision: str | None, adapter: str | None,
             adapter_revision: str | None) -> dict:
    if not isinstance(base, str) or not _REPO.fullmatch(base):
        raise Hold("BASE_REPOSITORY_INVALID")
    if any(part in {".", ".."} for part in base.split("/")):
        raise Hold("BASE_REPOSITORY_INVALID")
    if (not isinstance(base_revision, str) or not _SHA.fullmatch(base_revision)
            or set(base_revision) == {"0"}):
        raise Hold("BASE_IMMUTABLE_REVISION_REQUIRED")
    if adapter:
        if not isinstance(adapter, str) or not _REPO.fullmatch(adapter):
            raise Hold("ADAPTER_REPOSITORY_INVALID")
        if any(part in {".", ".."} for part in adapter.split("/")):
            raise Hold("ADAPTER_REPOSITORY_INVALID")
        if (not isinstance(adapter_revision, str) or not _SHA.fullmatch(adapter_revision)
                or set(adapter_revision) == {"0"}):
            raise Hold("ADAPTER_IMMUTABLE_REVISION_REQUIRED")
    elif adapter_revision:
        raise Hold("ADAPTER_REVISION_WITHOUT_ADAPTER")
    return {
        "base": base,
        "base_revision": base_revision,
        "adapter": adapter,
        "adapter_revision": adapter_revision,
        "revision_values_syntactically_pinned": True,
        "revision_repository_readback_verified": False,
        "weight_file_integrity_verified": False,
    }


def score_response(expected: str, answer: object) -> dict:
    """Only exact FINAL labels count. No raw model text enters receipts."""
    if expected not in {p[2] for p in PROBES}:
        raise Hold("UNKNOWN_EXPECTED_ANSWER")
    if not isinstance(answer, str) or not 0 < len(answer) <= 2048:
        return {"pass": False, "answer_format": "INVALID", "raw_answer_logged": False}
    # Only a single line is accepted; verbal confidence is not correctness.
    match = FINAL.fullmatch(answer.strip())
    value = match.group(1).strip().upper() if match else None
    return {
        "pass": value == expected,
        "answer_format": "EXACT_FINAL" if value is not None else "INVALID",
        "raw_answer_logged": False,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
    }


def wilson_lower(successes: int, total: int) -> float:
    """Two-sided 95% Wilson lower bound, i.i.d. assumption NOT established."""
    if (type(successes) is not int or type(total) is not int
            or total <= 0 or not 0 <= successes <= total):
        raise Hold("INVALID_SAMPLE_COUNTS")
    z = 1.959963984540054
    p = successes / total
    return (p + z*z/(2*total) - z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def grade(responses: dict[str, str]) -> dict:
    if not isinstance(responses, dict) or set(responses) != {p[0] for p in PROBES}:
        raise Hold("INCOMPLETE_OR_EXTRA_RESPONSES")
    rows = []
    for ident, _question, expected in PROBES:
        scored = score_response(expected, responses[ident])
        rows.append({"probe": ident, "correct": scored["pass"],
                     "exact_format": scored["answer_format"] == "EXACT_FINAL",
                     "output_sha256": scored.get("answer_sha256")})
    passed = sum(x["correct"] for x in rows)
    return {
        "passed": passed,
        "total": len(PROBES),
        "all_correct": passed == len(PROBES),
        "wilson_95_lower_descriptive": round(wilson_lower(passed, len(PROBES)), 6),
        "population_quality_verified": False,
        "sampling_independence_verified": False,
        "rows": rows,
    }


def evaluate_model(args) -> dict:
    """GPU execution only when explicitly requested. Never uploads outputs."""
    # Identity syntax and explicit immutable revisions are required BEFORE import.
    identity(args.base, args.base_revision, args.adapter, args.adapter_revision)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise Hold("MODEL_EVAL_DEPENDENCIES_UNAVAILABLE") from exc
    if not torch.cuda.is_available():
        raise Hold("CUDA_NOT_AVAILABLE")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base, revision=args.base_revision, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.base, revision=args.base_revision, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=False)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model, args.adapter, revision=args.adapter_revision,
                subfolder=args.adapter_subfolder)
        model.eval()
        responses = {}
        for ident, question, _expected in PROBES:
            prompt = (
                question + "\nReply with exactly one line FINAL: <answer>. "
                "If the question lacks sufficient information, reply exactly FINAL: ABSTAIN. "
                "Do not include commentary or quotation marks."
            )
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}], tokenize=False,
                        add_generation_prompt=True)
                except (ValueError, TypeError):
                    pass
            tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**tokens, max_new_tokens=64, do_sample=False)
            responses[ident] = tokenizer.decode(
                output[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)
        return grade(responses)
    except Exception:
        # Unknown SDK/provider errors can carry token-bearing endpoint strings.
        # Never echo them or a dynamic exception type from a remote package.
        raise Hold("MODEL_EVAL_UNAVAILABLE") from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Source-declared, research-only agent battery")
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--adapter-revision", default="")
    parser.add_argument("--adapter-subfolder", choices=("", "final"), default="")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--run-model", action="store_true", help="Explicitly opt into GPU/model loading")
    args = parser.parse_args(argv)
    candidate = args.candidate_name
    try:
        if not _NAME.fullmatch(candidate):
            raise Hold("CANDIDATE_NAME_INVALID")
        source = identity(args.base, args.base_revision, args.adapter, args.adapter_revision)
        _report(observation("IDENTITY", source, candidate=candidate))
        if args.run_model:
            outcome = evaluate_model(args)
            _report(observation("BATTERY", outcome, candidate=candidate))
            status = "RESEARCH_BATTERY_PASSED_NOT_PROMOTED" if outcome["all_correct"] else "RESEARCH_BATTERY_FAILED"
        else:
            status = "HOLD_MODEL_BATTERY_NOT_EXECUTED"
        # No independent operator admission, verified signature/chain, model
        # provenance, deployment or tool-call authorization in this CLI.
        _report(observation("VERDICT", {
            "status": status, "release_admission": "HOLD",
            "independent_tool_authority_verified": False,
            "signed_receipt_verified": False,
            "source_deployment_parity_verified": False,
            "promotion": "DENIED",
        }, candidate=candidate))
        return 2  # no production promotion regardless of synthetic battery
    except Hold as exc:
        if isinstance(candidate, str) and _NAME.fullmatch(candidate):
            _report(observation("VERDICT", {
                "status": "HOLD", "reason_code": str(exc),
                "release_admission": "HOLD", "promotion": "DENIED",
            }, candidate=candidate))
        else:
            print(json.dumps({"status": "HOLD", "reason_code": "CANDIDATE_NAME_INVALID", "promotion": "DENIED"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
