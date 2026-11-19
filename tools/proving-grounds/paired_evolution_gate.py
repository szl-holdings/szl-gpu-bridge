#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Research-only, frozen-family paired evaluator for agent/code evolution.

An LLM proposal, benchmark score, hash or p-value is not permission to run
training, deploy code or publish a quality claim. No code execution, network,
model imports, credentials, provider writes or promotion API exists here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "szl.frozen-paired-evolution/v1"
REPORT_SCHEMA = "szl.frozen-paired-evolution-observation/v1"
REPOSITORY = "szl-holdings/szl-gpu-bridge"
REV = re.compile(r"[0-9a-f]{40}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
LABEL = re.compile(r"[a-z][a-z0-9-]{0,47}\Z")
MAX_INPUT_BYTES = 2_000_000
MAX_CASES = 10_000
MAX_CANDIDATES = 32
MAX_OUTCOMES = 100_000
MIN_CASES = 64
CONTROL_DEGRADATION_PPM = 50_000


class Hold(ValueError):
    """Only static failure codes; never include an input record or filename."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise Hold(code)


def exact(value: object, keys: set[str], code: str) -> dict:
    require(type(value) is dict and set(value) == keys, code)
    return value


def pinned(value: object, pattern: re.Pattern) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None and any(
        ch != "0" for ch in value
    )


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def unique_pairs(pairs: list) -> dict:
    found = {}
    for key, value in pairs:
        if key in found:
            raise Hold("DUPLICATE_JSON_KEY")
        found[key] = value
    return found


def no_nonfinite(_: str) -> None:
    raise Hold("NONFINITE_JSON")


def load_input(path: Path) -> dict:
    """Reject ambiguous JSON, symlinks and unbounded files before parsing."""
    if path.is_symlink() or not path.is_file():
        raise Hold("INPUT_FILE_UNAVAILABLE")
    try:
        if not 0 < path.stat().st_size <= MAX_INPUT_BYTES:
            raise Hold("INPUT_BYTE_BUDGET")
        with path.open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
        if not 0 < len(raw) <= MAX_INPUT_BYTES:
            raise Hold("INPUT_BYTE_BUDGET")
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=no_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise Hold("INPUT_READ_OR_JSON_UNAVAILABLE") from None


def evaluate(payload: object) -> dict:
    """Produce an unsigned *research* signal conditional on unverified assumptions.

    For a frozen family of m candidates, paired differences D_i are in [-1, 1].
    The one-sided Hoeffding + Bonferroni lower bound for each candidate is:
        mean(D) - sqrt(2 * ln(m / alpha) / n).
    Familywise coverage requires independently sampled cases and every
    candidate fixed before the first holdout exposure. This function cannot
    authenticate either fact, or whether m includes omitted experiments.
    """
    data = exact(
        payload,
        {
            "schema",
            "repository",
            "source_revision",
            "plan_revision",
            "baseline_revision",
            "holdout_commitment_sha256",
            "registered_candidate_count",
            "holdout_exposure_rounds",
            "alpha_ppm",
            "min_effect_ppm",
            "cases",
            "candidates",
        },
        "ROOT_FIELDS_INVALID",
    )
    require(data["schema"] == SCHEMA and data["repository"] == REPOSITORY, "SCHEMA_INVALID")
    for key in ("source_revision", "plan_revision", "baseline_revision"):
        require(pinned(data[key], REV), "REVISION_NOT_IMMUTABLE")
    require(
        pinned(data["holdout_commitment_sha256"], SHA),
        "HOLDOUT_COMMITMENT_INVALID",
    )
    m = data["registered_candidate_count"]
    require(type(m) is int and 1 <= m <= MAX_CANDIDATES, "FAMILY_SIZE_INVALID")
    require(data["holdout_exposure_rounds"] == 1 and type(data["holdout_exposure_rounds"]) is int,
            "HOLDOUT_REUSE_NOT_PRE_REGISTERED")
    alpha_ppm = data["alpha_ppm"]
    margin_ppm = data["min_effect_ppm"]
    require(type(alpha_ppm) is int and 100 <= alpha_ppm <= 50_000, "ALPHA_INVALID")
    require(type(margin_ppm) is int and 0 <= margin_ppm <= 500_000, "EFFECT_MARGIN_INVALID")

    cases = data["cases"]
    candidates = data["candidates"]
    require(type(cases) is list and MIN_CASES <= len(cases) <= MAX_CASES, "CASE_COVERAGE_INVALID")
    require(type(candidates) is list and len(candidates) == m, "FAMILY_NOT_COMPLETE")
    n = len(cases)
    require(n * m <= MAX_OUTCOMES, "FAMILY_BYTE_BUDGET")
    case_ids: list[str] = []
    baseline: list[bool] = []
    broken: list[bool] = []
    for case in cases:
        case = exact(
            case,
            {"case_id", "baseline_correct", "broken_control_correct"},
            "CASE_FIELDS_INVALID",
        )
        require(pinned(case["case_id"], SHA), "CASE_ID_INVALID")
        for key in ("baseline_correct", "broken_control_correct"):
            require(type(case[key]) is bool, "CASE_RESULT_INVALID")
        case_ids.append(case["case_id"])
        baseline.append(case["baseline_correct"])
        broken.append(case["broken_control_correct"])
    require(len(set(case_ids)) == n, "DUPLICATE_HOLDOUT_CASE")
    base_correct = sum(baseline)
    control_correct = sum(broken)
    control_gap = base_correct - control_correct
    control_detected = control_gap * 1_000_000 >= CONTROL_DEGRADATION_PPM * n

    observed_ids: set[str] = set()
    observed_revisions: set[str] = set()
    assessments = []
    alpha = alpha_ppm / 1_000_000
    penalty = math.sqrt(2 * math.log(m / alpha) / n)
    for candidate in candidates:
        candidate = exact(
            candidate,
            {"id", "revision", "frozen_before_holdout", "rows"},
            "CANDIDATE_FIELDS_INVALID",
        )
        ident = candidate["id"]
        revision = candidate["revision"]
        require(type(ident) is str and LABEL.fullmatch(ident) is not None, "CANDIDATE_ID_INVALID")
        require(ident not in observed_ids, "DUPLICATE_CANDIDATE")
        require(
            pinned(revision, REV) and revision != data["baseline_revision"],
            "CANDIDATE_REVISION_INVALID",
        )
        require(revision not in observed_revisions, "DUPLICATE_CANDIDATE_REVISION")
        require(
            candidate["frozen_before_holdout"] is True,
            "CANDIDATE_NOT_FROZEN_BEFORE_HOLDOUT",
        )
        observed_ids.add(ident)
        observed_revisions.add(revision)
        rows = candidate["rows"]
        require(type(rows) is list and len(rows) == n, "CANDIDATE_COVERAGE_INVALID")
        wins = losses = ties = 0
        for index, item in enumerate(rows):
            item = exact(item, {"case_id", "correct"}, "OUTCOME_FIELDS_INVALID")
            require(item["case_id"] == case_ids[index], "PAIRED_CASE_ALIGNMENT_INVALID")
            require(type(item["correct"]) is bool, "OUTCOME_NOT_BOOLEAN")
            difference = int(item["correct"]) - int(baseline[index])
            wins += difference == 1
            losses += difference == -1
            ties += difference == 0
        observed_gain = (wins - losses) / n
        lower_bound = observed_gain - penalty
        signal = control_detected and lower_bound > margin_ppm / 1_000_000
        assessments.append(
            {
                "id": ident,
                "revision": revision,
                "sample_count": n,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "observed_paired_gain": round(observed_gain, 8),
                "conditional_familywise_lower_bound": round(lower_bound, 8),
                "signal": "RESEARCH_REVIEW_ONLY" if signal else "HOLD_NO_SUPPORTED_GAIN",
            }
        )
    assessments.sort(key=lambda x: x["id"])
    return {
        "schema": REPORT_SCHEMA,
        "state": (
            "RESEARCH_SIGNAL_REQUIRES_INDEPENDENT_REVIEW"
            if any(x["signal"] == "RESEARCH_REVIEW_ONLY" for x in assessments)
            else "HOLD_NO_SUPPORTED_GAIN_OR_CONTROL"
        ),
        "method": "PAIRED_HOEFFDING_ONE_SIDED_BONFERRONI_FROZEN_FAMILY",
        "input_sha256": hashlib.sha256(canonical(data)).hexdigest(),
        "source_revision": data["source_revision"],
        "plan_revision": data["plan_revision"],
        "holdout_commitment_sha256": data["holdout_commitment_sha256"],
        "cases": n,
        "candidate_family_size": m,
        "alpha_ppm": alpha_ppm,
        "min_effect_ppm": margin_ppm,
        "baseline_correct": base_correct,
        "broken_control_correct": control_correct,
        "negative_control_sensitivity_declared": control_detected,
        "candidates": assessments,
        "input_authenticity_verified": False,
        "pre_registration_independently_verified": False,
        "holdout_independence_verified": False,
        "population_quality_verified": False,
        "lean_theorem_proven_by_this_check": False,
        "release_authorized": False,
        "model_promotion_authorized": False,
        "provider_writes": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Research-only frozen paired-evolution gate")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = evaluate(load_input(args.input))
    except Hold as error:
        print(json.dumps({"state": "HOLD", "reason_code": str(error), "release_authorized": False}))
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    # Never turn successful source-only research analysis into a release PASS.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
