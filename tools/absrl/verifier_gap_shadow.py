#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure, opt-in verifier-gap shadow for candidate ABS-RL / GRPO reward audits.

This is not a TRL reward, an oracle, a validator of external provenance, a
training launcher, a signer, or an authorization API. It never reads prompts or
private datasets. The active ABS-RL worker remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re

PLAN_SCHEMA = "szl.absrl-verifier-gap-plan/v1"
OBSERVATION_SCHEMA = "szl.absrl-verifier-gap-observation/v1"
REPORT_SCHEMA = "szl.absrl-verifier-gap-shadow/v1"
MAX_CASES = 2048
MIN_CASES = 8  # Source-only input bound, not a statistically adequate study.
_REV = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PLAN_FIELDS = frozenset({
    "schema", "source_revision", "dataset_revision", "oracle_revision",
    "proxy_revision", "oracle_owner_digest", "proxy_owner_digest", "cases",
})
_OBSERVATION_FIELDS = frozenset({
    "schema", "source_revision", "dataset_revision", "oracle_revision",
    "proxy_revision", "rows",
})
_CASE_FIELDS = frozenset({"case_sha256", "stratum"})
_ROW_FIELDS = frozenset({"case_sha256", "oracle_result", "proxy_reward_half_units"})
_STRATA = frozenset({"answerable", "unanswerable"})
_OUTCOMES = frozenset({"CORRECT", "INCORRECT", "UNKNOWN", "MISSING"})


class GapHold(ValueError):
    """Fixed error code only, without raw task identifiers or model output."""


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise GapHold(reason)


def _shape(value: object, keys: frozenset[str], reason: str) -> dict:
    _need(type(value) is dict and set(value) == keys, reason)
    return value


def _pin(value: object, regex: re.Pattern[str], size: int) -> bool:
    return (
        type(value) is str and regex.fullmatch(value) is not None
        and value != "0" * size
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def plan_commitment(plan: object) -> str:
    """Caller-declared plan identity. A SHA-256 is not a signed freeze record."""
    _shape(plan, _PLAN_FIELDS, "PLAN_FIELDS_INVALID")
    return hashlib.sha256(_canonical(plan)).hexdigest()


def _hold(reason: str, **counts: object) -> dict:
    return {
        "schema": REPORT_SCHEMA, "state": "HOLD", "reason": reason,
        **counts,
        "trusted_oracle_proven": False,
        "independent_evaluator_proven": False,
        "pre_registration_proven": False,
        "dataset_rights_verified": False,
        "model_quality_verified": False,
        "training_authorized": False,
        "release_authorized": False,
        "provider_writes_performed": False,
    }


def _analyze(plan: object, observation: object, *,
             expected_plan_sha256: object, expected_source_revision: object) -> dict:
    plan = _shape(plan, _PLAN_FIELDS, "PLAN_FIELDS_INVALID")
    obs = _shape(observation, _OBSERVATION_FIELDS, "OBSERVATION_FIELDS_INVALID")
    _need(plan["schema"] == PLAN_SCHEMA and obs["schema"] == OBSERVATION_SCHEMA,
          "SCHEMA_INVALID")
    _need(_pin(expected_source_revision, _REV, 40), "EXPECTED_SOURCE_INVALID")
    _need(_pin(expected_plan_sha256, _DIGEST, 64), "EXPECTED_PLAN_SHA_INVALID")
    _need(plan["source_revision"] == expected_source_revision,
          "SOURCE_IDENTITY_NOT_EXPECTED")
    for key in ("source_revision", "dataset_revision", "oracle_revision", "proxy_revision"):
        _need(_pin(plan[key], _REV, 40), "PLAN_REVISION_INVALID")
        _need(obs[key] == plan[key], "OBSERVED_REVISION_MISMATCH")
    _need(plan["oracle_revision"] != plan["proxy_revision"],
          "ORACLE_PROXY_REVISION_NOT_DISTINCT")
    for key in ("oracle_owner_digest", "proxy_owner_digest"):
        _need(_pin(plan[key], _DIGEST, 64), "OWNER_IDENTITY_INVALID")
    _need(plan["oracle_owner_digest"] != plan["proxy_owner_digest"],
          "EVALUATOR_OWNER_COLLISION")
    _need(plan_commitment(plan) == expected_plan_sha256,
          "FROZEN_PLAN_COMMITMENT_MISMATCH")

    cases = plan["cases"]
    _need(type(cases) is list and MIN_CASES <= len(cases) <= MAX_CASES,
          "FROZEN_COHORT_COVERAGE_INVALID")
    expected: dict[str, str] = {}
    for case in cases:
        case = _shape(case, _CASE_FIELDS, "CASE_SCHEMA_INVALID")
        key = case["case_sha256"]
        _need(_pin(key, _DIGEST, 64), "CASE_ID_INVALID")
        _need(key not in expected, "DUPLICATE_FROZEN_CASE")
        _need(type(case["stratum"]) is str and case["stratum"] in _STRATA,
              "CASE_STRATUM_INVALID")
        expected[key] = case["stratum"]
    _need(set(expected.values()) == _STRATA, "COHORT_STRATUM_COVERAGE_INVALID")

    rows = obs["rows"]
    _need(type(rows) is list and len(rows) == len(expected),
          "OBSERVATION_POPULATION_INCOMPLETE")
    seen: set[str] = set()
    for row in rows:
        row = _shape(row, _ROW_FIELDS, "OUTCOME_SCHEMA_INVALID")
        key = row["case_sha256"]
        _need(_pin(key, _DIGEST, 64) and key in expected,
              "UNKNOWN_OR_SUBSTITUTED_CASE")
        _need(key not in seen, "DUPLICATE_OBSERVED_CASE")
        seen.add(key)
    _need(seen == set(expected), "OBSERVATION_CASES_NOT_EXACT")

    totals = {
        "frozen_cases": len(expected), "attempted": 0, "missing": 0,
        "unknown": 0, "oracle_correct": 0, "oracle_incorrect": 0,
        "optimistic_proxy_credit": 0, "pessimistic_proxy_credit": 0,
        "stratum_reward_mismatch": 0,
        "answerable_oracle_correct": 0,
        "answerable_oracle_incorrect": 0,
        "unanswerable_oracle_correct": 0,
        "unanswerable_oracle_incorrect": 0,
    }
    for row in rows:
        result = row["oracle_result"]
        _need(type(result) is str and result in _OUTCOMES,
              "ORACLE_RESULT_INVALID")
        reward = row["proxy_reward_half_units"]
        if result in ("UNKNOWN", "MISSING"):
            _need(reward is None, "UNKNOWN_OR_MISSING_WITH_PROXY_REWARD")
            totals["unknown" if result == "UNKNOWN" else "missing"] += 1
            continue
        _need(type(reward) is int and reward in (0, 1, 2),
              "PROXY_REWARD_INVALID")
        totals["attempted"] += 1
        if result == "INCORRECT":
            totals["oracle_incorrect"] += 1
            totals[expected[row["case_sha256"]] + "_oracle_incorrect"] += 1
            totals["optimistic_proxy_credit"] += int(reward > 0)
        else:
            totals["oracle_correct"] += 1
            totals[expected[row["case_sha256"]] + "_oracle_correct"] += 1
            expected_reward = 2 if expected[row["case_sha256"]] == "answerable" else 1
            totals["pessimistic_proxy_credit"] += int(reward == 0)
            totals["stratum_reward_mismatch"] += int(reward != expected_reward)

    reason = None
    if totals["attempted"] == 0:
        reason = "ZERO_ATTEMPTED_DENOMINATOR"
    elif totals["missing"] or totals["unknown"]:
        reason = "INCOMPLETE_ORACLE_COVERAGE"
    elif not totals["oracle_correct"] or not totals["oracle_incorrect"]:
        reason = "ORACLE_NEGATIVE_CONTROL_MISSING"
    elif totals["optimistic_proxy_credit"]:
        reason = "OPTIMISTIC_PROXY_REWARD_ON_INCORRECT_ORACLE"
    elif any(totals[s + "_oracle_incorrect"] == 0 for s in _STRATA):
        reason = "STRATUM_NEGATIVE_CONTROL_MISSING"
    elif any(totals[s + "_oracle_correct"] == 0 for s in _STRATA):
        reason = "STRATUM_POSITIVE_CONTROL_MISSING"
    elif totals["stratum_reward_mismatch"]:
        reason = "PROXY_ORACLE_REWARD_DISAGREEMENT"

    # Even a fully agreeing caller-supplied record remains an unsigned shadow.
    report = _hold(reason or "DECLARED_PROXY_ORACLE_AGREEMENT_ONLY", **totals)
    if reason is None:
        report["state"] = "SOURCE_SHADOW_CONSISTENT_NOT_ATTESTED"
    report["plan_sha256"] = expected_plan_sha256
    return report


def review_reward_gap(plan: object, observation: object, *,
                      expected_plan_sha256: object,
                      expected_source_revision: object) -> dict:
    """Compare declared proxy rewards with declared oracle outcomes, offline."""
    try:
        return _analyze(
            plan, observation,
            expected_plan_sha256=expected_plan_sha256,
            expected_source_revision=expected_source_revision,
        )
    except (GapHold, TypeError, ValueError, OverflowError, RecursionError) as error:
        reason = str(error) if type(error) is GapHold else "MALFORMED_INPUT"
        return _hold(reason)
