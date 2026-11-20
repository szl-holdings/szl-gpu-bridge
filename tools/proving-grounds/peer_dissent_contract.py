#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure two-model dissent witness for the existing SZL paired research cycle.

Kimi/Mistral declarations are untrusted. This program has no provider, agent
runtime, credential, filesystem, tool-call, merge or deployment capability.
Passing a finite structural review never constitutes independent verification.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re

SCHEMA = "szl.peer-dissent-request/v1"
REPORT_SCHEMA = "szl.peer-dissent-observation/v1"
REPOSITORY = "szl-holdings/szl-gpu-bridge"
CONTROL_IDS = frozenset(
    {
        "missing-signer",
        "source-revision-mismatch",
        "unauthorized-action",
        "stale-evidence",
        "provider-response-uncertain",
    }
)
ROOT_FIELDS = frozenset(
    {
        "schema",
        "repository",
        "source_revision",
        "patch_sha256",
        "evaluation_suite_sha256",
        "counterexample_suite_sha256",
        "observed_at_utc",
        "reviewers",
        "controls",
    }
)
REVIEWER_FIELDS = frozenset(
    {
        "role",
        "family",
        "model",
        "source_revision",
        "patch_sha256",
        "prompt_sha256",
        "response_sha256",
        "verdict",
        "highest_risk",
    }
)
CONTROL_FIELDS = frozenset({"id", "expected", "observed", "fixture_sha256"})
REVISION = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
MODEL = re.compile(r"[a-z0-9][a-z0-9._:-]{2,79}\Z")
MAX_BYTES = 32768
MAX_AGE_SECONDS = 900


class Hold(ValueError):
    """Static refusal reasons; never include caller-provided strings."""


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise Hold("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _nonfinite(_):
    raise Hold("NONFINITE_JSON")


def _exact(value, fields, reason):
    if type(value) is not dict or set(value) != fields:
        raise Hold(reason)
    return value


def _pin(value, pattern):
    return (
        type(value) is str
        and pattern.fullmatch(value) is not None
        and any(ch != "0" for ch in value)
    )


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _observed(text, now):
    if type(text) is not str or len(text) > 40:
        raise Hold("OBSERVATION_TIME_INVALID")
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Hold("OBSERVATION_TIME_INVALID") from exc
    if moment.tzinfo is None or not isinstance(now, datetime) or now.tzinfo is None:
        raise Hold("OBSERVATION_TIME_INVALID")
    age = (
        now.astimezone(timezone.utc) - moment.astimezone(timezone.utc)
    ).total_seconds()
    if not 0 <= age <= MAX_AGE_SECONDS:
        raise Hold("OBSERVATION_STALE_OR_FUTURE")


def _reviewers(rows, source_rev, patch):
    if type(rows) is not list or len(rows) != 2:
        raise Hold("REVIEWER_COVERAGE_INVALID")
    roles, families, prompts, responses = set(), set(), set(), set()
    normalized = []
    for row in rows:
        row = _exact(row, REVIEWER_FIELDS, "REVIEWER_FIELDS_INVALID")
        role, family, model = row["role"], row["family"], row["model"]
        if type(role) is not str or role not in {"proposer", "adversary"}:
            raise Hold("REVIEWER_IDENTITY_INVALID")
        if type(family) is not str or family not in {"kimi", "mistral"}:
            raise Hold("REVIEWER_IDENTITY_INVALID")
        if type(model) is not str or MODEL.fullmatch(model) is None:
            raise Hold("REVIEWER_IDENTITY_INVALID")
        if not (
            (family == "kimi" and model.startswith("kimi-"))
            or (
                family == "mistral"
                and model.startswith(("mistral-", "ministral-", "devstral-"))
            )
        ):
            raise Hold("REVIEWER_FAMILY_MISMATCH")
        if row["source_revision"] != source_rev or row["patch_sha256"] != patch:
            raise Hold("REVIEW_SOURCE_MISMATCH")
        if not _pin(row["prompt_sha256"], DIGEST) or not _pin(
            row["response_sha256"], DIGEST
        ):
            raise Hold("REVIEW_DIGEST_INVALID")
        roles.add(role)
        families.add(family)
        prompts.add(row["prompt_sha256"])
        responses.add(row["response_sha256"])
        if (
            type(row["verdict"]) is not str
            or row["verdict"] not in {"REVIEW", "HOLD"}
            or type(row["highest_risk"]) is not str
            or row["highest_risk"] not in {"NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        ):
            raise Hold("REVIEW_CLAIM_INVALID")
        normalized.append((role, family, row["verdict"], row["highest_risk"]))
    if roles != {"proposer", "adversary"} or families != {"kimi", "mistral"}:
        raise Hold("REVIEWER_COVERAGE_INVALID")
    # Required roles and families being present does not bind a model to
    # the correct role; a swapped pair could otherwise look like agreement.
    if {(role, family) for role, family, _, _ in normalized} != {
        ("proposer", "kimi"),
        ("adversary", "mistral"),
    }:
        raise Hold("REVIEWER_IDENTITY_INVALID")
    # Hash inequality is a syntactic check, not proof of independent thinking.
    if len(prompts) != 2 or len(responses) != 2:
        raise Hold("DECLARED_REVIEWS_NOT_DISTINCT")
    return sorted(normalized)


def _controls(rows):
    if type(rows) is not list or len(rows) != len(CONTROL_IDS):
        raise Hold("COUNTEREXAMPLE_COVERAGE_INVALID")
    observed = set()
    digests = set()
    failures = []
    for row in rows:
        row = _exact(row, CONTROL_FIELDS, "COUNTEREXAMPLE_FIELDS_INVALID")
        ident = row["id"]
        if type(ident) is not str or ident not in CONTROL_IDS or ident in observed:
            raise Hold("COUNTEREXAMPLE_COVERAGE_INVALID")
        if row["expected"] != "HOLD" or type(row["expected"]) is not str:
            raise Hold("COUNTEREXAMPLE_EXPECTATION_INVALID")
        if type(row["observed"]) is not str or row["observed"] not in {
            "HOLD",
            "READY",
            "UNKNOWN",
        }:
            raise Hold("COUNTEREXAMPLE_OBSERVATION_INVALID")
        digest = row["fixture_sha256"]
        if not _pin(digest, DIGEST) or digest in digests:
            raise Hold("COUNTEREXAMPLE_FIXTURE_INVALID")
        observed.add(ident)
        digests.add(digest)
        if row["observed"] != "HOLD":
            failures.append(ident)
    if observed != CONTROL_IDS:
        raise Hold("COUNTEREXAMPLE_COVERAGE_INVALID")
    return sorted(failures)


def assess(raw: bytes, *, now: datetime) -> dict:
    """Fail-closed classification of untrusted declarative peer reports."""
    result = {
        "schema": REPORT_SCHEMA,
        "state": "HOLD",
        "reason": "INPUT_UNOBSERVED",
        "input_sha256": None,
        "source_revision": None,
        "patch_sha256": None,
        "reviewers_declared": 0,
        "control_denominator": len(CONTROL_IDS),
        "controls_claimed_blocked": 0,
        "dissent_count": None,
        "peer_identity_verified": False,
        "peer_independence_verified": False,
        "reviewer_output_bytes_verified": False,
        "test_execution_verified": False,
        "counterexample_bytes_verified": False,
        "source_bytes_verified": False,
        "signer_identity_verified": False,
        "quality_or_novelty_proven": False,
        "release_authorized": False,
        "cloud_action_authorized": False,
        "provider_writes": False,
    }
    try:
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES:
            raise Hold("INPUT_BYTE_BUDGET")
        data = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite
        )
        _exact(data, ROOT_FIELDS, "ROOT_FIELDS_INVALID")
        if data["schema"] != SCHEMA or data["repository"] != REPOSITORY:
            raise Hold("SCHEMA_OR_REPOSITORY_INVALID")
        if not _pin(data["source_revision"], REVISION):
            raise Hold("SOURCE_REVISION_INVALID")
        for name in (
            "patch_sha256",
            "evaluation_suite_sha256",
            "counterexample_suite_sha256",
        ):
            if not _pin(data[name], DIGEST):
                raise Hold("SOURCE_DIGEST_INVALID")
        _observed(data["observed_at_utc"], now)
        reviewers = _reviewers(
            data["reviewers"], data["source_revision"], data["patch_sha256"]
        )
        failures = _controls(data["controls"])
        result.update(
            input_sha256=hashlib.sha256(_canonical(data)).hexdigest(),
            source_revision=data["source_revision"],
            patch_sha256=data["patch_sha256"],
            reviewers_declared=len(reviewers),
            controls_claimed_blocked=len(CONTROL_IDS) - len(failures),
            dissent_count=sum(v != "REVIEW" for _, _, v, _ in reviewers),
        )
        if failures:
            raise Hold("COUNTEREXAMPLE_NOT_BLOCKED")
        if any(v != "REVIEW" for _, _, v, _ in reviewers):
            raise Hold("REVIEWER_DISSENT_HOLD")
        if any(r in {"MEDIUM", "HIGH", "CRITICAL"} for _, _, _, r in reviewers):
            raise Hold("REVIEWER_RISK_HOLD")
        result.update(
            state="REVIEW_CANDIDATE_ONLY",
            reason="TWO_PEER_DECLARATIONS_REQUIRE_INDEPENDENT_PROOF",
        )
    except Hold as exc:
        result["reason"] = str(exc)
    except (
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        OverflowError,
        ValueError,
    ):
        result["reason"] = "INPUT_JSON_INVALID"
    return result
