"""Bounded evidence-obligation advisor. No provider calls, effects, or authorization.

A read may inform an obligation; it does not satisfy the obligation. This is
an exact finite least-cost advisory plan over CALLER-SUPPLIED candidate costs,
not a claim of real-world optimality, source authenticity, or runtime readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

REQUEST_SCHEMA = "szl.evidence-obligation-request/v1"
RESULT_SCHEMA = "szl.evidence-obligation-advice/v1"
MAX_BYTES = 1_000_000
MAX_OBLIGATIONS = 16
MAX_OBSERVATIONS = 24
MAX_STATES = 4096
MAX_COST_UNITS = 1_000_000_000
ID = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,79}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceHold(ValueError):
    """Fixed, non-sensitive refusal code for unsupported advice inputs."""


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise EvidenceHold("DUPLICATE_JSON_MEMBER")
        result[key] = value
    return result


def _constant(_value):
    raise EvidenceHold("NONFINITE_JSON")


def load_strict(raw: bytes):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
        raise EvidenceHold("INPUT_BYTE_BUDGET")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        if isinstance(error, EvidenceHold):
            raise
        raise EvidenceHold("INVALID_JSON") from error
    if not isinstance(value, dict):
        raise EvidenceHold("INPUT_NOT_OBJECT")
    return value


def _id(value):
    if not isinstance(value, str) or ID.fullmatch(value) is None:
        raise EvidenceHold("INVALID_OBLIGATION_IDENTIFIER")
    return value


def _bounded_int(value):
    if type(value) is not int or not 0 <= value <= MAX_COST_UNITS:
        raise EvidenceHold("INVALID_COST_UNITS")
    return value


def _result(state, *, missing, excluded, chosen=(), cost=None, reason=None):
    return {
        "schema": RESULT_SCHEMA,
        "state": state,
        "reason": reason,
        "scope": "ADVISORY_ONLY_CALLER_DECLARED_COSTS_AND_PERMISSIONS",
        "evidence_obligations_open": list(missing),
        "selected_read_observations": list(chosen),
        "proposed_cost_units": cost,
        "excluded_observations": excluded,
        "authority_granted": False,
        "remote_effects_performed": 0,
        "requirements_satisfied_by_advice": 0,
        "signature": "UNSIGNED",
        "quality_or_promotion_qualified": False,
    }


def advise(document: dict) -> dict:
    """Find a minimum-cost read-only *information proposal*, never execute it.

    Complete dynamic programming examines all admitted finite candidate subsets.
    If its search-state cap is reached, it abstains without declaring optimality.
    """
    if not isinstance(document, dict) or document.get("schema") != REQUEST_SCHEMA:
        raise EvidenceHold("UNSUPPORTED_QUERY_SCHEMA")
    if set(document) != {
        "schema",
        "operation_id",
        "authority_state",
        "budget_units",
        "obligations",
        "observations",
    }:
        raise EvidenceHold("QUERY_FIELDS_UNAVAILABLE_OR_UNKNOWN")
    _id(document["operation_id"])
    authority = document["authority_state"]
    if authority not in ("ALLOW", "DENY", "UNKNOWN"):
        raise EvidenceHold("INVALID_AUTHORITY_STATE")
    budget = _bounded_int(document["budget_units"])
    obligations = document["obligations"]
    observations = document["observations"]
    if not isinstance(obligations, list) or len(obligations) > MAX_OBLIGATIONS:
        raise EvidenceHold("OBLIGATION_COUNT_BOUND")
    if not isinstance(observations, list) or len(observations) > MAX_OBSERVATIONS:
        raise EvidenceHold("OBSERVATION_COUNT_BOUND")
    states, ids = {}, set()
    for entry in obligations:
        if not isinstance(entry, dict) or set(entry) != {
            "id",
            "state",
            "evidence_digest",
        }:
            raise EvidenceHold("INVALID_OBLIGATION")
        name = _id(entry["id"])
        if name in ids:
            raise EvidenceHold("DUPLICATE_OBLIGATION")
        ids.add(name)
        state = entry["state"]
        digest = entry["evidence_digest"]
        if state not in ("OPEN", "STALE", "UNKNOWN", "OBSERVED"):
            raise EvidenceHold("INVALID_OBLIGATION_STATE")
        if state == "OBSERVED":
            # A digest binds the caller's assertion. It proves no signer, expiry
            # or authority to this advisor.
            if not isinstance(digest, str) or DIGEST.fullmatch(digest) is None:
                raise EvidenceHold("OBSERVED_DIGEST_UNAVAILABLE")
        elif digest is not None:
            raise EvidenceHold("UNRESOLVED_OBLIGATION_CLAIMS_EVIDENCE")
        states[name] = state
    missing = tuple(sorted(k for k, state in states.items() if state != "OBSERVED"))
    excluded, admitted = {}, []
    seen = set()
    for item in observations:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "mode",
            "admission_state",
            "cost_units",
            "may_inform",
        }:
            raise EvidenceHold("INVALID_OBSERVATION")
        name = _id(item["id"])
        if name in seen:
            raise EvidenceHold("DUPLICATE_OBSERVATION")
        seen.add(name)
        cost = _bounded_int(item["cost_units"])
        targets = item["may_inform"]
        if not isinstance(targets, list) or not targets:
            raise EvidenceHold("INVALID_OBSERVATION_TARGETS")
        covered = set()
        for target in targets:
            key = _id(target)
            if key not in states or key in covered:
                raise EvidenceHold("INVALID_OBSERVATION_TARGETS")
            covered.add(key)
        if item["mode"] != "READ_ONLY":
            excluded[name] = "NOT_A_READ_ONLY_OBSERVATION"
        elif item["admission_state"] != "ALLOW":
            if item["admission_state"] not in ("DENY", "UNKNOWN"):
                raise EvidenceHold("INVALID_OBSERVATION_ADMISSION")
            excluded[name] = "READ_ADMISSION_" + item["admission_state"]
        else:
            admitted.append((name, cost, covered))
    if authority == "DENY":
        return _result(
            "BLOCKED_POLICY",
            missing=missing,
            excluded=excluded,
            reason="OPERATION_DENIED",
        )
    if authority == "UNKNOWN":
        return _result(
            "HOLD_AUTHORITY",
            missing=missing,
            excluded=excluded,
            reason="OPERATION_AUTHORITY_UNKNOWN",
        )
    if not missing:
        return _result(
            "NO_MISSING_INPUT_OBLIGATIONS",
            missing=(),
            excluded=excluded,
            reason="NO_EXECUTION_OR_RELEASE_AUTHORIZED",
        )
    bit = {name: 1 << index for index, name in enumerate(missing)}
    candidates = sorted(
        (name, cost, sum(bit.get(target, 0) for target in covered))
        for name, cost, covered in admitted
    )
    candidates = [(name, cost, mask) for name, cost, mask in candidates if mask]
    if not candidates:
        return _result(
            "HOLD_NO_READ_COVERAGE",
            missing=missing,
            excluded=excluded,
            reason="NO_ADMITTED_READ_COULD_INFORM_MISSING_OBLIGATIONS",
        )
    # Store (cost, count, names). The tuple provides deterministic tie-breaking.
    best = {0: (0, 0, ())}
    for name, cost, covers in candidates:
        addition = {}
        for mask, previous in best.items():
            next_mask = mask | covers
            total = previous[0] + cost
            if next_mask == mask or total > budget:
                continue
            candidate = (total, previous[1] + 1, previous[2] + (name,))
            previous_held = best.get(next_mask)
            pending = addition.get(next_mask)
            if (previous_held is None or candidate < previous_held) and (
                pending is None or candidate < pending
            ):
                addition[next_mask] = candidate
        best.update(addition)
        if len(best) > MAX_STATES:
            return _result(
                "ABSTAIN_SEARCH_BOUND",
                missing=missing,
                excluded=excluded,
                reason="FINITE_SEARCH_CAP_REACHED",
            )
    goal = (1 << len(missing)) - 1
    if goal not in best:
        return _result(
            "HOLD_UNCOVERED_OR_OVER_BUDGET",
            missing=missing,
            excluded=excluded,
            reason="NO_ADMITTED_PLAN_WITHIN_BUDGET",
        )
    optimum = best[goal]
    return _result(
        "ADVISORY_CANDIDATE",
        missing=missing,
        excluded=excluded,
        chosen=optimum[2],
        cost=optimum[0],
        reason="LEAST_COST_OVER_SUBMITTED_FINITE_READS_ONLY",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.input.is_symlink() or not args.input.is_file():
            raise EvidenceHold("INPUT_FILE_UNAVAILABLE")
        with args.input.open("rb") as source:
            raw = source.read(MAX_BYTES + 1)
        result = advise(load_strict(raw))
        result["input_bytes_sha256"] = hashlib.sha256(raw).hexdigest()
        serialized = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        with args.output.open("xb") as output:
            output.write(serialized)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "excluded_observations"},
                allow_nan=False,
            )
        )
        return 0
    except (EvidenceHold, OSError) as error:
        reason = str(error) if isinstance(error, EvidenceHold) else type(error).__name__
        print(json.dumps({"state": "HOLD", "reason": reason}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
