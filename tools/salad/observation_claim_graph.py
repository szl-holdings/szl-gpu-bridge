#!/usr/bin/env python3
"""Fail-closed, offline claim graph for the existing SZL one-shot training watch.

This is a read-only consumer, NOT another observer, cloud controller, publisher,
signature, benchmark, or model evaluator. The producer keeps exclusive ownership.
Private job identifiers must not be copied from its output to public surfaces.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


SCHEMA = "szl.training-observation-claim-graph/v1"
WATCH = "szl.training-watch-once/v2"
CANARY_WATCH = "szl.training-watch-once/v3-candidate-unadmitted"
MAX_BYTES = 4_000_000
MAX_GROUPS = 1000
MAX_CANDIDATES = 500
GROUP_COUNT_FIELDS = (
    "allocating_count",
    "creating_count",
    "running_count",
    "stopping_count",
)
REPO = re.compile(r"^SZLHOLDINGS/[A-Za-z0-9_.-]+$")
GROUP = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DIGEST = re.compile(r"^[a-f0-9]{64}$")


class CoverageHold(ValueError):
    """Deterministic diagnostics, never raw exception data or provider URLs."""


def _require(ok, reason):
    if not ok:
        raise CoverageHold(reason)


def _utc(value):
    _require(isinstance(value, str) and len(value) <= 64, "INVALID_OBSERVATION_TIME")
    try:
        point = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise CoverageHold("INVALID_OBSERVATION_TIME") from None
    _require(
        point.tzinfo is not None and point.utcoffset() == timedelta(0),
        "INVALID_OBSERVATION_TIME",
    )
    return point


def strict_json(raw):
    _require(
        isinstance(raw, bytes) and 0 < len(raw) <= MAX_BYTES,
        "INVALID_INPUT_BYTE_BUDGET",
    )

    def unique(pairs):
        data = {}
        for name, value in pairs:
            _require(name not in data, "DUPLICATE_JSON_KEY")
            data[name] = value
        return data

    def constant(_):
        raise CoverageHold("NONFINITE_JSON")

    def float_value(raw_number):
        value = float(raw_number)
        _require(math.isfinite(value), "NONFINITE_JSON")
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=constant,
            parse_float=float_value,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, CoverageHold):
            raise
        raise CoverageHold("INVALID_JSON") from None
    _require(isinstance(value, dict), "INVALID_REPORT_OBJECT")
    return value


def _watch(report):
    _require(
        isinstance(report, dict)
        and report.get("watch_version") in (WATCH, CANARY_WATCH),
        "UNRECOGNIZED_WATCH_VERSION",
    )
    _utc(report.get("observed_at_utc"))
    _require(isinstance(report.get("providers"), dict), "PROVIDERS_UNOBSERVED")
    return report


def _count(value):
    # Missing and boolean counts are unknown, NEVER assumed to be zero.
    return value if type(value) is int and value >= 0 else None


def salad_coverage(report):
    """Reconcile provider-reported population with rows retained by the watch."""
    salad = report["providers"].get("salad")
    if not isinstance(salad, dict):
        return {
            "status": "UNOBSERVED",
            "coverage_complete": False,
            "project_discovery_complete": False,
            "projects": [],
            "omitted_count": None,
            "captured_count": 0,
            "findings": ["SALAD_PROVIDER_UNOBSERVED"],
        }, {}
    groups = salad.get("groups", [])
    coverage = salad.get("coverage", [])
    _require(
        isinstance(groups, list)
        and len(groups) <= MAX_GROUPS
        and isinstance(coverage, list)
        and len(coverage) <= 100,
        "INVALID_SALAD_CATALOG",
    )
    indexed = {}
    by_project = Counter()
    for row in groups:
        _require(isinstance(row, dict), "INVALID_SALAD_GROUP")
        project, name = row.get("project"), row.get("name")
        _require(
            isinstance(project, str)
            and GROUP.fullmatch(project) is not None
            and isinstance(name, str)
            and GROUP.fullmatch(name) is not None,
            "INVALID_SALAD_GROUP_ID",
        )
        identity = (project, name)
        _require(identity not in indexed, "DUPLICATE_SALAD_GROUP")
        counts = row.get("counts")
        _require(isinstance(counts, dict), "INVALID_SALAD_COUNTS")
        indexed[identity] = {
            "project": project,
            "name": name,
            "classification": row.get("classification")
            if row.get("classification")
            in ("TRAINING", "EVALUATION", "INFERENCE", "UNKNOWN")
            else "UNKNOWN",
            "classification_basis": row.get("classification_basis"),
            "model_repo": row.get("model_repo")
            if isinstance(row.get("model_repo"), str)
            and REPO.fullmatch(row["model_repo"]) is not None
            else None,
            "configured_max_steps": _count(row.get("configured_max_steps")),
            "group_status": row["group_status"]
            if row.get("group_status")
            in ("running", "stopped", "deploying", "suspended")
            else None,
            "desired_replicas": _count(row.get("desired_replicas")),
            "counts": {name: _count(counts.get(name)) for name in GROUP_COUNT_FIELDS},
            "trainer_process_liveness_verified": False,
        }
        by_project[project] += 1

    projects = []
    names = set()
    reasons = []
    total_returned = total_omitted = 0
    has_unknown_scope = False
    for row in coverage:
        _require(
            isinstance(row, dict)
            and isinstance(row.get("project"), str)
            and GROUP.fullmatch(row["project"]) is not None
            and row["project"] not in names,
            "INVALID_PROJECT_COVERAGE",
        )
        project = row["project"]
        names.add(project)
        captured = by_project[project]
        listing = row.get("list")
        if listing != "OBSERVED":
            projects.append(
                {
                    "project": project,
                    "state": "UNOBSERVED",
                    "captured_count": captured,
                    "returned_count": None,
                    "omitted_count": None,
                    "listing_complete": False,
                }
            )
            has_unknown_scope = True
            continue
        pagination = row.get("pagination")
        _require(isinstance(pagination, dict), "INVALID_PAGINATION")
        returned, pages = (
            _count(pagination.get("returned")),
            _count(pagination.get("pages")),
        )
        truncated = pagination.get("truncated")
        _require(
            returned is not None
            and pages is not None
            and pages > 0
            and type(truncated) is bool
            and returned >= captured,
            "INVALID_PAGINATION",
        )
        omitted = returned - captured
        capture = row.get("group_capture")
        if report.get("watch_version") == CANARY_WATCH:
            _require(isinstance(capture, dict), "V3_GROUP_CAPTURE_MISSING")
            ceiling = _count(capture.get("capture_limit"))
            omitted_names = capture.get("omitted_names")
            flagged_names = capture.get("omitted_with_nonzero_provider_counts")
            unknown_names = capture.get("omitted_with_unknown_counts")
            _require(
                ceiling is not None
                and ceiling > 0
                and ceiling >= captured
                and capture.get("returned_count") == returned
                and type(capture.get("returned_count")) is int
                and capture.get("captured_count") == captured
                and type(capture.get("captured_count")) is int
                and capture.get("omitted_count") == omitted
                and type(capture.get("omitted_count")) is int
                and capture.get("listing_complete") is (not truncated)
                and capture.get("coverage_complete") is (not truncated and omitted == 0)
                and isinstance(omitted_names, list)
                and len(omitted_names) == omitted
                and len(set(omitted_names)) == omitted
                and all(
                    isinstance(n, str) and GROUP.fullmatch(n) for n in omitted_names
                )
                and not set(omitted_names).intersection(
                    name for proj, name in indexed if proj == project
                )
                and isinstance(flagged_names, list)
                and isinstance(unknown_names, list)
                and all(
                    isinstance(n, str) and n in omitted_names for n in flagged_names
                )
                and all(
                    isinstance(n, str) and n in omitted_names for n in unknown_names
                ),
                "V3_GROUP_CAPTURE_MISMATCH",
            )
        total_returned += returned
        total_omitted += omitted
        if omitted:
            reasons.append("SALAD_GROUP_DETAIL_OMISSION")
        if truncated:
            reasons.append("SALAD_PAGINATION_TRUNCATED")
        projects.append(
            {
                "project": project,
                "state": "OBSERVED",
                "captured_count": captured,
                "returned_count": returned,
                "omitted_count": omitted,
                "listing_complete": not truncated,
            }
        )

    _require(
        all(row["project"] in names for row in indexed.values()),
        "GROUP_WITHOUT_PROJECT_COVERAGE",
    )
    declared = salad.get("projects")
    if isinstance(declared, list):
        _require(all(isinstance(p, str) for p in declared), "INVALID_DECLARED_PROJECTS")
        missed_projects = sorted(set(declared) - names)
    else:
        missed_projects = []
        has_unknown_scope = True
    if missed_projects:
        reasons.append("SALAD_PROJECTS_NOT_CHECKED")
        has_unknown_scope = True
    discovery = salad.get("project_listing")
    project_discovery_complete = (
        isinstance(discovery, dict)
        and discovery.get("status") in (None, "OBSERVED")
        and type(discovery.get("returned")) is int
        and discovery["returned"] >= len(declared or [])
        and type(discovery.get("truncated")) is bool
        and not discovery["truncated"]
        and _count(discovery.get("pages")) is not None
        and discovery.get("pages") > 0
    )
    if not project_discovery_complete:
        reasons.append("ORG_PROJECT_DISCOVERY_UNOBSERVED")
    if (
        has_unknown_scope
        or not projects
        or any(p["state"] != "OBSERVED" for p in projects)
    ):
        reasons.append("SALAD_LISTING_SCOPE_INCOMPLETE")
    return (
        {
            "status": "PARTIAL" if reasons else "OBSERVED",
            "captured_count": len(indexed),
            "returned_count": total_returned if not has_unknown_scope else None,
            "omitted_count": total_omitted if not has_unknown_scope else None,
            "projects": projects,
            "unobserved_projects": missed_projects,
            "project_discovery_complete": project_discovery_complete,
            "known_project_group_details_complete": bool(projects)
            and all(
                row["state"] == "OBSERVED"
                and row["listing_complete"]
                and row["omitted_count"] == 0
                for row in projects
            ),
            "coverage_complete": not reasons,
            "findings": sorted(set(reasons)),
            "provider_instance_counts": {
                key: (
                    sum(item["counts"][key] for item in indexed.values())
                    if all(item["counts"][key] is not None for item in indexed.values())
                    else None
                )
                for key in GROUP_COUNT_FIELDS
            },
            "groups_with_unknown_counts": sum(
                any(v is None for v in x["counts"].values()) for x in indexed.values()
            ),
            "trainer_process_liveness_verified": False,
        },
        indexed,
    )


def instance_coverage(report, indexed):
    """Validate bounded selected instance GET coverage, never trainer liveness.

    Provider group and instance requests are sequential, not atomic. A count
    divergence is evidence to reconcile, not a restart or training claim.
    """
    salad = report["providers"].get("salad")
    if not isinstance(salad, dict):
        return {
            "state": "UNOBSERVED",
            "eligible_count": None,
            "attempted_count": None,
            "observed_count": 0,
            "unattempted_count": None,
            "coverage_complete_of_selected_eligible": False,
            "provider_count_drift": None,
            "findings": [],
            "temporal_atomic": False,
            "trainer_process_liveness_verified": False,
        }
    scope = salad.get("instance_scope")
    if scope is None:
        _require(report["watch_version"] != CANARY_WATCH, "V3_INSTANCE_SCOPE_MISSING")
        return {
            "state": "NOT_DECLARED_V2",
            "eligible_count": None,
            "attempted_count": None,
            "observed_count": 0,
            "unattempted_count": None,
            "coverage_complete_of_selected_eligible": False,
            "provider_count_drift": None,
            "findings": [],
            "temporal_atomic": False,
            "trainer_process_liveness_verified": False,
        }
    _require(isinstance(scope, dict), "INVALID_INSTANCE_SCOPE")
    eligible = _count(scope.get("eligible_count"))
    attempted = _count(scope.get("attempted_count"))
    limit = _count(scope.get("limit"))
    all_names = scope.get("eligible_names")
    omitted = scope.get("unattempted_names")
    inspections = salad.get("instance_inspections")
    _require(
        eligible is not None
        and attempted is not None
        and limit is not None
        and 1 <= limit <= 100
        and attempted <= eligible
        and attempted <= limit
        and eligible <= MAX_GROUPS
        and type(scope.get("provider_denial_stop")) is bool
        and isinstance(all_names, list)
        and len(all_names) == eligible
        and all(isinstance(x, str) and GROUP.fullmatch(x) for x in all_names)
        and len(set(all_names)) == len(all_names)
        and isinstance(omitted, list)
        and omitted == all_names[attempted:]
        and isinstance(inspections, list)
        and len(inspections) == attempted,
        "INVALID_INSTANCE_SCOPE",
    )
    checked = set()
    observed = 0
    complete_observations = 0
    drift = 0
    downloading_unmappable = 0
    for index, entry in enumerate(inspections):
        _require(
            isinstance(entry, dict)
            and isinstance(entry.get("project"), str)
            and isinstance(entry.get("group"), str)
            and entry["group"] == all_names[index]
            and (entry["project"], entry["group"]) in indexed
            and (entry["project"], entry["group"]) not in checked,
            "INSTANCE_IDENTITY_MISMATCH",
        )
        checked.add((entry["project"], entry["group"]))
        state = entry.get("status")
        _require(state in ("OBSERVED", "UNOBSERVED"), "INVALID_INSTANCE_STATUS")
        if state == "OBSERVED":
            observed += 1
            proof = entry.get("instance_evidence")
            _require(isinstance(proof, dict), "INVALID_INSTANCE_EVIDENCE")
            returned = _count(proof.get("returned_count"))
            captured = _count(proof.get("captured_count"))
            _require(
                returned is not None
                and captured is not None
                and returned >= captured
                and type(proof.get("coverage_complete")) is bool
                and _count(proof.get("unobserved_status_count")) is not None
                and _count(proof.get("conflicting_status_fields")) is not None,
                "INVALID_INSTANCE_EVIDENCE",
            )
            matched = entry.get("group_instance_count_reconciliation")
            _require(
                matched
                in (
                    "PROVIDER_COUNTS_AGREE_NOT_TRAINER_LIVENESS",
                    "PROVIDER_COUNT_DRIFT",
                    "UNOBSERVED",
                    "PROVIDER_COUNTS_NOT_COMPARABLE_DOWNLOADING",
                ),
                "INVALID_COUNT_RECONCILIATION",
            )
            if matched == "PROVIDER_COUNT_DRIFT":
                drift += 1
            if matched == "PROVIDER_COUNTS_NOT_COMPARABLE_DOWNLOADING":
                downloading_unmappable += 1
            if (
                proof["coverage_complete"]
                and not proof["unobserved_status_count"]
                and not proof["conflicting_status_fields"]
            ):
                complete_observations += 1
    full = (
        attempted == eligible
        and observed == eligible
        and complete_observations == observed
        and not scope["provider_denial_stop"]
    )
    reasons = []
    if attempted < eligible:
        reasons.append("INSTANCE_GET_ELIGIBLE_OMISSIONS")
    if observed < attempted or complete_observations < observed:
        reasons.append("INSTANCE_GET_PARTIAL_OBSERVATIONS")
    if scope["provider_denial_stop"]:
        reasons.append("INSTANCE_PROVIDER_DENIAL_STOP")
    if drift:
        reasons.append("SEQUENTIAL_PROVIDER_COUNTS_DIFFER_NOT_ATOMIC")
    if downloading_unmappable:
        reasons.append("DOWNLOADING_INSTANCE_HAS_NO_GROUP_COUNT_COUNTER")
    return {
        "state": "OBSERVED_WITH_LIMITS",
        "eligible_count": eligible,
        "attempted_count": attempted,
        "observed_count": observed,
        "unattempted_count": eligible - attempted,
        "coverage_complete_of_selected_eligible": full,
        "provider_count_drift": drift,
        "provider_count_not_comparable_downloading": downloading_unmappable,
        "findings": reasons,
        "temporal_atomic": False,
        "trainer_process_liveness_verified": False,
    }


def _trainer(row):
    if not isinstance(row, dict) or row.get("status") != "OBSERVED":
        return None
    info = row.get("trainer_state")
    if (
        not isinstance(info, dict)
        or info.get("integrity") != "PINNED_OBJECT_BYTES_VERIFIED"
    ):
        return None
    step, target = info.get("global_step"), info.get("max_steps")
    sha = info.get("trainer_state_sha256")
    if (
        type(step) is not int
        or type(target) is not int
        or target < 1
        or step < 0
        or type(row.get("latest_checkpoint")) is not int
        or step != row.get("latest_checkpoint")
        or not isinstance(sha, str)
        or DIGEST.fullmatch(sha) is None
        or not isinstance(row.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", row["revision"]) is None
    ):
        return None
    return (step, target, sha)


def model_claims(report, previous, indexed):
    now = report["providers"].get("huggingface", {})
    old = previous["providers"].get("huggingface", {}) if previous else {}
    current = now.get("training_candidates", []) if isinstance(now, dict) else []
    earlier = old.get("training_candidates", []) if isinstance(old, dict) else []
    _require(
        isinstance(current, list)
        and len(current) <= MAX_CANDIDATES
        and isinstance(earlier, list)
        and len(earlier) <= MAX_CANDIDATES,
        "INVALID_TRAINING_CANDIDATES",
    )
    oldrows = {}
    for oldrow in earlier:
        _require(
            isinstance(oldrow, dict)
            and isinstance(oldrow.get("id"), str)
            and REPO.fullmatch(oldrow["id"]) is not None
            and oldrow["id"] not in oldrows,
            "INVALID_PRIOR_CANDIDATE_ID",
        )
        oldrows[oldrow["id"]] = oldrow
    seen = set()
    result = []
    for row in current:
        _require(
            isinstance(row, dict)
            and isinstance(row.get("id"), str)
            and REPO.fullmatch(row["id"]) is not None
            and row["id"] not in seen,
            "INVALID_CANDIDATE_ID",
        )
        repo = row["id"]
        seen.add(repo)
        proof = _trainer(row)
        before = _trainer(oldrows.get(repo))
        mapped = [
            g
            for g in indexed.values()
            if g["model_repo"] == repo
            and g["classification"] == "TRAINING"
            and g["classification_basis"]
            in ("COMMAND_TARGET", "COMMAND_TRAINING_TARGET")
            and g["configured_max_steps"] is not None
        ]
        binding = (
            "UNOBSERVED"
            if not mapped
            else "TARGET_MISMATCH"
            if proof and any(g["configured_max_steps"] != proof[1] for g in mapped)
            else "DECLARED_COMMAND_TARGET_ONLY"
        )
        entry = {
            "repo": repo,
            "observed_revision": (
                row["revision"]
                if isinstance(row.get("revision"), str)
                and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", row["revision"])
                else None
            ),
            "trainer_state": "PRODUCER_REPORTED_PINNED_BYTES"
            if proof
            else "UNVERIFIED",
            "trainer_state_independently_reverified_by_graph": False,
            "observed_step": proof[0] if proof else None,
            "target_steps": proof[1] if proof else None,
            "mapped_provider_groups": [
                {
                    "project": g["project"],
                    "name": g["name"],
                    "desired_replicas": g["desired_replicas"],
                    "counts": g["counts"],
                    "process_liveness_verified": False,
                }
                for g in mapped
            ],
            "command_target_binding": binding,
            "checkpoint_step_delta": (
                proof[0] - before[0]
                if proof and before and proof[1] == before[1]
                else None
            ),
            "trainer_run_continuity_verified": False,
            "target_changed_since_prior": bool(
                proof and before and proof[1] != before[1]
            ),
            "observed_terminal_step": bool(proof and proof[0] == proof[1]),
            "final_adapter_filename_present": (
                row.get("status") == "OBSERVED" and row.get("final_adapter") is True
            ),
            "gates": {
                "training_execution_verified": False,
                "artifact_integrity_verified": False,
                "independent_backup_verified": False,
                "evaluated_quality_verified": False,
                "deployed_verified": False,
                "production_qualified": False,
            },
            "interpretation": (
                "The producer reported a pinned checkpoint state; this reducer did "
                "not reread Hub bytes, establish same-run continuity, trainer liveness, "
                "final-artifact integrity or quality."
            ),
        }
        result.append(entry)
    return result


def derive(current, previous=None):
    _watch(current)
    if previous is not None:
        _watch(previous)
        _require(
            _utc(previous["observed_at_utc"]) < _utc(current["observed_at_utc"]),
            "OBSERVATION_TIME_ORDER_INVALID",
        )
    coverage, indexed = salad_coverage(current)
    instance = instance_coverage(current, indexed)
    claims = model_claims(current, previous, indexed)
    hf = current["providers"].get("huggingface", {})
    capped = (
        isinstance(hf, dict)
        and type(hf.get("datasets_count_returned")) is int
        and type(hf.get("dataset_limit")) is int
        and hf["datasets_count_returned"] >= hf["dataset_limit"]
    )
    findings = list(coverage["findings"])
    findings.extend(instance["findings"])
    if current["watch_version"] == CANARY_WATCH:
        findings.append("CANDIDATE_RELEASE_NOT_ADMITTED")
    if capped:
        findings.append("HF_DATASET_LIST_CAPPED")
    if any(x["command_target_binding"] == "TARGET_MISMATCH" for x in claims):
        findings.append("DECLARED_TARGET_MISMATCH")
    if any(
        x["checkpoint_step_delta"] is not None and x["checkpoint_step_delta"] < 0
        for x in claims
    ):
        findings.append("CHECKPOINT_COUNTER_REGRESSION_OR_REPLACEMENT_UNRESOLVED")
    if any(x["target_changed_since_prior"] for x in claims):
        findings.append("TARGET_CHANGED_COMPARISON_UNAVAILABLE")
    return {
        "schema": SCHEMA,
        "observed_at_utc": current["observed_at_utc"],
        "prior_observed_at_utc": previous.get("observed_at_utc") if previous else None,
        "producer_watch_version": current["watch_version"],
        "previous_watch_version": previous["watch_version"] if previous else None,
        "producer_admission": (
            "OWNER_REVIEWED_V2_OBSERVER"
            if current["watch_version"] == WATCH
            else "V3_CANDIDATE_UNADMITTED_MANUAL_CANARY_ONLY"
        ),
        "provider_coverage": {
            "salad": coverage,
            "salad_instances": instance,
            "hf_dataset_listing_capped": bool(capped),
        },
        "model_claims": claims,
        "findings": sorted(set(findings)),
        "authority": {
            "read_only_offline_reducer": True,
            "new_scheduler": False,
            "signed": False,
            "training_started": False,
            "cloud_mutations": False,
            "qualification_conferred": False,
            "public_publication": False,
            "candidate_release_deployed": False,
        },
        "scope": "Private unsigned metadata graph; no node liveness, durable backup, quality, deployment or settlement.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True)
    parser.add_argument("--previous")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        current_raw = Path(args.current).read_bytes()
        prior_raw = Path(args.previous).read_bytes() if args.previous else None
        graph = derive(
            strict_json(current_raw), strict_json(prior_raw) if prior_raw else None
        )
        graph["input_sha256"] = {
            "current": hashlib.sha256(current_raw).hexdigest(),
            "previous": hashlib.sha256(prior_raw).hexdigest() if prior_raw else None,
        }
        raw = (
            json.dumps(graph, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
        output = Path(args.output)
        with output.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print(
            json.dumps(
                {
                    "status": "PRIVATE_GRAPH_WRITTEN",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "issues": graph["findings"],
                    "salad_groups_captured": graph["provider_coverage"]["salad"][
                        "captured_count"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    except (CoverageHold, OSError, ValueError) as exc:
        # Never print paths or provider exception bodies; keep diagnostics finite.
        code = str(exc) if isinstance(exc, CoverageHold) else type(exc).__name__
        print(json.dumps({"status": "HOLD", "reason": code}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
