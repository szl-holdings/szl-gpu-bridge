"""Bounded read-only SaladCloud fleet witness, not a trainer-completion claim.

The provider key is accepted only through SALAD_API_KEY in the runner environment.
Provider response data is aggressively reduced to non-secret state/replica fields.
No provider POST/PATCH/DELETE, no redirects, no retry after access denial.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

PROVIDER_URL = (
    "https://api.salad.com/api/public/organizations/"
    "alloyscape/projects/default/containers?per_page=100"
)
CHAIN_URL = (
    "https://huggingface.co/api/datasets/SZLHOLDINGS/frontier-54-runs/commits/main"
)
MAX_BYTES = 3_000_000
MAX_GROUPS = 300
MAX_SELECTED_INSTANCE_CALLS = 5
MAX_INSTANCES_PER_GROUP = 64
SOURCE_PRIORITY_GROUPS = frozenset(
    {
        "szl-train-flagship54-v2",
        "szl-train-flagship54-khipu-r2",
        "szl-train-absrl-grpo-r1",
        "szl-train-a11oy-mini-r1",
        "szl-train-prime-principle-r1",
    }
)
INSTANCE_URLS = {
    name: PROVIDER_URL.split("?", 1)[0] + "/" + name + "/instances"
    for name in SOURCE_PRIORITY_GROUPS
}
INSTANCE_STATES = frozenset(
    {
        "allocating",
        "downloading",
        "creating",
        "running",
        "stopping",
        "stopped",
        "failed",
        "error",
    }
)
ALLOWED_STATES = frozenset(
    {"running", "deploying", "pending", "stopped", "failed", "stopping"}
)
NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}\Z")


class Unavailable(ValueError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_json(url: str, *, token: str | None = None, opener=None) -> object:
    if url not in (PROVIDER_URL, CHAIN_URL, *INSTANCE_URLS.values()):
        raise Unavailable("UNAPPROVED_SOURCE")
    headers = {"Accept": "application/json", "User-Agent": "szl-fleet-evidence/1.0"}
    if token is not None:
        if url == CHAIN_URL:
            raise Unavailable("CREDENTIAL_SCOPE_REFUSED")
        headers["Salad-Api-Key"] = token
    elif url != CHAIN_URL:
        raise Unavailable("PROVIDER_KEY_REQUIRED")
    request = urllib.request.Request(url, headers=headers, method="GET")
    client = opener or urllib.request.build_opener(NoRedirect())
    with client.open(request, timeout=12) as response:
        if response.status != 200:
            raise Unavailable("PROVIDER_NON_200")
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise Unavailable("RESPONSE_BYTE_BUDGET")
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Unavailable("INVALID_JSON_RESPONSE") from exc


def nonnegative_int(value) -> int | None:
    return value if type(value) is int and value >= 0 else None


def summarize_fleet(payload: object) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise Unavailable("INVALID_FLEET_COLLECTION")
    items = payload["items"]
    if len(items) > MAX_GROUPS:
        raise Unavailable("FLEET_ITEM_BUDGET")
    seen = set()
    state_counts: Counter = Counter()
    priorities: Counter = Counter()
    known_replicas = 0
    known_running_instances = 0
    unknown_replicas = 0
    unknown_instance_counts = 0
    unpinned_image_refs = 0
    groups = []
    for item in items:
        if not isinstance(item, dict):
            raise Unavailable("INVALID_FLEET_ITEM")
        name = item.get("name")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name in seen:
            raise Unavailable("INVALID_OR_DUPLICATE_GROUP")
        seen.add(name)
        state = item.get("current_state")
        if not isinstance(state, dict):
            raise Unavailable("INVALID_GROUP_STATE")
        status = state.get("status")
        if not isinstance(status, str) or status not in ALLOWED_STATES:
            status = "unknown"
        state_counts[status] += 1
        replica_count = nonnegative_int(item.get("replicas"))
        if replica_count is None:
            unknown_replicas += 1
        else:
            known_replicas += replica_count
        counts = state.get("instance_status_counts")
        running = nonnegative_int(
            counts.get("running_count") if isinstance(counts, dict) else None
        )
        if running is None:
            unknown_instance_counts += 1
        else:
            known_running_instances += running
        container = item.get("container")
        image = container.get("image") if isinstance(container, dict) else None
        immutable_image_ref = bool(isinstance(image, str) and DIGEST_RE.search(image))
        if not immutable_image_ref:
            unpinned_image_refs += 1
        priority = item.get("priority")
        priorities[
            priority
            if isinstance(priority, str)
            and priority in {"high", "medium", "low", "lowest"}
            else "unknown"
        ] += 1
        groups.append(
            {
                "name": name,
                "state": status,
                "configured_replicas": replica_count,
                "provider_running_instances": running,
                "immutable_image_reference": immutable_image_ref,
            }
        )
    # A provider's item order is not an immutable identity. Canonicalize by
    # group name before hashing or selecting bounded per-group review samples.
    groups.sort(key=lambda group: group["name"])
    by_intent = {}
    for group in groups:
        intent = (
            "unknown"
            if group["configured_replicas"] is None
            else "parked_zero_replicas"
            if group["configured_replicas"] == 0
            else "positive_replica_intent"
        )
        by_intent.setdefault(group["state"], Counter())[intent] += 1
    # The provider's collection endpoint does not prove completeness for every
    # project/account. Do not misstate first-response coverage as an estate census.
    canonical = json.dumps(
        groups, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    anomalies = [
        group
        for group in groups
        if group["state"] in {"failed", "unknown"}
        or (
            group["state"] == "running"
            and group["configured_replicas"] is not None
            and group["configured_replicas"] > 0
            and group["provider_running_instances"] == 0
        )
    ]
    return {
        "provider_observation": "OBSERVED_GROUP_METADATA",
        "provider_collection_scope": "ONE_BOUNDED_RESPONSE_NOT_PROVEN_EXHAUSTIVE",
        "groups_observed": len(groups),
        "group_state_counts": dict(sorted(state_counts.items())),
        "state_by_replica_intent": {
            state: dict(sorted(intents.items()))
            for state, intents in sorted(by_intent.items())
        },
        "training_named_group_states": dict(
            sorted(
                Counter(
                    g["state"] for g in groups if g["name"].startswith("szl-train-")
                ).items()
            )
        ),
        "training_named_groups_observed": sum(
            g["name"].startswith("szl-train-") for g in groups
        ),
        "source_priority_group_sample": [
            g for g in groups if g["name"] in SOURCE_PRIORITY_GROUPS
        ],
        "priority_counts": dict(sorted(priorities.items())),
        "configured_replicas_known_subtotal": known_replicas,
        "configured_replicas_unknown_groups": unknown_replicas,
        "running_instances_provider_reported_subtotal": known_running_instances,
        "running_instance_count_unknown_groups": unknown_instance_counts,
        "groups_without_immutable_image_reference": unpinned_image_refs,
        "group_metadata_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        "anomalies_sample": anomalies[:12],
        "anomalies_total": len(anomalies),
        "trainer_steps_observed": False,
        "training_liveness_verified": False,
        "ready_instances_independently_verified": False,
        "billable_runtime_state_verified": False,
        "source_deployment_parity_verified": False,
    }


def summarize_instances(payload: object) -> dict:
    """Bounded provider instance state without machine identities or host data."""
    if not isinstance(payload, dict) or not isinstance(payload.get("instances"), list):
        raise Unavailable("INVALID_INSTANCE_COLLECTION")
    instances = payload["instances"]
    if len(instances) > MAX_INSTANCES_PER_GROUP:
        raise Unavailable("INSTANCE_LIST_BUDGET")
    states: Counter = Counter()
    ready = started = unknown_ready = 0
    for value in instances:
        if not isinstance(value, dict):
            raise Unavailable("INVALID_INSTANCE_ITEM")
        state = value.get("state")
        if not isinstance(state, str) or state not in INSTANCE_STATES:
            state = "unknown"
        states[state] += 1
        if value.get("ready") is True:
            ready += 1
        elif value.get("ready") is not False:
            unknown_ready += 1
        if value.get("started") is True:
            started += 1
    return {
        "state": "PROVIDER_INSTANCE_METADATA_OBSERVED",
        "instance_count": len(instances),
        "instance_states": dict(sorted(states.items())),
        "ready_count_provider_reported": ready,
        "started_count_provider_reported": started,
        "ready_state_unknown_instances": unknown_ready,
        "trainer_process_verified": False,
        "source_image_digest_verified": False,
        "provider_group_snapshot_atomic": False,
    }


def reconcile_selected_instances(group: dict, sample: dict) -> dict:
    """Reconcile two non-atomic provider GETs; NEVER attest trainer liveness.

    A reported mismatch means investigation is warranted, not that the same
    instant was sampled or that paid GPUs, source parity, or training exist.
    Only normalized, allowlisted group and instance fields are exposed.
    """
    if sample.get("state") != "PROVIDER_INSTANCE_METADATA_OBSERVED":
        raise Unavailable("INSTANCE_READBACK_UNAVAILABLE")
    desired = group.get("configured_replicas")
    collection_running = group.get("provider_running_instances")
    if desired is not None and nonnegative_int(desired) is None:
        raise Unavailable("GROUP_REPLICA_FIELD_INVALID")
    if collection_running is not None and nonnegative_int(collection_running) is None:
        raise Unavailable("GROUP_RUNNING_FIELD_INVALID")
    observed_instances = sample["instance_count"]
    running = sample["instance_states"].get("running", 0)
    ready = sample["ready_count_provider_reported"]
    reasons = []
    if desired is None:
        reasons.append("DESIRED_REPLICAS_UNAVAILABLE")
    elif desired == 0 and observed_instances > 0:
        reasons.append("ZERO_REPLICA_GROUP_HAS_INSTANCE_SAMPLE")
    elif desired > 0 and observed_instances == 0:
        reasons.append("POSITIVE_REPLICA_INTENT_WITHOUT_INSTANCE_SAMPLE")
    if collection_running is None:
        reasons.append("COLLECTION_RUNNING_COUNT_UNAVAILABLE")
    elif collection_running != running:
        reasons.append("NONATOMIC_RUNNING_COUNT_DIFFERENCE")
    if group["state"] == "running" and desired is not None and desired > 0:
        if running == 0:
            reasons.append("RUNNING_LABEL_WITHOUT_RUNNING_INSTANCE")
    if ready > running:
        reasons.append("READY_REPORT_DOES_NOT_ESTABLISH_RUNNING")
    if not group["immutable_image_reference"]:
        reasons.append("IMAGE_REFERENCE_NOT_IMMUTABLE")
    return {
        "state": "HOLD" if reasons else "STRUCTURAL_ONLY",
        "scope": "TWO_NONATOMIC_PROVIDER_GETS_NOT_A_TRAINER_WITNESS",
        "reason_codes": sorted(set(reasons)),
        "desired_replicas": desired,
        "collection_running_count": collection_running,
        "instance_sample_count": observed_instances,
        "instance_sample_running_count": running,
        "instance_sample_ready_count": ready,
        "trainer_liveness_verified": False,
        "checkpoint_bytes_verified": False,
        "source_image_digest_verified": False,
        "billing_verified": False,
        "production_admission": "HOLD",
    }


def summarize_chain(payload: object, *, now: datetime) -> dict:
    if not isinstance(payload, list):
        raise Unavailable("INVALID_CHAIN_COLLECTION")
    fresh = 0
    latest = None
    for record in payload[:100]:
        if not isinstance(record, dict):
            continue
        value = record.get("date")
        if not isinstance(value, str):
            continue
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if stamp.tzinfo is None:
            continue
        stamp = stamp.astimezone(timezone.utc)
        if latest is None or latest < stamp:
            latest = stamp
        if 0 <= (now - stamp).total_seconds() <= 7200:
            fresh += 1
    return {
        "source": "PUBLIC_HF_COMMIT_COLLECTION",
        "sampled_commits": min(len(payload), 100),
        "recent_commit_count_in_sample": fresh,
        "latest_sampled_commit_utc": latest.isoformat() if latest else None,
        "per_lane_heartbeat_verified": False,
        "checkpoint_durability_verified": False,
        "training_completion_verified": False,
        "scope": "GLOBAL_CHAIN_RECENCY_NOT_LANE_LIVENESS",
    }


def collect(key: str | None, *, reader=read_json, now: datetime | None = None):
    now = now or datetime.now(timezone.utc)
    report = {
        "schema": "szl.salad-fleet-evidence/v2",
        "observed_utc": now.isoformat(),
        "salad_organization": "alloyscape",
        "salad_project": "default",
        "provider_writes": False,
        "training_completion_verified": False,
        "scaling_authorized": False,
        "release_admission": "HOLD",
    }
    drift = []
    if not key:
        report["fleet"] = {
            "provider_observation": "UNAVAILABLE",
            "reason": "KEY_NOT_CONFIGURED",
        }
        drift.append("SALAD_KEY_NOT_CONFIGURED")
    else:
        try:
            report["fleet"] = summarize_fleet(reader(PROVIDER_URL, token=key))
            # A provider group marked running without any running instance is
            # operational drift, NOT a silent 'green' fleet-watch success.
            # Preserve the full bounded sample while emitting one stable,
            # non-secret reason code; never infer trainer liveness from status.
            if report["fleet"]["anomalies_total"] > 0:
                drift.append("SALAD_PROVIDER_STATE_INSTANCE_MISMATCH")
            if report["fleet"]["groups_observed"] == 0:
                drift.append("SALAD_EMPTY_COLLECTION_NOT_PROVEN_COMPLETE")
            # Up to five immutable-name allowlisted API GETs; 401/403/429
            # stops subsequent reads without retry or credential fallback.
            samples = report["fleet"].get("source_priority_group_sample", [])
            readbacks = {}
            access_stopped = False
            for group in samples[:MAX_SELECTED_INSTANCE_CALLS]:
                name = group["name"]
                if access_stopped:
                    readbacks[name] = {
                        "state": "UNAVAILABLE",
                        "reason": "PROVIDER_ACCESS_STOPPED_AFTER_DENIAL",
                    }
                    continue
                try:
                    readbacks[name] = summarize_instances(
                        reader(INSTANCE_URLS[name], token=key)
                    )
                    readbacks[name]["reconciliation"] = (
                        reconcile_selected_instances(group, readbacks[name])
                    )
                except urllib.error.HTTPError as exc:
                    code = exc.code if type(exc.code) is int else None
                    readbacks[name] = {
                        "state": "UNAVAILABLE",
                        "reason": "PROVIDER_HTTP_" + str(code),
                    }
                    if code in (401, 403, 429):
                        access_stopped = True
                        drift.append("SALAD_SELECTED_INSTANCES_HTTP_" + str(code))
                except Unavailable as exc:
                    readbacks[name] = {"state": "UNAVAILABLE", "reason": str(exc)}
                except (OSError, TimeoutError) as exc:
                    readbacks[name] = {
                        "state": "UNAVAILABLE",
                        "reason": type(exc).__name__,
                    }
            report["fleet"]["selected_instance_readbacks"] = readbacks
            report["fleet"]["selected_instance_access_stopped"] = access_stopped
        except urllib.error.HTTPError as exc:
            status = exc.code if type(exc.code) is int else None
            code = "SALAD_HTTP_" + str(status)
            report["fleet"] = {"provider_observation": "UNAVAILABLE", "reason": code}
            drift.append(code)
        except Unavailable as exc:
            report["fleet"] = {
                "provider_observation": "UNAVAILABLE",
                "reason": str(exc),
            }
            drift.append("SALAD_" + str(exc))
        except (OSError, TimeoutError) as exc:
            code = "SALAD_TRANSPORT_" + type(exc).__name__
            report["fleet"] = {"provider_observation": "UNAVAILABLE", "reason": code}
            drift.append(code)
    try:
        report["receipt_chain"] = summarize_chain(reader(CHAIN_URL), now=now)
    except (Unavailable, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        code = "CHAIN_OBSERVATION_UNAVAILABLE_" + type(exc).__name__
        report["receipt_chain"] = {
            "source": "PUBLIC_HF_COMMIT_COLLECTION",
            "state": "UNAVAILABLE",
            "reason": code,
            "per_lane_heartbeat_verified": False,
        }
        drift.append(code)
    return report, drift


def emit(report: dict, drift: list[str], *, output_path: str | None = None) -> None:
    def compact(value):
        return json.dumps(
            value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )

    if output_path:
        # One-line JSON outputs cannot inject a multiline runner-output delimiter.
        with Path(output_path).open("a", encoding="utf-8", newline="\n") as file:
            file.write("has_drift=" + ("true" if drift else "false") + "\n")
            file.write("drift=" + compact(drift) + "\n")
            file.write("report=" + compact(report) + "\n")
    print(
        compact(
            {
                "fleet": report.get("fleet"),
                "receipt_chain": report.get("receipt_chain"),
                "release_admission": "HOLD",
                "drift": drift,
            }
        )
    )


if __name__ == "__main__":
    observation, blockers = collect(os.environ.get("SALAD_API_KEY"))
    emit(observation, blockers, output_path=os.environ.get("GITHUB_OUTPUT"))
