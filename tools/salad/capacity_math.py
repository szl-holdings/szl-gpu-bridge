#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SZL Holdings — original source-only capacity mathematics
"""Evidence-bounded queueing and fleet-capacity experiments.

STD-LIB ONLY, offline, read-only. This module neither reads live provider
credentials nor mutates GPUs, clusters, deployments, datasets, DNS or money.

Inputs are user-supplied data, never independently authenticated telemetry.
All derived estimates are hypothetical planning quantities, not admission.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import sys

INPUT_SCHEMA = "szl.capacity-sample/v1"
OUTPUT_SCHEMA = "szl.capacity-experiment/v1"
MAX_BYTES = 131072
MAX_SAMPLES = 256
MIN_SAMPLES = 20
Z95 = 1.959963984540054
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA64 = re.compile(r"[0-9a-f]{64}\Z")
_UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\Z")
_REPOSITORY = "szl-holdings/szl-gpu-bridge"
_SOURCE_TYPES = frozenset({"SYNTHETIC", "OPERATOR_REPORTED_UNVERIFIED"})


class Hold(ValueError):
    """Stable diagnostic only. Never includes inputs or potential secrets."""


def _keys(value, fields, code):
    if type(value) is not dict or set(value) != set(fields):
        raise Hold(code)
    return value


def _integer(value, low, high, code):
    if type(value) is not int or value < low or value > high:
        raise Hold(code)
    return value


def _number(value, low, high, code, *, exclusive_low=False):
    if type(value) not in (int, float):
        raise Hold(code)
    if (
        not math.isfinite(value)
        or value > high
        or (value <= low if exclusive_low else value < low)
    ):
        raise Hold(code)
    return float(value)


def _samples(value, code, *, positive=True):
    if type(value) is not list or not MIN_SAMPLES <= len(value) <= MAX_SAMPLES:
        raise Hold(code)
    return [_number(number, 0, 3600, code, exclusive_low=positive) for number in value]


def _time(value, code):
    if type(value) is not str or _UTC.fullmatch(value) is None:
        raise Hold(code)
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise Hold(code) from None
    return result


def _object_pairs(items):
    record = {}
    for key, value in items:
        if key in record:
            raise Hold("DUPLICATE_JSON_KEY")
        record[key] = value
    return record


def _nonfinite(_):
    raise Hold("NONFINITE_JSON")


def read_input(path):
    """One local bounded JSON file; symlinks or duplicate keys are refused."""
    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise Hold("SOURCE_FILE_UNAVAILABLE")
        if source.stat().st_size > MAX_BYTES or source.stat().st_size == 0:
            raise Hold("SOURCE_FILE_BYTE_BUDGET")
        with source.open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
    except OSError:
        raise Hold("SOURCE_FILE_UNAVAILABLE") from None
    if not 0 < len(raw) <= MAX_BYTES:
        raise Hold("SOURCE_FILE_BYTE_BUDGET")
    try:
        data = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError):
        raise Hold("SOURCE_JSON_INVALID") from None
    return data, hashlib.sha256(raw).hexdigest()


def _cv2(values):
    mean = statistics.fmean(values)
    # Squared coefficient of variation. Population form is a descriptive
    # statistic of this bounded sample, not an unbiased population inference.
    return statistics.pvariance(values) / (mean * mean)


def _percentile(values, fraction):
    # Nearest-rank empirical quantile. No confidence interval is claimed.
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _wilson_upper(failures, total):
    """Two-sided 95% Wilson upper bound; i.i.d. assumption unverified."""
    p = failures / total
    z2 = Z95 * Z95
    denominator = 1.0 + z2 / total
    centre = p + z2 / (2 * total)
    radius = Z95 * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total)
    return min(1.0, (centre + radius) / denominator)


def evaluate(spec, *, as_of=None, input_sha256=None):
    """Generate transparent estimates without permitting actions."""
    top = _keys(
        spec,
        ("schema", "source", "window", "capacity", "requests"),
        "INVALID_INPUT_FIELDS",
    )
    if top["schema"] != INPUT_SCHEMA:
        raise Hold("SCHEMA_UNSUPPORTED")
    source = _keys(
        top["source"],
        ("repository", "revision", "sample_sha256", "kind"),
        "SOURCE_BINDING_FIELDS",
    )
    if (
        source["repository"] != _REPOSITORY
        or type(source["revision"]) is not str
        or _SHA40.fullmatch(source["revision"]) is None
        or source["revision"] == "0" * 40
        or type(source["sample_sha256"]) is not str
        or _SHA64.fullmatch(source["sample_sha256"]) is None
        or source["sample_sha256"] == "0" * 64
        or type(source["kind"]) is not str
        or source["kind"] not in _SOURCE_TYPES
    ):
        raise Hold("SOURCE_BINDING_UNVERIFIED")
    window = _keys(top["window"], ("started_at", "ended_at"), "WINDOW_FIELDS")
    begin = _time(window["started_at"], "WINDOW_TIME_INVALID")
    end = _time(window["ended_at"], "WINDOW_TIME_INVALID")
    duration = (end - begin).total_seconds()
    if not 30 <= duration <= 86400:
        raise Hold("WINDOW_DURATION_INVALID")
    now = as_of or datetime.now(timezone.utc)
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise Hold("REFERENCE_CLOCK_INVALID")
    now = now.astimezone(timezone.utc)
    age = (now - end).total_seconds()
    if age < -5:
        raise Hold("WINDOW_IN_FUTURE")
    capacity = _keys(
        top["capacity"],
        (
            "ready_replicas",
            "min_replicas",
            "max_replicas",
            "target_utilization",
            "inflight_limit",
        ),
        "CAPACITY_FIELDS",
    )
    ready = _integer(capacity["ready_replicas"], 1, 10000, "REPLICA_COUNT_INVALID")
    lower = _integer(capacity["min_replicas"], 1, 10000, "REPLICA_BOUND_INVALID")
    upper = _integer(capacity["max_replicas"], 1, 10000, "REPLICA_BOUND_INVALID")
    if not lower <= ready <= upper:
        raise Hold("REPLICA_BOUNDS_INCONSISTENT")
    target = _number(
        capacity["target_utilization"],
        0,
        0.95,
        "UTILIZATION_TARGET_INVALID",
        exclusive_low=True,
    )
    limit = _integer(capacity["inflight_limit"], 1, 100000, "INFLIGHT_LIMIT_INVALID")
    requests = _keys(
        top["requests"],
        (
            "arrivals",
            "completed",
            "failed",
            "service_time_seconds",
            "response_time_seconds",
            "interarrival_seconds",
            "baseline_rtt_seconds",
            "sample_rtt_seconds",
        ),
        "REQUEST_FIELDS",
    )
    arrived = _integer(requests["arrivals"], 1, 10000000, "ARRIVALS_INVALID")
    completed = _integer(requests["completed"], 1, 10000000, "COMPLETIONS_INVALID")
    failed = _integer(requests["failed"], 0, completed, "FAILURES_INVALID")
    service = _samples(requests["service_time_seconds"], "SERVICE_SAMPLE_INVALID")
    response = _samples(requests["response_time_seconds"], "RESPONSE_SAMPLE_INVALID")
    gaps = _samples(requests["interarrival_seconds"], "INTERARRIVAL_SAMPLE_INVALID")
    if len(service) > completed or len(response) > completed or len(gaps) >= arrived:
        raise Hold("SAMPLE_COUNT_INCONSISTENT")
    baseline = _number(
        requests["baseline_rtt_seconds"],
        0,
        3600,
        "BASELINE_RTT_INVALID",
        exclusive_low=True,
    )
    sample = _number(
        requests["sample_rtt_seconds"],
        0,
        3600,
        "SAMPLED_RTT_INVALID",
        exclusive_low=True,
    )
    if baseline > sample:
        raise Hold("RTT_BASELINE_EXCEEDS_SAMPLE")
    arrival_rate = arrived / duration
    completed_rate = completed / duration
    mean_service = statistics.fmean(service)
    mean_response = statistics.fmean(response)
    offered = arrival_rate * mean_service
    rho = offered / ready
    ca2 = _cv2(gaps)
    cs2 = _cv2(service)
    if not all(math.isfinite(v) for v in (rho, offered, ca2, cs2)):
        raise Hold("NUMERIC_OVERFLOW")
    # Same algebra as HPA's ratio, but rho comes from measured SAMPLE MEANS,
    # not Kubernetes CPU metrics. No control-loop parity or admission claim.
    raw_replicas = math.ceil(offered / target)
    bounded_replicas = min(upper, max(lower, raw_replicas))
    if raw_replicas > upper:
        planning_state = "CAPACITY_CEILING_EXCEEDED"
    elif raw_replicas < ready:
        planning_state = "DOWNSCALE_NOT_EVALUATED"
    else:
        planning_state = "ADVISORY_ONLY"
    if ready == 1 and rho < 1:
        kingman = {
            "state": "G_G_1_APPROXIMATION",
            "mean_wait_seconds": (rho / (1 - rho) * (ca2 + cs2) / 2 * mean_service),
            "assumption": "ONE_STATIONARY_SINGLE_SERVER_QUEUE_NOT_VERIFIED",
        }
    else:
        kingman = {
            "state": (
                "MULTISERVER_NOT_G_G_1"
                if ready > 1
                else "NO_STATIONARY_FINITE_WAIT_AT_OR_ABOVE_ONE"
            ),
            "mean_wait_seconds": None,
            "assumption": "KINGMAN_NOT_APPLICABLE",
        }
    # Netflix Vegas form, separate from its distributed control algorithm.
    vegas_queue = limit * (1.0 - baseline / sample)
    # If two descriptive means disagree under this approximate queue model,
    # expose the residual rather than silently treating the model as calibrated.
    residual = (
        mean_response - mean_service - kingman["mean_wait_seconds"]
        if kingman["mean_wait_seconds"] is not None
        else None
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "state": "SYNTHETIC_ANALYSIS"
        if source["kind"] == "SYNTHETIC"
        else "UNVERIFIED_OPERATOR_SAMPLE",
        "source": {
            "repository": _REPOSITORY,
            "declared_revision": source["revision"],
            "declared_telemetry_sha256": source["sample_sha256"],
            "input_sha256": input_sha256,
            "source_revision_verified": False,
            "telemetry_authenticity_verified": False,
        },
        "observation": {
            "started_at": window["started_at"],
            "ended_at": window["ended_at"],
            "window_seconds": duration,
            "seconds_since_window": age,
            "current_eligible": False,
            "arrivals": arrived,
            "completions": completed,
            "failed": failed,
            "arrival_rate_per_second": arrival_rate,
            "completion_rate_per_second": completed_rate,
            "backlog_delta_count": arrived - completed,
            "service_samples": len(service),
            "response_samples": len(response),
            "interarrival_samples": len(gaps),
        },
        "queue": {
            "mean_service_seconds": mean_service,
            "mean_response_seconds": mean_response,
            "empirical_p95_response_seconds": _percentile(response, 0.95),
            "arrival_cv_squared": ca2,
            "service_cv_squared": cs2,
            "mean_offered_concurrency_little_identity": arrival_rate * mean_response,
            "mean_offered_work_per_second": offered,
            "rho_per_ready_replica": rho,
            "kingman": kingman,
            "kingman_mean_response_residual_seconds": residual,
            "residual_scope": "UNVERIFIED_SAMPLE_VS_APPROXIMATION_NO_SIGNIFICANCE_TEST",
            "netflix_vegas_queue_signal": vegas_queue,
            "wilson_95_two_sided_upper_failure_fraction": _wilson_upper(
                failed, completed
            ),
            "little_assumption": "STATIONARITY_AND_FLOW_CONSERVATION_NOT_PROVED",
            "vegas_assumption": "COMPARABLE_MIN_RTT_AND_SAMPLED_RTT_NOT_VERIFIED",
            "wilson_assumption": "IID_BERNOULLI_NOT_VERIFIED",
            "p95_scope": "DESCRIPTIVE_SAMPLE_NOT_SLA_OR_CONFIDENCE_BOUND",
        },
        "capacity": {
            "ready_replicas": ready,
            "target_utilization": target,
            "minimum_replicas": lower,
            "maximum_replicas": upper,
            "hpa_like_raw_replicas": raw_replicas,
            "bounded_planning_replicas": bounded_replicas,
            "planning_state": planning_state,
            "scale_action": "NONE",
            "proposed_gpu_replica_change_authorized": False,
            "gpu_cost_measured": False,
        },
        "authority": {
            "provider_writes": 0,
            "training_executed": False,
            "model_quality_verified": False,
            "production_qualified": False,
            "signed_evidence": False,
            "settlement_verified": False,
            "promotion_authorized": False,
        },
        "scope": "READ_ONLY_EXPLORATORY_MATH_NO_CONTROLLER",
        "limitations": [
            "Operator inputs and source hashes are declared, not read and authenticated.",
            "Little's law requires stationary long-run conditions, not proven by this window.",
            "Kingman is an approximate single-server model; multi-GPU use prohibited.",
            "Vegas estimate is not Netflix's actual distributed controller or an admission policy.",
            "HPA-like candidate uses offered work, not Kubernetes CPU/pod metrics or stabilization.",
            "Observed p95 and Wilson bounds cannot certify latency or safety under correlated loads.",
            "No cost, energy, model-evaluation, settlement, or cloud deployment is inferred.",
        ],
    }


def render_html(data):
    """Python-only static report, escaping even if unexpected text is injected."""
    if (
        type(data) is not dict
        or data.get("schema") != OUTPUT_SCHEMA
        or data.get("scope") != "READ_ONLY_EXPLORATORY_MATH_NO_CONTROLLER"
        or type(data.get("authority")) is not dict
        or data["authority"].get("provider_writes") != 0
        or data["authority"].get("promotion_authorized") is not False
        or data["authority"].get("production_qualified") is not False
        or type(data.get("capacity")) is not dict
        or data["capacity"].get("scale_action") != "NONE"
    ):
        raise Hold("RENDER_CONTRACT_INVALID")

    def esc(value):
        return html.escape(str(value), quote=True)

    def row(label, value):
        return (
            "<tr><th scope='row'>"
            + esc(label)
            + "</th><td>"
            + esc(value)
            + "</td></tr>"
        )

    q = data["queue"]
    capacity = data["capacity"]
    observation = data["observation"]
    fields = [
        ("Evidence state", data["state"]),
        ("Original observation window", observation["ended_at"]),
        ("Arrival rate (/s)", q and observation["arrival_rate_per_second"]),
        ("Empirical p95 response (s)", q["empirical_p95_response_seconds"]),
        ("Offered work (replica-seconds / s)", q["mean_offered_work_per_second"]),
        ("Estimated utilization / ready replica", q["rho_per_ready_replica"]),
        (
            "Little: modeled mean inflight",
            q["mean_offered_concurrency_little_identity"],
        ),
        ("Kingman queue state", q["kingman"]["state"]),
        ("Kingman approximate wait (s)", q["kingman"]["mean_wait_seconds"]),
        (
            "Unverified response-model residual (s)",
            q["kingman_mean_response_residual_seconds"],
        ),
        ("Vegas delay queue proxy", q["netflix_vegas_queue_signal"]),
        (
            "Wilson 95% upper failure fraction",
            q["wilson_95_two_sided_upper_failure_fraction"],
        ),
        ("HPA-like raw replica count", capacity["hpa_like_raw_replicas"]),
        ("Planning status", capacity["planning_state"]),
        ("Actual GPU action", capacity["scale_action"]),
        ("Production qualification", data["authority"]["production_qualified"]),
    ]
    css = (
        "body{margin:0;background:#0c1721;color:#e8f3f7;font:1rem/1.6 system-ui,sans-serif}"
        "main{max-width:980px;margin:auto;padding:clamp(1rem,4vw,3rem)}"
        "h1{font-size:2rem;line-height:1.2}.warn{padding:1rem;border:1px solid #bf995f;"
        "background:#33291d;border-radius:.6rem}table{width:100%;border-collapse:collapse}"
        "td,th{padding:.6rem;border-bottom:1px solid #354657;text-align:left;overflow-wrap:anywhere}"
        "a{color:#9bdae1}a:focus-visible{outline:3px solid #f3c873}p{color:#b8d0db}"
        "div.table{overflow-x:auto}"
    )
    body = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta http-equiv='Content-Security-Policy' "
        "content=\"default-src 'none'; style-src 'unsafe-inline'; script-src 'none'; "
        "img-src 'none'; form-action 'none'; base-uri 'none'\">"
        "<title>SZL advisory capacity mathematics</title>"
        f"<style>{css}</style></head><body><main>"
        "<h1>Capacity mathematics — exploratory, not deployment authority</h1>"
        "<p class='warn' role='status'><strong>NO AUTOSCALING AUTHORIZED.</strong>"
        " Telemetry and Git SHA fields are unverified assertions. The output cannot"
        " allocate GPUs, qualify models, or authorize production.</p>"
        "<div class='table'><table><caption>Derived estimates, subject to assumptions</caption>"
        "<tbody>"
        + "".join(row(*fields_) for fields_ in fields)
        + "</tbody></table></div>"
        "<p>Research: Little (1961), Kingman (1961), Netflix concurrency-limits,"
        " Kubernetes HPA, and OpenTelemetry HTTP request duration. "
        "See the source repository's docs/runbooks/capacity-mathematics.md.</p>"
        "</main></body></html>"
    )
    return body.encode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only, offline queueing math experiment"
    )
    parser.add_argument(
        "--input", required=True, help="local, bounded JSON fixture; no URL or token"
    )
    parser.add_argument("--format", choices=("json", "html"), default="json")
    args = parser.parse_args(argv)
    try:
        specimen, raw_sha = read_input(args.input)
        result = evaluate(specimen, input_sha256=raw_sha)
        output = (
            render_html(result).decode()
            if args.format == "html"
            else (json.dumps(result, sort_keys=True, allow_nan=False, indent=2) + "\n")
        )
        sys.stdout.write(output)
        return 0
    except (Hold, OverflowError) as exc:
        code = str(exc) if isinstance(exc, Hold) else "NUMERIC_OVERFLOW"
        # Never include original exception text, file content or credentials.
        sys.stderr.write("CAPACITY_HOLD:" + code + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
