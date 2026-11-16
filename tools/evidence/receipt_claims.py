"""Conservative claims from the explicit abstention/refusal receipt schema.

No relabeling of questions, model-quality scoring, meter validation or result inheritance.
"""

from __future__ import annotations

import math


class ClaimHold(ValueError):
    """A contradictory or unsupported receipt cannot be summarized as a result."""


def battery_summary(receipt: dict) -> dict:
    if not isinstance(receipt, dict) or not isinstance(receipt.get("battery"), dict):
        raise ClaimHold("BATTERY_SCHEMA_UNAVAILABLE")
    battery = receipt["battery"]
    total, reported, probes = (
        battery.get("of"),
        battery.get("abstained_or_refused"),
        battery.get("probes"),
    )
    if (
        type(total) is not int
        or total <= 0
        or type(reported) is not int
        or not isinstance(probes, list)
        or len(probes) != total
    ):
        raise ClaimHold("AGGREGATE_DENOMINATOR_UNAVAILABLE")
    if any(
        not isinstance(p, dict) or type(p.get("abstained_or_refused")) is not bool
        for p in probes
    ):
        raise ClaimHold("PROBE_VERDICTS_UNAVAILABLE")
    observed = sum(p["abstained_or_refused"] for p in probes)
    if observed != reported:
        raise ClaimHold("AGGREGATE_COUNT_CONTRADICTION")
    rows = receipt.get("rows")
    categories = None
    if (
        isinstance(rows, list)
        and len(rows) == total
        and all(
            isinstance(r, dict)
            and isinstance(r.get("kind"), str)
            and r.get("kind") in {"answerable", "unanswerable"}
            and isinstance(r.get("verdict"), str)
            and r.get("verdict") in {"PASS", "FAIL"}
            for r in rows
        )
    ):
        categories = {}
        for kind in ("answerable", "unanswerable"):
            subset = [r for r in rows if r["kind"] == kind]
            categories[kind] = (
                {
                    "passed": sum(r["verdict"] == "PASS" for r in subset),
                    "total": len(subset),
                }
                if subset
                else None
            )
    readings = {}
    for name in ("joules_measured", "avg_watts", "wall_hours"):
        value = receipt.get(name)
        if value is not None and (
            type(value) not in (float, int)
            or (isinstance(value, float) and not math.isfinite(value))
            or value < 0
        ):
            raise ClaimHold("INVALID_RECORDED_METER_VALUE")
        readings[name] = value
    return {
        "abstained_or_refused": observed,
        "probe_count": total,
        "metric": "recorded abstention/refusal count, not answer accuracy",
        "subgroups": categories,
        "subgroup_scope": "explicit labels only; unavailable is not 0/0",
        "recorded_meter_fields": readings,
        "energy_independently_verified": False,
        "current_weight_binding_verified": False,
        "model_qualified": False,
    }
