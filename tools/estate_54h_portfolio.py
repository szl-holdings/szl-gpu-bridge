"""Fail-closed source-only plan for SZL's public 54-hour model portfolio.

Never dispatches workers, acquires leases, grants rights, signs receipts,
downloads weights/dataset rows, schedules compute, or mints model rewards.
Use the existing single-writer Salad admission controller after independent
owner, rights, hardware, signer, holdout, and budget qualification.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re

SCHEMA = "szl.public-training-metadata-audit/v1"
REPORT_SCHEMA = "szl.estate-54h-review-only/v1"
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
HF_ID = re.compile(r"SZLHOLDINGS/[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
ARTIFACT_SIGNALS = {
    "ADAPTER_FILES_LISTED",
    "WEIGHT_FILES_LISTED",
    "NO_RECOGNIZED_WEIGHT_FILE",
}
MAX_INPUT_BYTES = 3_000_000
MAX_ASSETS = 1000
MAX_AGE = timedelta(days=7)
REQUESTED_HOURS = 54


class Hold(ValueError):
    """A source/data uncertainty that is not training authorization."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise Hold(reason)


def _pairs_strict(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        _require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def parse_public_snapshot(raw: bytes) -> dict:
    _require(
        type(raw) is bytes and 0 < len(raw) <= MAX_INPUT_BYTES,
        "SNAPSHOT_BYTES_INVALID",
    )
    try:
        data = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_strict,
            parse_constant=lambda _: _require(False, "NONFINITE_JSON"),
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise Hold("SNAPSHOT_INVALID_JSON") from exc
    _require(type(data) is dict, "SNAPSHOT_NOT_OBJECT")
    return data


def _utc(value: str) -> datetime:
    _require(type(value) is str, "OBSERVATION_TIME_UNAVAILABLE")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Hold("OBSERVATION_TIME_INVALID") from exc
    _require(
        parsed.tzinfo is not None and parsed.utcoffset() is not None, "CLOCK_NAIVE"
    )
    return parsed.astimezone(timezone.utc)


def validate_snapshot(s: dict, *, now: datetime) -> None:
    """Metadata integrity check only; never grants rights from a Hub card."""
    _require(type(s) is dict and s.get("schema") == SCHEMA, "SNAPSHOT_SCHEMA")
    _require(
        s.get("scope")
        == "Hugging Face anonymous public author search; NOT authenticated private inventory",
        "SNAPSHOT_AUTHORITY_SCOPE",
    )
    _require(
        s.get("provider_write_operations") == 0
        and type(s.get("provider_write_operations")) is int
        and s.get("training_jobs_started") == 0
        and type(s.get("training_jobs_started")) is int,
        "SNAPSHOT_EFFECT_CLAIM",
    )
    _require(
        s.get("all_training_admitted") is False
        and s.get("rights_clearance") == "UNVERIFIED",
        "UNAUTHORIZED_ADMISSION_CLAIM",
    )
    _require(now.tzinfo is not None and now.utcoffset() is not None, "CLOCK_NAIVE")
    age = now.astimezone(timezone.utc) - _utc(s.get("observed_at_utc"))
    _require(timedelta(0) <= age <= MAX_AGE, "SNAPSHOT_STALE_OR_FUTURE")
    counts = s.get("counts")
    items = s.get("public_assets")
    _require(type(counts) is dict and type(items) is dict, "ASSET_LIST_MISSING")
    _require(
        set(counts) == {"models", "datasets"} and set(items) == {"models", "datasets"},
        "ASSET_KIND_INVALID",
    )
    for kind in ("models", "datasets"):
        entries = items[kind]
        _require(
            type(entries) is list and 0 < len(entries) <= MAX_ASSETS,
            "ASSET_COUNT_BUDGET",
        )
        _require(
            type(counts[kind]) is int and counts[kind] == len(entries),
            "ASSET_COUNT_CONTRADICTED",
        )
        seen = set()
        for row in entries:
            _require(type(row) is dict, "ASSET_RECORD_INVALID")
            rid = row.get("repo")
            sha = row.get("immutable_hub_sha")
            _require(
                type(rid) is str and HF_ID.fullmatch(rid) is not None, "ASSET_NAMESPACE"
            )
            _require(rid not in seen, "DUPLICATE_ASSET_ID")
            seen.add(rid)
            _require(
                type(sha) is str and SHA40.fullmatch(sha) is not None,
                "ASSET_SHA_UNVERIFIED",
            )
            _require(
                row.get("rights_review") == "NOT_PERFORMED"
                and row.get("training_admitted") is False,
                "PUBLIC_METADATA_NOT_RIGHTS_APPROVAL",
            )
            if kind == "models":
                _require(
                    set(row)
                    == {
                        "repo",
                        "immutable_hub_sha",
                        "artifact_signal",
                        "weight_file_count",
                        "license_tags",
                        "rights_review",
                        "training_admitted",
                    },
                    "MODEL_METADATA_SCHEMA",
                )
                _require(
                    row["artifact_signal"] in ARTIFACT_SIGNALS,
                    "MODEL_ARTIFACT_SIGNAL",
                )
                _require(
                    type(row["weight_file_count"]) is int
                    and 0 <= row["weight_file_count"] <= 100_000,
                    "WEIGHT_FILE_COUNT_INVALID",
                )
                _require(
                    type(row["license_tags"]) is list
                    and all(
                        type(t) is str and 0 < len(t) <= 120
                        for t in row["license_tags"]
                    ),
                    "MODEL_LICENSE_TAG_INVALID",
                )
                _require(
                    (row["artifact_signal"] == "NO_RECOGNIZED_WEIGHT_FILE")
                    == (row["weight_file_count"] == 0),
                    "MODEL_SIGNAL_CONTRADICTED",
                )
            else:
                _require(
                    set(row)
                    == {
                        "repo",
                        "immutable_hub_sha",
                        "declared_license",
                        "rights_review",
                        "holdout_verified",
                        "training_admitted",
                    },
                    "DATASET_METADATA_SCHEMA",
                )
                _require(
                    row.get("holdout_verified") is False,
                    "PUBLIC_METADATA_NOT_HOLDOUT_APPROVAL",
                )
                lic = row.get("declared_license")
                _require(
                    lic is None or (type(lic) is str and 0 < len(lic) <= 240),
                    "DATASET_LICENSE_INVALID",
                )


def review_portfolio(snapshot: dict, *, now: datetime) -> dict:
    """Return 54-hour REVIEW REQUESTS; never training instructions or ADMIT."""
    validate_snapshot(snapshot, now=now)
    datasets = snapshot["public_assets"]["datasets"]
    models = snapshot["public_assets"]["models"]
    ds_unknown = sum(x["declared_license"] in (None, "other") for x in datasets)
    signals = Counter(row["artifact_signal"] for row in models)
    entries = []
    for model in models:
        reasons = [
            "MODEL_LICENSE_AND_BASE_UNVERIFIED",
            "DATASET_RIGHTS_NOT_INDEPENDENTLY_CLEARED",
            "HOLDOUT_AND_CONTAMINATION_UNVERIFIED",
            "TRAINER_OWNER_SIGNER_LEASE_UNVERIFIED",
            "GPU_CAPACITY_AND_SPEND_CAP_UNVERIFIED",
            "INDEPENDENT_EVALUATOR_REWARD_UNVERIFIED",
        ]
        if model["artifact_signal"] == "ADAPTER_FILES_LISTED":
            reasons.append("ADAPTER_BASE_IDENTITY_UNVERIFIED")
        elif model["artifact_signal"] == "NO_RECOGNIZED_WEIGHT_FILE":
            reasons.append("NO_RECOGNIZED_WEIGHT_FILE")
        entries.append(
            {
                "model": model["repo"],
                "model_sha": model["immutable_hub_sha"],
                "artifact_signal": model["artifact_signal"],
                "desired_training_hours": REQUESTED_HOURS,
                "duration_state": "REQUESTED_NOT_AUTHORIZED",
                "verdict": "HOLD",
                "reason_codes": reasons,
                "dispatched": False,
                "reward_state": "UNAVAILABLE_NO_INDEPENDENT_EVAL",
            }
        )
    _require(len(entries) == snapshot["counts"]["models"], "PORTFOLIO_COVERAGE")
    return {
        "schema": REPORT_SCHEMA,
        "observed_at_utc": now.astimezone(timezone.utc).isoformat(),
        "metadata_snapshot_observed_at_utc": snapshot["observed_at_utc"],
        "portfolio_model_count": len(models),
        "portfolio_dataset_count": len(datasets),
        "model_file_signals": dict(sorted(signals.items())),
        "datasets_without_specific_license_declaration": ds_unknown,
        "requested_hours_per_model": REQUESTED_HOURS,
        "training_jobs_started": 0,
        "provider_writes": 0,
        "model_promotions": 0,
        "reward_grants": 0,
        "all_models_admitted": False,
        "all_dataset_uses_approved": False,
        "status": "SOURCE_REVIEW_ONLY_ALL_HELD",
        "control_plane": (
            "One logical metadata registry and lazy artifact loading; "
            "existing source-owned bridge and Hatun/Khipu/Lean gates only."
        ),
        "remarks": [
            "A file suffix is neither a complete model nor verified trainable weights.",
            "Declared dataset licenses do not establish consent or compatible training use.",
            "No external foundation-model pretraining corpora or closed weights supplied.",
            "Rewards require signed independent evaluator results and no holdout leakage.",
            "This plan has no GPU executor, API credentials, lease, signing or network operations.",
        ],
        "review_entries": entries,
    }


def write_exclusive(path: Path, report: dict) -> str:
    _require(path.parent.is_dir() and not path.is_symlink(), "OUTPUT_DIR_INVALID")
    raw = (
        json.dumps(report, sort_keys=True, ensure_ascii=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("ascii")
    with path.open("xb") as fd:
        fd.write(raw)
        fd.flush()
        os.fsync(fd.fileno())
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-snapshot", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.public_snapshot)
    _require(source.is_file() and not source.is_symlink(), "INPUT_PATH_NOT_FILE")
    _require(source.stat().st_size <= MAX_INPUT_BYTES, "INPUT_FILE_OVER_BUDGET")
    raw = source.read_bytes()
    report = review_portfolio(
        parse_public_snapshot(raw), now=datetime.now(timezone.utc)
    )
    report["public_snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
    digest = write_exclusive(Path(args.out), report)
    print(
        json.dumps(
            {
                "verdict": report["status"],
                "models": report["portfolio_model_count"],
                "datasets": report["portfolio_dataset_count"],
                "models_dispatched": report["training_jobs_started"],
                "receipt_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
