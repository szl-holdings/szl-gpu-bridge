"""Offline candidate recipe integrity for the SZL model portfolio.

This module is deliberately not a training adapter, data loader, scheduler,
rights authority, signature verifier, evaluator or training dispatch mechanism.
A syntactically consistent review candidate ALWAYS remains on HOLD.
"""

from __future__ import annotations

from datetime import datetime
import re

from tools import estate_54h_portfolio as portfolio

RECIPE_SCHEMA = "szl.portfolio-candidate-review/v1"
SHA64 = re.compile(r"[a-f0-9]{64}\Z")
MAX_REQUESTED_WALL_MINUTES = 54 * 60
MAX_STEPS_LIMIT = 10_000_000


class CandidateHold(ValueError):
    """Fail-closed source-level format or source binding conflict."""


def require(ok: bool, code: str) -> None:
    if not ok:
        raise CandidateHold(code)


def exact_sha(value: object, *, field: str) -> None:
    require(type(value) is str and portfolio.SHA40.fullmatch(value) is not None, field)


def digest_set(items: object, label: str) -> set[str]:
    require(
        type(items) is list and 0 < len(items) <= 2000, label + "_EMPTY_OR_OVER_BUDGET"
    )
    seen: set[str] = set()
    for item in items:
        require(
            type(item) is str and SHA64.fullmatch(item) is not None,
            label + "_DIGEST_INVALID",
        )
        require(item not in seen, label + "_DUPLICATE_DIGEST")
        seen.add(item)
    return seen


def review_candidate(recipe: dict, snapshot: dict, *, now: datetime) -> dict:
    """Validate declared references, never infer data use rights or trainability."""
    try:
        portfolio.validate_snapshot(snapshot, now=now)
    except portfolio.Hold as exc:
        raise CandidateHold("SOURCE_SNAPSHOT_INVALID") from exc
    require(
        type(recipe) is dict and recipe.get("schema") == RECIPE_SCHEMA, "RECIPE_SCHEMA"
    )
    require(
        set(recipe)
        == {
            "schema",
            "adapter",
            "adapter_sha",
            "base_model",
            "base_sha",
            "adapter_config_base",
            "datasets",
            "train_example_sha256",
            "heldout_example_sha256",
            "max_steps",
            "max_wall_minutes",
            "evaluation_recipe",
            "training_dispatch_enabled",
        },
        "RECIPE_FIELDS",
    )
    model_idx = {row["repo"]: row for row in snapshot["public_assets"]["models"]}
    dataset_idx = {row["repo"]: row for row in snapshot["public_assets"]["datasets"]}
    adapter, base = recipe["adapter"], recipe["base_model"]
    require(type(adapter) is str and adapter in model_idx, "ADAPTER_ID_UNOBSERVED")
    require(type(base) is str and base in model_idx, "BASE_ID_UNOBSERVED")
    require(adapter != base, "ADAPTER_BASE_SELF_REFERENCE")
    exact_sha(recipe["adapter_sha"], field="ADAPTER_SHA_INVALID")
    exact_sha(recipe["base_sha"], field="BASE_SHA_INVALID")
    require(
        model_idx[adapter]["immutable_hub_sha"] == recipe["adapter_sha"],
        "ADAPTER_REVISION_DRIFT",
    )
    require(
        model_idx[base]["immutable_hub_sha"] == recipe["base_sha"],
        "BASE_REVISION_DRIFT",
    )
    require(
        model_idx[adapter]["artifact_signal"] == "ADAPTER_FILES_LISTED",
        "ADAPTER_WEIGHT_SIGNAL_UNQUALIFIED",
    )
    require(
        model_idx[base]["artifact_signal"] == "WEIGHT_FILES_LISTED",
        "BASE_WEIGHT_SIGNAL_UNQUALIFIED",
    )
    require(
        type(recipe["adapter_config_base"]) is str
        and recipe["adapter_config_base"] == base,
        "ADAPTER_DECLARED_BASE_MISMATCH",
    )

    ds = recipe["datasets"]
    require(type(ds) is list and 0 < len(ds) <= 30, "DATASET_COUNT_BUDGET")
    ids: set[str] = set()
    for row in ds:
        require(
            type(row) is dict and set(row) == {"id", "sha", "role"},
            "DATASET_BINDING_SCHEMA",
        )
        rid = row.get("id")
        require(type(rid) is str and rid in dataset_idx, "DATASET_ID_UNOBSERVED")
        require(rid not in ids, "DUPLICATE_DATASET_ID")
        ids.add(rid)
        exact_sha(row.get("sha"), field="DATASET_SHA_INVALID")
        require(
            row["sha"] == dataset_idx[rid]["immutable_hub_sha"],
            "DATASET_REVISION_DRIFT",
        )
        require(
            type(row.get("role")) is str and row["role"] in ("train", "evaluation"),
            "DATASET_ROLE_INVALID",
        )
    require(
        any(row["role"] == "train" for row in ds)
        and any(row["role"] == "evaluation" for row in ds),
        "TRAIN_AND_EVAL_DATASETS_REQUIRED",
    )
    train = digest_set(recipe["train_example_sha256"], "TRAIN")
    heldout = digest_set(recipe["heldout_example_sha256"], "HELDOUT")
    require(not train.intersection(heldout), "DECLARED_TRAIN_HELDOUT_COLLISION")
    require(
        type(recipe["max_steps"]) is int and 0 < recipe["max_steps"] <= MAX_STEPS_LIMIT,
        "STEP_BOUND_INVALID",
    )
    require(
        type(recipe["max_wall_minutes"]) is int
        and 0 < recipe["max_wall_minutes"] <= MAX_REQUESTED_WALL_MINUTES,
        "WALLTIME_OVER_54H_OR_UNBOUNDED",
    )
    require(
        type(recipe["evaluation_recipe"]) is str
        and re.fullmatch(r"[a-z][a-z0-9_-]{2,79}", recipe["evaluation_recipe"])
        is not None,
        "EVALUATOR_BINDING_UNVERIFIED",
    )
    require(
        recipe["training_dispatch_enabled"] is False,
        "EXECUTION_FLAG_DENIED",
    )
    # A syntactically valid declaration is still NOT an independent check of
    # adapter bytes, base architecture, licenses, contamination or signatures.
    return {
        "schema": "szl.portfolio-candidate-source-hold/v1",
        "adapter": adapter,
        "adapter_sha": recipe["adapter_sha"],
        "base_model": base,
        "base_sha": recipe["base_sha"],
        "dataset_count": len(ds),
        "declared_train_digest_count": len(train),
        "declared_heldout_digest_count": len(heldout),
        "requested_max_steps": recipe["max_steps"],
        "requested_wall_minutes": recipe["max_wall_minutes"],
        "verdict": "HOLD",
        "release_approved": False,
        "training_dispatched": False,
        "reward_grants": 0,
        "unverified": [
            "ADAPTER_BASE_ARCHITECTURE_COMPATIBILITY_UNVERIFIED",
            "ADAPTER_AND_BASE_FILE_BYTES_UNVERIFIED",
            "RIGHTS_CONSENT_AND_LICENSE_COMPATIBILITY",
            "SPLIT_BYTES_NEAR_DUPLICATE_AND_CONTAMINATION",
            "INDEPENDENT_EVALUATOR_AND_REWARD",
            "OWNER_SIGNER_LEASE_GPU_CAPACITY_AND_BUDGET",
        ],
    }
