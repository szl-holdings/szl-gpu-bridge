#!/usr/bin/env python3
"""Verify-first contract for a single governed SZL-Nemo v3 GPU attempt.

This module has no network or ML-framework imports.  The dispatcher and runner
load it only after the DSSE envelope has verified against the pinned engine key.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from datetime import datetime
from typing import Any

from frontier_contract import ContractError

NEMO_V3_PAYLOAD_TYPE = "application/vnd.szl.gpu-bridge.nemo-v3.jobspec.v1+json"
NEMO_V3_KIND = "szl-nemo-governed-v3"
_SHA = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_JOB = re.compile(r"^job-[0-9]{4}-nemo-v3-[a-z0-9][a-z0-9-]{2,64}$")
_SAFE_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]*$")
_HOLDOUT_NAMES = ("original-v2", "shadow-v2", "challenge-v3")
_ALLOWED_TOP = {
    "jobId", "kind", "createdAt", "expiresAt", "source", "base", "dataset",
    "recipe", "gates", "outputs", "evaluation", "notes",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def record_ids_sha256(ids: list[str]) -> str:
    return sha256_bytes(("\n".join(ids) + "\n").encode("utf-8"))


def _object(parent: dict[str, Any], key: str, required: set[str], allowed: set[str]) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ContractError(f"{key} must be an object")
    missing = required - value.keys()
    extra = value.keys() - allowed
    if missing:
        raise ContractError(f"{key} missing required fields: {sorted(missing)}")
    if extra:
        raise ContractError(f"{key} contains unsupported fields: {sorted(extra)}")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ContractError(f"{field} must be ISO-8601")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{field} is not valid ISO-8601") from exc


def _repo(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _REPO.fullmatch(value):
        raise ContractError(f"{field} must be owner/name")
    return value


def _revision(value: Any, field: str, *, exact40: bool = False) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value) or (exact40 and len(value) != 40):
        suffix = "exactly 40" if exact40 else "40-64"
        raise ContractError(f"{field} must be {suffix} lowercase hex")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ContractError(f"{field} must be lowercase sha256 hex")
    return value


def _path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_PATH.fullmatch(value):
        raise ContractError(f"{field} must be a safe relative path")
    candidate = pathlib.PurePosixPath(value)
    if value.startswith("/") or ".." in candidate.parts:
        raise ContractError(f"{field} must not be absolute or traverse parents")
    return value


def _pinned_file(value: Any, field: str, *, require_records: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{field} must be an object")
    required = {"path", "sha256", "bytes"}
    if require_records:
        required |= {"name", "recordIds", "recordIdsSha256"}
    missing = required - value.keys()
    if missing:
        raise ContractError(f"{field} missing fields: {sorted(missing)}")
    _path(value.get("path"), f"{field}.path")
    _sha256(value.get("sha256"), f"{field}.sha256")
    if not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
        raise ContractError(f"{field}.bytes must be a positive integer")
    if require_records:
        ids = value.get("recordIds")
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise ContractError(f"{field}.recordIds must be a non-empty unique array")
        if not all(isinstance(item, str) and item for item in ids):
            raise ContractError(f"{field}.recordIds contains an invalid identifier")
        _sha256(value.get("recordIdsSha256"), f"{field}.recordIdsSha256")
        if record_ids_sha256(ids) != value["recordIdsSha256"]:
            raise ContractError(f"{field}.recordIdsSha256 does not bind recordIds")
    return value


def validate_nemo_v3_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ContractError("Nemo v3 spec must be an object")
    required = _ALLOWED_TOP - {"notes"}
    missing = required - spec.keys()
    extra = spec.keys() - _ALLOWED_TOP
    if missing:
        raise ContractError(f"spec missing required fields: {sorted(missing)}")
    if extra:
        raise ContractError(f"spec contains unsupported fields: {sorted(extra)}")
    if spec.get("kind") != NEMO_V3_KIND:
        raise ContractError(f"kind must be {NEMO_V3_KIND!r}")
    if not isinstance(spec.get("jobId"), str) or not _JOB.fullmatch(spec["jobId"]):
        raise ContractError("jobId does not match the Nemo v3 idempotency pattern")
    created = _timestamp(spec.get("createdAt"), "createdAt")
    expires = _timestamp(spec.get("expiresAt"), "expiresAt")
    if expires <= created:
        raise ContractError("expiresAt must be later than createdAt")

    source = _object(spec, "source", {"repoId", "revision", "licenseId"}, {"repoId", "revision", "licenseId"})
    if source["repoId"] != "szl-holdings/a11oy" or source["licenseId"].lower() != "apache-2.0":
        raise ContractError("source must be Apache-2.0 szl-holdings/a11oy")
    _revision(source["revision"], "source.revision", exact40=True)

    base = _object(
        spec,
        "base",
        {"repoId", "revision", "licenseId", "licenseAcknowledgement", "trustRemoteCode"},
        {"repoId", "revision", "licenseId", "licenseAcknowledgement", "trustRemoteCode"},
    )
    _repo(base["repoId"], "base.repoId")
    _revision(base["revision"], "base.revision")
    if not isinstance(base["licenseId"], str) or not base["licenseId"].strip():
        raise ContractError("base.licenseId is required")
    if not isinstance(base["licenseAcknowledgement"], str) or len(base["licenseAcknowledgement"].strip()) < 20:
        raise ContractError("base.licenseAcknowledgement must be explicit")
    if not isinstance(base["trustRemoteCode"], bool):
        raise ContractError("base.trustRemoteCode must be boolean")

    dataset = _object(
        spec,
        "dataset",
        {"provenance", "rightsBasis", "train", "holdouts", "preregistration"},
        {"provenance", "rightsBasis", "train", "holdouts", "preregistration"},
    )
    if dataset["rightsBasis"] != "PROJECT_AUTHORED_SCENARIOS":
        raise ContractError("dataset rights basis is not admitted")
    if not isinstance(dataset["provenance"], str) or len(dataset["provenance"].strip()) < 40:
        raise ContractError("dataset provenance must be auditable")
    _pinned_file(dataset["train"], "dataset.train")
    _pinned_file(dataset["preregistration"], "dataset.preregistration")
    holdouts = dataset["holdouts"]
    if not isinstance(holdouts, list) or len(holdouts) != 3:
        raise ContractError("dataset.holdouts must contain exactly three frozen suites")
    names: list[str] = []
    ids: list[str] = []
    for index, value in enumerate(holdouts):
        item = _pinned_file(value, f"dataset.holdouts[{index}]", require_records=True)
        if item.get("name") not in _HOLDOUT_NAMES:
            raise ContractError("holdout suite name is not allowed")
        names.append(item["name"])
        ids.extend(item["recordIds"])
    if tuple(names) != _HOLDOUT_NAMES:
        raise ContractError("holdout suite order must be original-v2, shadow-v2, challenge-v3")
    if len(ids) != len(set(ids)):
        raise ContractError("record identifiers overlap across holdout suites")

    recipe_required = {
        "maxSeqLength", "loraR", "loraAlpha", "loraDropout", "targetModules", "batchSize",
        "gradAccum", "epochs", "learningRate", "optimizer", "gradientCheckpointing", "seed",
        "warmupRatio", "weightDecay", "lrSchedulerType",
    }
    recipe = _object(spec, "recipe", recipe_required, recipe_required)
    integer_ranges = {
        "maxSeqLength": (256, 4096), "loraR": (1, 64), "loraAlpha": (1, 256),
        "batchSize": (1, 1), "gradAccum": (1, 64),
    }
    for field, (minimum, maximum) in integer_ranges.items():
        value = recipe[field]
        if not isinstance(value, int) or not minimum <= value <= maximum:
            raise ContractError(f"recipe.{field} is outside the fixed range")
    if not isinstance(recipe["loraDropout"], (int, float)) or not 0 <= recipe["loraDropout"] <= 0.2:
        raise ContractError("recipe.loraDropout is outside the fixed range")
    if not isinstance(recipe["targetModules"], list) or not recipe["targetModules"] or len(recipe["targetModules"]) != len(set(recipe["targetModules"])):
        raise ContractError("recipe.targetModules must be a non-empty unique array")
    if not all(isinstance(item, str) and item for item in recipe["targetModules"]):
        raise ContractError("recipe.targetModules contains an invalid item")
    if not isinstance(recipe["epochs"], (int, float)) or not 0 < recipe["epochs"] <= 8:
        raise ContractError("recipe.epochs is outside the fixed range")
    if not isinstance(recipe["learningRate"], (int, float)) or not 0 < recipe["learningRate"] <= 0.001:
        raise ContractError("recipe.learningRate is outside the fixed range")
    if recipe["optimizer"] not in {"adamw_8bit", "paged_adamw_8bit"}:
        raise ContractError("recipe.optimizer is unsupported")
    if recipe["gradientCheckpointing"] not in {"unsloth", "true"}:
        raise ContractError("recipe.gradientCheckpointing is unsupported")
    if not isinstance(recipe["seed"], int):
        raise ContractError("recipe.seed must be an integer")
    for field, maximum in (("warmupRatio", 0.2), ("weightDecay", 0.2)):
        value = recipe[field]
        if not isinstance(value, (int, float)) or not 0 <= value <= maximum:
            raise ContractError(f"recipe.{field} is outside the fixed range")
    if recipe["lrSchedulerType"] not in {"linear", "cosine"}:
        raise ContractError("recipe.lrSchedulerType is unsupported")

    gates_required = {
        "minFreeVramGb", "minFreeDiskGb", "maxWallclockMinutes", "maxDatasetRows",
        "maxTemperatureC", "maxUtilizationPct",
    }
    gates = _object(spec, "gates", gates_required, gates_required)
    if not isinstance(gates["minFreeVramGb"], (int, float)) or gates["minFreeVramGb"] < 5:
        raise ContractError("gates.minFreeVramGb cannot be weakened below 5 GB")
    if not isinstance(gates["minFreeDiskGb"], (int, float)) or gates["minFreeDiskGb"] < 20:
        raise ContractError("gates.minFreeDiskGb cannot be weakened below 20 GB")
    if not isinstance(gates["maxWallclockMinutes"], int) or not 10 <= gates["maxWallclockMinutes"] <= 360:
        raise ContractError("gates.maxWallclockMinutes is outside the fixed range")
    if not isinstance(gates["maxDatasetRows"], int) or not 24 <= gates["maxDatasetRows"] <= 5000:
        raise ContractError("gates.maxDatasetRows is outside the fixed range")
    if not isinstance(gates["maxTemperatureC"], int) or not 40 <= gates["maxTemperatureC"] <= 80:
        raise ContractError("gates.maxTemperatureC is outside the fixed range")
    if not isinstance(gates["maxUtilizationPct"], int) or not 0 <= gates["maxUtilizationPct"] <= 30:
        raise ContractError("gates.maxUtilizationPct is outside the fixed range")

    outputs = _object(
        spec,
        "outputs",
        {"candidateId", "receiptsRepoId", "private", "publishCandidate"},
        {"candidateId", "receiptsRepoId", "private", "publishCandidate"},
    )
    if not isinstance(outputs["candidateId"], str) or not outputs["candidateId"].startswith("SZL-Nemo-v3-"):
        raise ContractError("outputs.candidateId must identify SZL-Nemo-v3")
    _repo(outputs["receiptsRepoId"], "outputs.receiptsRepoId")
    if outputs["private"] is not True or outputs["publishCandidate"] is not False:
        raise ContractError("Nemo v3 candidate publication must remain disabled")

    evaluation = _object(
        spec,
        "evaluation",
        {"requiredPassRate", "maxDegenerateRate", "maxNewTokens", "requireExactRecordOrder"},
        {"requiredPassRate", "maxDegenerateRate", "maxNewTokens", "requireExactRecordOrder"},
    )
    if evaluation["requiredPassRate"] != 1.0 or evaluation["maxDegenerateRate"] != 0.0:
        raise ContractError("Nemo v3 requires all holdouts to pass with no degeneration")
    if not isinstance(evaluation["maxNewTokens"], int) or not 32 <= evaluation["maxNewTokens"] <= 512:
        raise ContractError("evaluation.maxNewTokens is outside the fixed range")
    if evaluation["requireExactRecordOrder"] is not True:
        raise ContractError("exact holdout record order is mandatory")
    return spec
