#!/usr/bin/env python3
"""Shared verify-first contract helpers for the SZL GPU bridge.

This module deliberately has no network or ML-framework imports.  It is safe to
load before a job has been trusted.  The dispatcher and frontier runner both
verify the DSSE envelope against the pinned engine key before inspecting any
job field.
"""
from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import re
from datetime import datetime
from typing import Any, Iterable

V1_PAYLOAD_TYPE = "application/vnd.szl.gpu-bridge.jobspec.v1+json"
V2_PAYLOAD_TYPE = "application/vnd.szl.gpu-bridge.jobspec.v2+json"
_ALLOWED_TOP_LEVEL_V2 = {
    "jobId",
    "kind",
    "createdAt",
    "expiresAt",
    "base",
    "dataset",
    "recipe",
    "gates",
    "outputs",
    "eval",
    "notes",
}
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_JOB_ID = re.compile(r"^job-[0-9]{4}-[a-z0-9][a-z0-9-]{2,80}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]*$")
_ALLOWED_QUANTS = {"q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"}


class ContractError(ValueError):
    """A signed envelope or job contract failed a fail-closed check."""


def canonicalize(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pae(payload_type: str, payload: bytes) -> bytes:
    encoded_type = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (
        len(encoded_type),
        encoded_type,
        len(payload),
        payload,
    )


def derive_key_id(spki: bytes) -> str:
    return hashlib.sha256(spki).hexdigest()[:16]


def _decode_b64(value: Any, field: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} must be a non-empty base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ContractError(f"{field} is not valid base64: {exc}") from exc


def verify_envelope(
    envelope: dict[str, Any],
    engine_pin: dict[str, Any],
    *,
    allowed_payload_types: Iterable[str] = (V1_PAYLOAD_TYPE, V2_PAYLOAD_TYPE),
) -> tuple[dict[str, Any], bytes, str]:
    """Verify an envelope before any payload field is trusted.

    Returns ``(spec, exact_payload_bytes, payload_type)``.  The exact payload
    bytes are retained so receipts can pin what was actually signed rather than
    a language-specific re-serialization.
    """
    if not isinstance(envelope, dict):
        raise ContractError("envelope must be an object")
    payload_type = envelope.get("payloadType")
    if payload_type not in set(allowed_payload_types):
        raise ContractError(f"unsupported payloadType {payload_type!r}")

    spki = _decode_b64(envelope.get("publicKeySpkiBase64"), "publicKeySpkiBase64")
    pin_spki = _decode_b64(engine_pin.get("publicKeySpkiBase64"), "engine pin publicKeySpkiBase64")
    if spki != pin_spki:
        raise ContractError("envelope public key differs from the pinned engine key")
    derived = derive_key_id(spki)
    if derived != engine_pin.get("keyId"):
        raise ContractError(
            f"engine pin is mislabeled: derived keyId {derived} != {engine_pin.get('keyId')!r}"
        )

    signatures = envelope.get("signatures")
    if not isinstance(signatures, list) or len(signatures) != 1:
        raise ContractError("exactly one DSSE signature is required")
    signature = _decode_b64(signatures[0].get("sig"), "signatures[0].sig")
    payload = _decode_b64(envelope.get("payload"), "payload")

    try:
        from nacl.signing import VerifyKey

        if len(spki) != 44:
            raise ContractError(f"unexpected Ed25519 SPKI length {len(spki)}")
        VerifyKey(spki[-32:]).verify(pae(payload_type, payload), signature)
    except ContractError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ContractError(f"DSSE signature verification failed: {exc}") from exc

    try:
        spec = json.loads(payload)
    except Exception as exc:  # noqa: BLE001
        raise ContractError(f"signed payload is not JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise ContractError("signed payload must decode to an object")
    return spec, payload, payload_type


def _required_object(parent: dict[str, Any], key: str, required: set[str], allowed: set[str]) -> dict[str, Any]:
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


def _iso8601(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ContractError(f"{field} must be an ISO-8601 string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{field} is not valid ISO-8601: {value!r}") from exc


def _repo_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _REPO_ID.fullmatch(value):
        raise ContractError(f"{field} must be an owner/name Hub identifier")
    return value


def _revision(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ContractError(f"{field} must be an immutable 40-64 character lowercase hex revision")
    return value


def _positive_number(value: Any, field: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be numeric")
    number = float(value)
    if number < 0 if allow_zero else number <= 0:
        comparator = "non-negative" if allow_zero else "positive"
        raise ContractError(f"{field} must be {comparator}")
    return number


def validate_v2_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate the security-critical v2 contract without trusting JSON Schema alone."""
    if not isinstance(spec, dict):
        raise ContractError("v2 spec must be an object")
    missing = {
        "jobId", "kind", "createdAt", "expiresAt", "base", "dataset",
        "recipe", "gates", "outputs", "eval",
    } - spec.keys()
    extra = spec.keys() - _ALLOWED_TOP_LEVEL_V2
    if missing:
        raise ContractError(f"spec missing required fields: {sorted(missing)}")
    if extra:
        raise ContractError(f"spec contains unsupported fields: {sorted(extra)}")
    if spec.get("kind") != "unsloth-frontier-sft-v2":
        raise ContractError("kind must be 'unsloth-frontier-sft-v2'")
    if not isinstance(spec.get("jobId"), str) or not _JOB_ID.fullmatch(spec["jobId"]):
        raise ContractError("jobId does not match the v2 idempotency pattern")
    created = _iso8601(spec.get("createdAt"), "createdAt")
    expires = _iso8601(spec.get("expiresAt"), "expiresAt")
    if expires <= created:
        raise ContractError("expiresAt must be later than createdAt")

    base = _required_object(
        spec,
        "base",
        {"repoId", "revision", "licenseId"},
        {"repoId", "revision", "licenseId", "trustRemoteCode"},
    )
    _repo_id(base["repoId"], "base.repoId")
    _revision(base["revision"], "base.revision")
    if not isinstance(base["licenseId"], str) or not base["licenseId"].strip():
        raise ContractError("base.licenseId must be a non-empty SPDX-style identifier")
    if "trustRemoteCode" in base and not isinstance(base["trustRemoteCode"], bool):
        raise ContractError("base.trustRemoteCode must be boolean")

    dataset = _required_object(
        spec,
        "dataset",
        {"repoId", "revision", "file", "sha256", "provenance", "format", "licenseId"},
        {"repoId", "revision", "file", "sha256", "provenance", "format", "licenseId"},
    )
    _repo_id(dataset["repoId"], "dataset.repoId")
    _revision(dataset["revision"], "dataset.revision")
    if not isinstance(dataset["file"], str) or not _SAFE_RELATIVE.fullmatch(dataset["file"]):
        raise ContractError("dataset.file must be a safe relative Hub path")
    if dataset["file"].startswith("/") or ".." in pathlib.PurePosixPath(dataset["file"]).parts:
        raise ContractError("dataset.file must not be absolute or traverse parent directories")
    if not isinstance(dataset["sha256"], str) or not _SHA256.fullmatch(dataset["sha256"]):
        raise ContractError("dataset.sha256 must be lowercase sha256 hex")
    if dataset["format"] != "messages-jsonl":
        raise ContractError("dataset.format must be 'messages-jsonl'")
    if not isinstance(dataset["licenseId"], str) or not dataset["licenseId"].strip():
        raise ContractError("dataset.licenseId must be a non-empty SPDX-style identifier")
    if not isinstance(dataset["provenance"], str) or len(dataset["provenance"].strip()) < 20:
        raise ContractError("dataset.provenance must contain an auditable lineage statement")

    recipe_required = {
        "maxSeqLength", "loraR", "loraAlpha", "loraDropout", "targetModules",
        "batchSize", "gradAccum", "epochs", "learningRate", "optimizer",
        "gradientCheckpointing", "seed", "packing", "packingStrategy",
        "assistantOnlyLoss", "useRsLoRA", "warmupRatio", "weightDecay",
        "lrSchedulerType", "expectedChatTemplateSha256",
    }
    recipe = _required_object(spec, "recipe", recipe_required, recipe_required)
    if not isinstance(recipe["maxSeqLength"], int) or not 256 <= recipe["maxSeqLength"] <= 131072:
        raise ContractError("recipe.maxSeqLength must be an integer from 256 to 131072")
    if not isinstance(recipe["loraR"], int) or not 1 <= recipe["loraR"] <= 256:
        raise ContractError("recipe.loraR must be an integer from 1 to 256")
    if not isinstance(recipe["loraAlpha"], int) or recipe["loraAlpha"] <= 0:
        raise ContractError("recipe.loraAlpha must be a positive integer")
    dropout = _positive_number(recipe["loraDropout"], "recipe.loraDropout", allow_zero=True)
    if dropout > 0.5:
        raise ContractError("recipe.loraDropout must not exceed 0.5")
    modules = recipe["targetModules"]
    if not isinstance(modules, list) or not modules or not all(isinstance(v, str) and v for v in modules):
        raise ContractError("recipe.targetModules must be a non-empty string array")
    for field in ("batchSize", "gradAccum"):
        if not isinstance(recipe[field], int) or recipe[field] <= 0:
            raise ContractError(f"recipe.{field} must be a positive integer")
    if _positive_number(recipe["epochs"], "recipe.epochs") > 20:
        raise ContractError("recipe.epochs must not exceed 20")
    _positive_number(recipe["learningRate"], "recipe.learningRate")
    if recipe["optimizer"] not in {"adamw_8bit", "adamw_torch"}:
        raise ContractError("recipe.optimizer is unsupported")
    if recipe["gradientCheckpointing"] not in {"unsloth", "true", "false"}:
        raise ContractError("recipe.gradientCheckpointing is unsupported")
    if not isinstance(recipe["seed"], int):
        raise ContractError("recipe.seed must be an integer")
    for field in ("packing", "assistantOnlyLoss", "useRsLoRA"):
        if not isinstance(recipe[field], bool):
            raise ContractError(f"recipe.{field} must be boolean")
    if recipe["packingStrategy"] not in {"ffd", "wrapped"}:
        raise ContractError("recipe.packingStrategy must be ffd or wrapped")
    warmup = _positive_number(recipe["warmupRatio"], "recipe.warmupRatio", allow_zero=True)
    if warmup > 0.5:
        raise ContractError("recipe.warmupRatio must not exceed 0.5")
    _positive_number(recipe["weightDecay"], "recipe.weightDecay", allow_zero=True)
    if recipe["lrSchedulerType"] not in {"linear", "cosine", "constant_with_warmup"}:
        raise ContractError("recipe.lrSchedulerType is unsupported")
    expected_template = recipe["expectedChatTemplateSha256"]
    if not isinstance(expected_template, str) or not _SHA256.fullmatch(expected_template):
        raise ContractError("recipe.expectedChatTemplateSha256 must be lowercase sha256 hex")

    gates = _required_object(
        spec,
        "gates",
        {"minFreeVramGb", "minFreeDiskGb", "maxWallclockMinutes", "maxDatasetRows"},
        {"minFreeVramGb", "minFreeDiskGb", "maxWallclockMinutes", "maxDatasetRows", "abortOnNanLoss"},
    )
    _positive_number(gates["minFreeVramGb"], "gates.minFreeVramGb")
    _positive_number(gates["minFreeDiskGb"], "gates.minFreeDiskGb")
    if not isinstance(gates["maxWallclockMinutes"], int) or gates["maxWallclockMinutes"] <= 0:
        raise ContractError("gates.maxWallclockMinutes must be a positive integer")
    if not isinstance(gates["maxDatasetRows"], int) or gates["maxDatasetRows"] <= 0:
        raise ContractError("gates.maxDatasetRows must be a positive integer")
    if "abortOnNanLoss" in gates and not isinstance(gates["abortOnNanLoss"], bool):
        raise ContractError("gates.abortOnNanLoss must be boolean")

    outputs = _required_object(
        spec,
        "outputs",
        {"modelRepoId", "receiptsRepoId", "private", "exports"},
        {"modelRepoId", "receiptsRepoId", "private", "exports", "checkpointBucketId"},
    )
    _repo_id(outputs["modelRepoId"], "outputs.modelRepoId")
    _repo_id(outputs["receiptsRepoId"], "outputs.receiptsRepoId")
    if not isinstance(outputs["private"], bool):
        raise ContractError("outputs.private must be boolean")
    if "checkpointBucketId" in outputs:
        _repo_id(outputs["checkpointBucketId"], "outputs.checkpointBucketId")
    exports = outputs["exports"]
    if not isinstance(exports, dict) or set(exports) != {"adapter", "merged16bit", "ggufQuantizations", "requireReloadSmoke"}:
        raise ContractError("outputs.exports must contain exactly adapter, merged16bit, ggufQuantizations, requireReloadSmoke")
    if exports["adapter"] is not True:
        raise ContractError("outputs.exports.adapter must be true")
    if not isinstance(exports["merged16bit"], bool) or not isinstance(exports["requireReloadSmoke"], bool):
        raise ContractError("merged16bit and requireReloadSmoke must be boolean")
    quants = exports["ggufQuantizations"]
    if not isinstance(quants, list) or len(quants) != len(set(quants)):
        raise ContractError("ggufQuantizations must be a unique array")
    if any(quant not in _ALLOWED_QUANTS for quant in quants):
        raise ContractError(f"unsupported GGUF quantization; allowed={sorted(_ALLOWED_QUANTS)}")
    if (exports["merged16bit"] or quants) and not exports["requireReloadSmoke"]:
        raise ContractError("merged or GGUF exports require reload smoke verification")

    evaluation = _required_object(
        spec,
        "eval",
        {
            "suite", "heldOutFraction", "seed", "maxGenerations", "maxNewTokens",
            "requiredJsonKeys", "convictionCeiling", "maxDegenerateRate",
            "minJsonValidRate", "minRequiredKeysRate", "minCeilingRespectRate",
        },
        {
            "suite", "heldOutFraction", "seed", "maxGenerations", "maxNewTokens",
            "requiredJsonKeys", "convictionCeiling", "maxDegenerateRate",
            "minJsonValidRate", "minRequiredKeysRate", "minCeilingRespectRate",
        },
    )
    if evaluation["suite"] != "frontier-heldout-v2":
        raise ContractError("eval.suite must be frontier-heldout-v2")
    heldout = _positive_number(evaluation["heldOutFraction"], "eval.heldOutFraction")
    if heldout > 0.5:
        raise ContractError("eval.heldOutFraction must not exceed 0.5")
    if not isinstance(evaluation["seed"], int):
        raise ContractError("eval.seed must be an integer")
    for field in ("maxGenerations", "maxNewTokens"):
        if not isinstance(evaluation[field], int) or evaluation[field] <= 0:
            raise ContractError(f"eval.{field} must be a positive integer")
    keys = evaluation["requiredJsonKeys"]
    if (
        not isinstance(keys, list)
        or not keys
        or len(keys) != len(set(keys))
        or not all(isinstance(v, str) and v for v in keys)
    ):
        raise ContractError("eval.requiredJsonKeys must be a non-empty unique string array")
    ceiling = _positive_number(evaluation["convictionCeiling"], "eval.convictionCeiling", allow_zero=True)
    if ceiling > 1:
        raise ContractError("eval.convictionCeiling must be <= 1")
    degenerate = _positive_number(evaluation["maxDegenerateRate"], "eval.maxDegenerateRate", allow_zero=True)
    if degenerate > 1:
        raise ContractError("eval.maxDegenerateRate must be <= 1")
    for field in ("minJsonValidRate", "minRequiredKeysRate", "minCeilingRespectRate"):
        rate = _positive_number(evaluation[field], f"eval.{field}", allow_zero=True)
        if rate > 1:
            raise ContractError(f"eval.{field} must be <= 1")

    return spec


def load_pin(path: str | pathlib.Path) -> dict[str, Any]:
    value = json.loads(pathlib.Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ContractError("engine pin must be an object")
    return value
