#!/usr/bin/env python3
"""Execute one signed, preregistered SZL-Nemo v3 GPU attempt.

The job trains only the pinned project-authored training file.  The original v2,
shadow v2, and preregistered v3 challenge suites are downloaded separately,
verified byte-for-byte, and never passed to the trainer.  The result is either a
signed terminal BLOCKED receipt or a signed
QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW receipt.  Candidate upload, publication,
deployment, and promotion are forbidden in this runner.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

from frontier_contract import load_engine_pin_for_envelope, verify_envelope
from frontier_job import (
    ROOT,
    _build_sft_config,
    _build_sft_trainer,
    _gradient_checkpointing,
    _verify_hub_license,
    blocked,
    deliver_receipt,
    now_iso,
)
from frontier_runtime import (
    artifact_manifest,
    is_degenerate_text,
    manifest_digest,
    normalize_conversation,
    sha256_file,
    stack_fingerprint,
)
from nemo_v3_contract import (
    NEMO_V3_PAYLOAD_TYPE,
    expected_engine_key_id,
    record_ids_sha256,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

ISOLATION_MARKER = "credentialless-networkless-container"
SENSITIVE_ENVIRONMENT_NAMES = {
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "ACTIONS_RUNTIME_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
}
PINNED_OFFLINE_TOKENIZERS: dict[tuple[str, str], dict[str, tuple[int, str]]] = {
    (
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
    ): {
        "chat_template.jinja": (
            10504,
            "ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862",
        ),
        "special_tokens_map.json": (
            420,
            "e3a4f63da745f02317a45e00e6476c17fc66ac41faf14bb1b0be1f3211b0ca53",
        ),
        "tokenizer.json": (
            17077484,
            "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
        ),
        "tokenizer_config.json": (
            188034,
            "48de4056b0b17de26e03232fdc1f55b70595c9354ceb2ed061f724f45620aa41",
        ),
    },
}


def _raw_url(repo_id: str, revision: str, path: str) -> str:
    owner, name = repo_id.split("/", 1)
    parts = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    return f"https://raw.githubusercontent.com/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(name, safe='')}/{revision}/{parts}"


def _download_pinned(
    spec: dict[str, Any], descriptor: dict[str, Any], target: pathlib.Path
) -> pathlib.Path:
    input_cache = os.environ.get("SZL_INPUT_CACHE", "").strip()
    if input_cache:
        cached = pathlib.Path(input_cache) / descriptor["path"]
        if not cached.is_file():
            raise RuntimeError(f"pinned offline input is absent: {descriptor['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        digest = sha256_file(target)
        total = target.stat().st_size
        if total != descriptor["bytes"] or digest != descriptor["sha256"]:
            raise RuntimeError(
                f"pinned offline input mismatch for {descriptor['path']}: "
                f"bytes={total}, sha256={digest}"
            )
        return target

    request = urllib.request.Request(
        _raw_url(
            spec["source"]["repoId"], spec["source"]["revision"], descriptor["path"]
        ),
        headers={
            "User-Agent": "szl-gpu-bridge-nemo-v3/1.0",
            "Cache-Control": "no-cache",
        },
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with (
        urllib.request.urlopen(request, timeout=90) as response,
        target.open("xb") as output,
    ):
        if response.status != 200:
            raise RuntimeError(f"source download returned HTTP {response.status}")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            output.write(chunk)
    if total != descriptor["bytes"] or digest.hexdigest() != descriptor["sha256"]:
        raise RuntimeError(
            f"pinned source mismatch for {descriptor['path']}: bytes={total}, sha256={digest.hexdigest()}"
        )
    return target


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return rows


def _validate_train(rows: list[dict[str, Any]], maximum: int) -> None:
    if not 24 <= len(rows) <= maximum:
        raise ValueError(f"training row count {len(rows)} outside admitted range")
    ids: set[str] = set()
    for index, row in enumerate(rows):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in ids:
            raise ValueError(f"training row {index} has invalid or duplicate record_id")
        ids.add(record_id)
        if (
            row.get("rights_basis") != "PROJECT_AUTHORED_SCENARIOS"
            or row.get("split") != "TRAIN"
        ):
            raise ValueError(
                f"training row {record_id} is not rights-admitted TRAIN data"
            )
        messages = row.get("messages")
        normalize_conversation(messages, require_final_assistant=True)


def _validate_holdout(rows: list[dict[str, Any]], descriptor: dict[str, Any]) -> None:
    ids: list[str] = []
    for index, row in enumerate(rows):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"holdout row {index} has no record_id")
        ids.append(record_id)
        if row.get("rights_basis") != "PROJECT_AUTHORED_SCENARIOS":
            raise ValueError(f"holdout row {record_id} has no admitted rights basis")
        if row.get("split") not in {"EVAL", "SHADOW_EVAL", "CHALLENGE_EVAL"}:
            raise ValueError(f"holdout row {record_id} has an invalid split")
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or messages[-1].get("role") != "user"
        ):
            raise ValueError(
                f"holdout row {record_id} must end with an unanswered user message"
            )
        expected = row.get("expected")
        if not isinstance(expected, dict):
            raise ValueError(f"holdout row {record_id} has no rubric")
        for key in ("required_terms", "forbidden_terms"):
            terms = expected.get(key)
            if not isinstance(terms, list) or not all(
                isinstance(term, str) and term for term in terms
            ):
                raise ValueError(f"holdout row {record_id} has an invalid {key}")
    if (
        ids != descriptor["recordIds"]
        or record_ids_sha256(ids) != descriptor["recordIdsSha256"]
    ):
        raise ValueError(f"holdout identity/order mismatch for {descriptor['name']}")


def _gpu_state() -> dict[str, Any]:
    output = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=20,
        )
        .strip()
        .splitlines()[0]
    )
    name, free_mib, utilization, temperature = [
        item.strip() for item in output.split(",", 3)
    ]
    return {
        "name": name,
        "free_vram_gb": int(free_mib) / 1024.0,
        "utilization_pct": int(utilization),
        "temperature_c": int(temperature),
        "observed_at": now_iso(),
    }


def _admit_gpu(spec: dict[str, Any], stage: str) -> dict[str, Any]:
    state = _gpu_state()
    gates = spec["gates"]
    if state["free_vram_gb"] < gates["minFreeVramGb"]:
        blocked(
            spec,
            f"gate:{stage}:vram",
            f"free VRAM {state['free_vram_gb']:.2f} GB below {gates['minFreeVramGb']} GB",
            extra=state,
        )
    if state["utilization_pct"] > gates["maxUtilizationPct"]:
        blocked(
            spec,
            f"gate:{stage}:utilization",
            f"GPU utilization {state['utilization_pct']}% exceeds {gates['maxUtilizationPct']}%",
            extra=state,
        )
    if state["temperature_c"] > gates["maxTemperatureC"]:
        blocked(
            spec,
            f"gate:{stage}:temperature",
            f"GPU temperature {state['temperature_c']} C exceeds {gates['maxTemperatureC']} C",
            extra=state,
        )
    return state


def _generate(
    model: Any, tokenizer: Any, messages: list[dict[str, Any]], max_new_tokens: int
) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    start = encoded["input_ids"].shape[1]
    return tokenizer.decode(output[0][start:], skip_special_tokens=True)


def _evaluate_suite(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_new_tokens: int
) -> tuple[list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    passes = 0
    for row in rows:
        text = _generate(model, tokenizer, row["messages"], max_new_tokens)
        folded = text.casefold()
        required = row["expected"]["required_terms"]
        forbidden = row["expected"]["forbidden_terms"]
        missing = [term for term in required if term.casefold() not in folded]
        present_forbidden = [term for term in forbidden if term.casefold() in folded]
        degenerate = is_degenerate_text(text)
        passed = not missing and not present_forbidden and not degenerate
        passes += int(passed)
        results.append(
            {
                "record_id": row["record_id"],
                "state": "PASS" if passed else "FAIL",
                "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "response_bytes": len(text.encode("utf-8")),
                "missing_required_terms": missing,
                "present_forbidden_terms": present_forbidden,
                "degenerate": degenerate,
            }
        )
    return results, passes


def _result_receipt(
    spec: dict[str, Any], exact_payload: bytes, *, state: str, evidence: dict[str, Any]
) -> dict[str, Any]:
    return {
        "kind": "szl-nemo-v3-governed-training",
        "v": 1,
        "jobId": spec["jobId"],
        "candidateId": spec["outputs"]["candidateId"],
        "state": state,
        "at": now_iso(),
        "source": spec["source"],
        "base": {key: spec["base"][key] for key in ("repoId", "revision", "licenseId")},
        "signed_job_payload_sha256": hashlib.sha256(exact_payload).hexdigest(),
        "training_rights_basis": spec["dataset"]["rightsBasis"],
        "evaluation": evidence,
        "effects": {
            "candidate_uploaded": False,
            "published": False,
            "deployed": False,
            "promoted": False,
        },
        "decision": (
            "SEPARATE_PROMOTION_REVIEW_REQUIRED"
            if state == "QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW"
            else "TERMINAL_FAILURE_NO_AUTOMATIC_RETRY"
        ),
    }


def _require_remote_code_isolation(spec: dict[str, Any]) -> None:
    if not spec["base"]["trustRemoteCode"]:
        return
    if os.environ.get("SZL_EXECUTION_ISOLATION") != ISOLATION_MARKER:
        raise RuntimeError("trusted remote code requires the isolated container lane")
    if os.environ.get("SZL_RECEIPT_TRANSPORT") != "local-unsigned-outbox":
        raise RuntimeError("isolated remote code must emit an unsigned receipt intent")
    for name in SENSITIVE_ENVIRONMENT_NAMES:
        if os.environ.get(name):
            raise RuntimeError(
                f"isolated remote code received sensitive environment: {name}"
            )
    for name in (
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"isolated remote code requires {name}=1")
    if (ROOT / "keys" / "laptop_key.pem").exists():
        raise RuntimeError("isolated remote code can access the laptop signing key")


def _verified_offline_tokenizer_snapshot(spec: dict[str, Any]) -> pathlib.Path:
    """Resolve and verify the exact tokenizer bytes admitted for offline loading."""
    base = spec["base"]
    identity = (base["repoId"], base["revision"])
    artifact_manifest = PINNED_OFFLINE_TOKENIZERS.get(identity)
    if artifact_manifest is None:
        raise RuntimeError(
            "base model/revision has no admitted offline tokenizer manifest"
        )

    cache_value = os.environ.get("HF_HUB_CACHE", "").strip()
    if not cache_value:
        raise RuntimeError("HF_HUB_CACHE is required for pinned offline tokenizer")
    cache_root = pathlib.Path(cache_value)
    if not cache_root.is_absolute():
        raise RuntimeError("HF_HUB_CACHE must be an absolute path")
    try:
        resolved_cache = cache_root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("HF_HUB_CACHE does not resolve") from exc
    if not resolved_cache.is_dir():
        raise RuntimeError("HF_HUB_CACHE is not a directory")

    owner, name = base["repoId"].split("/", 1)
    snapshot = (
        resolved_cache / f"models--{owner}--{name}" / "snapshots" / base["revision"]
    )
    try:
        resolved_snapshot = snapshot.resolve(strict=True)
        resolved_snapshot.relative_to(resolved_cache)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "pinned tokenizer snapshot does not resolve inside cache"
        ) from exc
    if not resolved_snapshot.is_dir() or snapshot.is_symlink():
        raise RuntimeError("pinned tokenizer snapshot is not an immutable directory")

    for relative_path, (expected_bytes, expected_sha256) in artifact_manifest.items():
        candidate = resolved_snapshot / relative_path
        try:
            resolved_candidate = candidate.resolve(strict=True)
            resolved_candidate.relative_to(resolved_snapshot)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"pinned tokenizer artifact is absent or escapes snapshot: {relative_path}"
            ) from exc
        if not resolved_candidate.is_file() or candidate.is_symlink():
            raise RuntimeError(
                f"pinned tokenizer artifact is not an immutable file: {relative_path}"
            )
        observed_bytes = resolved_candidate.stat().st_size
        observed_sha256 = sha256_file(resolved_candidate)
        if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"pinned tokenizer artifact mismatch for {relative_path}: "
                f"bytes={observed_bytes}, sha256={observed_sha256}"
            )
    return resolved_snapshot


def _require_loaded_tokenizer(
    tokenizer: Any,
    tokenizer_base_type: type[Any],
    expected_snapshot: pathlib.Path,
) -> str:
    if tokenizer is None:
        raise RuntimeError("base model loader returned no tokenizer")
    if not isinstance(tokenizer, tokenizer_base_type):
        raise RuntimeError(
            f"base model loader returned unsupported tokenizer type: {type(tokenizer)!r}"
        )
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if not isinstance(name_or_path, str) or not name_or_path:
        raise RuntimeError("base tokenizer does not expose its source snapshot")
    try:
        observed_snapshot = pathlib.Path(name_or_path).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("base tokenizer source snapshot does not resolve") from exc
    if observed_snapshot != expected_snapshot:
        raise RuntimeError(
            "base tokenizer source snapshot is not the verified snapshot"
        )
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise RuntimeError("base tokenizer has no chat template")
    return template


def _canonical_device_map_name(parameter_name: str) -> str:
    for prefix in ("base_model.model.", "model."):
        if parameter_name.startswith(prefix):
            return parameter_name[len(prefix) :].rsplit(".", 1)[0]
    return parameter_name.rsplit(".", 1)[0]


def _device_map_assignment(model: Any, parameter_name: str) -> Any:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict) or not device_map:
        raise RuntimeError("model does not expose an immutable device map")
    module_name = _canonical_device_map_name(parameter_name)
    candidates = [
        (name, assignment)
        for name, assignment in device_map.items()
        if module_name == name or module_name.startswith(f"{name}.")
    ]
    if not candidates:
        raise RuntimeError(
            f"model device map does not bind parameter: {parameter_name}"
        )
    return max(candidates, key=lambda item: len(item[0]))[1]


def _offloaded_meta_backing(model: Any, parameter_name: str, parameter: Any) -> Any:
    if bool(getattr(parameter, "requires_grad", False)):
        raise RuntimeError(f"trainable parameter is meta: {parameter_name}")
    if str(_device_map_assignment(model, parameter_name)).lower() != "cpu":
        raise RuntimeError(
            f"meta parameter is not assigned to CPU offload: {parameter_name}"
        )

    module_name, tensor_name = parameter_name.rsplit(".", 1)
    module = dict(model.named_modules()).get(module_name)
    if module is None:
        raise RuntimeError(f"meta parameter owner is absent: {parameter_name}")
    hook = getattr(module, "_hf_hook", None)
    if hook is None or type(hook).__name__ != "AlignDevicesHook":
        raise RuntimeError(
            f"meta parameter lacks the expected offload hook: {parameter_name}"
        )
    if not bool(getattr(hook, "offload", False)):
        raise RuntimeError(f"meta parameter hook is not offloading: {parameter_name}")
    if str(getattr(hook, "execution_device", "")) not in {"0", "cuda:0"}:
        raise RuntimeError(
            f"meta parameter has an unexpected execution device: {parameter_name}"
        )
    weights_map = getattr(hook, "weights_map", None)
    if weights_map is None or tensor_name not in weights_map:
        raise RuntimeError(f"meta parameter has no backing data: {parameter_name}")
    backing = weights_map[tensor_name]
    if bool(getattr(backing, "is_meta", False)):
        raise RuntimeError(f"meta parameter backing is also meta: {parameter_name}")
    if bool(getattr(backing, "requires_grad", False)):
        raise RuntimeError(
            f"meta parameter backing unexpectedly requires gradients: {parameter_name}"
        )
    if str(getattr(getattr(backing, "device", None), "type", "")) != "cpu":
        raise RuntimeError(f"meta parameter backing is not on CPU: {parameter_name}")
    if getattr(backing, "shape", None) != getattr(parameter, "shape", None) or getattr(
        backing, "dtype", None
    ) != getattr(parameter, "dtype", None):
        raise RuntimeError(
            f"meta parameter backing metadata mismatches: {parameter_name}"
        )
    return backing


def _require_nemo_model_materialization(
    model: Any, target_modules: list[str], *, phase: str
) -> list[str]:
    if not target_modules or any(
        not isinstance(name, str) or not name for name in target_modules
    ):
        raise RuntimeError("LoRA target module list is invalid")
    seen_targets = {name: 0 for name in target_modules}
    meta_names: list[str] = []
    for parameter_name, parameter in model.named_parameters():
        target = next(
            (name for name in target_modules if f".{name}." in f".{parameter_name}."),
            None,
        )
        if target is not None:
            seen_targets[target] += 1
            if bool(getattr(parameter, "is_meta", False)):
                raise RuntimeError(
                    f"{phase} LoRA target parameter is meta: {parameter_name}"
                )
        if bool(getattr(parameter, "requires_grad", False)) and bool(
            getattr(parameter, "is_meta", False)
        ):
            raise RuntimeError(f"{phase} trainable parameter is meta: {parameter_name}")
        if bool(getattr(parameter, "is_meta", False)):
            _offloaded_meta_backing(model, parameter_name, parameter)
            meta_names.append(parameter_name)
    missing = sorted(name for name, count in seen_targets.items() if count == 0)
    if missing:
        raise RuntimeError(f"{phase} LoRA target modules are absent: {missing}")
    return meta_names


def _materialize_nemo_lm_head_for_trainer(
    model: Any, *, tensor_setter: Any | None = None
) -> None:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise RuntimeError("model has no output embedding weight")
    weight = output_embeddings.weight
    if not bool(getattr(weight, "is_meta", False)):
        if bool(getattr(weight, "requires_grad", False)):
            raise RuntimeError("materialized lm_head unexpectedly requires gradients")
        return

    parameter_name = next(
        (
            name
            for name, parameter in model.named_parameters()
            if parameter is weight and name.endswith("lm_head.weight")
        ),
        None,
    )
    if parameter_name is None:
        raise RuntimeError("meta lm_head is not a named model parameter")
    backing = _offloaded_meta_backing(model, parameter_name, weight)
    if tensor_setter is None:
        from accelerate.utils.modeling import set_module_tensor_to_device

        tensor_setter = set_module_tensor_to_device
    tensor_setter(output_embeddings, "weight", "cpu", value=backing)
    materialized = output_embeddings.weight
    if bool(getattr(materialized, "is_meta", False)):
        raise RuntimeError("lm_head remains meta after materialization")
    if str(getattr(getattr(materialized, "device", None), "type", "")) != "cpu":
        raise RuntimeError("lm_head did not materialize on CPU")
    if bool(getattr(materialized, "requires_grad", False)):
        raise RuntimeError("materialized lm_head unexpectedly requires gradients")
    if getattr(materialized, "shape", None) != getattr(
        backing, "shape", None
    ) or getattr(materialized, "dtype", None) != getattr(backing, "dtype", None):
        raise RuntimeError("materialized lm_head metadata changed")


def _prepare_nemo_assistant_labels(
    dataset: Any, tokenizer: Any, *, max_length: int
) -> Any:
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length < 1
    ):
        raise RuntimeError("assistant-label maximum length is invalid")
    column_names = getattr(dataset, "column_names", None)
    if not isinstance(column_names, list) or "messages" not in column_names:
        raise RuntimeError("assistant-label dataset has no messages column")

    def encode(example: dict[str, Any]) -> dict[str, list[int]]:
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise RuntimeError("assistant-label row must have exactly three messages")
        roles = [
            message.get("role") if isinstance(message, dict) else None
            for message in messages
        ]
        if roles != ["system", "user", "assistant"]:
            raise RuntimeError("assistant-label row must be system, user, assistant")

        context_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=False
        )
        full_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        for name, values in (("context", context_ids), ("full", full_ids)):
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, int) for value in values)
            ):
                raise RuntimeError(
                    f"assistant-label {name} tokenization is not a flat integer list"
                )
        if len(full_ids) > max_length:
            raise RuntimeError(
                f"assistant-label row exceeds signed maximum length: {len(full_ids)}"
            )
        if len(context_ids) >= len(full_ids):
            raise RuntimeError("assistant-label row has no supervised assistant tokens")
        if full_ids[: len(context_ids)] != context_ids:
            raise RuntimeError(
                "assistant-label context is not an exact prefix of the full row"
            )

        labels = [-100] * len(context_ids) + full_ids[len(context_ids) :]
        if len(labels) != len(full_ids) or not any(value != -100 for value in labels):
            raise RuntimeError("assistant-label mask has no supervised tokens")
        return {
            "attention_mask": [1] * len(full_ids),
            "input_ids": full_ids,
            "labels": labels,
        }

    prepared = dataset.map(
        encode,
        batched=False,
        desc="Apply exact assistant-only labels",
    )
    prepared_columns = getattr(prepared, "column_names", None)
    required_columns = {"attention_mask", "input_ids", "labels", "messages"}
    if not isinstance(prepared_columns, list) or not required_columns.issubset(
        prepared_columns
    ):
        raise RuntimeError("assistant-label dataset preparation is incomplete")
    return prepared


def _complete_terminal_evaluation_failure(
    spec: dict[str, Any],
    exact_payload: bytes,
    job_root: pathlib.Path,
    evidence: dict[str, Any],
) -> int:
    """Emit a terminal failure for upload, with no automatic retry."""
    evidence = dict(evidence)
    evidence["stack"] = stack_fingerprint()
    receipt = _result_receipt(
        spec,
        exact_payload,
        state="EVALUATION_FAILED_NOT_PROMOTED_NOT_SIGNED",
        evidence=evidence,
    )
    return deliver_receipt(receipt, "nemo-v3-terminal.signed.json", spec)


def main(spec_path: str) -> int:
    envelope = json.loads(pathlib.Path(spec_path).read_text(encoding="utf-8-sig"))
    pin = load_engine_pin_for_envelope(ROOT / "keys", envelope)
    try:
        spec, exact_payload, payload_type = verify_envelope(
            envelope,
            pin,
            allowed_payload_types=(NEMO_V3_PAYLOAD_TYPE,),
        )
        validate_nemo_v3_spec(spec)
        require_nemo_v3_dispatchable(
            spec,
            expected_execution_bridge_revision=os.environ.get(
                "SZL_EXECUTION_BRIDGE_REVISION"
            ),
        )
        if "authorization" in spec and pin.get("keyId") != expected_engine_key_id(spec):
            raise ValueError("Nemo v3 engine authorization key mismatch")
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED Nemo v3 contract: {exc}")
        return 3
    if payload_type != NEMO_V3_PAYLOAD_TYPE:
        return 3

    try:
        _require_remote_code_isolation(spec)
    except RuntimeError as exc:
        print(f"LOCAL ISOLATION REQUIRED: {exc}")
        return 4

    if datetime.fromisoformat(spec["expiresAt"].replace("Z", "+00:00")) < datetime.now(
        timezone.utc
    ):
        blocked(spec, "expiry", f"job expired at {spec['expiresAt']}")
    free_disk_gb = shutil.disk_usage(str(ROOT)).free / 1e9
    if free_disk_gb < spec["gates"]["minFreeDiskGb"]:
        blocked(
            spec,
            "gate:disk",
            f"free disk {free_disk_gb:.1f} GB below {spec['gates']['minFreeDiskGb']} GB",
        )
    initial_gpu = _admit_gpu(spec, "initial")

    job_root = ROOT / "jobs" / spec["jobId"]
    source_root = job_root / "source"
    release_root = job_root / "candidate"
    trainer_output = job_root / "trainer-output"
    for path in (source_root, release_root, trainer_output):
        path.mkdir(parents=True, exist_ok=True)

    try:
        train_path = _download_pinned(
            spec, spec["dataset"]["train"], source_root / "train.jsonl"
        )
        prereg_path = _download_pinned(
            spec,
            spec["dataset"]["preregistration"],
            source_root / "preregistration.json",
        )
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        if (
            not isinstance(prereg, dict)
            or prereg.get("schema") != "szl.nemo-v3-preregistration/v1"
        ):
            raise ValueError("preregistration schema mismatch")
        train_rows = _load_jsonl(train_path)
        _validate_train(train_rows, spec["gates"]["maxDatasetRows"])
        holdout_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for descriptor in spec["dataset"]["holdouts"]:
            path = _download_pinned(
                spec, descriptor, source_root / f"{descriptor['name']}.jsonl"
            )
            rows = _load_jsonl(path)
            _validate_holdout(rows, descriptor)
            holdout_rows.append((descriptor, rows))
    except Exception as exc:  # noqa: BLE001
        blocked(spec, "dataset:immutable-inputs", str(exc))

    try:
        base_license = _verify_hub_license(
            repo_id=spec["base"]["repoId"],
            revision=spec["base"]["revision"],
            expected=spec["base"]["licenseId"],
            repo_type="model",
        )
    except Exception as exc:  # noqa: BLE001
        blocked(
            spec, "gate:base-license", f"pinned base license verification failed: {exc}"
        )

    started = time.monotonic()
    recipe = spec["recipe"]
    try:
        import torch
        import unsloth
        from datasets import Dataset
        from transformers import PreTrainedTokenizerBase, TrainerCallback
        from trl import SFTConfig, SFTTrainer

        FastLanguageModel = unsloth.FastLanguageModel
        if not torch.cuda.is_available():
            blocked(spec, "gate:cuda", "CUDA is unavailable")
        torch.cuda.reset_peak_memory_stats()
        tokenizer_snapshot = _verified_offline_tokenizer_snapshot(spec)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=spec["base"]["repoId"],
            tokenizer_name=str(tokenizer_snapshot),
            revision=spec["base"]["revision"],
            max_seq_length=recipe["maxSeqLength"],
            load_in_4bit=True,
            trust_remote_code=spec["base"]["trustRemoteCode"],
        )
        _require_loaded_tokenizer(
            tokenizer,
            PreTrainedTokenizerBase,
            tokenizer_snapshot,
        )
        _require_nemo_model_materialization(
            model, recipe["targetModules"], phase="before adapter construction"
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=recipe["loraR"],
            lora_alpha=recipe["loraAlpha"],
            lora_dropout=recipe["loraDropout"],
            bias="none",
            target_modules=recipe["targetModules"],
            use_gradient_checkpointing=_gradient_checkpointing(
                recipe["gradientCheckpointing"]
            ),
            random_state=recipe["seed"],
            max_seq_length=recipe["maxSeqLength"],
            use_rslora=True,
            loftq_config=None,
        )
        _require_nemo_model_materialization(
            model, recipe["targetModules"], phase="after adapter construction"
        )
        full_train = _prepare_nemo_assistant_labels(
            Dataset.from_list(train_rows),
            tokenizer,
            max_length=recipe["maxSeqLength"],
        )
        split = full_train.train_test_split(
            test_size=min(0.15, max(1 / len(full_train), 0.05)), seed=recipe["seed"]
        )

        class GateCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = logs or {}
                loss = logs.get("loss")
                if isinstance(loss, (int, float)) and not math.isfinite(loss):
                    blocked(
                        spec,
                        "train:non-finite-loss",
                        f"non-finite loss at step {state.global_step}",
                    )
                if (
                    time.monotonic() - started
                    > spec["gates"]["maxWallclockMinutes"] * 60
                ):
                    blocked(
                        spec,
                        "train:wallclock",
                        "signed maximum wall-clock duration exceeded",
                    )
                observed = _gpu_state()
                if observed["temperature_c"] > spec["gates"]["maxTemperatureC"]:
                    blocked(
                        spec,
                        "train:temperature",
                        "GPU temperature exceeded signed maximum",
                        extra=observed,
                    )
                return control

        frontier_recipe = {
            **recipe,
            "packing": False,
            "packingStrategy": "ffd",
            "assistantOnlyLoss": True,
            "useRsLoRA": True,
        }
        config = _build_sft_config(SFTConfig, frontier_recipe, torch, trainer_output)
        _materialize_nemo_lm_head_for_trainer(model)
        _require_nemo_model_materialization(
            model, recipe["targetModules"], phase="before trainer construction"
        )
        trainer = _build_sft_trainer(
            SFTTrainer,
            model=model,
            tokenizer=tokenizer,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            callbacks=[GateCallback()],
            config=config,
        )
        train_output = trainer.train()
        eval_raw = trainer.evaluate()
        if not math.isfinite(float(train_output.training_loss)) or not math.isfinite(
            float(eval_raw.get("eval_loss", float("nan")))
        ):
            blocked(
                spec,
                "eval:non-finite-loss",
                "training or validation loss is non-finite",
            )

        adapter_dir = release_root / "adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        adapter_files = artifact_manifest(adapter_dir)
        if not any(
            item["path"] == "adapter_model.safetensors" for item in adapter_files
        ):
            blocked(spec, "artifact:safetensors", "adapter_model.safetensors is absent")
        if any(
            item["path"].endswith((".bin", ".pt", ".pth", ".pkl", ".pickle"))
            for item in adapter_files
        ):
            blocked(
                spec,
                "artifact:pickle",
                "candidate contains a pickle-compatible artifact",
            )

        FastLanguageModel.for_inference(model)
        suite_results: dict[str, Any] = {}
        total = total_passes = total_degenerate = 0
        for descriptor, rows in holdout_rows:
            results, passes = _evaluate_suite(
                model, tokenizer, rows, spec["evaluation"]["maxNewTokens"]
            )
            degenerate = sum(int(item["degenerate"]) for item in results)
            suite_results[descriptor["name"]] = {
                "rows": len(rows),
                "passes": passes,
                "pass_rate": passes / len(rows),
                "degenerate": degenerate,
                "record_ids_sha256": descriptor["recordIdsSha256"],
                "results": results,
            }
            total += len(rows)
            total_passes += passes
            total_degenerate += degenerate
        pass_rate = total_passes / total
        if pass_rate != spec["evaluation"]["requiredPassRate"] or total_degenerate != 0:
            evidence = {
                "state": "FAIL",
                "rows": total,
                "passes": total_passes,
                "pass_rate": pass_rate,
                "degenerate": total_degenerate,
                "suites": suite_results,
            }
            return _complete_terminal_evaluation_failure(
                spec, exact_payload, job_root, evidence
            )

        final_gpu = _gpu_state()
        evidence = {
            "state": "PASS",
            "rows": total,
            "passes": total_passes,
            "pass_rate": pass_rate,
            "degenerate": total_degenerate,
            "suites": suite_results,
            "training_loss": float(train_output.training_loss),
            "validation_loss": float(eval_raw["eval_loss"]),
            "adapter_manifest_sha256": manifest_digest(adapter_files),
            "adapter_files": adapter_files,
            "stack": stack_fingerprint(),
            "base_license": base_license,
            "gpu_initial": initial_gpu,
            "gpu_final": final_gpu,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "preregistration_sha256": sha256_file(prereg_path),
        }
        receipt = _result_receipt(
            spec,
            exact_payload,
            state="QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW",
            evidence=evidence,
        )
        return deliver_receipt(receipt, "nemo-v3-qualified.signed.json", spec)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        blocked(spec, "runtime:terminal", f"{type(exc).__name__}: {exc}")
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: runjob_nemo_v3.py <signed-job-spec.json>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
