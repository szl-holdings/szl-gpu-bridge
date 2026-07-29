#!/usr/bin/env python3
"""Pure evidence and artifact helpers for frontier training jobs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
from typing import Any, Iterable

PACKAGE_EVIDENCE = (
    "torch",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "unsloth",
    "unsloth-zoo",
    "bitsandbytes",
    "xformers",
    "huggingface-hub",
    "pynacl",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chat_template_evidence(tokenizer: Any) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        template = ""
    encoded = template.encode("utf-8")
    return {
        "present": bool(template),
        "sha256": sha256_bytes(encoded),
        "bytes": len(encoded),
        "hasGenerationBlocks": "{% generation %}" in template
        and "{% endgeneration %}" in template,
    }


def validate_expected_chat_template(
    evidence: dict[str, Any], expected_sha256: str | None
) -> None:
    if expected_sha256 and evidence.get("sha256") != expected_sha256:
        raise ValueError(
            f"chat-template sha256 {evidence.get('sha256')} != pinned {expected_sha256}"
        )


def artifact_manifest(root: str | pathlib.Path) -> list[dict[str, Any]]:
    root_path = pathlib.Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"artifact root does not exist: {root_path}")
    result: list[dict[str, Any]] = []
    for path in sorted(root_path.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinked artifact refused: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root_path).as_posix()
        result.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def manifest_digest(entries: Iterable[dict[str, Any]]) -> str:
    body = json.dumps(list(entries), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256_bytes(body)


def _command_evidence(command: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        text = (completed.stdout or completed.stderr or "").strip()
        return {
            "available": True,
            "returncode": completed.returncode,
            "output": text[:4000],
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)[:500]}


def stack_fingerprint(packages: Iterable[str] = PACKAGE_EVIDENCE) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    evidence = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": versions,
        "nvidiaSmi": _command_evidence(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    image_reference = os.environ.get("SZL_CONTAINER_IMAGE_REFERENCE")
    image_id = os.environ.get("SZL_CONTAINER_IMAGE_ID")
    image_revision = os.environ.get("SZL_CONTAINER_IMAGE_REVISION")
    build_receipt_sha256 = os.environ.get("SZL_CONTAINER_IMAGE_BUILD_RECEIPT_SHA256")
    dockerfile_sha256 = os.environ.get("SZL_CONTAINER_IMAGE_DOCKERFILE_SHA256")
    launcher_sha256 = os.environ.get("SZL_LAUNCHER_SHA256")
    if launcher_sha256 and not re.fullmatch(r"[0-9a-f]{64}", launcher_sha256):
        raise RuntimeError("isolated launcher identity is not immutable")
    evidence["launcherSha256"] = launcher_sha256 or None
    if image_reference or image_id:
        if not image_reference or not image_id or not image_revision:
            raise RuntimeError("container image evidence is incomplete")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_reference):
            raise RuntimeError("container image reference is not an approved local ID")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise RuntimeError("container image ID is not immutable")
        if not re.fullmatch(r"[0-9a-f]{40}", image_revision):
            raise RuntimeError("container image revision is not immutable")
        if bool(build_receipt_sha256) != bool(dockerfile_sha256):
            raise RuntimeError("container local-build evidence is incomplete")
        container_image = {
            "reference": image_reference,
            "id": image_id,
            "revision": image_revision,
        }
        if build_receipt_sha256:
            if not re.fullmatch(r"[0-9a-f]{64}", build_receipt_sha256):
                raise RuntimeError("container image build receipt is not immutable")
            if not re.fullmatch(r"[0-9a-f]{64}", dockerfile_sha256 or ""):
                raise RuntimeError("container image Dockerfile is not immutable")
            container_image["localBuild"] = {
                "receiptSha256": build_receipt_sha256,
                "dockerfileSha256": dockerfile_sha256,
            }
        evidence["containerImage"] = container_image
    else:
        if image_revision or build_receipt_sha256 or dockerfile_sha256:
            raise RuntimeError("container image evidence is incomplete")
        evidence["containerImage"] = None
    lock_path = pathlib.Path(__file__).resolve().parent / "stack-freeze.txt"
    if lock_path.exists():
        evidence["stackFreeze"] = {
            "path": lock_path.name,
            "sha256": sha256_file(lock_path),
            "bytes": lock_path.stat().st_size,
        }
    else:
        evidence["stackFreeze"] = None
    return evidence


def export_plan(spec: dict[str, Any]) -> list[dict[str, Any]]:
    exports = spec["outputs"]["exports"]
    plan: list[dict[str, Any]] = [{"kind": "adapter", "path": "adapter"}]
    if exports["merged16bit"]:
        plan.append({"kind": "merged_16bit", "path": "merged-16bit"})
    for quantization in exports["ggufQuantizations"]:
        plan.append(
            {
                "kind": "gguf",
                "quantization": quantization,
                "path": f"gguf/{quantization}",
            }
        )
    return plan


def normalize_conversation(
    messages: Any,
    *,
    require_final_assistant: bool = False,
) -> list[dict[str, Any]]:
    """Validate and normalize one conversational training row."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index} has an unsupported role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} content must be a non-empty string")
        normalized.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in normalized):
        raise ValueError("conversation requires at least one user message")
    if require_final_assistant and normalized[-1]["role"] != "assistant":
        raise ValueError("training conversation must end with an assistant target")
    return normalized


def prompt_messages(messages: Any) -> list[dict[str, Any]]:
    normalized = normalize_conversation(messages, require_final_assistant=True)
    normalized = normalized[:-1]
    if not normalized or normalized[-1]["role"] == "assistant":
        raise ValueError("could not derive a generation prompt from messages")
    return normalized


def extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def is_degenerate_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 8:
        return True
    compact = re.sub(r"\s+", "", stripped)
    if not compact:
        return True
    unique_ratio = len(set(compact)) / len(compact)
    if unique_ratio < 0.03:
        return True
    if re.search(r"(.)\1{15,}", compact, re.DOTALL):
        return True
    return False


def model_card(
    spec: dict[str, Any],
    *,
    training_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    artifacts: list[dict[str, Any]],
    chat_template: dict[str, Any],
) -> str:
    name = spec["outputs"]["modelRepoId"].split("/", 1)[-1]
    heldout_loss = eval_metrics.get("heldOutLoss")
    yaml_loss = "null" if heldout_loss is None else str(heldout_loss)
    artifact_digest = manifest_digest(artifacts)
    checkpoint_bucket = ""
    if spec["outputs"].get("checkpointBucketId"):
        checkpoint_bucket = f"buckets:\n- {spec['outputs']['checkpointBucketId']}\n"
    return f"""---
base_model: {spec["base"]["repoId"]}
datasets:
- {spec["dataset"]["repoId"]}
{checkpoint_bucket}library_name: peft
license: {spec["base"]["licenseId"]}
pipeline_tag: text-generation
tags:
- unsloth
- peft
- lora
- governed-training
- szl-holdings
model-index:
- name: {name}
  results:
  - task:
      type: text-generation
    dataset:
      name: {spec["dataset"]["repoId"]} held-out split
      type: {spec["dataset"]["repoId"]}
    metrics:
    - type: loss
      value: {yaml_loss}
      name: Held-out loss
---

# {name}

This repository was produced by the SZL GPU Bridge under a DSSE-signed
`unsloth-frontier-sft-v2` job. It is an adaptation of the immutable base
`{spec["base"]["repoId"]}@{spec["base"]["revision"]}`; it is not from-scratch
pretraining.

## Reproducibility anchors

- Base license metadata: `{spec["base"]["licenseId"]}`
- Dataset: `{spec["dataset"]["repoId"]}@{spec["dataset"]["revision"]}`
- Dataset license metadata: `{spec["dataset"]["licenseId"]}`
- Dataset file sha256: `{spec["dataset"]["sha256"]}`
- Chat-template sha256: `{chat_template["sha256"]}`
- Artifact-manifest sha256: `{artifact_digest}`
- Training recipe seed: `{spec["recipe"]["seed"]}`

## Measured run evidence

```json
{json.dumps({"training": training_metrics, "evaluation": eval_metrics}, indent=2, sort_keys=True)}
```

## Limitations

The evaluation is a deterministic held-out split plus structural doctrine checks.
It does not establish general intelligence, safety, theoremhood, financial skill,
or performance outside the recorded test frame. Training and evaluation receipts
are signed attestations by the laptop key, not cryptographic proofs of computation.
"""
