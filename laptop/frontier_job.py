#!/usr/bin/env python3
"""Execution helpers for the governed frontier training runner."""
from __future__ import annotations

import base64
import gc
import hashlib
import inspect
import json
import pathlib
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

from frontier_contract import canonicalize
from frontier_runtime import (
    chat_template_evidence,
    extract_json_object,
    is_degenerate_text,
    normalize_conversation,
    prompt_messages,
    sha256_file,
)

ROOT = pathlib.Path(__file__).resolve().parent

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def laptop_keys():
    from nacl.signing import SigningKey

    pem = (ROOT / "keys" / "laptop_key.pem").read_text(encoding="utf-8-sig")
    seed = base64.b64decode("".join(line for line in pem.splitlines() if "-----" not in line))
    signing_key = SigningKey(seed)
    public = json.loads((ROOT / "keys" / "laptop_pubkey.json").read_text(encoding="utf-8-sig"))
    return signing_key, public


def sign_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    signing_key, public = laptop_keys()
    body = canonicalize(receipt).encode("utf-8")
    signature = signing_key.sign(body).signature
    return {
        "receipt": receipt,
        "bodyBase64": base64.b64encode(body).decode("ascii"),
        "signatureBase64": base64.b64encode(signature).decode("ascii"),
        "publicKeySpkiBase64": public["publicKeySpkiBase64"],
        "keyId": public["keyId"],
        "scheme": "ed25519-over-exact-bytes-v2",
    }


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def upload_receipt(signed: dict[str, Any], name: str, spec: dict[str, Any]) -> Any:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        spec["outputs"]["receiptsRepoId"],
        repo_type="dataset",
        exist_ok=True,
        private=spec["outputs"]["private"],
    )
    local = ROOT / "jobs" / spec["jobId"] / "receipts" / name
    write_json(local, signed)
    return api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=f"{spec['jobId']}/{name}",
        repo_id=spec["outputs"]["receiptsRepoId"],
        repo_type="dataset",
        commit_message=f"receipt({spec['jobId']}): {name}",
    )


def blocked(spec: dict[str, Any], stage: str, reason: str, *, extra: dict[str, Any] | None = None) -> None:
    receipt = {
        "kind": "szl-frontier-training-blocked",
        "v": 2,
        "jobId": spec["jobId"],
        "verdict": "BLOCKED",
        "stage": stage,
        "reason": reason,
        "at": now_iso(),
        "extra": extra or {},
        "doctrine": {
            "failClosed": True,
            "note": "a refused or failed job is a result, never a success to infer",
        },
    }
    signed = sign_receipt(receipt)
    upload_receipt(signed, "blocked_receipt.signed.json", spec)
    raise SystemExit(0)


def probe_vram_gb() -> float:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.strip().splitlines()[0]) / 1024.0


def _gradient_checkpointing(value: str) -> str | bool:
    if value == "unsloth":
        return "unsloth"
    return value == "true"


def _filtered_kwargs(callable_obj: Any, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in parameters}


def _build_sft_config(SFTConfig: Any, recipe: dict[str, Any], torch: Any, output_dir: pathlib.Path):
    values = {
        "per_device_train_batch_size": recipe["batchSize"],
        "gradient_accumulation_steps": recipe["gradAccum"],
        "num_train_epochs": recipe["epochs"],
        "learning_rate": recipe["learningRate"],
        "optim": recipe["optimizer"],
        "seed": recipe["seed"],
        "output_dir": str(output_dir),
        "logging_steps": 10,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "report_to": [],
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "fp16": not bool(torch.cuda.is_bf16_supported()),
        "warmup_ratio": recipe["warmupRatio"],
        "weight_decay": recipe["weightDecay"],
        "lr_scheduler_type": recipe["lrSchedulerType"],
        "packing": recipe["packing"],
        "packing_strategy": recipe["packingStrategy"],
        "assistant_only_loss": recipe["assistantOnlyLoss"],
        "max_length": recipe["maxSeqLength"],
        "max_seq_length": recipe["maxSeqLength"],
    }
    return SFTConfig(**_filtered_kwargs(SFTConfig, values))


def _build_sft_trainer(
    SFTTrainer: Any,
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    callbacks: list[Any],
    config: Any,
):
    values = {
        "model": model,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "callbacks": callbacks,
        "args": config,
    }
    return SFTTrainer(**_filtered_kwargs(SFTTrainer, values))


def _license_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.strip().lower()} if value.strip() else set()
    if isinstance(value, list):
        return {
            item.strip().lower()
            for item in value
            if isinstance(item, str) and item.strip()
        }
    return set()


def _verify_hub_license(
    *,
    repo_id: str,
    revision: str,
    expected: str,
    repo_type: str,
) -> dict[str, Any]:
    """Load card metadata from the exact signed revision and match its license."""
    from huggingface_hub import DatasetCard, ModelCard, hf_hub_download

    readme = hf_hub_download(
        repo_id=repo_id,
        filename="README.md",
        repo_type=None if repo_type == "model" else repo_type,
        revision=revision,
    )
    card_class = ModelCard if repo_type == "model" else DatasetCard
    card = card_class.load(readme)
    metadata = card.data.to_dict()
    observed = _license_values(metadata.get("license"))
    expected_normalized = expected.strip().lower()
    if not observed:
        raise RuntimeError(f"{repo_type} card at the pinned revision has no license metadata")
    if expected_normalized not in observed:
        raise RuntimeError(
            f"signed license {expected_normalized!r} does not match pinned Hub metadata {sorted(observed)!r}"
        )
    return {
        "repoId": repo_id,
        "revision": revision,
        "repoType": repo_type,
        "expected": expected_normalized,
        "observed": sorted(observed),
        "readmeSha256": sha256_file(readme),
    }


def _validate_dataset_rows(dataset: Any) -> None:
    for index in range(len(dataset)):
        row = dataset[index]
        if not isinstance(row, dict) or "messages" not in row:
            raise ValueError(f"dataset row {index} has no messages field")
        try:
            normalize_conversation(row["messages"], require_final_assistant=True)
        except ValueError as exc:
            raise ValueError(f"dataset row {index} is malformed: {exc}") from exc


def _generate(model: Any, tokenizer: Any, messages: list[dict[str, Any]], *, max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    start = encoded["input_ids"].shape[1]
    return tokenizer.decode(output[0][start:], skip_special_tokens=True)


def _evaluate_generations(model: Any, tokenizer: Any, eval_dataset: Any, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = min(config["maxGenerations"], len(eval_dataset))
    records: list[dict[str, Any]] = []
    json_valid = required_keys_valid = ceiling_valid = abstains = degenerate = 0
    for index in range(total):
        messages = eval_dataset[index].get("messages")
        prompt = prompt_messages(messages)
        text = _generate(model, tokenizer, prompt, max_new_tokens=config["maxNewTokens"])
        obj = extract_json_object(text)
        is_json = obj is not None
        keys_ok = bool(obj is not None and all(key in obj for key in config["requiredJsonKeys"]))
        conviction = obj.get("conviction") if obj else None
        ceiling_ok = conviction is None or (
            isinstance(conviction, (int, float))
            and not isinstance(conviction, bool)
            and float(conviction) <= config["convictionCeiling"]
        )
        is_degenerate = is_degenerate_text(text)
        json_valid += int(is_json)
        required_keys_valid += int(keys_ok)
        ceiling_valid += int(ceiling_ok)
        abstains += int(bool(obj and obj.get("action") == "ABSTAIN"))
        degenerate += int(is_degenerate)
        records.append(
            {
                "index": index,
                "responseSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "responseBytes": len(text.encode("utf-8")),
                "jsonValid": is_json,
                "requiredKeysValid": keys_ok,
                "ceilingValid": ceiling_ok,
                "degenerate": is_degenerate,
            }
        )
    def rate(count: int) -> float | None:
        return count / total if total else None

    return (
        {
            "generationsChecked": total,
            "jsonValidRate": rate(json_valid),
            "requiredKeysRate": rate(required_keys_valid),
            "ceilingRespectRate": rate(ceiling_valid),
            "abstainGenerations": abstains,
            "degenerateRate": rate(degenerate),
        },
        records,
    )


def _reload_smoke_merged(
    merged_dir: pathlib.Path,
    prompt: list[dict[str, Any]],
    *,
    expected_template_sha256: str,
    max_new_tokens: int,
    trust_remote_code: bool,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=trust_remote_code)
    evidence = chat_template_evidence(tokenizer)
    if evidence["sha256"] != expected_template_sha256:
        raise RuntimeError(
            f"merged tokenizer template drifted: {evidence['sha256']} != {expected_template_sha256}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    with torch.no_grad():
        text = _generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    if is_degenerate_text(text):
        raise RuntimeError("merged reload produced empty/repeated-character output")
    evidence_out = {
        "kind": "merged_16bit",
        "ok": True,
        "responseSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "responseBytes": len(text.encode("utf-8")),
        "chatTemplate": evidence,
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return evidence_out


def _llama_cli() -> str | None:
    for candidate in ("llama-cli.exe", "llama-cli"):
        found = shutil.which(candidate)
        if found:
            return found
    home = pathlib.Path.home()
    candidates = (
        ROOT / "llama.cpp" / "llama-cli.exe",
        home / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe",
        home / "szl-forge" / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe",
        home / "szl-forge" / "llama.cpp" / "build" / "bin" / "llama-cli.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _reload_smoke_gguf(
    root: pathlib.Path,
    prompt: str,
    *,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    cli = _llama_cli()
    if not cli:
        raise RuntimeError("GGUF was requested but llama-cli is not installed for reload smoke")
    files = sorted(root.rglob("*.gguf"))
    if not files:
        raise RuntimeError("GGUF export produced no .gguf file")
    evidence = []
    for file in files:
        completed = subprocess.run(
            [
                cli,
                "-m",
                str(file),
                "-p",
                prompt,
                "-n",
                str(min(max_new_tokens, 128)),
                "--temp",
                "0",
                "--seed",
                "3407",
            ],
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        text = (completed.stdout or "")[-4000:]
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama-cli failed for {file.name}: {(completed.stderr or text)[-1000:]}"
            )
        if is_degenerate_text(text):
            raise RuntimeError(f"GGUF reload produced degenerate output for {file.name}")
        evidence.append(
            {
                "kind": "gguf",
                "file": file.relative_to(root).as_posix(),
                "ok": True,
                "responseSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "responseBytes": len(text.encode("utf-8")),
            }
        )
    return evidence


def _sync_checkpoint_bucket(output_dir: pathlib.Path, spec: dict[str, Any]) -> dict[str, Any] | None:
    bucket_id = spec["outputs"].get("checkpointBucketId")
    if not bucket_id:
        return None
    from huggingface_hub import create_bucket, sync_bucket

    create_bucket(bucket_id, private=spec["outputs"]["private"], exist_ok=True)
    destination = f"hf://buckets/{bucket_id}/{spec['jobId']}/checkpoints"
    plan = sync_bucket(str(output_dir), destination)
    return {
        "bucketId": bucket_id,
        "prefix": f"{spec['jobId']}/checkpoints",
        "operations": getattr(plan, "summary", lambda: None)(),
        "note": "mutable working storage; release artifacts remain in versioned model repositories",
    }
