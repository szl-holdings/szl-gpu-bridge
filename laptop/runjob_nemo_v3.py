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
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

from frontier_contract import load_pin, verify_envelope
from frontier_job import (
    ROOT,
    _build_sft_config,
    _build_sft_trainer,
    _gradient_checkpointing,
    _verify_hub_license,
    blocked,
    now_iso,
    sign_receipt,
    upload_receipt,
    write_json,
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
    record_ids_sha256,
    validate_nemo_v3_spec,
)


def _raw_url(repo_id: str, revision: str, path: str) -> str:
    owner, name = repo_id.split("/", 1)
    parts = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    return f"https://raw.githubusercontent.com/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(name, safe='')}/{revision}/{parts}"


def _download_pinned(
    spec: dict[str, Any], descriptor: dict[str, Any], target: pathlib.Path
) -> pathlib.Path:
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


def main(spec_path: str) -> int:
    envelope = json.loads(pathlib.Path(spec_path).read_text(encoding="utf-8-sig"))
    pin = load_pin(ROOT / "keys" / "engine_pubkey.json")
    try:
        spec, exact_payload, payload_type = verify_envelope(
            envelope,
            pin,
            allowed_payload_types=(NEMO_V3_PAYLOAD_TYPE,),
        )
        validate_nemo_v3_spec(spec)
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED Nemo v3 contract: {exc}")
        return 3
    if payload_type != NEMO_V3_PAYLOAD_TYPE:
        return 3

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
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel

        if not torch.cuda.is_available():
            blocked(spec, "gate:cuda", "CUDA is unavailable")
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=spec["base"]["repoId"],
            revision=spec["base"]["revision"],
            max_seq_length=recipe["maxSeqLength"],
            load_in_4bit=True,
            trust_remote_code=spec["base"]["trustRemoteCode"],
        )
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template:
            blocked(spec, "gate:chat-template", "base tokenizer has no chat template")
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
        full_train = Dataset.from_list(train_rows)
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
            receipt = _result_receipt(
                spec,
                exact_payload,
                state="EVALUATION_FAILED_NOT_PROMOTED_NOT_SIGNED",
                evidence=evidence,
            )
            signed = sign_receipt(receipt)
            write_json(job_root / "receipts" / "nemo-v3-terminal.signed.json", signed)
            upload_receipt(signed, "nemo-v3-terminal.signed.json", spec)
            return 6

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
        signed = sign_receipt(receipt)
        write_json(job_root / "receipts" / "nemo-v3-qualified.signed.json", signed)
        upload_receipt(signed, "nemo-v3-qualified.signed.json", spec)
        return 0
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
