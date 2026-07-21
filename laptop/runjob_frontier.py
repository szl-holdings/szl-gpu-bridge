#!/usr/bin/env python3
"""Verify, execute, evaluate, export, and publish one signed frontier job."""

from __future__ import annotations

import base64
import gc
import hashlib
import json
import math
import pathlib
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any

from frontier_contract import (
    V2_PAYLOAD_TYPE,
    load_pin,
    validate_v2_spec,
    verify_envelope,
)
from frontier_job import (
    ROOT,
    _build_sft_config,
    _build_sft_trainer,
    _evaluate_generations,
    _gradient_checkpointing,
    _reload_smoke_gguf,
    _reload_smoke_merged,
    _sync_checkpoint_bucket,
    _validate_dataset_rows,
    _verify_hub_license,
    blocked,
    now_iso,
    probe_vram_gb,
    sign_receipt,
    upload_receipt,
    write_json,
)
from frontier_runtime import (
    artifact_manifest,
    chat_template_evidence,
    export_plan,
    manifest_digest,
    model_card,
    prompt_messages,
    sha256_file,
    stack_fingerprint,
    validate_expected_chat_template,
)


def main(spec_path: str) -> int:
    envelope = json.loads(pathlib.Path(spec_path).read_text(encoding="utf-8-sig"))
    pin = load_pin(ROOT / "keys" / "engine_pubkey.json")
    try:
        spec, payload, payload_type = verify_envelope(
            envelope,
            pin,
            allowed_payload_types=(V2_PAYLOAD_TYPE,),
        )
        validate_v2_spec(spec)
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED frontier contract: {exc}")
        return 3
    if payload_type != V2_PAYLOAD_TYPE:
        return 3

    job_root = ROOT / "jobs" / spec["jobId"]
    output_dir = job_root / "trainer-output"
    release_dir = job_root / "release"
    output_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)

    if datetime.fromisoformat(spec["expiresAt"].replace("Z", "+00:00")) < datetime.now(
        timezone.utc
    ):
        blocked(spec, "expiry", f"spec expired at {spec['expiresAt']}")
    free_disk_gb = shutil.disk_usage(str(ROOT)).free / 1e9
    if free_disk_gb < spec["gates"]["minFreeDiskGb"]:
        blocked(
            spec,
            "gate:disk",
            f"free disk {free_disk_gb:.1f} GB < required {spec['gates']['minFreeDiskGb']} GB",
        )
    try:
        free_vram_gb = probe_vram_gb()
    except Exception as exc:  # noqa: BLE001
        blocked(spec, "gate:vram-probe", f"nvidia-smi probe failed: {exc}")
    if free_vram_gb < spec["gates"]["minFreeVramGb"]:
        blocked(
            spec,
            "gate:vram",
            f"free VRAM {free_vram_gb:.1f} GB < required {spec['gates']['minFreeVramGb']} GB",
        )

    stack = stack_fingerprint()

    try:
        base_license_evidence = _verify_hub_license(
            repo_id=spec["base"]["repoId"],
            revision=spec["base"]["revision"],
            expected=spec["base"]["licenseId"],
            repo_type="model",
        )
        dataset_license_evidence = _verify_hub_license(
            repo_id=spec["dataset"]["repoId"],
            revision=spec["dataset"]["revision"],
            expected=spec["dataset"]["licenseId"],
            repo_type="dataset",
        )
    except Exception as exc:  # noqa: BLE001
        blocked(
            spec,
            "gate:license-metadata",
            f"pinned Hub license verification failed: {exc}",
        )

    try:
        from huggingface_hub import hf_hub_download

        dataset_path = hf_hub_download(
            repo_id=spec["dataset"]["repoId"],
            filename=spec["dataset"]["file"],
            repo_type="dataset",
            revision=spec["dataset"]["revision"],
        )
    except Exception as exc:  # noqa: BLE001
        blocked(spec, "dataset:download", f"pinned dataset download failed: {exc}")
    observed_dataset_sha = sha256_file(dataset_path)
    if observed_dataset_sha != spec["dataset"]["sha256"]:
        blocked(
            spec,
            "gate:dataset-hash",
            f"dataset sha {observed_dataset_sha} != pinned {spec['dataset']['sha256']}",
        )

    started = time.time()
    recipe = spec["recipe"]
    try:
        import torch
        from datasets import load_dataset
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=spec["base"]["repoId"],
            revision=spec["base"]["revision"],
            max_seq_length=recipe["maxSeqLength"],
            load_in_4bit=True,
            trust_remote_code=spec["base"].get("trustRemoteCode", False),
        )
        template = chat_template_evidence(tokenizer)
        validate_expected_chat_template(template, recipe["expectedChatTemplateSha256"])
        if recipe["assistantOnlyLoss"] and not template["hasGenerationBlocks"]:
            blocked(
                spec,
                "gate:assistant-mask-template",
                "assistantOnlyLoss requires a pinned chat template with generation/endgeneration blocks",
                extra={"chatTemplate": template},
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
            use_rslora=recipe["useRsLoRA"],
            loftq_config=None,
        )

        full_dataset = load_dataset("json", data_files=dataset_path, split="train")
        if len(full_dataset) > spec["gates"]["maxDatasetRows"]:
            blocked(
                spec,
                "gate:dataset-rows",
                f"dataset has {len(full_dataset)} rows > signed maximum {spec['gates']['maxDatasetRows']}",
            )
        if len(full_dataset) < 4:
            blocked(
                spec,
                "gate:dataset-rows",
                "dataset requires at least four rows for train/eval split",
            )
        if "messages" not in full_dataset.column_names:
            blocked(
                spec,
                "gate:dataset-format",
                "messages-jsonl dataset has no messages column",
            )
        try:
            _validate_dataset_rows(full_dataset)
        except ValueError as exc:
            blocked(spec, "gate:dataset-conversation", str(exc))
        split = full_dataset.train_test_split(
            test_size=spec["eval"]["heldOutFraction"],
            seed=spec["eval"]["seed"],
        )
        train_dataset, eval_dataset = split["train"], split["test"]

        class GateCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                logs = logs or {}
                loss = logs.get("loss")
                if (
                    spec["gates"].get("abortOnNanLoss", True)
                    and isinstance(loss, (int, float))
                    and not math.isfinite(loss)
                ):
                    blocked(
                        spec,
                        "train:non-finite-loss",
                        f"non-finite loss at step {state.global_step}",
                    )
                if time.time() - started > spec["gates"]["maxWallclockMinutes"] * 60:
                    blocked(
                        spec,
                        "train:wallclock",
                        f"exceeded signed wallclock limit of {spec['gates']['maxWallclockMinutes']} minutes",
                    )
                return control

        sft_config = _build_sft_config(SFTConfig, recipe, torch, output_dir)
        trainer = _build_sft_trainer(
            SFTTrainer,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[GateCallback()],
            config=sft_config,
        )
        train_output = trainer.train()
        eval_raw = trainer.evaluate()
        heldout_loss = float(eval_raw.get("eval_loss", float("nan")))
        if not math.isfinite(float(train_output.training_loss)) or not math.isfinite(
            heldout_loss
        ):
            blocked(
                spec,
                "eval:non-finite-loss",
                "training or held-out loss is non-finite",
                extra={
                    "trainLoss": float(train_output.training_loss),
                    "heldOutLoss": heldout_loss,
                },
            )

        FastLanguageModel.for_inference(model)
        generation_metrics, generation_records = _evaluate_generations(
            model,
            tokenizer,
            eval_dataset,
            spec["eval"],
        )
        if (generation_metrics["degenerateRate"] or 0) > spec["eval"][
            "maxDegenerateRate"
        ]:
            blocked(
                spec,
                "eval:degenerate-output",
                f"degenerate output rate {generation_metrics['degenerateRate']} exceeds signed maximum {spec['eval']['maxDegenerateRate']}",
                extra={"generationEvidence": generation_records},
            )
        rate_thresholds = {
            "jsonValidRate": spec["eval"]["minJsonValidRate"],
            "requiredKeysRate": spec["eval"]["minRequiredKeysRate"],
            "ceilingRespectRate": spec["eval"]["minCeilingRespectRate"],
        }
        for metric_name, minimum in rate_thresholds.items():
            observed = generation_metrics.get(metric_name)
            if observed is None or observed < minimum:
                blocked(
                    spec,
                    f"eval:{metric_name}",
                    f"{metric_name} {observed!r} is below signed minimum {minimum}",
                    extra={"generationEvidence": generation_records},
                )

        adapter_dir = release_dir / "adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        exports = spec["outputs"]["exports"]
        if exports["merged16bit"]:
            merged_dir = release_dir / "merged-16bit"
            model.save_pretrained_merged(
                str(merged_dir),
                tokenizer,
                save_method="merged_16bit",
            )
        gguf_roots: list[pathlib.Path] = []
        for quantization in exports["ggufQuantizations"]:
            gguf_dir = release_dir / "gguf" / quantization
            model.save_pretrained_gguf(
                str(gguf_dir),
                tokenizer=tokenizer,
                quantization_method=quantization,
            )
            gguf_roots.append(gguf_dir)

        first_prompt = prompt_messages(eval_dataset[0]["messages"])
        prompt_text = tokenizer.apply_chat_template(
            first_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        train_loss_value = float(train_output.training_loss)
        global_steps = int(getattr(trainer.state, "global_step", 0))
        train_rows = len(train_dataset)
        heldout_rows = len(eval_dataset)
        peak_vram_value = (
            float(torch.cuda.max_memory_allocated()) / 1e9
            if torch.cuda.is_available()
            else None
        )
        del trainer, train_output, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        export_smoke: list[dict[str, Any]] = [
            {
                "kind": "adapter",
                "ok": True,
                "basis": "trained in-memory model passed held-out deterministic generation checks",
            }
        ]
        if exports["merged16bit"]:
            export_smoke.append(
                _reload_smoke_merged(
                    release_dir / "merged-16bit",
                    first_prompt,
                    expected_template_sha256=template["sha256"],
                    max_new_tokens=min(spec["eval"]["maxNewTokens"], 128),
                    trust_remote_code=spec["base"].get("trustRemoteCode", False),
                )
            )
        for gguf_root in gguf_roots:
            export_smoke.extend(
                _reload_smoke_gguf(
                    gguf_root,
                    prompt_text,
                    max_new_tokens=spec["eval"]["maxNewTokens"],
                )
            )

        bucket_evidence = _sync_checkpoint_bucket(output_dir, spec)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        blocked(spec, "frontier-runner", f"training/export pipeline failed: {exc}")

    train_minutes = (time.time() - started) / 60
    training_metrics = {
        "label": "MEASURED",
        "finalTrainLoss": train_loss_value,
        "trainMinutes": round(train_minutes, 2),
        "peakVramGb": round(peak_vram_value, 2)
        if peak_vram_value is not None
        else None,
        "steps": global_steps,
        "trainRows": train_rows,
    }
    eval_metrics = {
        "label": "MEASURED",
        "heldOutLoss": heldout_loss,
        "heldOutRows": heldout_rows,
        **generation_metrics,
        "limits": "single deterministic held-out split; structural doctrine checks do not measure general capability",
    }

    model_artifacts = artifact_manifest(release_dir)
    artifact_digest = manifest_digest(model_artifacts)
    artifact_body = {
        "jobId": spec["jobId"],
        "generatedAt": now_iso(),
        "entries": model_artifacts,
        "manifestSha256": artifact_digest,
        "exportPlan": export_plan(spec),
        "reloadSmoke": export_smoke,
    }
    write_json(release_dir / "artifact-manifest.json", artifact_body)
    (release_dir / "README.md").write_text(
        model_card(
            spec,
            training_metrics=training_metrics,
            eval_metrics=eval_metrics,
            artifacts=model_artifacts,
            chat_template=template,
        ),
        encoding="utf-8",
    )

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        spec["outputs"]["modelRepoId"],
        exist_ok=True,
        private=spec["outputs"]["private"],
    )
    release_commit = api.upload_folder(
        folder_path=str(release_dir),
        repo_id=spec["outputs"]["modelRepoId"],
        commit_message=f"release({spec['jobId']}): validated frontier artifacts",
    )
    release_oid = getattr(release_commit, "oid", None) or getattr(
        release_commit, "commit_id", None
    )

    training_receipt = {
        "kind": "szl-frontier-training-receipt",
        "v": 2,
        "jobId": spec["jobId"],
        "specPayloadSha256": hashlib.sha256(payload).hexdigest(),
        "base": spec["base"],
        "dataset": spec["dataset"],
        "licenseEvidence": {
            "base": base_license_evidence,
            "dataset": dataset_license_evidence,
        },
        "recipe": spec["recipe"],
        "measured": training_metrics,
        "chatTemplate": template,
        "stack": stack,
        "checkpointBucket": bucket_evidence,
        "artifactManifestSha256": artifact_digest,
        "modelReleaseCommit": release_oid,
        "at": now_iso(),
    }
    signed_training = sign_receipt(training_receipt)
    training_body_sha = hashlib.sha256(
        base64.b64decode(signed_training["bodyBase64"])
    ).hexdigest()
    eval_receipt = {
        "kind": "szl-frontier-eval-receipt",
        "v": 2,
        "jobId": spec["jobId"],
        "trainingReceiptBodySha256": training_body_sha,
        "suite": spec["eval"],
        "measured": eval_metrics,
        "generationEvidence": generation_records,
        "reloadSmoke": export_smoke,
        "artifactManifestSha256": artifact_digest,
        "modelReleaseCommit": release_oid,
        "at": now_iso(),
    }
    signed_eval = sign_receipt(eval_receipt)

    for name, signed in (
        ("training_receipt.signed.json", signed_training),
        ("eval_receipt.signed.json", signed_eval),
    ):
        local = job_root / "receipts" / name
        write_json(local, signed)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=name,
            repo_id=spec["outputs"]["modelRepoId"],
            commit_message=f"receipt({spec['jobId']}): {name}",
        )
        upload_receipt(signed, name, spec)

    public_key = ROOT / "keys" / "laptop_pubkey.json"
    if public_key.exists():
        api.upload_file(
            path_or_fileobj=str(public_key),
            path_in_repo="owner_pubkey.json",
            repo_id=spec["outputs"]["modelRepoId"],
            commit_message=f"identity({spec['jobId']}): owner receipt key",
        )
    print(
        f"job {spec['jobId']} COMPLETE — validated artifacts + signed receipt chain uploaded; "
        f"release commit {release_oid or 'UNKNOWN'}"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: runjob_frontier.py <signed-job-spec.json>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
