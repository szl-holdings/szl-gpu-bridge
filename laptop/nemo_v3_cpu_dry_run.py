#!/usr/bin/env python3
"""CPU dry-run for the Nemotron trainer boundary.

This path never dispatches a GPU job, never starts a hosted training job, and
never reads a DSSE signing key.  Verification is an injected callback.  The
model is a randomly initialized tiny Llama, and training is one short CPU
epoch.  The checks are the ones that stopped governed attempts before
``train()``: SFTConfig aliases, the tokenizer and its chat template, dtype and
4-bit policy, dataset field mapping, meta-tensor materialization, and a mocked
DSSE payload check.
"""

from __future__ import annotations

import math
import pathlib
import tempfile
from collections.abc import Callable
from typing import Any

from frontier_job import _build_sft_config, _build_sft_trainer
from runjob_nemo_v3 import (
    _materialize_nemo_lm_head_for_trainer,
    _prepare_nemo_assistant_labels,
    _unsloth_load_kwargs,
)

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }} {{ message['content'] }}{{ eos_token }}"
    "{% endfor %}"
)


class DsseVerificationError(RuntimeError):
    """Injected DSSE verification refused the synthetic envelope."""


def verify_synthetic_trainer_envelope(
    envelope: dict[str, Any], verify: Callable[[dict[str, Any]], dict[str, Any]]
) -> dict[str, Any]:
    """Validate a trainer payload after an injected, keyless DSSE check."""

    if not callable(verify):
        raise DsseVerificationError(
            "DSSE verification must be injected; signing keys are not read"
        )
    payload = verify(envelope)
    if not isinstance(payload, dict):
        raise DsseVerificationError("DSSE payload is not an object")
    recipe = payload.get("recipe")
    dataset = payload.get("dataset")
    if not isinstance(recipe, dict) or not isinstance(dataset, dict):
        raise DsseVerificationError("DSSE payload is missing recipe or dataset")
    if dataset.get("field") not in {"messages", "conversations"}:
        raise DsseVerificationError("dataset field mapping is not a chat column")
    length = recipe.get("maxSeqLength")
    if isinstance(length, bool) or not isinstance(length, int) or length < 8:
        raise DsseVerificationError("recipe maxSeqLength is not trainable")
    if recipe.get("packingStrategy") not in {"ffd", "bfd", "wrapped"}:
        raise DsseVerificationError("recipe packing strategy is not admitted")
    return payload


def _tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = [
        "system",
        "user",
        "assistant",
        "hello",
        "question",
        "answer",
        "one",
        "two",
    ]
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
    for word in words:
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(WordLevel(vocab, unk_token="<pad>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<pad>",
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def _require_chat_template(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        raise RuntimeError("base tokenizer has no chat template")
    return template


def _tiny_model(vocab_size: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    return LlamaForCausalLM(config)


class AlignDevicesHook:
    """Stand-in for the Accelerate hook Unsloth leaves on a frozen lm_head."""

    def __init__(self, backing: Any) -> None:
        self.execution_device = "0"
        self.offload = True
        self.weights_map = {"weight": backing}


def _park_lm_head_on_meta(model: Any) -> None:
    import torch

    module = model.get_output_embeddings()
    backing = module.weight.detach().to("cpu").clone()
    backing.requires_grad_(False)
    meta = torch.empty_like(module.weight, device="meta")
    module.weight = torch.nn.Parameter(meta, requires_grad=False)
    module._hf_hook = AlignDevicesHook(backing)
    model.hf_device_map = {"lm_head": "cpu"}


def _rows() -> list[dict[str, Any]]:
    answers = ("one", "two", "one", "two")

    def row(record_id: str, answer: str) -> dict[str, Any]:
        return {
            "record_id": record_id,
            "messages": [
                {"role": "system", "content": "system hello"},
                {"role": "user", "content": "user question"},
                {"role": "assistant", "content": f"assistant {answer}"},
            ],
        }

    return [
        row(f"dry-{index}", answer) for index, answer in enumerate(answers, start=1)
    ]


def _recipe(max_length: int) -> dict[str, Any]:
    return {
        "batchSize": 1,
        "gradAccum": 1,
        "epochs": 1,
        "learningRate": 1e-4,
        "optimizer": "adamw_torch",
        "seed": 7,
        "warmupRatio": 0.0,
        "weightDecay": 0.0,
        "lrSchedulerType": "linear",
        "packing": False,
        "packingStrategy": "ffd",
        "assistantOnlyLoss": True,
        "maxSeqLength": max_length,
    }


def run_cpu_dry_run(
    envelope: dict[str, Any],
    verify: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Construct the real TRL 0.23.1 trainer and take a few CPU steps."""

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    payload = verify_synthetic_trainer_envelope(envelope, verify)
    quant_error = None
    try:
        _unsloth_load_kwargs(torch, load_in_4bit=True)
    except RuntimeError as exc:
        quant_error = str(exc)
    cpu_load = _unsloth_load_kwargs(torch, load_in_4bit=False)
    if quant_error is None:
        raise RuntimeError("CPU 4-bit quantization was accepted")
    if cpu_load["load_in_4bit"] or cpu_load["dtype"] != torch.float32:
        raise RuntimeError("CPU load policy is not full-precision")

    tokenizer = _tiny_tokenizer()
    template = _require_chat_template(tokenizer)
    model = _tiny_model(max(len(tokenizer), 32))
    _park_lm_head_on_meta(model)
    if not bool(model.get_output_embeddings().weight.is_meta):
        raise RuntimeError("dry-run did not plant a meta lm_head")
    _materialize_nemo_lm_head_for_trainer(model)
    materialized = model.get_output_embeddings().weight
    if bool(materialized.is_meta) or materialized.device.type != "cpu":
        raise RuntimeError("lm_head was not materialized on CPU")
    # The offload map was only evidence for the meta placeholder.  Training
    # sees one CPU device after the frozen weight has real storage.
    if hasattr(model, "hf_device_map"):
        del model.hf_device_map
    hook_owner = model.get_output_embeddings()
    if hasattr(hook_owner, "_hf_hook"):
        del hook_owner._hf_hook

    prepared = _prepare_nemo_assistant_labels(
        Dataset.from_list(_rows()), tokenizer, max_length=32
    )
    columns = set(prepared.column_names)
    if not {"input_ids", "labels", "messages"}.issubset(columns):
        raise RuntimeError("assistant-label dataset is missing trainer columns")
    supervised = 0
    for row in prepared:
        if len(row["input_ids"]) != len(row["labels"]):
            raise RuntimeError("label width does not match input ids")
        if not any(value != -100 for value in row["labels"]):
            raise RuntimeError("assistant region was fully masked")
        supervised += sum(value != -100 for value in row["labels"])
    split = prepared.train_test_split(test_size=0.5, seed=7)

    with tempfile.TemporaryDirectory() as directory:
        config = _build_sft_config(
            SFTConfig, _recipe(32), torch, pathlib.Path(directory)
        )
        if config.eval_strategy != "epoch":
            raise RuntimeError("evaluation strategy was not bound to eval_strategy")
        if getattr(config, "max_seq_length", None) not in (None, config.max_length):
            raise RuntimeError("retired max_seq_length survived construction")
        if config.max_length != 32 or config.packing_strategy != "bfd":
            raise RuntimeError("sequence length or packing strategy was not bound")
        if config.bf16 or config.fp16:
            raise RuntimeError("CPU dry-run requested mixed precision")
        trainer = _build_sft_trainer(
            SFTTrainer,
            model=model,
            tokenizer=tokenizer,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            callbacks=[],
            config=config,
        )
        trained = trainer.train()
        loss = float(trained.training_loss)

    if not math.isfinite(loss):
        raise RuntimeError("CPU dry-run loss is not finite")
    return {
        "training_loss": loss,
        "global_steps": int(trainer.state.global_step),
        "chat_template_present": bool(template),
        "supervised_tokens": supervised,
        "eval_strategy": config.eval_strategy,
        "max_length": config.max_length,
        "packing_strategy": config.packing_strategy,
        "bf16": bool(config.bf16),
        "fp16": bool(config.fp16),
        "cpu_dtype": str(cpu_load["dtype"]),
        "quantization_refused": quant_error,
        "dataset_field": payload["dataset"]["field"],
        "lm_head_device": materialized.device.type,
        "gpu_job_launched": False,
    }


# ---------------------------------------------------------------------------
# Nemotron-H (hybrid Mamba-2 + attention) CPU dry-run on the same boundary.
# ---------------------------------------------------------------------------

NEMOTRON_TEST_TEMPLATE = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined "
    "else True -%}"
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}assistant "
    "{% if enable_thinking %}think {% else %}nothink {% endif %}"
    "{% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}"
    "{% else %}{{ message['role'] }} {{ message['content'] }}{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant "
    "{% if enable_thinking %}think {% else %}nothink {% endif %}{% endif %}"
)
NANO_4B = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
NANO_4B_REVISION = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"


def _tiny_nemotron_h(vocab_size: int):
    """Random-init native NemotronH with Mamba, MLP and attention mixers."""

    from transformers import NemotronHConfig, NemotronHForCausalLM

    config = NemotronHConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        layers_block_type=["mamba", "mlp", "attention", "mlp"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        ssm_state_size=8,
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=1,
        chunk_size=8,
        expand=2,
        conv_kernel=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    return NemotronHForCausalLM(config)


def _nemotron_tokenizer(tokenizer_dir: str | None):
    import nemotron_h

    entry = nemotron_h.nemotron_base(NANO_4B, NANO_4B_REVISION)
    if tokenizer_dir:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        return tokenizer, entry, "pinned-nano-4b-tokenizer"
    tokenizer = _tiny_tokenizer()
    tokenizer.chat_template = NEMOTRON_TEST_TEMPLATE
    fixture = {
        **entry,
        "chat_template_sha256": nemotron_h.chat_template_sha256(tokenizer),
    }
    return tokenizer, fixture, "tiny-fixture-tokenizer"


def _nemotron_spec(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "base": {
            "repoId": NANO_4B,
            "revision": NANO_4B_REVISION,
            "licenseId": entry["license_id"],
            "licenseAcknowledgement": (
                "Base weights are used under the NVIDIA Nemotron Open Model "
                f"License. {entry['notice']}"
            ),
            "trustRemoteCode": False,
        }
    }


def _refusal(fn: Callable[[], Any]) -> str:
    try:
        fn()
    except RuntimeError as exc:
        return str(exc)
    raise RuntimeError("a stripped guardrail or mismatched target was accepted")


def run_nemotron_cpu_dry_run(tokenizer_dir: str | None = None) -> dict[str, Any]:
    """Tiny Nemotron-H through the Nemo v3 label, target and trainer path."""

    import importlib.util

    import nemotron_h
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from runjob_nemo_v3 import (
        _nemotron_gradient_probe,
        _nemotron_guardrails,
        _nemotron_pre_load_gate,
        _nemotron_signed_targets,
    )
    from trl import SFTConfig, SFTTrainer

    tokenizer, entry, source = _nemotron_tokenizer(tokenizer_dir)
    spec = _nemotron_spec(entry)
    if _nemotron_pre_load_gate(spec)["revision"] != NANO_4B_REVISION:
        raise RuntimeError("pre-load gate did not bind the pinned revision")
    refusals = {
        "wrong_license": _refusal(
            lambda: _nemotron_pre_load_gate(
                {"base": {**spec["base"], "licenseId": "apache-2.0"}}
            )
        ),
        "notice_stripped": _refusal(
            lambda: _nemotron_pre_load_gate(
                {
                    "base": {
                        **spec["base"],
                        "licenseAcknowledgement": "Used under an NVIDIA licence.",
                    }
                }
            )
        ),
        "wrong_revision": _refusal(
            lambda: _nemotron_pre_load_gate(
                {"base": {**spec["base"], "revision": "f" * 40}}
            )
        ),
        "replaced_template": _refusal(
            lambda: _nemotron_guardrails(
                spec, {**entry, "chat_template_sha256": "0" * 64}, tokenizer
            )
        ),
    }
    guard = _nemotron_guardrails(spec, entry, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _tiny_nemotron_h(max(len(tokenizer), 32))
    signed = ["in_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    refusals["dense_targets_on_hybrid"] = _refusal(
        lambda: _nemotron_signed_targets(
            model,
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            load_in_4bit=False,
        )
    )
    evidence = _nemotron_signed_targets(model, signed, load_in_4bit=False)
    reasoning = {
        "control": entry["reasoning_control"],
        "enable_thinking": False,
    }
    prompt = [{"role": "user", "content": "user question"}]
    on = nemotron_h.render_chat_text(
        tokenizer,
        prompt,
        reasoning={**reasoning, "enable_thinking": True},
        add_generation_prompt=True,
    )
    off = nemotron_h.render_chat_text(
        tokenizer, prompt, reasoning=reasoning, add_generation_prompt=True
    )
    if on == off:
        raise RuntimeError("enable_thinking did not change the rendered prompt")

    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=evidence["lora_targets"],
            task_type="CAUSAL_LM",
        ),
    )
    prepared = _prepare_nemo_assistant_labels(
        Dataset.from_list(_rows()), tokenizer, max_length=64, reasoning=reasoning
    )
    split = prepared.train_test_split(test_size=0.5, seed=7)
    probe = _nemotron_gradient_probe(model, split["train"])
    before = {
        n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n
    }
    with tempfile.TemporaryDirectory() as directory:
        config = _build_sft_config(
            SFTConfig, _recipe(64), torch, pathlib.Path(directory)
        )
        trainer = _build_sft_trainer(
            SFTTrainer,
            model=model,
            tokenizer=tokenizer,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            callbacks=[],
            config=config,
        )
        trained = trainer.train()
        adapter = pathlib.Path(directory) / "adapter"
        model.save_pretrained(str(adapter))
        notice = pathlib.Path(nemotron_h.write_nvidia_notice(adapter, entry))
        notice_text = notice.read_text(encoding="utf-8")
        saved = sorted(p.name for p in adapter.iterdir())
    updated = sorted(
        {
            n.split(".lora_B")[0].rsplit(".", 1)[-1]
            for n, p in model.named_parameters()
            if n in before and not torch.equal(before[n], p.detach())
        }
    )
    loss = float(trained.training_loss)
    if not math.isfinite(loss):
        raise RuntimeError("Nemotron-H CPU dry-run loss is not finite")
    if set(updated) != set(evidence["lora_targets"]):
        raise RuntimeError(f"LoRA targets did not all train: {updated}")
    return {
        "global_steps": int(trainer.state.global_step),
        "training_loss": loss,
        "lora_targets": evidence["lora_targets"],
        "lora_targets_updated": updated,
        "layout": evidence["layout"],
        "fused_mamba_kernel": evidence["fused_mamba_kernel"],
        "mamba_ssm_installed": importlib.util.find_spec("mamba_ssm") is not None,
        "mamba_path": "torch reference (CPU)",
        "gradient_probe": probe,
        "guardrails": guard,
        "refusals": refusals,
        "prompt_tail_thinking_on": on[-24:],
        "prompt_tail_thinking_off": off[-24:],
        "tokenizer_source": source,
        "notice_in_artifacts": entry["notice"] in notice_text,
        "saved_files": saved,
        "gpu_job_launched": False,
    }


if __name__ == "__main__":
    import json
    import os
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    result = run_nemotron_cpu_dry_run(os.environ.get("SZL_NEMOTRON_TOKENIZER_DIR"))
    print(json.dumps(result, indent=2, default=str))
