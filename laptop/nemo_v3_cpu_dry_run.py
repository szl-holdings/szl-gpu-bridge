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

import hashlib
import math
import pathlib
import pickle
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
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}"
    "{{ message['role'] }} {{ message['content'] }}{{ eos_token }}"
    "{% endgeneration %}"
    "{% else %}"
    "{{ message['role'] }} {{ message['content'] }}{{ eos_token }}"
    "{% endif %}"
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


def _allow_pyarrow26_dataset_fingerprint() -> Callable[[], None]:
    """datasets 5.2 dill-pickles the Arrow table to fingerprint it.

    PyArrow 26 declares MonthDayNano as a builtin, and that pickle fails. When
    it does, hash the table shape instead so the dry-run can still train.
    """

    import datasets.arrow_dataset as arrow_dataset
    import datasets.fingerprint as fingerprint

    if getattr(arrow_dataset.generate_fingerprint, "_szl_arrow_fallback", False):
        return lambda: None
    original = arrow_dataset.generate_fingerprint

    def generate(dataset):
        try:
            return original(dataset)
        except (pickle.PicklingError, RecursionError):
            table = dataset._data
            payload = f"{table.schema}|{table.num_rows}|{table.nbytes}".encode()
            return hashlib.sha256(payload).hexdigest()

    generate._szl_arrow_fallback = True
    arrow_dataset.generate_fingerprint = generate
    fingerprint.generate_fingerprint = generate

    def restore() -> None:
        arrow_dataset.generate_fingerprint = original
        fingerprint.generate_fingerprint = original

    return restore


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

    restore_fingerprint = _allow_pyarrow26_dataset_fingerprint()
    try:
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
    finally:
        restore_fingerprint()
