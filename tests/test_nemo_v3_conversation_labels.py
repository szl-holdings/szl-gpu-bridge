from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import runjob_nemo_v3  # noqa: E402
from frontier_job import _build_sft_config, _build_sft_trainer  # noqa: E402

ATTEMPT_14_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-14-reviewed.json"


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": "assistant"},
    ]


class _Dataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.column_names = sorted({key for row in rows for key in row})

    def map(self, function, *, batched: bool, desc: str):
        if batched or desc != "Apply exact assistant-only labels":
            raise AssertionError("unexpected dataset mapping contract")
        return _Dataset([{**row, **function(row)} for row in self.rows])


class _Tokenizer:
    def __init__(
        self,
        *,
        context_ids: list[int] | None = None,
        full_ids: list[int] | None = None,
    ) -> None:
        self.context_ids = context_ids or [10, 11, 12]
        self.full_ids = full_ids or [10, 11, 12, 20, 21]
        self.calls: list[tuple[list[dict[str, str]], bool, bool]] = []

    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool
    ) -> list[int]:
        self.calls.append((messages, tokenize, add_generation_prompt))
        if not tokenize or add_generation_prompt:
            raise AssertionError("unexpected chat-template options")
        return list(self.context_ids if len(messages) == 2 else self.full_ids)


class NemoV3ConversationLabelTests(unittest.TestCase):
    def test_exact_context_prefix_is_masked_and_assistant_region_is_supervised(
        self,
    ) -> None:
        rows = [{"messages": _messages(), "record_id": "train:1"}]
        original = copy.deepcopy(rows)
        tokenizer = _Tokenizer()
        prepared = runjob_nemo_v3._prepare_nemo_assistant_labels(
            _Dataset(rows), tokenizer, max_length=8
        )

        self.assertEqual(rows, original)
        self.assertEqual(len(tokenizer.calls), 2)
        row = prepared.rows[0]
        self.assertEqual(row["input_ids"], [10, 11, 12, 20, 21])
        self.assertEqual(row["attention_mask"], [1, 1, 1, 1, 1])
        self.assertEqual(row["labels"], [-100, -100, -100, 20, 21])
        self.assertEqual(row["messages"], original[0]["messages"])

    def test_wrong_role_order_or_extra_assistant_turn_fails_closed(self) -> None:
        cases = (
            _messages()[:-1],
            [*_messages(), {"role": "assistant", "content": "extra"}],
            [
                {"role": "system", "content": "system"},
                {"role": "assistant", "content": "assistant"},
                {"role": "user", "content": "user"},
            ],
        )
        for messages in cases:
            with self.subTest(roles=[message["role"] for message in messages]):
                with self.assertRaisesRegex(RuntimeError, "three messages|system"):
                    runjob_nemo_v3._prepare_nemo_assistant_labels(
                        _Dataset([{"messages": messages}]),
                        _Tokenizer(),
                        max_length=8,
                    )

    def test_prefix_drift_empty_completion_and_length_overflow_fail_closed(
        self,
    ) -> None:
        cases = (
            (_Tokenizer(context_ids=[10, 99], full_ids=[10, 11, 20]), 8, "prefix"),
            (_Tokenizer(context_ids=[10, 11], full_ids=[10, 11]), 8, "supervised"),
            (_Tokenizer(context_ids=[10, 11], full_ids=[10, 11, 20]), 2, "maximum"),
        )
        for tokenizer, max_length, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    runjob_nemo_v3._prepare_nemo_assistant_labels(
                        _Dataset([{"messages": _messages()}]),
                        tokenizer,
                        max_length=max_length,
                    )

    def test_non_integer_or_nested_tokenization_fails_closed(self) -> None:
        for invalid in ([], [1, "2"], [[1, 2]]):
            with self.subTest(invalid=invalid):
                tokenizer = _Tokenizer(context_ids=invalid, full_ids=[1, 2, 3])
                tokenizer.context_ids = invalid
                with self.assertRaisesRegex(RuntimeError, "flat integer list"):
                    runjob_nemo_v3._prepare_nemo_assistant_labels(
                        _Dataset([{"messages": _messages()}]),
                        tokenizer,
                        max_length=8,
                    )

    def test_messages_column_and_max_length_contract_are_required(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "messages column"):
            runjob_nemo_v3._prepare_nemo_assistant_labels(
                _Dataset([{"text": "not conversational"}]),
                _Tokenizer(),
                max_length=8,
            )
        for invalid in (0, True, 1.5):
            with self.subTest(max_length=invalid):
                with self.assertRaisesRegex(RuntimeError, "maximum length"):
                    runjob_nemo_v3._prepare_nemo_assistant_labels(
                        _Dataset([{"messages": _messages()}]),
                        _Tokenizer(),
                        max_length=invalid,
                    )

    def test_preparation_precedes_split_and_trainer_construction(self) -> None:
        source = (ROOT / "laptop" / "runjob_nemo_v3.py").read_text(encoding="utf-8")
        main_source = source[source.index("def main(spec_path: str)") :]
        prepare = main_source.index("_prepare_nemo_assistant_labels(")
        self.assertLess(prepare, main_source.index("train_test_split("))
        self.assertLess(prepare, main_source.index("trainer = _build_sft_trainer("))

    @unittest.skipUnless(
        os.environ.get("SZL_PINNED_CONVERSATION_PROBE") == "1",
        "requires the pinned immutable GPU image, snapshot, and training file",
    )
    def test_exact_snapshot_builds_trainer_offline_with_assistant_labels(self) -> None:
        for name in runjob_nemo_v3.SENSITIVE_ENVIRONMENT_NAMES:
            self.assertFalse(os.environ.get(name))
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

        import torch
        import unsloth
        from datasets import Dataset
        from transformers import PreTrainedTokenizerBase
        from trl import SFTConfig, SFTTrainer

        spec = json.loads(ATTEMPT_14_PATH.read_text(encoding="utf-8"))
        train_path = pathlib.Path(os.environ["SZL_PINNED_TRAIN_PATH"])
        rows = [
            json.loads(line)
            for line in train_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 36)
        snapshot = runjob_nemo_v3._verified_offline_tokenizer_snapshot(spec)
        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=spec["base"]["repoId"],
            tokenizer_name=str(snapshot),
            revision=spec["base"]["revision"],
            max_seq_length=spec["recipe"]["maxSeqLength"],
            load_in_4bit=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        runjob_nemo_v3._require_loaded_tokenizer(
            tokenizer, PreTrainedTokenizerBase, snapshot
        )
        runjob_nemo_v3._require_nemo_model_materialization(
            model,
            spec["recipe"]["targetModules"],
            phase="pinned pre-adapter",
        )
        model = unsloth.FastLanguageModel.get_peft_model(
            model,
            r=spec["recipe"]["loraR"],
            lora_alpha=spec["recipe"]["loraAlpha"],
            lora_dropout=spec["recipe"]["loraDropout"],
            bias="none",
            target_modules=spec["recipe"]["targetModules"],
            use_gradient_checkpointing="unsloth",
            random_state=spec["recipe"]["seed"],
            max_seq_length=spec["recipe"]["maxSeqLength"],
            use_rslora=True,
            loftq_config=None,
        )
        prepared = runjob_nemo_v3._prepare_nemo_assistant_labels(
            Dataset.from_list(rows),
            tokenizer,
            max_length=spec["recipe"]["maxSeqLength"],
        )
        self.assertEqual(len(prepared), 36)
        for row in prepared:
            self.assertEqual(len(row["input_ids"]), len(row["labels"]))
            self.assertTrue(any(value != -100 for value in row["labels"]))
        split = prepared.train_test_split(test_size=0.15, seed=spec["recipe"]["seed"])
        recipe = {
            **spec["recipe"],
            "packing": False,
            "packingStrategy": "ffd",
            "assistantOnlyLoss": True,
            "useRsLoRA": True,
        }
        with tempfile.TemporaryDirectory() as output:
            config = _build_sft_config(SFTConfig, recipe, torch, pathlib.Path(output))
            runjob_nemo_v3._materialize_nemo_lm_head_for_trainer(model)
            trainer = _build_sft_trainer(
                SFTTrainer,
                model=model,
                tokenizer=tokenizer,
                train_dataset=split["train"],
                eval_dataset=split["test"],
                callbacks=[],
                config=config,
            )
        self.assertIsNotNone(trainer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
