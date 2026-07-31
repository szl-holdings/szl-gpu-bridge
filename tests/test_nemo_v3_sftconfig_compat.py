from __future__ import annotations

import copy
import importlib.metadata
import inspect
import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import frontier_job  # noqa: E402

ATTEMPT_13_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-13-reviewed.json"


class _Cuda:
    @staticmethod
    def is_bf16_supported() -> bool:
        return True


class _Torch:
    cuda = _Cuda()


def _recipe() -> dict[str, object]:
    spec = json.loads(ATTEMPT_13_PATH.read_text(encoding="utf-8"))
    return {
        **spec["recipe"],
        "packing": False,
        "packingStrategy": "ffd",
        "assistantOnlyLoss": True,
        "useRsLoRA": True,
    }


def _recording_config(*, strategy_name: str | None, both: bool = False):
    class RecordingConfig:
        calls: list[dict[str, object]] = []

        def __init__(self, **kwargs: object) -> None:
            type(self).calls.append(kwargs)
            self.values = kwargs

    names = [
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "learning_rate",
        "optim",
        "seed",
        "output_dir",
        "logging_steps",
        "save_strategy",
        "report_to",
        "bf16",
        "fp16",
        "warmup_ratio",
        "weight_decay",
        "lr_scheduler_type",
        "packing",
        "packing_strategy",
        "assistant_only_loss",
        "max_length",
        "max_seq_length",
    ]
    if strategy_name is not None:
        names.append(strategy_name)
    if both:
        names.extend(
            name
            for name in frontier_job._SFT_CONFIG_STRATEGY_ALIASES
            if name not in names
        )
    RecordingConfig.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in names]
        + [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)]
    )
    return RecordingConfig


class NemoV3SftConfigCompatibilityTests(unittest.TestCase):
    def test_alias_normalizes_to_exact_explicit_signature(self) -> None:
        values = {"evaluation_strategy": "epoch", "seed": 3407}
        for strategy_name in frontier_job._SFT_CONFIG_STRATEGY_ALIASES:
            with self.subTest(strategy_name=strategy_name):
                config = _recording_config(strategy_name=strategy_name)
                normalized = frontier_job._normalize_sft_config_kwargs(config, values)
                self.assertEqual(normalized, {strategy_name: "epoch", "seed": 3407})
                self.assertNotIn(
                    next(
                        name
                        for name in frontier_job._SFT_CONFIG_STRATEGY_ALIASES
                        if name != strategy_name
                    ),
                    normalized,
                )

    def test_both_value_aliases_are_ambiguous(self) -> None:
        config = _recording_config(strategy_name="eval_strategy")
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            frontier_job._normalize_sft_config_kwargs(
                config,
                {"eval_strategy": "epoch", "evaluation_strategy": "epoch"},
            )
        self.assertEqual(config.calls, [])

    def test_missing_strategy_value_fails_before_constructor(self) -> None:
        config = _recording_config(strategy_name="eval_strategy")
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            frontier_job._normalize_sft_config_kwargs(config, {"seed": 3407})
        self.assertEqual(config.calls, [])

    def test_unknown_or_ambiguous_signature_fails_before_constructor(self) -> None:
        for config in (
            _recording_config(strategy_name=None),
            _recording_config(strategy_name="eval_strategy", both=True),
        ):
            with self.subTest(signature=inspect.signature(config)):
                with self.assertRaisesRegex(RuntimeError, "signature must expose"):
                    frontier_job._build_sft_config(
                        config, _recipe(), _Torch(), pathlib.Path("output")
                    )
                self.assertEqual(config.calls, [])

    def test_unsupported_field_is_not_hidden_by_var_kwargs(self) -> None:
        config = _recording_config(strategy_name="eval_strategy")
        with self.assertRaisesRegex(RuntimeError, "unsupported fields"):
            frontier_job._normalize_sft_config_kwargs(
                config,
                {"evaluation_strategy": "epoch", "unknown_setting": True},
            )
        self.assertEqual(config.calls, [])

    def test_build_preserves_recipe_and_all_other_config_values(self) -> None:
        recipe = _recipe()
        original = copy.deepcopy(recipe)
        config_type = _recording_config(strategy_name="eval_strategy")
        with tempfile.TemporaryDirectory() as directory:
            config = frontier_job._build_sft_config(
                config_type, recipe, _Torch(), pathlib.Path(directory)
            )
        self.assertEqual(recipe, original)
        values = config.values
        self.assertEqual(values["eval_strategy"], "epoch")
        self.assertNotIn("evaluation_strategy", values)
        self.assertEqual(values["per_device_train_batch_size"], recipe["batchSize"])
        self.assertEqual(values["gradient_accumulation_steps"], recipe["gradAccum"])
        self.assertEqual(values["num_train_epochs"], recipe["epochs"])
        self.assertEqual(values["learning_rate"], recipe["learningRate"])
        self.assertEqual(values["optim"], recipe["optimizer"])
        self.assertEqual(values["seed"], recipe["seed"])
        self.assertEqual(values["max_length"], recipe["maxSeqLength"])
        self.assertEqual(values["max_seq_length"], recipe["maxSeqLength"])
        self.assertTrue(values["bf16"])
        self.assertFalse(values["fp16"])

    def test_attempt_13_science_license_and_input_hashes_are_unchanged(self) -> None:
        spec = json.loads(ATTEMPT_13_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            spec["base"],
            {
                "repoId": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                "revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
                "licenseId": "nvidia-nemotron-open-model-license",
                "licenseAcknowledgement": (
                    "I accept the pinned NVIDIA Nemotron Open Model License, "
                    "preserve NVIDIA attribution, and keep the upstream base license "
                    "distinct from the downstream Apache-2.0 adapter source."
                ),
                "trustRemoteCode": True,
            },
        )
        self.assertEqual(
            spec["dataset"]["train"]["sha256"],
            "a81e5742d8146dfb67a0754e45b578765b5c6212ff6725b8157035b49c0e1c9a",
        )
        self.assertEqual(
            [item["sha256"] for item in spec["dataset"]["holdouts"]],
            [
                "caeb07c94929c24a47fd12f35cbc9021523308dc9fcc684bd444ffcf4a367b0d",
                "1b8578051b7b829595493615bdf11e24d01a1837a23d6191c8e88e21cce990ac",
                "1b23d20406eb96d3c58741ac57c802eb723a19802c7f83b8500b99a71d15c35f",
            ],
        )
        self.assertEqual(
            spec["dataset"]["preregistration"]["sha256"],
            "41a27921ff3a377442e2cf7b4ffe569324de73cedb56b2485e50a6e8057cfacd",
        )

    @unittest.skipUnless(
        os.environ.get("SZL_PINNED_IMAGE_PROBE") == "1",
        "requires the pinned immutable GPU image",
    )
    def test_pinned_unsloth_trl_interface_offline(self) -> None:
        for secret_name in (
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HF_ORG_TOKEN",
            "HF_ORG_TOKEN1",
        ):
            self.assertNotIn(secret_name, os.environ)

        import unsloth  # noqa: F401
        from trl import SFTConfig

        self.assertEqual(importlib.metadata.version("trl"), "0.23.1")
        self.assertEqual(importlib.metadata.version("transformers"), "4.57.6")
        signature = inspect.signature(SFTConfig)
        self.assertIn("eval_strategy", signature.parameters)
        self.assertNotIn("evaluation_strategy", signature.parameters)
        with self.assertRaisesRegex(TypeError, "evaluation_strategy"):
            SFTConfig(evaluation_strategy="epoch")

        with tempfile.TemporaryDirectory() as directory:
            config = frontier_job._build_sft_config(
                SFTConfig, _recipe(), _Torch(), pathlib.Path(directory)
            )
        self.assertEqual(config.eval_strategy, "epoch")
        self.assertEqual(config.per_device_train_batch_size, 1)
        self.assertEqual(config.gradient_accumulation_steps, 8)
        self.assertEqual(config.num_train_epochs, 2)
        self.assertEqual(config.learning_rate, 0.0001)
        self.assertEqual(config.optim, "paged_adamw_8bit")
        self.assertEqual(config.seed, 3407)
        self.assertEqual(config.max_length, 768)
        self.assertEqual(config.max_seq_length, 768)
        self.assertFalse(config.packing)
        self.assertEqual(config.packing_strategy, "ffd")
        self.assertTrue(config.assistant_only_loss)


if __name__ == "__main__":
    unittest.main(verbosity=2)
