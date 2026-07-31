from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import runjob_nemo_v3  # noqa: E402

ATTEMPT_14_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-14-reviewed.json"


class _Device:
    def __init__(self, kind: str) -> None:
        self.type = kind


class _Tensor:
    def __init__(
        self,
        *,
        device: str,
        is_meta: bool,
        requires_grad: bool = False,
        shape: tuple[int, ...] = (8, 4),
        dtype: str = "bfloat16",
    ) -> None:
        self.device = _Device(device)
        self.dtype = dtype
        self.is_meta = is_meta
        self.requires_grad = requires_grad
        self.shape = shape


class AlignDevicesHook:
    def __init__(self, backing: _Tensor | None) -> None:
        self.execution_device = 0
        self.offload = True
        self.weights_map = {} if backing is None else {"weight": backing}


class _Module:
    def __init__(self, weight: _Tensor, backing: _Tensor | None = None) -> None:
        self.weight = weight
        if backing is not None or weight.is_meta:
            self._hf_hook = AlignDevicesHook(backing)


class _Model:
    def __init__(
        self,
        *,
        backing: _Tensor | None = None,
        lm_head: _Tensor | None = None,
        lm_head_assignment: str = "cpu",
        target: _Tensor | None = None,
    ) -> None:
        self.target = target or _Tensor(device="cuda", is_meta=False)
        self.output = _Module(
            lm_head or _Tensor(device="meta", is_meta=True),
            backing or _Tensor(device="cpu", is_meta=False),
        )
        self.hf_device_map = {
            "backbone.layers.0": 0,
            "lm_head": lm_head_assignment,
        }

    def get_output_embeddings(self) -> _Module:
        return self.output

    def named_modules(self):
        return [
            ("", self),
            ("backbone.layers.0.q_proj", _Module(self.target)),
            ("lm_head", self.output),
        ]

    def named_parameters(self):
        return [
            ("backbone.layers.0.q_proj.weight", self.target),
            ("lm_head.weight", self.output.weight),
        ]


class NemoV3MetaTensorCompatibilityTests(unittest.TestCase):
    def test_exact_cpu_offload_backing_is_admitted_and_materialized(self) -> None:
        model = _Model()
        observed = runjob_nemo_v3._require_nemo_model_materialization(
            model, ["q_proj"], phase="test"
        )
        self.assertEqual(observed, ["lm_head.weight"])

        calls: list[tuple[object, str, str, object]] = []

        def setter(module, name, device, *, value):
            calls.append((module, name, device, value))
            module.weight = value

        runjob_nemo_v3._materialize_nemo_lm_head_for_trainer(
            model, tensor_setter=setter
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], model.output)
        self.assertEqual(calls[0][1:3], ("weight", "cpu"))
        self.assertIs(model.output.weight, calls[0][3])
        self.assertFalse(model.output.weight.is_meta)
        self.assertEqual(model.output.weight.device.type, "cpu")

    def test_meta_trainable_or_target_parameter_fails_closed(self) -> None:
        for target in (
            _Tensor(device="meta", is_meta=True),
            _Tensor(device="meta", is_meta=True, requires_grad=True),
        ):
            with self.subTest(requires_grad=target.requires_grad):
                model = _Model(target=target)
                with self.assertRaisesRegex(RuntimeError, "LoRA target|trainable"):
                    runjob_nemo_v3._require_nemo_model_materialization(
                        model, ["q_proj"], phase="test"
                    )

    def test_missing_or_tampered_offload_backing_fails_closed(self) -> None:
        cases = (
            _Model(backing=None),
            _Model(backing=_Tensor(device="meta", is_meta=True)),
            _Model(backing=_Tensor(device="cuda", is_meta=False)),
            _Model(backing=_Tensor(device="cpu", is_meta=False, shape=(7, 4))),
        )
        cases[0].output._hf_hook.weights_map.clear()
        for model in cases:
            with self.subTest(backing=model.output._hf_hook.weights_map):
                with self.assertRaises(RuntimeError):
                    runjob_nemo_v3._require_nemo_model_materialization(
                        model, ["q_proj"], phase="test"
                    )

    def test_mutable_or_ambiguous_offload_assignment_fails_before_setter(self) -> None:
        for model in (
            _Model(lm_head_assignment="disk"),
            _Model(backing=_Tensor(device="cpu", is_meta=False, requires_grad=True)),
        ):
            called = False

            def setter(*args, **kwargs):
                nonlocal called
                called = True

            with self.subTest(device_map=model.hf_device_map):
                with self.assertRaises(RuntimeError):
                    runjob_nemo_v3._materialize_nemo_lm_head_for_trainer(
                        model, tensor_setter=setter
                    )
                self.assertFalse(called)

    def test_missing_target_module_fails_closed(self) -> None:
        model = _Model()
        with self.assertRaisesRegex(RuntimeError, "target modules are absent"):
            runjob_nemo_v3._require_nemo_model_materialization(
                model, ["k_proj"], phase="test"
            )

    def test_materialization_precedes_trainer_construction(self) -> None:
        source = (ROOT / "laptop" / "runjob_nemo_v3.py").read_text(encoding="utf-8")
        main_source = source[source.index("def main(spec_path: str)") :]
        self.assertLess(
            main_source.index("_materialize_nemo_lm_head_for_trainer(model)"),
            main_source.index("trainer = _build_sft_trainer("),
        )

    def test_attempt_14_science_license_tokenizer_and_inputs_are_unchanged(
        self,
    ) -> None:
        spec = json.loads(ATTEMPT_14_PATH.read_text(encoding="utf-8"))
        before = copy.deepcopy(spec)
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
        self.assertEqual(spec, before)

    @unittest.skipUnless(
        os.environ.get("SZL_PINNED_META_PROBE") == "1",
        "requires the pinned immutable GPU image and snapshot",
    )
    def test_pinned_snapshot_offload_boundary_offline(self) -> None:
        for name in runjob_nemo_v3.SENSITIVE_ENVIRONMENT_NAMES:
            self.assertFalse(os.environ.get(name))
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

        import unsloth

        spec = json.loads(ATTEMPT_14_PATH.read_text(encoding="utf-8"))
        snapshot = runjob_nemo_v3._verified_offline_tokenizer_snapshot(spec)
        model, _ = unsloth.FastLanguageModel.from_pretrained(
            model_name=spec["base"]["repoId"],
            tokenizer_name=str(snapshot),
            revision=spec["base"]["revision"],
            max_seq_length=spec["recipe"]["maxSeqLength"],
            load_in_4bit=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        before = runjob_nemo_v3._require_nemo_model_materialization(
            model,
            spec["recipe"]["targetModules"],
            phase="pinned pre-adapter",
        )
        self.assertEqual(len(before), 26)
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
        runjob_nemo_v3._materialize_nemo_lm_head_for_trainer(model)
        after = runjob_nemo_v3._require_nemo_model_materialization(
            model,
            spec["recipe"]["targetModules"],
            phase="pinned pre-trainer",
        )
        self.assertEqual(len(after), 25)
        self.assertFalse(model.get_output_embeddings().weight.is_meta)
        self.assertEqual(model.get_output_embeddings().weight.device.type, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
