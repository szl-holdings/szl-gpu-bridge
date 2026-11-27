"""Nemotron-H detection, license preflight, caps, pool hooks, and CPU dry-run."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

from nemo_v3_contract import (  # noqa: E402
    LICENSE_ACK_MIN_CHARS,
    NEMOTRON_OPEN_MODEL_LICENSE_ID,
)

NANO_4B = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
NANO_9B = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
MODULES = [
    "in_proj",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
]


def load_nemotron():
    path = ROOT / "tools" / "salad" / "nemotron_h.py"
    spec = importlib.util.spec_from_file_location("nemotron_h_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nemotron = load_nemotron()


class _Config:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class _Model:
    def __init__(self, model_type):
        self.config = _Config(model_type=model_type, architectures=[])


class NemotronHTrainerTests(unittest.TestCase):
    def test_detected_modules_match_nano_patterns(self):
        self.assertEqual(len(NANO_4B), 42)
        self.assertEqual(len(NANO_9B), 56)
        for pattern in (NANO_4B, NANO_9B):
            observed = nemotron.lora_target_modules(
                _Config(model_type="nemotron_h", hybrid_override_pattern=pattern)
            )
            self.assertEqual(observed, MODULES)
        native = _Config(
            model_type="nemotron_h",
            layers_block_type=["linear_attention", "full_attention", "mlp"],
        )
        custom = _Config(
            model_type="nemotron_h",
            layers_block_type=["mamba", "attention", "mlp"],
        )
        self.assertEqual(nemotron.lora_target_modules(native), MODULES)
        self.assertEqual(nemotron.lora_target_modules(custom), MODULES)
        agreed = _Config(
            model_type="nemotron_h",
            hybrid_override_pattern="M*-",
            layers_block_type=["linear_attention", "full_attention", "mlp"],
        )
        self.assertEqual(nemotron.lora_target_modules(agreed), MODULES)

    def test_unknown_architecture_fails_closed(self):
        refused = (
            _Config(model_type="llama", hybrid_override_pattern="M*-"),
            _Config(model_type="nemotron_h", hybrid_override_pattern="ME"),
            _Config(model_type="nemotron_h", layers_block_type=["moe"]),
            _Config(model_type="nemotron_h"),
            _Config(
                model_type="nemotron_h",
                hybrid_override_pattern="M",
                layers_block_type=["mlp"],
            ),
        )
        for config in refused:
            with self.subTest(config=config.__dict__):
                with self.assertRaises(nemotron.NemotronHError):
                    nemotron.lora_target_modules(config)
        self.assertEqual(
            nemotron.lora_targets_for_model(_Model("llama"), ["q_proj"]),
            ["q_proj"],
        )

    def test_thinking_toggle_is_the_system_prompt(self):
        enabled = nemotron.messages_for_thinking(
            [{"role": "user", "content": "hello"}], True
        )
        disabled = nemotron.messages_for_thinking(
            [{"role": "user", "content": "hello"}], False
        )
        self.assertEqual(enabled[0]["content"], "/think")
        self.assertEqual(disabled[0]["content"], "/no_think")
        kept = nemotron.messages_for_thinking(
            [
                {"role": "system", "content": "Be precise"},
                {"role": "user", "content": "hello"},
            ],
            False,
        )
        self.assertEqual(kept[0]["content"], "Be precise\n/no_think")
        with self.assertRaises(nemotron.NemotronHError):
            nemotron.messages_for_thinking(
                [
                    {"role": "system", "content": "keep /think inside"},
                    {"role": "user", "content": "x"},
                ],
                False,
            )
        with self.assertRaises(nemotron.NemotronHError):
            nemotron.messages_for_thinking([{"role": "user", "content": "x"}], 1)

    def test_license_preflight_requires_id_acknowledgement_and_notice(self):
        self.assertEqual(nemotron.LICENSE_ID, NEMOTRON_OPEN_MODEL_LICENSE_ID)
        self.assertEqual(nemotron.LICENSE_ID, "nvidia-nemotron-open-model-license")
        self.assertEqual(nemotron.ACK_MIN_CHARS, LICENSE_ACK_MIN_CHARS)
        base = {
            "licenseId": nemotron.LICENSE_ID,
            "licenseAcknowledgement": "I accept the pinned NVIDIA Nemotron Open Model License.",
        }
        with tempfile.TemporaryDirectory() as directory:
            notice = Path(directory) / "NOTICE"
            notice.write_text("NVIDIA notice\n", encoding="utf-8")
            self.assertEqual(nemotron.preflight_license(base, notice), notice)
            empty = Path(directory) / "EMPTY"
            empty.write_text("   \n", encoding="utf-8")
            with self.assertRaisesRegex(nemotron.NemotronHError, "empty"):
                nemotron.preflight_license(base, empty)
            link = Path(directory) / "LINK"
            link.symlink_to(notice)
            with self.assertRaisesRegex(nemotron.NemotronHError, "absent"):
                nemotron.preflight_license(base, link)
            with self.assertRaisesRegex(nemotron.NemotronHError, "license ID"):
                nemotron.preflight_license(
                    {**base, "licenseId": "nvidia-open-model-license"}, notice
                )
            with self.assertRaisesRegex(nemotron.NemotronHError, "explicit"):
                nemotron.preflight_license(
                    {**base, "licenseAcknowledgement": "too short"}, notice
                )

    def test_pool_hooks_append_metrics_and_complete_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            env = {nemotron.METRICS_ENV: str(metrics)}
            self.assertIsNone(nemotron.append_eval_record(1, 0.5, environ={}))
            nemotron.append_eval_record(3, 0.25, environ=env)
            self.assertEqual(
                json.loads(metrics.read_text(encoding="utf-8")),
                {"eval_loss": 0.25, "step": 3},
            )
            with self.assertRaises(nemotron.NemotronHError):
                nemotron.append_eval_record(1, float("nan"), environ=env)
            target = Path(directory) / "ckpt"
            request = Path(directory) / "request.json"
            request.write_text(
                json.dumps({"seq": 2, "label": "0002", "dir": str(target)}),
                encoding="utf-8",
            )
            seen = []

            def save(path):
                seen.append(path)
                path.mkdir(parents=True, exist_ok=True)
                (path / "weights").write_text("adapter", encoding="utf-8")

            nemotron.honor_checkpoint_request(
                save, environ={nemotron.CHECKPOINT_REQUEST_ENV: str(request)}
            )
            self.assertTrue((target / "COMPLETE").is_file())
            nemotron.honor_checkpoint_request(
                save, environ={nemotron.CHECKPOINT_REQUEST_ENV: str(request)}
            )
            self.assertEqual(len(seen), 1)
            request.write_text("{", encoding="utf-8")
            with self.assertRaises(nemotron.NemotronHError):
                nemotron.honor_checkpoint_request(
                    save, environ={nemotron.CHECKPOINT_REQUEST_ENV: str(request)}
                )

    def test_reviewed_jobspec_uses_config_caps_and_fixed_ranges(self):
        path = ROOT / "jobspecs" / "nemotron-h-nano-4b-y2-reviewed.json"
        text = path.read_text(encoding="utf-8")
        self.assertTrue(path.name.endswith("-reviewed.json"))
        self.assertFalse(path.name.startswith("nemo-v3-"))
        for forbidden in ("per_run_usd", "per_day_usd", "per_month_usd"):
            self.assertNotIn(forbidden, text)
        spec = json.loads(text)
        self.assertEqual(
            spec["base"]["revision"],
            "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        )
        self.assertEqual(spec["expectedHybridPattern"], NANO_4B)
        self.assertEqual(spec["recipe"]["targetModules"], MODULES)
        self.assertIs(spec["recipe"]["enableThinking"], True)
        caps = json.loads((ROOT / "config" / "dispatch-caps.v1.json").read_text())
        self.assertEqual(
            [caps["per_run_usd"], caps["per_day_usd"], caps["per_month_usd"]],
            [100, 500, 5000],
        )
        with tempfile.TemporaryDirectory() as directory:
            notice = Path(directory) / "NOTICE"
            notice.write_text(
                "NVIDIA notice retained with the base.\n", encoding="utf-8"
            )
            resolved, provenance = nemotron.review_launch_spec(
                spec, ROOT, notice_path=notice, provider="massed"
            )
            self.assertEqual(
                resolved,
                {"per_run_usd": 100.0, "per_day_usd": 500.0, "per_month_usd": 5000.0},
            )
            self.assertTrue(
                all(str(item).startswith("dispatch:") for item in provenance.values())
            )
            tighter, _ = nemotron.resolved_caps(
                spec, ROOT, extra_sources=(("lane", {"per_run_usd": 40}),)
            )
            self.assertEqual(tighter["per_run_usd"], 40.0)
            self.assertEqual(tighter["per_day_usd"], 500.0)
            wider, _ = nemotron.resolved_caps(
                spec, ROOT, extra_sources=(("lane", {"per_run_usd": 250}),)
            )
            self.assertEqual(wider["per_run_usd"], 100.0)
            with self.assertRaises(nemotron.NemotronHError):
                nemotron.review_launch_spec(
                    spec, ROOT, notice_path=notice, provider="runpod"
                )
            broken = json.loads(text)
            broken["providers"] = ["salad"]
            with self.assertRaises(nemotron.NemotronHError):
                nemotron.review_launch_spec(
                    broken, ROOT, notice_path=notice, provider="salad"
                )
            broken = json.loads(text)
            broken["trainingImage"] = broken["trainingImage"] + "-moved"
            with self.assertRaises(nemotron.NemotronHError):
                nemotron.review_launch_spec(
                    broken, ROOT, notice_path=notice, provider="salad"
                )
            broken = json.loads(text)
            broken["recipe"]["loraR"] = 65
            with self.assertRaisesRegex(nemotron.NemotronHError, "loraR"):
                nemotron.review_launch_spec(
                    broken, ROOT, notice_path=notice, provider="salad"
                )
            broken = json.loads(text)
            broken["recipe"]["targetModules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
            with self.assertRaisesRegex(nemotron.NemotronHError, "targetModules"):
                nemotron.review_launch_spec(
                    broken, ROOT, notice_path=notice, provider="salad"
                )
            broken = json.loads(text)
            broken["caps"]["per_run_usd"] = 100
            with self.assertRaisesRegex(nemotron.NemotronHError, "hardcoded"):
                nemotron.review_launch_spec(
                    broken, ROOT, notice_path=notice, provider="salad"
                )

    def test_pin_mismatches_are_exact(self):
        observed = nemotron.requirement_pin_mismatches(
            (ROOT / "training-image" / "requirements.lock").read_text(encoding="utf-8"),
            (ROOT / "requirements.txt").read_text(encoding="utf-8"),
            (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        )
        mamba = (
            "mamba-ssm: absent from training-image/requirements.lock. "
            "The N-14 overlay records 2.3.2.post1+cu.12.6.torch.2.14 "
            "wheel sha256 329557d964d58ebfcaf44cb17dc043356a24dd5c8ab9b68d96dc0fbf89911614 "
            "for this torch build. Digest 8eb91965 was not readable "
            "anonymously, so this is the overlay record, not a manifest read-back."
        )
        causal = (
            "causal-conv1d: absent from training-image/requirements.lock. "
            "The N-14 overlay records 1.7.0+cu.12.6.torch.2.14 "
            "wheel sha256 75f42d6ab5c87490d04a9a58f9e3c191c9cd88b31e8b626fc057099e01832d60 "
            "for this torch build. Digest 8eb91965 was not readable "
            "anonymously, so this is the overlay record, not a manifest read-back."
        )
        expected = [
            mamba,
            causal,
            "transformers: requirements.txt and pyproject.toml pin 5.17.0; image pins 5.22.2",
            "peft: requirements.txt and pyproject.toml pin 0.21.0; image pins 0.21.2",
            "trl: requirements.txt and pyproject.toml pin 1.14.0; image pins 1.17.1",
            (
                "torch: requirements.txt and pyproject.toml do not pin "
                "2.14.1+cu126 (image pin). The CPU dry-run installs "
                "torch==2.14.1+cpu."
            ),
        ]
        self.assertEqual(observed, expected)

    def test_cpu_dry_run_trains_two_steps_without_downloading_weights(self):
        source = inspect.getsource(nemotron._cpu_dry_run)
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("NVIDIA-Nemotron", source)
        offline = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
        previous = {name: os.environ.get(name) for name in offline}
        for name in offline:
            os.environ[name] = "1"
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = nemotron.cpu_dry_run(directory)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "detected_modules",
                        "losses",
                        "global_steps",
                        "lora_updated",
                        "checkpoint_complete",
                        "thinking_on",
                        "thinking_off",
                        "weights_downloaded",
                        "gpu_job_launched",
                        "quantization",
                        "metrics",
                    )
                },
                indent=2,
            )
        )
        self.assertEqual(result["detected_modules"], MODULES)
        self.assertEqual(result["global_steps"], 2)
        self.assertEqual(len(result["losses"]), 2)
        self.assertTrue(all(isinstance(loss, float) for loss in result["losses"]))
        self.assertTrue(result["lora_updated"])
        self.assertTrue(result["checkpoint_complete"])
        self.assertFalse(result["weights_downloaded"])
        self.assertFalse(result["gpu_job_launched"])
        self.assertTrue(result["thinking_on"].startswith("/think|"))
        self.assertIn("thinking=True", result["thinking_on"])
        self.assertTrue(result["thinking_off"].startswith("/no_think|"))
        self.assertIn("thinking=False", result["thinking_off"])
        self.assertEqual(set(result["adapter_target_modules"]), set(MODULES))
        self.assertTrue(result["adapter_safetensors"])
        lines = [json.loads(line) for line in result["metrics"].splitlines()]
        self.assertEqual([line["step"] for line in lines], [1, 2])
        self.assertEqual(lines[0]["eval_loss"], result["losses"][0])

    def test_name_gate_does_not_treat_every_nemotron_as_hybrid(self):
        self.assertTrue(
            nemotron.base_requires_nemotron_h_preflight(
                "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
            )
        )
        self.assertTrue(
            nemotron.base_requires_nemotron_h_preflight(
                "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
            )
        )
        self.assertFalse(
            nemotron.base_requires_nemotron_h_preflight(
                "nvidia/Llama-3.1-Nemotron-70B-Instruct"
            )
        )
        self.assertFalse(
            nemotron.base_requires_nemotron_h_preflight(
                "nvidia/Nemotron-4-340B-Instruct"
            )
        )

    def test_eager_out_proj_ignores_a_fused_kernel_tensor(self):
        torch = importlib.import_module("torch")
        config_mod = importlib.import_module(
            "transformers.models.nemotron_h.configuration_nemotron_h"
        )
        modeling = importlib.import_module(
            "transformers.models.nemotron_h.modeling_nemotron_h"
        )
        config = config_mod.NemotronHConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            mamba_num_heads=4,
            mamba_head_dim=8,
            n_groups=2,
            ssm_state_size=8,
            conv_kernel=4,
            chunk_size=8,
            intermediate_size=64,
            vocab_size=128,
            max_position_embeddings=64,
            num_hidden_layers=1,
            hybrid_override_pattern="M",
            num_nextn_predict_layers=0,
            use_cache=False,
            mlp_hidden_act="silu",
        )
        mixer = modeling.NemotronHMamba2Mixer(config, layer_idx=0)
        mixer.train()
        sentinel = torch.zeros(1, 4, 32)
        original = modeling.mamba2_split_conv1d_scan_combined

        def fused_kernel(*args, **kwargs):
            return sentinel

        modeling.mamba2_split_conv1d_scan_combined = fused_kernel
        try:
            states = torch.randn(1, 4, 32)
            with torch.no_grad():
                self.assertIs(mixer(states), sentinel)
            nemotron.use_eager_out_proj(mixer)
            output = mixer(states)
            self.assertIsNot(output, sentinel)
            output.sum().backward()
            gradient = mixer.out_proj.weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0)
        finally:
            modeling.mamba2_split_conv1d_scan_combined = original


if __name__ == "__main__":
    unittest.main()
