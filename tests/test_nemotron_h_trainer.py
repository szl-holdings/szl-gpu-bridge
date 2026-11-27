"""Nemotron-H boundary in the Nemo v3 runner: pins, targets, guardrails.

Pure-Python checks run without torch.  The tiny Nemotron-H CPU dry-run runs
when torch, PEFT, TRL and a Transformers build with the native
``NemotronHForCausalLM`` are installed, and skips otherwise.  No signing key
is opened, nothing is downloaded, and no GPU job is launched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import nemotron_h  # noqa: E402
import runjob_nemo_v3  # noqa: E402

NANO_4B = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
NANO_4B_REV = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
NANO_9B = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
NANO_9B_REV = "6533e8de2c68e4536bf7c411d7a3ce5734111476"
HYBRID_TARGETS = [
    "in_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
]


class _Leaf:
    pass


def _mixer(class_name: str, children: list[str]):
    obj = type(class_name, (), {})()
    obj.named_children = lambda: [(c, _Leaf()) for c in children]
    return obj


class _Config:
    def __init__(self, model_type: str, architectures: list[str]) -> None:
        self.model_type = model_type
        self.architectures = architectures


class _Model:
    def __init__(self, config, mixers) -> None:
        self.config = config
        self._mixers = mixers

    def named_modules(self):
        yield "", self
        for index, mixer in enumerate(self._mixers):
            yield f"model.layers.{index}.mixer", mixer

    def parameters(self):
        return iter(())


def _hybrid() -> _Model:
    return _Model(
        _Config("nemotron_h", ["NemotronHForCausalLM"]),
        [
            _mixer("NemotronHMamba2Mixer", ["in_proj", "conv1d", "norm", "out_proj"]),
            _mixer("NemotronHMLP", ["up_proj", "down_proj"]),
            _mixer("NemotronHAttention", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ],
    )


class _Tokenizer:
    def __init__(self, template) -> None:
        self.chat_template = template

    def apply_chat_template(self, messages, **kwargs):
        return (
            "|".join(m["content"] for m in messages)
            + f"|{kwargs.get('enable_thinking')}"
        )


def _spec(**base):
    entry = nemotron_h.nemotron_base(NANO_4B)
    value = {
        "repoId": NANO_4B,
        "revision": NANO_4B_REV,
        "licenseId": entry["license_id"],
        "licenseAcknowledgement": f"Used under the NVIDIA licence. {entry['notice']}",
        "trustRemoteCode": False,
    }
    value.update(base)
    return {"base": value}


class NemotronHPolicy(unittest.TestCase):
    def test_dense_targets_unchanged(self) -> None:
        targets, evidence = nemotron_h.lora_target_modules(
            _Model(_Config("llama", ["LlamaForCausalLM"]), [])
        )
        self.assertEqual(
            targets,
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.assertEqual(evidence, {"family": "dense"})

    def test_hybrid_targets(self) -> None:
        targets, evidence = nemotron_h.lora_target_modules(_hybrid())
        self.assertEqual(targets, HYBRID_TARGETS)
        self.assertFalse(evidence["fused_mamba_kernel"])
        self.assertIn("out_proj", evidence["excluded"])

    def test_pins_and_notice_lines(self) -> None:
        four = nemotron_h.nemotron_base(NANO_4B, NANO_4B_REV)
        nine = nemotron_h.nemotron_base(NANO_9B, NANO_9B_REV)
        self.assertEqual(
            four["notice"],
            "Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License.",
        )
        self.assertEqual(four["reasoning_control"], "enable_thinking")
        self.assertEqual(nine["reasoning_control"], "system_marker")
        self.assertNotEqual(four["notice"], nine["notice"])
        with self.assertRaises(RuntimeError):
            nemotron_h.nemotron_base(NANO_4B, "0" * 40)

    def test_reasoning_toggle_fails_closed(self) -> None:
        reasoning = {"control": "enable_thinking", "enable_thinking": False}
        with self.assertRaises(RuntimeError):
            nemotron_h.render_chat_text(_Tokenizer(None), [], reasoning=reasoning)
        with self.assertRaises(RuntimeError):
            nemotron_h.render_chat_text(
                _Tokenizer("no toggle"), [], reasoning=reasoning
            )
        text = nemotron_h.render_chat_text(
            _Tokenizer("{{ enable_thinking }}"),
            [{"role": "user", "content": "q"}],
            reasoning=reasoning,
        )
        self.assertEqual(text, "q|False")

    def test_guardrails_refuse_stripping(self) -> None:
        template = "pinned"
        base = {
            **nemotron_h.nemotron_base(NANO_4B),
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        }
        guard = {
            "nvidia_guardrails": "kept",
            "license_id": base["license_id"],
            "notice": base["notice"],
            "system_prompt": "preserve",
        }
        nemotron_h.require_nemotron_guardrails(base, _Tokenizer(template), guard)
        for bad in (
            None,
            {**guard, "notice": ""},
            {**guard, "nvidia_guardrails": "off"},
            {**guard, "system_prompt": "strip"},
        ):
            with self.assertRaises(RuntimeError):
                nemotron_h.require_nemotron_guardrails(base, _Tokenizer(template), bad)
        with self.assertRaises(RuntimeError):
            nemotron_h.require_nemotron_guardrails(base, _Tokenizer("other"), guard)
        with tempfile.TemporaryDirectory() as directory:
            text = pathlib.Path(
                nemotron_h.write_nvidia_notice(directory, base)
            ).read_text(encoding="utf-8")
        self.assertIn(base["notice"], text)


class RunnerGates(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = runjob_nemo_v3

    def test_pre_load_gate(self) -> None:
        self.assertIsNone(
            self.runner._nemotron_pre_load_gate(_spec(repoId="unsloth/x"))
        )
        self.assertEqual(
            self.runner._nemotron_pre_load_gate(_spec())["revision"], NANO_4B_REV
        )
        for bad in (
            _spec(licenseId="apache-2.0"),
            _spec(licenseAcknowledgement="Used under an NVIDIA licence, no notice."),
            _spec(trustRemoteCode=True),
            _spec(revision="1" * 40),
        ):
            with self.assertRaises(RuntimeError):
                self.runner._nemotron_pre_load_gate(bad)

    def test_signed_targets_must_match_graph(self) -> None:
        evidence = self.runner._nemotron_signed_targets(
            _hybrid(), list(HYBRID_TARGETS), load_in_4bit=True
        )
        self.assertEqual(evidence["lora_targets"], HYBRID_TARGETS)
        for signed in (
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            HYBRID_TARGETS + ["out_proj"],
            HYBRID_TARGETS[:-1],
        ):
            with self.assertRaises(RuntimeError):
                self.runner._nemotron_signed_targets(
                    _hybrid(), signed, load_in_4bit=False
                )


def _native_stack() -> bool:
    for name in ("torch", "trl", "peft", "transformers", "datasets"):
        if importlib.util.find_spec(name) is None:
            return False
    import transformers

    return hasattr(transformers, "NemotronHForCausalLM")


class NemotronHCpuDryRun(unittest.TestCase):
    def test_tiny_nemotron_h_trains(self) -> None:
        if not _native_stack():
            self.skipTest("torch/TRL/PEFT with native NemotronH is not installed")
        import nemo_v3_cpu_dry_run

        result = nemo_v3_cpu_dry_run.run_nemotron_cpu_dry_run()
        self.assertEqual(result["global_steps"], 2)
        self.assertTrue(math.isfinite(result["training_loss"]))
        self.assertEqual(sorted(HYBRID_TARGETS), result["lora_targets_updated"])
        self.assertTrue(result["notice_in_artifacts"])
        self.assertEqual(len(result["refusals"]), 5)
        self.assertFalse(result["gpu_job_launched"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
