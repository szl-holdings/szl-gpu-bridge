"""Check the worker's actual call sites against its installed training-image APIs.

Set SZL_TEST_PINNED_RUNTIME=1 in a network-disabled candidate container. These
checks do not download weights, instantiate a model/Trainer, or execute CUDA.
"""

import ast
import copy
import importlib.metadata
import inspect
import os
from pathlib import Path
import re
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "salad" / "salad_worker.py"
DOCKERFILE = ROOT / "tools" / "salad" / "Dockerfile"


@unittest.skipUnless(
    os.environ.get("SZL_TEST_PINNED_RUNTIME") == "1",
    "requires the pinned training image and explicit CPU-only runtime smoke",
)
class PinnedRuntimeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer

        cls.constructors = {
            "SFTConfig": SFTConfig,
            "SFTTrainer": SFTTrainer,
            "LoraConfig": LoraConfig,
        }
        cls.tree = ast.parse(WORKER.read_text(encoding="utf-8"))
        fields = inspect.signature(SFTConfig).parameters
        cls.sequence_keys = [
            key for key in ("max_seq_length", "max_length") if key in fields
        ]
        if len(cls.sequence_keys) != 1:
            raise AssertionError("pinned TRL must expose exactly one sequence-bound field")

    def calls(self, name):
        result = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]
        self.assertTrue(result, f"no {name} call sites were inspected")
        return result

    def test_installed_core_versions_match_dockerfile(self):
        pins = dict(re.findall(r'"([a-z_]+)==([^"]+)"', DOCKERFILE.read_text()))
        for package in ("transformers", "trl", "peft", "datasets"):
            with self.subTest(package=package):
                self.assertEqual(importlib.metadata.version(package), pins[package])

    def test_every_actual_constructor_call_binds_to_installed_api(self):
        for name, constructor in self.constructors.items():
            signature = inspect.signature(constructor)
            for call in self.calls(name):
                with self.subTest(constructor=name, line=call.lineno):
                    self.assertFalse(
                        call.args, "review positional constructor arguments"
                    )
                    expanded = [kw for kw in call.keywords if kw.arg is None]
                    if name == "SFTConfig":
                        self.assertEqual(len(expanded), 1)
                        self.assertIsInstance(expanded[0].value, ast.Name)
                        self.assertEqual(expanded[0].value.id, "sequence_kw")
                    else:
                        self.assertFalse(expanded, "review expanded constructor kwargs")
                    values = {
                        kw.arg: object() for kw in call.keywords if kw.arg is not None
                    }
                    if name == "SFTConfig":
                        values[self.sequence_keys[0]] = object()
                    signature.bind(**values)

    def test_real_worker_config_constructs_with_cpu_test_overrides(self):
        # Use the worker's complete actual config expression. Only output location
        # and hardware selection change; there is no model or Trainer construction.
        for original in self.calls("SFTConfig"):
            call = copy.deepcopy(original)
            with tempfile.TemporaryDirectory() as directory:
                overrides = {"output_dir": directory, "bf16": False, "use_cpu": True}
                call.keywords = [
                    keyword for keyword in call.keywords if keyword.arg not in overrides
                ] + [
                    ast.keyword(arg=name, value=ast.Constant(value=value))
                    for name, value in overrides.items()
                ]
                expression = ast.fix_missing_locations(ast.Expression(body=call))
                config = eval(
                    compile(expression, str(WORKER), "eval"),
                    {
                        "__builtins__": {},
                        "SFTConfig": self.constructors["SFTConfig"],
                        "sequence_kw": {self.sequence_keys[0]: 2048},
                        "os": os,
                        "a": types.SimpleNamespace(
                            max_steps=2, save_steps=1, gradient_checkpointing=False
                        ),
                    },
                )
                self.assertEqual(getattr(config, self.sequence_keys[0]), 2048)
                self.assertEqual(config.max_steps, 2)
                self.assertFalse(config.bf16)
                self.assertTrue(config.use_cpu)

    def test_pinned_trl_rejects_the_other_sequence_keyword(self):
        unsupported = ({"max_length", "max_seq_length"} - set(self.sequence_keys)).pop()
        with self.assertRaisesRegex(TypeError, unsupported):
            self.constructors["SFTConfig"](
                output_dir="/tmp/salad-rejected-config",
                **{unsupported: 2048},
                use_cpu=True,
                bf16=False,
                report_to=[],
            )


if __name__ == "__main__":
    unittest.main()
