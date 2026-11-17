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
                    self.assertTrue(
                        all(keyword.arg is not None for keyword in call.keywords),
                        "review expanded constructor keyword arguments",
                    )
                    signature.bind(
                        **{keyword.arg: object() for keyword in call.keywords}
                    )

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
                        "os": os,
                        "a": types.SimpleNamespace(max_steps=2, save_steps=1),
                    },
                )
                self.assertEqual(config.max_seq_length, 2048)
                self.assertEqual(config.max_steps, 2)
                self.assertFalse(config.bf16)
                self.assertTrue(config.use_cpu)

    def test_pinned_trl_rejects_the_previous_max_length_keyword(self):
        with self.assertRaisesRegex(TypeError, "max_length"):
            self.constructors["SFTConfig"](
                output_dir="/tmp/salad-rejected-config",
                max_length=2048,
                use_cpu=True,
                bf16=False,
                report_to=[],
            )


if __name__ == "__main__":
    unittest.main()
