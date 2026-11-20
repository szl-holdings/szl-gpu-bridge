"""Offline denial tests for the unsafe ABS-RL v1 GRPO worker.

No torch, TRL, network, model weights, private data, provider actions or jobs.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "tools" / "absrl" / "reward_admission.py"
WORKER = ROOT / "tools" / "absrl" / "grpo_absention.py"

SPEC = importlib.util.spec_from_file_location("szl_absrl_reward_admission", GUARD)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class AbsRlTrainingHoldTests(unittest.TestCase):
    def test_gate_is_unconditionally_denial_only(self):
        for _ in range(3):
            with self.assertRaisesRegex(
                gate.TrainingAdmissionHold, "^ABS_RL_REWARD_V1_UNSAFE$"
            ):
                gate.require_training_admission()
        self.assertEqual(gate.TRAINING_HOLD_EXIT_CODE, 86)

    def test_gate_is_pure_and_has_no_runtime_override(self):
        tree = ast.parse(GUARD.read_text(encoding="utf-8"))
        forbidden = {
            "os",
            "sys",
            "subprocess",
            "socket",
            "requests",
            "torch",
            "trl",
            "huggingface_hub",
            "pathlib",
            "ctypes",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(
                    forbidden.intersection(x.name.split(".")[0] for x in node.names)
                )
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn(node.module.split(".")[0], forbidden)
        self.assertNotIn("getenv", GUARD.read_text(encoding="utf-8"))
        self.assertNotIn(
            "training_authorized = True", GUARD.read_text(encoding="utf-8")
        )

    def test_worker_calls_hold_before_argument_or_provider_side_effect(self):
        source = WORKER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        main = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
        )
        first = main.body[0]
        self.assertIsInstance(first, ast.Try)
        self.assertIsInstance(first.body[0], ast.Expr)
        self.assertIsInstance(first.body[0].value, ast.Call)
        self.assertIsInstance(first.body[0].value.func, ast.Name)
        self.assertEqual(first.body[0].value.func.id, "require_training_admission")
        self.assertTrue(
            any(
                isinstance(n, ast.Return)
                and isinstance(n.value, ast.Name)
                and n.value.id == "TRAINING_HOLD_EXIT_CODE"
                for handler in first.handlers
                for n in ast.walk(handler)
            )
        )
        parser_line = next(
            index
            for index, line in enumerate(source.splitlines(), 1)
            if "ap = argparse.ArgumentParser()" in line
        )
        self.assertLess(first.lineno, parser_line)
        self.assertNotIn("rewards_v2", source)

    def test_denial_without_installed_ml_packages(self):
        # -S excludes site-packages. A missing torch/TRL/HF SDK must never
        # prevent the unconditional source-level training denial.
        result = subprocess.run(
            [sys.executable, "-S", str(WORKER)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, gate.TRAINING_HOLD_EXIT_CODE)
        self.assertIn("FAIL-CLOSED: ABS_RL_REWARD_V1_UNSAFE", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_no_ml_or_provider_import_runs_before_main_guard(self):
        source = WORKER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        prohibited = {
            "torch",
            "transformers",
            "peft",
            "datasets",
            "huggingface_hub",
            "trl",
        }
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(
                    prohibited.intersection(
                        alias.name.split(".")[0] for alias in node.names
                    )
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn(node.module.split(".")[0], prohibited)
        self.assertNotIn("os.environ.setdefault", source.split("def main() -> int:")[0])

    def test_imported_main_has_no_ml_or_provider_import_side_effect(self):
        # Covers the direct Python-import/function entry point as well as
        # the script entry point tested above. No model, network or GPU.
        canary = (
            "import builtins, importlib.util, sys\n"
            "_original = builtins.__import__\n"
            "_forbidden = {'torch', 'trl', 'peft', 'datasets',\n"
            "              'transformers', 'huggingface_hub'}\n"
            "def _tripwire(name, *args, **kwargs):\n"
            "    if name.split('.')[0] in _forbidden:\n"
            "        raise RuntimeError('FORBIDDEN_PROVIDER_IMPORT:' + name)\n"
            "    return _original(name, *args, **kwargs)\n"
            "builtins.__import__ = _tripwire\n"
            "spec = importlib.util.spec_from_file_location(\n"
            "    'staged_absrl_worker', sys.argv[1]\n"
            ")\n"
            "worker = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(worker)\n"
            "assert worker.main() == 86\n"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", canary, str(WORKER)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FAIL-CLOSED: ABS_RL_REWARD_V1_UNSAFE", result.stderr)
        self.assertNotIn("FORBIDDEN_PROVIDER_IMPORT", result.stderr)

    def test_v1_reward_stays_explicitly_unqualified(self):
        source = WORKER.read_text(encoding="utf-8")
        self.assertIn("from rewards import abstention_reward", source)
        self.assertIn("from reward_admission import (", source)
        self.assertIn("require_training_admission()", source)
        self.assertIn("TRAINING_HOLD_EXIT_CODE", source)


if __name__ == "__main__":
    unittest.main()
