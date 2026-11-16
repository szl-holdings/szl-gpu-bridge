"""Recipe/API compatibility regressions; no real imports, cloud, or training."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase, main, mock

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "tools/a6000-migration/validate_training_runtime.py"
)
MODERN_FIELDS = {"output_dir", "max_length", "packing"}


def load():
    spec = importlib.util.spec_from_file_location("exact_recipe_preflight", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe(keyword="max_length"):
    return (
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 - <<'TRAIN'\n"
        "from trl import SFTConfig, SFTTrainer\n"
        f"cfg = SFTConfig(output_dir='out', {keyword}=2048, packing=True)\n"
        "TRAIN\n"
    ).encode()


class RecipeApiTests(TestCase):
    def invoke(self, raw, runtime_fields=None):
        module = load()
        fields = set(module.CONFIG_FIELDS) if runtime_fields is None else runtime_fields
        runner = mock.Mock(
            side_effect=[
                SimpleNamespace(returncode=0, stdout=""),
                SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"sft_config_fields": sorted(fields)}),
                ),
            ]
        )
        real_checks = module.run_checks

        def checks(**kwargs):
            return real_checks(runner=runner, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "recipe.sh"
            script.write_bytes(raw)
            contract = {
                "schema": module.SCHEMA,
                "repository": "szl-holdings/szl-gpu-bridge",
                "source_revision": "a" * 40,
                "recipe_sha256": hashlib.sha256(raw).hexdigest(),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "distributions": {key: "1.0.0" for key in module.REQUIRED},
            }
            path = root / "contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [str(SOURCE), "--contract", str(path), "--recipe", str(script)],
                ),
                mock.patch.object(
                    module, "verify_versions", return_value=contract["distributions"]
                ),
                mock.patch.object(module, "run_checks", side_effect=checks),
                mock.patch.object(sys, "stdout", io.StringIO()) as output,
            ):
                code = module.main()
            return code, json.loads(output.getvalue()), runner

    def test_legacy_keyword_cannot_pass_a_modern_runtime(self):
        code, report, _ = self.invoke(recipe("max_seq_length"))
        self.assertEqual(code, 78)
        self.assertEqual(report["reason"], "TRAINER_CONFIG_API_MISMATCH")

    def test_exact_modern_recipe_remains_accepted(self):
        code, report, _ = self.invoke(recipe(), MODERN_FIELDS)
        self.assertEqual(code, 0)
        self.assertFalse(report["training_started"])
        self.assertFalse(report["source_admission_verified"])
        self.assertEqual(report["recipe_config_fields"], sorted(MODERN_FIELDS))

    def test_actual_fields_not_unrelated_hardcoded_recipe_shape(self):
        code, report, _ = self.invoke(
            recipe("max_seq_length"), {"output_dir", "max_seq_length", "packing"}
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            report["recipe_config_fields"], ["max_seq_length", "output_dir", "packing"]
        )

    def test_dynamic_kwargs_are_not_guessed(self):
        raw = recipe().replace(b"packing=True", b"**settings")
        code, report, runner = self.invoke(raw)
        self.assertEqual(code, 78)
        self.assertEqual(report["reason"], "RECIPE_CONFIG_DYNAMIC_ARGUMENTS")
        runner.assert_not_called()

    def test_positional_config_arguments_require_explicit_review(self):
        raw = recipe().replace(b"output_dir='out', ", b"'out', ")
        code, report, runner = self.invoke(raw)
        self.assertEqual(code, 78)
        self.assertEqual(report["reason"], "RECIPE_CONFIG_DYNAMIC_ARGUMENTS")
        runner.assert_not_called()

    def test_unquoted_shell_expansion_is_not_static_recipe_evidence(self):
        code, report, runner = self.invoke(recipe().replace(b"<<'TRAIN'", b"<<TRAIN"))
        self.assertEqual(code, 78)
        self.assertEqual(report["reason"], "RECIPE_PYTHON_BLOCK_UNRESOLVED")
        runner.assert_not_called()

    def test_missing_heredoc_terminator_is_held(self):
        code, _, runner = self.invoke(recipe().removesuffix(b"TRAIN\n"))
        self.assertEqual(code, 78)
        runner.assert_not_called()

    def test_missing_config_call_is_held(self):
        raw = b"#!/bin/bash\npython3 - <<'PY'\nprint('not a trainer')\nPY\n"
        code, report, runner = self.invoke(raw)
        self.assertEqual(code, 78)
        self.assertEqual(report["reason"], "RECIPE_CONFIG_BINDING_UNRESOLVED")
        runner.assert_not_called()

    def test_rebound_config_name_is_held(self):
        raw = recipe().replace(b"cfg =", b"SFTConfig = custom_factory\ncfg =")
        code, _, runner = self.invoke(raw)
        self.assertEqual(code, 78)
        runner.assert_not_called()

    def test_config_import_alias_requires_separate_support(self):
        raw = (
            recipe()
            .replace(b"SFTConfig, SFTTrainer", b"SFTConfig as Config, SFTTrainer")
            .replace(b"cfg = SFTConfig", b"cfg = Config")
        )
        code, _, runner = self.invoke(raw)
        self.assertEqual(code, 78)
        runner.assert_not_called()

    def test_ambiguous_python_blocks_are_held(self):
        raw = recipe() + b"python3 - <<'OTHER'\nprint('another block')\nOTHER\n"
        code, _, runner = self.invoke(raw)
        self.assertEqual(code, 78)
        runner.assert_not_called()

    def test_syntax_error_never_reaches_environment_probe(self):
        code, _, runner = self.invoke(recipe().replace(b"cfg =", b"cfg = ="))
        self.assertEqual(code, 78)
        runner.assert_not_called()

    def test_recipe_code_is_parsed_never_executed(self):
        raw = recipe().replace(
            b"cfg =", b"raise RuntimeError('MUST_NOT_EXECUTE')\ncfg ="
        )
        code, report, _ = self.invoke(raw)
        self.assertEqual(code, 0)
        self.assertFalse(report["training_started"])

    def test_literal_comments_do_not_create_config_calls(self):
        raw = recipe().replace(b"cfg =", b"# SFTConfig(max_seq_length=2048)\ncfg =")
        code, _, _ = self.invoke(raw)
        self.assertEqual(code, 0)

    def test_canonical_recipe_can_be_bound(self):
        raw = (SOURCE.parent / "legs/szl-a11oy-code-train.sh").read_bytes()
        code, report, _ = self.invoke(raw)
        self.assertEqual(code, 0)
        self.assertIn("max_length", report["recipe_config_fields"])
        self.assertNotIn("max_seq_length", report["recipe_config_fields"])


if __name__ == "__main__":
    main()
