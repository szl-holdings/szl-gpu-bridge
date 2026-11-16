"""Offline tests; no real package install, network, model load or training."""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

P = (
    Path(__file__).resolve().parents[1]
    / "tools/a6000-migration/validate_training_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("runtime_preflight", P)
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.recipe = b"#!/bin/bash\nexit 0\n"
        self.c = {
            "schema": g.SCHEMA,
            "repository": "szl-holdings/szl-gpu-bridge",
            "source_revision": "a" * 40,
            "recipe_sha256": hashlib.sha256(self.recipe).hexdigest(),
            "python": "3.11",
            "distributions": {k: "1.0.0" for k in g.REQUIRED},
        }

    def validate(self):
        return g.validate_contract(self.c, self.recipe, "3.11")

    def test_valid_exact_contract(self):
        self.assertEqual(self.validate(), self.c["distributions"])

    def test_unknown_field_rejected(self):
        self.c["install"] = True
        self.assertRaises(g.RuntimeHold, self.validate)

    def test_missing_pin_rejected(self):
        del self.c["distributions"]["torch"]
        self.assertRaises(g.RuntimeHold, self.validate)

    def test_version_range_rejected(self):
        self.c["distributions"]["torch"] = "2.*"
        self.assertRaises(g.RuntimeHold, self.validate)

    def test_python_mismatch_rejected(self):
        self.c["python"] = "3.12"
        self.assertRaises(g.RuntimeHold, self.validate)

    def test_moving_source_rejected(self):
        self.c["source_revision"] = "main"
        self.assertRaises(g.RuntimeHold, self.validate)

    def test_recipe_drift_rejected(self):
        self.recipe += b"echo changed\n"
        self.assertRaises(g.RuntimeHold, self.validate)

    def test_wrong_repository_rejected(self):
        self.c["repository"] = "other/repo"
        self.assertRaises(g.RuntimeHold, self.validate)

    def test_duplicate_contract_rejected(self):
        self.assertRaises(g.RuntimeHold, g.parse_contract, b'{"schema":1,"schema":2}')

    def test_nonfinite_contract_rejected(self):
        self.assertRaises(g.RuntimeHold, g.parse_contract, b'{"schema":NaN}')

    def test_oversized_contract_rejected(self):
        self.assertRaises(g.RuntimeHold, g.parse_contract, b" " * 32769)

    def test_unexpected_torch_upgrade_rejected(self):
        pins = self.c["distributions"]

        def getter(n):
            return "2.15.0" if n == "torch" else pins[n]

        self.assertRaises(g.RuntimeHold, g.verify_versions, pins, getter)

    def test_unpinned_installed_vision_rejected(self):
        self.assertRaises(
            g.RuntimeHold, g.verify_versions, self.c["distributions"], lambda n: "1.0.0"
        )

    def test_missing_required_distribution(self):
        def missing(n):
            raise importlib.metadata.PackageNotFoundError(n)

        self.assertRaises(
            g.RuntimeHold, g.verify_versions, self.c["distributions"], missing
        )

    def test_absent_optional_distribution_permitted(self):
        def getter(n):
            if n in g.OPTIONAL:
                raise importlib.metadata.PackageNotFoundError(n)
            return self.c["distributions"][n]

        self.assertEqual(
            g.verify_versions(self.c["distributions"], getter), self.c["distributions"]
        )

    def test_pip_conflict_stops_before_import(self):
        runner = Mock(return_value=NS(returncode=1, stdout="private diagnostic"))
        self.assertRaises(g.RuntimeHold, g.run_checks, runner)
        self.assertEqual(runner.call_count, 1)

    def test_nms_import_crash_rejected(self):
        runner = Mock(side_effect=[NS(returncode=0), NS(returncode=1, stdout="")])
        self.assertRaises(g.RuntimeHold, g.run_checks, runner)

    def test_removed_config_keyword_rejected(self):
        runner = Mock(
            side_effect=[
                NS(returncode=0),
                NS(
                    returncode=0,
                    stdout=json.dumps({"sft_config_fields": ["max_seq_length"]}),
                ),
            ]
        )
        self.assertRaises(g.RuntimeHold, g.run_checks, runner)

    def test_timeout_rejected(self):
        runner = Mock(side_effect=subprocess.TimeoutExpired("test", 1))
        self.assertRaises(g.RuntimeHold, g.run_checks, runner)

    def test_smoke_stays_offline_and_cannot_install(self):
        runner = Mock(
            side_effect=[
                NS(returncode=0),
                NS(
                    returncode=0,
                    stdout=json.dumps({"sft_config_fields": sorted(g.CONFIG_FIELDS)}),
                ),
            ]
        )
        report = g.run_checks(runner)
        self.assertFalse(report["training_started"])
        self.assertFalse(report["packages_installed"])
        for call in runner.call_args_list:
            self.assertNotIn("install", call.args[0])
            self.assertEqual(call.kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertNotIn("HF_TOKEN", call.kwargs["env"])

    def test_invalid_smoke_response_rejected(self):
        runner = Mock(side_effect=[NS(returncode=0), NS(returncode=0, stdout="[]")])
        self.assertRaises(g.RuntimeHold, g.run_checks, runner)

    def test_real_cli_missing_contract_fails_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(P),
                    "--contract",
                    str(Path(directory) / "missing.json"),
                    "--recipe",
                    str(P),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertEqual(result.returncode, 78)
        self.assertFalse(json.loads(result.stdout)["training_started"])
        self.assertNotIn("Traceback", result.stderr)

    def test_real_cli_wrong_recipe_fails_before_environment_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(self.c))
            result = subprocess.run(
                [sys.executable, str(P), "--contract", str(path), "--recipe", str(P)],
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertEqual(result.returncode, 78)
        self.assertEqual(json.loads(result.stdout)["reason"], "RECIPE_BYTES_CHANGED")

    def test_extra_distribution_rejected(self):
        self.c["distributions"]["untrusted"] = "1.0"
        self.assertRaises(g.RuntimeHold, self.validate)


if __name__ == "__main__":
    unittest.main()
