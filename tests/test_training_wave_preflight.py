"""Training-wave planning gates: CPU-only, synthetic, no credentials or GPU."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.salad import training_wave_preflight as s

ROOT = Path(__file__).resolve().parents[1]
S = "a" * 64


def model(**changes):
    result = {
        "model_id": "SZLHOLDINGS/test-model",
        "lane_id": "test-lane",
        "intent": "train",
        "artifact_kind": "adapter",
        "base_model_id": "Qwen/test-base",
        "base_revision": "b" * 64,
        "dataset_id": "SZLHOLDINGS/test-data",
        "dataset_revision": "c" * 64,
        "recipe_sha256": "d" * 64,
        "trainer_commit_sha": "e" * 64,
        "eval_receipt_sha256": "f" * 64,
        "canary_receipt_sha256": "1" * 64,
        "license_rights_state": "OWNER_REPORTED_REVIEWED",
        "dependencies_confirmed": True,
        "gpu_sku": "test-48gb",
        "min_vram_gib": 40,
        "replicas": 2,
        "max_attempts": 1,
        "max_runtime_minutes": 90,
        "hourly_gpu_price_micro_usd": 1_500_000,
        "checkpoint_interval_steps": 100,
        "checkpoint_target": "SZLHOLDINGS/test-runs",
    }
    result.update(changes)
    return result


def plan(**changes):
    result = {
        "schema": s.SCHEMA,
        "mode": "DRY_RUN",
        "catalog_sha256": S,
        "auth_witness_state": "MISSING",
        "owner_budget_approved": False,
        "owner_budget_micro_usd": 50_000_000,
        "inflight_lane_ids": [],
        "models": [model()],
    }
    result.update(changes)
    return result


class TrainingWavePreflightTests(unittest.TestCase):
    def test_valid_structural_plan_stays_hold_and_no_authority(self):
        r = s.evaluate(plan())
        self.assertEqual(r["decision"], "HOLD_NO_COMPUTE")
        self.assertFalse(r["training_authorized"])
        self.assertFalse(r["provider_writes_authorized"])
        self.assertFalse(r["credential_checked"])
        self.assertFalse(r["source_data_authenticated"])
        self.assertEqual(r["row_results"][0]["state"], "STRUCTURAL_CANDIDATE_ONLY")
        self.assertIn("AUTH_MISSING", r["global_reasons"])
        self.assertIn("BUDGET_UNAPPROVED", r["global_reasons"])
        self.assertEqual(r["estimated_upper_cost_micro_usd"], 4_500_000)
        self.assertNotIn("SZLHOLDINGS/test-model", json.dumps(r))

    def test_reported_auth_valid_and_budget_do_not_authorize(self):
        r = s.evaluate(
            plan(
                auth_witness_state="EXTERNALLY_REPORTED_VALID",
                owner_budget_approved=True,
            )
        )
        self.assertEqual(r["decision"], "HOLD_NO_COMPUTE")
        self.assertEqual(r["global_reasons"], ["AUTH_UNVERIFIED"])
        self.assertFalse(r["training_authorized"])
        self.assertFalse(r["receipt_signatures_checked"])

    def test_401_has_explicit_no_retry_or_gpu_authority(self):
        r = s.evaluate(plan(auth_witness_state="DENIED_401"))
        self.assertIn("AUTH_DENIED_401", r["global_reasons"])
        self.assertEqual(r["decision"], "HOLD_NO_COMPUTE")

    def test_read_only_auth_is_not_training_capability(self):
        r = s.evaluate(plan(auth_witness_state="READ_ONLY"))
        self.assertIn("AUTH_READ_ONLY", r["global_reasons"])

    def test_unknown_auth_is_not_success(self):
        r = s.evaluate(plan(auth_witness_state="UNKNOWN"))
        self.assertIn("AUTH_UNVERIFIED", r["global_reasons"])

    def test_missing_license_and_eval_hold(self):
        m = model(
            license_rights_state="UNKNOWN",
            eval_receipt_sha256=None,
            canary_receipt_sha256=None,
            dependencies_confirmed=False,
        )
        r = s.evaluate(plan(models=[m]))
        self.assertIn("LICENSE_RIGHTS_UNVERIFIED", r["row_reasons"])
        self.assertIn("CANARY_OR_EVAL_MISSING", r["row_reasons"])

    def test_inflight_lane_does_not_override_running_training(self):
        r = s.evaluate(plan(inflight_lane_ids=["test-lane"]))
        self.assertIn("INFLIGHT_LANE", r["row_reasons"])

    def test_duplicate_training_identity_is_rejected(self):
        a = model()
        b = model(lane_id="other-lane", model_id="SZLHOLDINGS/other-output")
        r = s.evaluate(plan(models=[a, b]))
        self.assertIn("DUPLICATE_TRAINING_INTENT", r["row_reasons"])
        self.assertIn("DUPLICATE_TRAINING_INTENT", r["row_results"][1]["reasons"])

    def test_duplicate_output_repository_separately_rejected(self):
        other = model(lane_id="unique-lane", recipe_sha256="7" * 64)
        r = s.evaluate(plan(models=[model(), other]))
        self.assertIn("DUPLICATE_OUTPUT_REPO", r["row_reasons"])

    def test_max_attempts_upper_bound_includes_retry_exposure(self):
        r = s.evaluate(plan(models=[model(max_attempts=3)]))
        self.assertEqual(r["estimated_upper_cost_micro_usd"], 13_500_000)
        with self.assertRaises(s.PreflightError):
            s.evaluate(plan(models=[model(max_attempts=0)]))
        with self.assertRaises(s.PreflightError):
            s.evaluate(plan(models=[model(max_attempts=4)]))

    def test_duplicate_lane_id_separately_rejected(self):
        b = model(recipe_sha256="9" * 64, model_id="SZLHOLDINGS/other-model")
        r = s.evaluate(plan(models=[model(), b]))
        self.assertIn("DUPLICATE_LANE", r["row_reasons"])

    def test_no_negative_or_float_budget_is_accepted(self):
        for value in (-1, 0.5, True, s.MAX_MICRO_USD + 1):
            with self.subTest(value=value), self.assertRaises(s.PreflightError):
                s.evaluate(plan(owner_budget_micro_usd=value))

    def test_price_and_runtime_bounds_fail_closed(self):
        for change in (
            {"replicas": 0},
            {"replicas": True},
            {"hourly_gpu_price_micro_usd": 0},
            {"max_runtime_minutes": 0},
            {"min_vram_gib": -1},
        ):
            with self.subTest(change=change), self.assertRaises(s.PreflightError):
                s.evaluate(plan(models=[model(**change)]))

    def test_insufficient_reported_budget_is_detected(self):
        r = s.evaluate(
            plan(owner_budget_micro_usd=4_499_999, owner_budget_approved=True)
        )
        self.assertIn("BUDGET_INSUFFICIENT", r["global_reasons"])
        self.assertIn("AUTH_MISSING", r["global_reasons"])

    def test_wrong_or_malformed_canonical_identity_rejected(self):
        for change in (
            {"base_revision": "main"},
            {"model_id": "../secrets"},
            {"trainer_commit_sha": "main"},
            {"checkpoint_target": "https://example.com"},
        ):
            with self.subTest(change=change), self.assertRaises(s.PreflightError):
                s.evaluate(plan(models=[model(**change)]))

    def test_unbounded_model_count_is_rejected(self):
        fake = model()
        with self.assertRaises(s.PreflightError):
            s.evaluate(plan(models=[fake] * (s.MAX_MODELS + 1)))

    def test_schema_extras_rejected_no_secret_echo(self):
        manifest = plan()
        manifest["hf_token"] = "hf_this_is_not_a_real_token"
        with self.assertRaises(s.PreflightError):
            s.evaluate(manifest)

    def test_unhashable_json_types_fail_closed_without_traceback(self):
        for broken in ([1], {"surprise": True}):
            with (
                self.subTest(level="auth", value=broken),
                self.assertRaises(s.PreflightError),
            ):
                s.evaluate(plan(auth_witness_state=broken))
            for key in ("artifact_kind", "intent", "license_rights_state"):
                with (
                    self.subTest(level=key, value=broken),
                    self.assertRaises(s.PreflightError),
                ):
                    s.evaluate(plan(models=[model(**{key: broken})]))

    def test_json_duplicate_keys_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "input.json"
            p.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaises(s.PreflightError):
                s._bounded_json(p)

    def test_giant_file_fails_before_json_parse(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "big.json"
            p.write_bytes(b"x" * (s.MAX_INPUT_BYTES + 1))
            with self.assertRaises(s.PreflightError):
                s._bounded_json(p)

    def test_cli_writes_hold_and_nonzero_without_provider_calls(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            source, output = p / "source.json", p / "report.json"
            source.write_text(json.dumps(plan()), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/salad/training_wave_preflight.py"),
                    str(source),
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(proc.returncode, 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["training_authorized"])
            self.assertFalse(report["provider_writes_authorized"])
            self.assertNotIn("test-model", output.read_text(encoding="utf-8"))

    def test_large_401_row_dry_run_is_not_training(self):
        import hashlib

        rows = [
            model(
                model_id=f"SZLHOLDINGS/synthetic-{i}",
                lane_id=f"test-lane-{i}",
                recipe_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
            )
            for i in range(401)
        ]
        r = s.evaluate(
            plan(
                models=rows,
                owner_budget_approved=True,
                owner_budget_micro_usd=2_000_000_000,
                auth_witness_state="EXTERNALLY_REPORTED_VALID",
            )
        )
        self.assertEqual(r["plan_model_count"], 401)
        self.assertEqual(r["estimated_upper_cost_micro_usd"], 401 * 4_500_000)
        self.assertEqual(r["decision"], "HOLD_NO_COMPUTE")
        self.assertFalse(r["training_authorized"])
        self.assertNotIn("test-lane-400", json.dumps(r))

    def test_same_input_output_path_does_not_destroy_private_plan(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "operator-manifest.json"
            original = json.dumps(plan())
            source.write_text(original, encoding="utf-8")
            self.assertEqual(s.main([str(source), str(source)]), 1)
            self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_hardlink_output_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d) / "source.json", Path(d) / "alias.json"
            original = json.dumps(plan())
            source.write_text(original, encoding="utf-8")
            try:
                output.hardlink_to(source)
            except OSError:
                self.skipTest("hardlinks unsupported on test filesystem")
            self.assertEqual(s.main([str(source), str(output)]), 1)
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            self.assertEqual(output.read_text(encoding="utf-8"), original)

    def test_cli_invalid_input_is_nonzero_hold(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            source, output = p / "bad.json", p / "report.json"
            source.write_text("{}", encoding="utf-8")
            self.assertEqual(s.main([str(source), str(output)]), 1)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["reason"],
                "INPUT_INVALID",
            )

    def test_secret_data_never_appears_in_error_output(self):
        broken = plan()
        broken["token"] = "test_secret"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            source, output = p / "secret.json", p / "report.json"
            source.write_text(json.dumps(broken), encoding="utf-8")
            self.assertEqual(s.main([str(source), str(output)]), 1)
            self.assertNotIn("test_secret", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
