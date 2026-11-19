# SPDX-License-Identifier: Apache-2.0
"""Synthetic, zero-network checks of the research-only paired evolution gate."""

import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/proving-grounds/paired_evolution_gate.py"
spec = importlib.util.spec_from_file_location("paired_evolution_gate", MODULE_PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture(count=400, family_size=1):
    cases = []
    for i in range(count):
        cases.append(
            {
                "case_id": hashlib.sha256(f"opaque-salted-{i}".encode()).hexdigest(),
                "baseline_correct": i % 2 == 0,
                "broken_control_correct": False,
            }
        )
    candidates = [
        {
            "id": f"candidate-{i}",
            "revision": f"{i + 1:040x}",
            "frozen_before_holdout": True,
            "rows": [
                {"case_id": case["case_id"], "correct": True}
                for case in cases
            ],
        }
        for i in range(family_size)
    ]
    return {
        "schema": gate.SCHEMA,
        "repository": gate.REPOSITORY,
        "source_revision": "a" * 40,
        "plan_revision": "b" * 40,
        "baseline_revision": "c" * 40,
        "holdout_commitment_sha256": "d" * 64,
        "registered_candidate_count": family_size,
        "holdout_exposure_rounds": 1,
        "alpha_ppm": 50_000,
        "min_effect_ppm": 50_000,
        "cases": cases,
        "candidates": candidates,
    }


class PairedEvolutionTests(unittest.TestCase):
    def test_positive_synthetic_signal_is_never_release_authority(self):
        report = gate.evaluate(fixture())
        self.assertEqual(report["state"], "RESEARCH_SIGNAL_REQUIRES_INDEPENDENT_REVIEW")
        self.assertEqual(report["candidates"][0]["signal"], "RESEARCH_REVIEW_ONLY")
        self.assertEqual(report["candidates"][0]["wins"], 200)
        self.assertEqual(report["candidates"][0]["losses"], 0)
        self.assertEqual(report["candidates"][0]["ties"], 200)
        for key in (
            "input_authenticity_verified",
            "pre_registration_independently_verified",
            "holdout_independence_verified",
            "population_quality_verified",
            "lean_theorem_proven_by_this_check",
            "release_authorized",
            "model_promotion_authorized",
            "provider_writes",
        ):
            self.assertIs(report[key], False)

    def test_no_gain_stays_hold(self):
        payload = fixture()
        for row, case in zip(payload["candidates"][0]["rows"], payload["cases"]):
            row["correct"] = case["baseline_correct"]
        report = gate.evaluate(payload)
        self.assertEqual(report["state"], "HOLD_NO_SUPPORTED_GAIN_OR_CONTROL")
        self.assertEqual(report["candidates"][0]["wins"], 0)

    def test_negative_control_that_does_not_degrade_is_hold(self):
        payload = fixture()
        for row in payload["cases"]:
            row["broken_control_correct"] = row["baseline_correct"]
        report = gate.evaluate(payload)
        self.assertFalse(report["negative_control_sensitivity_declared"])
        self.assertEqual(report["state"], "HOLD_NO_SUPPORTED_GAIN_OR_CONTROL")

    def test_paired_hoeffding_matches_explicit_mathematical_bound(self):
        import math

        payload = fixture(count=400, family_size=1)
        report = gate.evaluate(payload)["candidates"][0]
        predicted = 0.5 - math.sqrt(2 * math.log(1 / 0.05) / 400)
        self.assertAlmostEqual(report["conditional_familywise_lower_bound"], predicted, 7)

    def test_more_predeclared_candidates_widen_not_shrink_the_bound(self):
        one = gate.evaluate(fixture(family_size=1))["candidates"][0]
        two = gate.evaluate(fixture(family_size=2))["candidates"][0]
        self.assertLess(
            two["conditional_familywise_lower_bound"],
            one["conditional_familywise_lower_bound"],
        )

    def test_registered_family_count_cannot_be_understated(self):
        payload = fixture(family_size=2)
        payload["registered_candidate_count"] = 1
        with self.assertRaisesRegex(gate.Hold, "FAMILY_NOT_COMPLETE"):
            gate.evaluate(payload)

    def test_adaptive_holdout_reuse_is_blocked(self):
        for value in (0, 2, True, 1.0, "1"):
            with self.subTest(value=value):
                payload = fixture()
                payload["holdout_exposure_rounds"] = value
                with self.assertRaisesRegex(gate.Hold, "HOLDOUT_REUSE"):
                    gate.evaluate(payload)

    def test_unfrozen_candidate_or_boolean_impersonator_rejected(self):
        for value in (False, 1, "true", None):
            with self.subTest(value=value):
                payload = fixture()
                payload["candidates"][0]["frozen_before_holdout"] = value
                with self.assertRaisesRegex(gate.Hold, "CANDIDATE_NOT_FROZEN"):
                    gate.evaluate(payload)

    def test_missing_reordered_or_duplicate_case_ids_are_blocked(self):
        original = fixture()
        for mode in ("missing", "reordered", "duplicate"):
            payload = copy.deepcopy(original)
            rows = payload["candidates"][0]["rows"]
            if mode == "missing":
                rows.pop()
            elif mode == "reordered":
                rows[0], rows[1] = rows[1], rows[0]
            else:
                payload["cases"][1]["case_id"] = payload["cases"][0]["case_id"]
            with self.subTest(mode=mode), self.assertRaises(gate.Hold):
                gate.evaluate(payload)

    def test_lying_baseline_result_is_not_recomputed_from_model_output(self):
        payload = fixture()
        payload["cases"][0]["baseline_correct"] = "yes"
        with self.assertRaisesRegex(gate.Hold, "CASE_RESULT_INVALID"):
            gate.evaluate(payload)

    def test_nonboolean_candidate_outcome_rejected(self):
        for bad in (1, 0, 0.5, "true", None):
            with self.subTest(bad=bad):
                payload = fixture()
                payload["candidates"][0]["rows"][0]["correct"] = bad
                with self.assertRaisesRegex(gate.Hold, "OUTCOME_NOT_BOOLEAN"):
                    gate.evaluate(payload)

    def test_duplicate_candidate_ids_or_revisions_rejected(self):
        for key in ("id", "revision"):
            payload = fixture(family_size=2)
            payload["candidates"][1][key] = payload["candidates"][0][key]
            with self.subTest(key=key), self.assertRaises(gate.Hold):
                gate.evaluate(payload)

    def test_unpinned_revisions_and_commitments_rejected(self):
        for field in ("source_revision", "plan_revision", "baseline_revision"):
            payload = fixture()
            payload[field] = "main"
            with self.subTest(field=field), self.assertRaisesRegex(
                gate.Hold, "REVISION_NOT_IMMUTABLE"
            ):
                gate.evaluate(payload)
        payload = fixture()
        payload["holdout_commitment_sha256"] = "0" * 64
        with self.assertRaisesRegex(gate.Hold, "HOLDOUT_COMMITMENT_INVALID"):
            gate.evaluate(payload)

    def test_manifest_id_and_evidence_digest_reject_silent_tampering(self):
        before = gate.evaluate(fixture())
        changed = fixture()
        changed["candidates"][0]["rows"][0]["correct"] = False
        after = gate.evaluate(changed)
        self.assertNotEqual(before["input_sha256"], after["input_sha256"])

    def test_unexpected_fields_and_candidate_prompt_injection_rejected(self):
        payload = fixture()
        payload["execute_cloud_job"] = True
        with self.assertRaisesRegex(gate.Hold, "ROOT_FIELDS_INVALID"):
            gate.evaluate(payload)
        payload = fixture()
        payload["candidates"][0]["id"] = "../do-what-I-say"
        with self.assertRaisesRegex(gate.Hold, "CANDIDATE_ID_INVALID"):
            gate.evaluate(payload)

    def test_alpha_bool_or_excessive_budget_fails(self):
        for val in (True, 0, 100_000, 0.05):
            payload = fixture()
            payload["alpha_ppm"] = val
            with self.subTest(val=val), self.assertRaisesRegex(gate.Hold, "ALPHA_INVALID"):
                gate.evaluate(payload)

    def test_duplicate_json_keys_and_nonfinite_values_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "spec.json"
            p.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
            with self.assertRaisesRegex(gate.Hold, "DUPLICATE_JSON_KEY"):
                gate.load_input(p)
            p.write_text('{"p":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(gate.Hold, "NONFINITE_JSON"):
                gate.load_input(p)

    def test_cli_never_emits_release_success_exit_code(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "eval.json"
            p.write_text(json.dumps(fixture()), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                rc = gate.main(["--input", str(p)])
            self.assertEqual(rc, 2)
            response = json.loads(output.getvalue())
            self.assertIs(response["release_authorized"], False)
            self.assertNotIn("opaque-salted-", output.getvalue())

    def test_short_and_symlink_input_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "short.json"
            p.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(gate.Hold, "CASE_COVERAGE_INVALID"):
                gate.evaluate({**fixture(), "cases": []})
            try:
                link = Path(td) / "linked.json"
                link.symlink_to(p)
            except OSError:
                self.skipTest("symlink unavailable")
            with self.assertRaisesRegex(gate.Hold, "INPUT_FILE_UNAVAILABLE"):
                gate.load_input(link)


if __name__ == "__main__":
    unittest.main()
