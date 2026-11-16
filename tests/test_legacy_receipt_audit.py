"""Offline negative controls using synthetic receipts; no GPU or network."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "tools/salad/legacy_receipt_audit.py"
SPEC = importlib.util.spec_from_file_location("legacy_receipt_audit", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
FAKE_SOURCE = b"synthetic test profile; not executable model code\n"
FAKE_SHA = hashlib.sha256(FAKE_SOURCE).hexdigest()


def fixture():
    return {
        "schema": "szl.salad-heldout/v1",
        "status": "OK",
        "kind": "MEASURED",
        "criterion_2_contamination": {"rate": 0.0, "rows_used": 40, "pass": True},
        "criterion_3_refusal": {
            "base_refusal_rate": 0.75,
            "adapter_refusal_rate": 0.75,
            "pass": True,
        },
        "criterion_4_neutral": {
            "base_accuracy": 0.05,
            "adapter_accuracy": 0.05,
            "pass": True,
        },
        "criterion_5_invented_id": {"invented_id_rate": 0.2, "tau": 0.8, "pass": True},
        "energy_j": 123456,
        "heldout_quality_all_pass": True,
    }


def run_audit(data):
    with patch.dict(M.PROFILES, {FAKE_SHA: "heldout-v1"}):
        return M.audit(json.dumps(data).encode(), FAKE_SOURCE)


class LegacyAuditTests(unittest.TestCase):
    def test_neutral_rule_cannot_fail(self):
        for observed in (0, 0.05, 0.5, 1):
            result = M.lower_bound_check(observed, 0.05, 0.10)
            self.assertTrue(result["comparison_pass"])
            self.assertFalse(result["can_fail_for_recorded_baseline"])
            self.assertEqual(result["qualification"], "UNINTERPRETABLE")

    def test_refusal_rule_can_fail(self):
        self.assertFalse(M.lower_bound_check(0.5, 0.75, 0.15)["comparison_pass"])
        self.assertTrue(
            M.lower_bound_check(0.75, 0.75, 0.15)["can_fail_for_recorded_baseline"]
        )

    def test_invented_id_rule_can_fail(self):
        self.assertFalse(M.upper_bound_check(1.0, 0.8)["comparison_pass"])
        self.assertTrue(M.upper_bound_check(0.2, 0.8)["can_fail"])

    def test_contamination_rule_can_fail(self):
        self.assertFalse(M.upper_bound_check(5 / 40, 0.1)["comparison_pass"])

    def test_units(self):
        correction = M.energy_correction(123456)
        self.assertEqual(correction["derived_energy_j"], 123.456)
        self.assertEqual(
            correction["classification"], "UNIT_CORRECTION_NOT_NEW_MEASUREMENT"
        )

    def test_invalid_energy(self):
        for value in (-1, True, 1.2, None, "123"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                M.energy_correction(value)

    def test_invalid_rate(self):
        for value in (-0.1, 1.1, True, float("nan"), float("inf"), "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                M.lower_bound_check(value, 0.5, 0.1)

    def test_duplicate_json(self):
        with self.assertRaises(ValueError):
            M.strict_json(b'{"status":"OK","status":"ERROR"}')

    def test_nonfinite_json(self):
        with self.assertRaises(ValueError):
            M.strict_json(b'{"x":NaN}')

    def test_nonobject_json(self):
        with self.assertRaises(ValueError):
            M.strict_json(b"[]")

    def test_unknown_source_never_qualifies(self):
        result = M.audit(json.dumps(fixture()).encode(), b"unknown source")
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertFalse(result["promotion_authorized"])

    def test_matching_profile_is_not_attestation(self):
        result = run_audit(fixture())
        self.assertEqual(result["status"], "AUDITED_LEGACY_NOT_QUALIFIED")
        self.assertFalse(result["source_match_is_execution_attestation"])
        self.assertFalse(result["gpu_rerun_performed"])
        self.assertEqual(result["quality_qualification"], "NOT_ESTABLISHED")
        self.assertTrue(any("BASE_ARM_NOT_ISOLATED" in s for s in result["findings"]))

    def test_empty_denominator_unavailable(self):
        data = fixture()
        data["criterion_2_contamination"]["rows_used"] = 0
        self.assertEqual(
            run_audit(data)["checks"]["contamination"]["qualification"], "UNAVAILABLE"
        )

    def test_boolean_denominator_rejected(self):
        data = fixture()
        data["criterion_2_contamination"]["rows_used"] = True
        with self.assertRaises(ValueError):
            run_audit(data)

    def test_denominator_above_fixture_bound(self):
        data = fixture()
        data["criterion_2_contamination"]["rows_used"] = 41
        with self.assertRaises(ValueError):
            run_audit(data)

    def test_forged_pass_flag_rejected(self):
        data = fixture()
        data["criterion_3_refusal"]["adapter_refusal_rate"] = 0
        with self.assertRaises(ValueError):
            run_audit(data)

    def test_truthy_string_rejected(self):
        data = fixture()
        data["criterion_4_neutral"]["pass"] = "true"
        with self.assertRaises(ValueError):
            run_audit(data)

    def test_forged_aggregate_rejected(self):
        data = fixture()
        data["heldout_quality_all_pass"] = False
        with self.assertRaises(ValueError):
            run_audit(data)

    def test_changed_threshold_rejected(self):
        data = fixture()
        data["criterion_5_invented_id"]["tau"] = 0.9
        with self.assertRaises(ValueError):
            run_audit(data)

    def test_schema_mismatch_rejected(self):
        data = fixture()
        data["schema"] = "szl.salad-heldout/v2"
        with self.assertRaises(ValueError):
            run_audit(data)

    def test_oversized_input_rejected(self):
        with self.assertRaises(ValueError):
            M.audit(b" " * (M.MAX_BYTES + 1), b"x")

    def test_cli_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "receipt.json").write_text(json.dumps(fixture()))
            (root / "source.py").write_bytes(FAKE_SOURCE)
            (root / "audit.json").write_text("keep original")
            with patch.dict(M.PROFILES, {FAKE_SHA: "heldout-v1"}):
                code = M.main(
                    [
                        "--receipt",
                        str(root / "receipt.json"),
                        "--source",
                        str(root / "source.py"),
                        "--output",
                        str(root / "audit.json"),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertEqual((root / "audit.json").read_text(), "keep original")


if __name__ == "__main__":
    unittest.main()
