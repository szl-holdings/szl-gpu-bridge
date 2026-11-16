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


def proving_fixture():
    return {
        "schema": "szl.proving-ground/v1",
        "status": "OK",
        "kind": "MEASURED",
        "energy_label": "MEASURED",
        "energy_j": 179593466,
        "unproven_tested": 2,
        "counts_declared": {"total": 2, "locked": 0, "unproven": 2},
        "verdicts": {"ABSTAINED": 0, "SKETCH_CANDIDATE": 1, "WEAK_OR_FAKE": 1},
        "receipts": [
            {"verdict": "SKETCH_CANDIDATE"},
            {"verdict": "WEAK_OR_FAKE"},
        ],
    }


def run_proving_audit(data):
    with patch.dict(M.PROFILES, {FAKE_SHA: "provingground-v1"}):
        return M.audit(json.dumps(data).encode(), FAKE_SOURCE)


class ProvingGroundAuditTests(unittest.TestCase):
    def test_raw_nvml_millijoules_are_derived_joules_not_new_measurement(self):
        result = run_proving_audit(proving_fixture())
        correction = result["unit_corrections"]["provingground_window"]
        self.assertEqual(correction["legacy_value"], 179593466)
        self.assertEqual(correction["derived_energy_j"], 179593.466)
        self.assertEqual(correction["actual_counter_unit"], "mJ")
        self.assertEqual(
            correction["classification"], "UNIT_CORRECTION_NOT_NEW_MEASUREMENT"
        )
        self.assertFalse(result["gpu_rerun_performed"])
        self.assertFalse(result["promotion_authorized"])

    def test_all_sketch_candidates_still_establish_no_kernel_proof(self):
        data = proving_fixture()
        data["receipts"] = [{"verdict": "SKETCH_CANDIDATE"}] * 2
        data["verdicts"].update(SKETCH_CANDIDATE=2, WEAK_OR_FAKE=0)
        result = run_proving_audit(data)
        self.assertFalse(result["kernel_verification_performed"])
        self.assertFalse(result["receipt_chain_verification_performed"])
        self.assertEqual(result["quality_qualification"], "NOT_ESTABLISHED")
        self.assertTrue(any("NO_KERNEL_VERIFICATION" in x for x in result["findings"]))

    def test_declared_unproven_coverage_mismatch_is_not_hidden(self):
        data = proving_fixture()
        data["counts_declared"] = {"total": 2, "locked": 1, "unproven": 1}
        result = run_proving_audit(data)
        self.assertFalse(result["probe_count_matches_declared_unproven"])
        self.assertTrue(any("PROBE_COUNT_MISMATCH" in x for x in result["findings"]))

    def test_inconsistent_row_count_is_rejected(self):
        data = proving_fixture()
        data["unproven_tested"] = 1
        with self.assertRaises(ValueError):
            run_proving_audit(data)

    def test_inconsistent_verdict_counts_are_rejected(self):
        data = proving_fixture()
        data["verdicts"].update(SKETCH_CANDIDATE=2, WEAK_OR_FAKE=0)
        with self.assertRaises(ValueError):
            run_proving_audit(data)

    def test_unknown_verdict_cannot_be_promoted_to_proof(self):
        data = proving_fixture()
        data["receipts"][0]["verdict"] = "KERNEL_VERIFIED"
        with self.assertRaises(ValueError):
            run_proving_audit(data)

    def test_boolean_counts_are_not_observations(self):
        for field in ("unproven_tested", "energy_j"):
            data = proving_fixture()
            data[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                run_proving_audit(data)
        data = proving_fixture()
        data["verdicts"]["SKETCH_CANDIDATE"] = True
        with self.assertRaises(ValueError):
            run_proving_audit(data)

    def test_empty_probe_set_never_qualifies(self):
        data = proving_fixture()
        data.update(unproven_tested=0, receipts=[])
        data["verdicts"] = dict.fromkeys(data["verdicts"], 0)
        result = run_proving_audit(data)
        self.assertTrue(any("EMPTY_PROBE_SET" in x for x in result["findings"]))
        self.assertEqual(result["quality_qualification"], "NOT_ESTABLISHED")

    def test_changed_archived_source_gets_no_unit_correction(self):
        with patch.dict(M.PROFILES, {FAKE_SHA: "provingground-v1"}):
            result = M.audit(json.dumps(proving_fixture()).encode(), FAKE_SOURCE + b"x")
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertNotIn("unit_corrections", result)

    def test_unmeasured_legacy_energy_is_not_relabelled(self):
        data = proving_fixture()
        data["energy_label"] = "NOT_MEASURED"
        with self.assertRaises(ValueError):
            run_proving_audit(data)

    def test_original_receipt_bytes_are_preserved_and_hash_bound(self):
        raw = json.dumps(proving_fixture()).encode()
        before = hashlib.sha256(raw).hexdigest()
        with patch.dict(M.PROFILES, {FAKE_SHA: "provingground-v1"}):
            result = M.audit(raw, FAKE_SOURCE)
        self.assertEqual(result["receipt_sha256"], before)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), before)
        self.assertTrue(result["original_receipt_unchanged"])
        self.assertFalse(result["source_match_is_execution_attestation"])


if __name__ == "__main__":
    unittest.main()
