"""Synthetic cases, never a regrading of a preserved model evaluation."""

import copy
import importlib.util
from pathlib import Path
import unittest

P = Path(__file__).resolve().parents[1] / "tools/evidence/receipt_claims.py"
SPEC = importlib.util.spec_from_file_location("receipt_claims", P)
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)


class ClaimTests(unittest.TestCase):
    def setUp(self):
        self.r = {
            "battery": {
                "of": 6,
                "abstained_or_refused": 0,
                "probes": [{"abstained_or_refused": False} for _ in range(6)],
            },
            "rows": [{"verdict": "ANSWER"} for _ in range(6)],
            "joules_measured": 0.0,
            "avg_watts": 0.0,
            "wall_hours": 1.71,
        }

    def test_six_unlabelled_probes_are_not_zero_denominators(self):
        s = g.battery_summary(self.r)
        self.assertEqual((s["abstained_or_refused"], s["probe_count"]), (0, 6))
        self.assertIsNone(s["subgroups"])

    def test_zero_meter_not_validated(self):
        self.assertFalse(g.battery_summary(self.r)["energy_independently_verified"])

    def test_positive_meter_not_automatically_validated(self):
        self.r["joules_measured"] = 42.0
        self.assertFalse(g.battery_summary(self.r)["energy_independently_verified"])

    def test_no_result_inheritance(self):
        self.assertFalse(g.battery_summary(self.r)["current_weight_binding_verified"])

    def test_no_production_qualification(self):
        self.assertFalse(g.battery_summary(self.r)["model_qualified"])

    def test_denominator_count_mismatch_rejected(self):
        self.r["battery"]["of"] = 7
        self.assertRaises(g.ClaimHold, g.battery_summary, self.r)

    def test_numerator_mismatch_rejected(self):
        self.r["battery"]["abstained_or_refused"] = 1
        self.assertRaises(g.ClaimHold, g.battery_summary, self.r)

    def test_boolean_denominator_rejected(self):
        self.r["battery"]["of"] = True
        self.assertRaises(g.ClaimHold, g.battery_summary, self.r)

    def test_string_verdict_rejected(self):
        self.r["battery"]["probes"][0]["abstained_or_refused"] = "false"
        self.assertRaises(g.ClaimHold, g.battery_summary, self.r)

    def test_no_text_inferred_labels(self):
        for row in self.r["rows"]:
            row["question"] = "unanswerable question"
        self.assertIsNone(g.battery_summary(self.r)["subgroups"])

    def test_malformed_label_is_unavailable(self):
        self.r["rows"][0]["kind"] = ["answerable"]
        self.assertIsNone(g.battery_summary(self.r)["subgroups"])

    def test_partial_labels_unavailable(self):
        self.r["rows"][0].update(kind="answerable", verdict="PASS")
        self.assertIsNone(g.battery_summary(self.r)["subgroups"])

    def test_explicit_complete_labels_supported(self):
        self.r["rows"] = [
            {
                "kind": "answerable" if i < 3 else "unanswerable",
                "verdict": "PASS" if i == 0 else "FAIL",
            }
            for i in range(6)
        ]
        self.assertEqual(
            g.battery_summary(self.r)["subgroups"]["answerable"],
            {"passed": 1, "total": 3},
        )

    def test_empty_subgroup_unavailable(self):
        self.r["rows"] = [{"kind": "answerable", "verdict": "FAIL"} for _ in range(6)]
        self.assertIsNone(g.battery_summary(self.r)["subgroups"]["unanswerable"])

    def test_nan_meter_rejected(self):
        self.r["joules_measured"] = float("nan")
        self.assertRaises(g.ClaimHold, g.battery_summary, self.r)

    def test_negative_meter_rejected(self):
        self.r["avg_watts"] = -1
        self.assertRaises(g.ClaimHold, g.battery_summary, self.r)

    def test_missing_meter_remains_unknown(self):
        del self.r["joules_measured"]
        self.assertIsNone(
            g.battery_summary(self.r)["recorded_meter_fields"]["joules_measured"]
        )

    def test_input_never_mutated(self):
        before = copy.deepcopy(self.r)
        g.battery_summary(self.r)
        self.assertEqual(before, self.r)


if __name__ == "__main__":
    unittest.main()
