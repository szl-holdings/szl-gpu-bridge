"""Offline evidence negatives; no credentials, GPU, network or publication."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "worker_energy", ROOT / "tools/evidence/worker_energy.py"
)
energy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(energy)


def fixture():
    return {
        "kind": energy.SCHEMA,
        "event": "complete",
        "detail": "fixture only",
        "ts": "2026-11-16T00:00:00Z",
        "adapter": "SZLHOLDINGS/fixture",
        "energy_start": {
            "label": "MEASURED",
            "energy_j": 37768237.571,
            "gpu": "fixture-gpu",
        },
        "energy_end": {
            "label": "MEASURED",
            "energy_j": 37769417.675,
            "gpu": "fixture-gpu",
        },
        "energy_j": {"label": "MEASURED", "joules": 1180.1, "method": energy.METHOD},
    }


def seal(receipt):
    value = copy.deepcopy(receipt)
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()
    return json.dumps(value).encode()


class WorkerEnergyTests(unittest.TestCase):
    def assert_hold(self, receipt, reason):
        with self.assertRaisesRegex(energy.EvidenceHold, "^" + reason + "$"):
            energy.inspect_receipt(seal(receipt))

    def test_explicit_delta_not_cumulative_boot_counter(self):
        row = energy.inspect_receipt(seal(fixture()))
        self.assertEqual(row["recorded_delta_joules"], 1180.1)
        self.assertNotEqual(row["recorded_delta_joules"], 37768237.571)
        self.assertFalse(row["physical_measurement_independently_verified"])
        self.assertFalse(row["training_completion_verified"])

    def test_boot_counter_alone_never_counts_as_job_energy(self):
        value = fixture()
        del value["energy_j"]
        del value["energy_end"]
        value["event"] = "boot"
        self.assert_hold(value, "EXPLICIT_DELTA_UNAVAILABLE")

    def test_nonmeasured_deltas_are_excluded(self):
        for label in (None, "SAMPLE", "MODELED", "UNAVAILABLE", "measured"):
            with self.subTest(label=label):
                value = fixture()
                value["energy_j"]["label"] = label
                self.assert_hold(value, "DELTA_NOT_MEASURED")

    def test_counter_labels_cannot_be_inferred(self):
        value = fixture()
        value["energy_start"]["label"] = "SAMPLE"
        self.assert_hold(value, "COUNTERS_NOT_MEASURED")

    def test_unknown_method_is_not_guessed(self):
        value = fixture()
        value["energy_j"]["method"] = "power times elapsed seconds"
        self.assert_hold(value, "DELTA_METHOD_UNSUPPORTED")

    def test_counter_pair_is_required(self):
        value = fixture()
        del value["energy_end"]
        self.assert_hold(value, "COUNTER_PAIR_UNAVAILABLE")

    def test_different_device_labels_are_rejected(self):
        value = fixture()
        value["energy_end"]["gpu"] = "other-device"
        self.assert_hold(value, "COUNTER_DEVICE_LABEL_MISMATCH")

    def test_counter_reset_is_not_zero_energy(self):
        value = fixture()
        value["energy_end"]["energy_j"] = 1
        self.assert_hold(value, "COUNTER_RESET")

    def test_counter_delta_must_match_declared_delta(self):
        value = fixture()
        value["energy_j"]["joules"] = 37768237.571
        self.assert_hold(value, "DELTA_COUNTER_MISMATCH")

    def test_zero_is_a_valid_recorded_delta(self):
        value = fixture()
        value["energy_end"]["energy_j"] = value["energy_start"]["energy_j"]
        value["energy_j"]["joules"] = 0
        self.assertEqual(
            energy.inspect_receipt(seal(value))["recorded_delta_joules"], 0
        )

    def test_booleans_strings_negative_and_null_are_not_energy(self):
        for item in (True, False, "1180.1", -1, None):
            with self.subTest(value=item):
                value = fixture()
                value["energy_j"]["joules"] = item
                self.assert_hold(value, "INVALID_ENERGY_NUMBER")

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        for raw in (
            b'{"kind":"a","kind":"b"}',
            b'{"value":NaN}',
            b'{"value":Infinity}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(energy.EvidenceHold):
                    energy.inspect_receipt(raw)
        value = fixture()
        value["energy_j"]["joules"] = float("inf")
        with self.assertRaises(energy.EvidenceHold):
            energy.inspect_receipt(seal(value))

    def test_integrity_mismatch_and_missing_hash_are_rejected(self):
        value = json.loads(seal(fixture()))
        value["detail"] = "tampered"
        with self.assertRaisesRegex(energy.EvidenceHold, "RECEIPT_HASH_MISMATCH"):
            energy.inspect_receipt(json.dumps(value).encode())
        with self.assertRaisesRegex(energy.EvidenceHold, "RECEIPT_HASH_UNAVAILABLE"):
            energy.inspect_receipt(json.dumps(fixture()).encode())

    def test_unsupported_shapes_and_byte_budgets_are_rejected(self):
        for raw in (b"", b"[]", b"null", b"not json", b"x" * (energy.MAX_BYTES + 1)):
            with self.subTest(size=len(raw)):
                with self.assertRaises(energy.EvidenceHold):
                    energy.inspect_receipt(raw)
        value = fixture()
        value["kind"] = "unreviewed-schema"
        self.assert_hold(value, "UNSUPPORTED_RECEIPT_SCHEMA")

    def test_no_arbitrary_numeric_energy_key_fallback(self):
        value = fixture()
        del value["energy_j"]
        value["energy_mj"] = 123456
        self.assert_hold(value, "EXPLICIT_DELTA_UNAVAILABLE")

    def test_unknown_adapter_is_not_invented(self):
        value = fixture()
        del value["adapter"]
        self.assertIsNone(energy.inspect_receipt(seal(value))["adapter"])

    def test_duplicate_receipt_is_counted_once_and_total_stays_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = seal(fixture())
            (root / "salad-a.json").write_bytes(raw)
            (root / "salad-b.json").write_bytes(
                json.dumps(json.loads(raw), indent=2).encode()
            )
            (root / "salad-bad.json").write_bytes(b"invalid")
            result = energy.reconcile(root)
            self.assertEqual(result["files_examined"], 3)
            self.assertEqual(result["valid_unique_delta_receipts"], 1)
            self.assertEqual(result["duplicate_receipts"], 1)
            self.assertEqual(result["recorded_device_delta_subtotal_joules"], 1180.1)
            self.assertIsNone(result["total_measured_joules"])
            self.assertEqual(result["estate_energy_state"], "UNVERIFIED")
            self.assertFalse(result["provider_mutation"])
            self.assertRegex(result["input_manifest_sha256"], r"^[0-9a-f]{64}$")

    def test_distinct_receipts_do_not_establish_nonoverlapping_physical_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(2):
                value = fixture()
                value["detail"] = str(i)
                (root / f"salad-{i}.json").write_bytes(seal(value))
            result = energy.reconcile(root)
            self.assertEqual(result["valid_unique_delta_receipts"], 2)
            self.assertEqual(result["recorded_device_delta_subtotal_joules"], 2360.2)
            self.assertIsNone(result["total_measured_joules"])

    def test_empty_coverage_is_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            result = energy.reconcile(Path(directory))
            self.assertIsNone(result["recorded_device_delta_subtotal_joules"])
            self.assertIsNone(result["total_measured_joules"])

    def test_file_budget_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(2):
                (root / f"salad-{i}.json").write_bytes(seal(fixture()))
            with patch.object(energy, "MAX_FILES", 1):
                with self.assertRaisesRegex(energy.EvidenceHold, "RECEIPT_FILE_BUDGET"):
                    energy.reconcile(root)

    def test_cli_refuses_to_overwrite_old_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "salad-a.json").write_bytes(seal(fixture()))
            out = root / "audit.json"
            out.write_text("original", encoding="utf-8")
            with patch("builtins.print"):
                code = energy.main(["--receipt-dir", str(root), "--output", str(out)])
            self.assertEqual(code, 2)
            self.assertEqual(out.read_text(), "original")

    def test_cli_writes_a_new_unsigned_local_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "salad-a.json").write_bytes(seal(fixture()))
            out = root / "audit.json"
            with patch("builtins.print"):
                code = energy.main(["--receipt-dir", str(root), "--output", str(out)])
            self.assertEqual(code, 0)
            result = json.loads(out.read_bytes())
            self.assertEqual(result["signature"], "UNSIGNED")
            self.assertFalse(result["training_completion_verified"])


if __name__ == "__main__":
    unittest.main()
