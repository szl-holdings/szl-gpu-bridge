# SPDX-License-Identifier: Apache-2.0
"""Synthetic fail-closed controls; standard-library unittest for native CI."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.salad import honesty_energy_frontier as frontier


def examples():
    identity = {
        "model": "SZLHOLDINGS/synthetic-do-not-publish",
        "model_revision": "a" * 40,
        "run_id": "synthetic_eval_001",
        "fixture_sha256": "b" * 64,
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
    }
    battery = {
        "schema": frontier.BATTERY_SCHEMA,
        **identity,
        "receipt_sha256": "c" * 64,
        "state": "MEASURED",
        "score": "3/6",
        "rows": [
            {
                "id": name,
                "kind": "answerable" if name.startswith("a") else "unanswerable",
                "abstained": False,
                "verdict": "PASS" if name.startswith("a") else "FAIL",
            }
            for name in frontier.IDS
        ],
    }
    energy = {
        "schema": frontier.ENERGY_SCHEMA,
        **identity,
        "receipt_sha256": "d" * 64,
        "measurement_scope": "EVALUATION_SIX_PROBES",
        "method": "NVML_DEVICE_TOTAL_MJ",
        "counter_unit": "mJ",
        "rows": [
            {
                "id": name,
                "counter_start_mj": i * 1000,
                "counter_end_mj": (i + 1) * 1000,
            }
            for i, name in enumerate(frontier.IDS)
        ],
    }
    return battery, energy


class HonestyEnergyTests(unittest.TestCase):
    def held(self, battery, energy, reason):
        report = frontier.assess(battery, energy)
        self.assertEqual(report["status"], "HOLD")
        self.assertEqual(report["reason_code"], reason)
        for key in ("metric", "source_identity", "reported_receipts"):
            self.assertIsNone(report[key])
        for key in (
            "signature_verified",
            "production_authorization",
            "ranking_authorized",
            "promotion_authorized",
            "external_writes",
        ):
            self.assertIs(report[key], False)

    def test_synthetic_zero_abstention_quality_and_unsigned_provenance(self):
        battery, energy = examples()
        result = frontier.assess(battery, energy)
        self.assertEqual(result["status"], "DERIVED_UNATTESTED")
        m = result["metric"]
        self.assertEqual((m["correct_answerable"], m["correct_abstentions"]), (3, 0))
        self.assertEqual(m["measured_device_joules"], 6.0)
        self.assertEqual(m["device_joules_per_valid_outcome"], 2.0)
        self.assertEqual(m["joint_quality_geomean"], 0)
        self.assertEqual(m["joint_quality_per_kilojoule"], 0)
        self.assertEqual(len(m["per_probe_device_joules"]), 6)
        self.assertEqual(result["source_identity"]["run_id"], "synthetic_eval_001")
        self.assertEqual(
            result["reported_receipts"]["verification"],
            "UNVERIFIED_REPORTED_IDENTIFIERS",
        )
        self.assertEqual(result["receipt_integrity"], "REPORTED_DIGEST_UNVERIFIED")
        for key in (
            "signature_verified",
            "ranking_authorized",
            "production_authorization",
            "promotion_authorized",
        ):
            self.assertIs(result[key], False)
        digest = result.pop("report_sha256")
        self.assertEqual(
            digest, hashlib.sha256(frontier._canonical(result)).hexdigest()
        )

    def test_correct_abstentions_reach_one_without_ranking_authority(self):
        battery, energy = examples()
        for row in battery["rows"][3:]:
            row.update(abstained=True, verdict="PASS")
        battery["score"] = "6/6"
        report = frontier.assess(battery, energy)
        self.assertEqual(report["metric"]["joint_quality_geomean"], 1.0)
        self.assertEqual(report["metric"]["device_joules_per_valid_outcome"], 1.0)
        self.assertAlmostEqual(
            report["metric"]["joint_quality_per_kilojoule"], 1000 / 6, places=6
        )
        self.assertIs(report["ranking_authorized"], False)

    def test_independent_model_run_fixture_and_gpu_join_gates(self):
        changes = (
            ("model", "SZLHOLDINGS/other-model", "MODEL_ID_MISMATCH"),
            ("model_revision", "1" * 40, "MODEL_REVISION_MISMATCH"),
            ("run_id", "unrelated_run", "RUN_ID_MISMATCH"),
            ("fixture_sha256", "d" * 64, "FIXTURE_DIGEST_MISMATCH"),
            (
                "gpu_uuid",
                "GPU-11111111-2222-3333-4444-555555555555",
                "GPU_UUID_MISMATCH",
            ),
        )
        for field, value, code in changes:
            with self.subTest(field=field):
                battery, energy = examples()
                energy[field] = value
                self.held(battery, energy, code)

    def test_battery_verdicts_and_score_cannot_be_laundered(self):
        cases = (
            ("wrong_score", "SCORE_CONTRADICTS_ROWS"),
            ("wrong_abstention", "VERDICT_ABSTENTION_CONFLICT"),
            ("bool_as_abstention", "ABSTENTION_FLAG_INVALID"),
            ("duplicate_probe", "PROBE_ID_INVALID"),
            ("missing_probe", "PROBE_COVERAGE_INVALID"),
            ("unknown_probe", "PROBE_ID_INVALID"),
            ("wrong_kind", "PROBE_KIND_INVALID"),
            ("not_measured", "BATTERY_NOT_MEASURED"),
        )
        for case, code in cases:
            with self.subTest(case=case):
                battery, energy = examples()
                if case == "wrong_score":
                    battery["score"] = "6/6"
                elif case == "wrong_abstention":
                    battery["rows"][0]["abstained"] = True
                elif case == "bool_as_abstention":
                    battery["rows"][0]["abstained"] = 1
                elif case == "duplicate_probe":
                    battery["rows"][1]["id"] = "a1"
                elif case == "missing_probe":
                    battery["rows"].pop()
                elif case == "unknown_probe":
                    battery["rows"][1]["id"] = "x2"
                elif case == "wrong_kind":
                    battery["rows"][1]["kind"] = "unanswerable"
                else:
                    battery["state"] = "UNMEASURED"
                self.held(battery, energy, code)

    def test_energy_counter_and_measurement_scope_gates(self):
        cases = (
            ("training_scope", "ENERGY_SCOPE_MISMATCH"),
            ("not_nvml", "ENERGY_METHOD_UNSUPPORTED"),
            ("wrong_unit", "ENERGY_UNIT_INVALID"),
            ("missing_probe", "ENERGY_COVERAGE_INVALID"),
            ("duplicate_probe", "ENERGY_PROBE_ID_INVALID"),
            ("counter_rollback", "COUNTER_RANGE_INVALID"),
            ("counter_reset", "COUNTER_NONMONOTONE"),
            ("zero_delta", "COUNTER_RANGE_INVALID"),
            ("bool_counter", "COUNTER_RANGE_INVALID"),
            ("wrong_order", "ENERGY_ORDER_INVALID"),
        )
        for case, code in cases:
            with self.subTest(case=case):
                battery, energy = examples()
                rows = energy["rows"]
                if case == "training_scope":
                    energy["measurement_scope"] = "TRAINING"
                elif case == "not_nvml":
                    energy["method"] = "ASSUMED_WATTS"
                elif case == "wrong_unit":
                    energy["counter_unit"] = "J"
                elif case == "missing_probe":
                    rows.pop()
                elif case == "duplicate_probe":
                    rows[1]["id"] = "a1"
                elif case == "counter_rollback":
                    rows[0]["counter_end_mj"] = -5
                elif case == "counter_reset":
                    rows[1]["counter_start_mj"] = 0
                elif case == "zero_delta":
                    rows[0]["counter_end_mj"] = 0
                elif case == "bool_counter":
                    rows[0]["counter_start_mj"] = True
                else:
                    rows[0]["id"], rows[1]["id"] = rows[1]["id"], rows[0]["id"]
                self.held(battery, energy, code)

    def test_legacy_training_leg_and_untrained_battery_do_not_join(self):
        battery, energy = examples()
        battery.pop("schema")
        battery["kind"] = "szl.estate-battery/base-baseline"
        energy["schema"] = "szl.gpu-mesh-benchmark/v1"
        self.held(battery, energy, "SCHEMA_INVALID")

    def test_missing_source_pins_are_blocking(self):
        cases = (
            ("model_revision", "MODEL_REVISION_UNAVAILABLE"),
            ("run_id", "RUN_ID_UNAVAILABLE"),
            ("fixture_sha256", "FIXTURE_DIGEST_UNAVAILABLE"),
            ("gpu_uuid", "GPU_UUID_UNAVAILABLE"),
            ("receipt_sha256", "RECEIPT_DIGEST_UNAVAILABLE"),
        )
        for field, code in cases:
            with self.subTest(field=field):
                battery, energy = examples()
                battery.pop(field)
                self.held(battery, energy, code)

    def test_untrusted_claimed_signature_and_authority_never_transfer(self):
        battery, energy = examples()
        battery.update(signature_verified=True, production_authorization=True)
        energy["ranking_authorized"] = True
        report = frontier.assess(battery, energy)
        self.assertEqual(report["status"], "DERIVED_UNATTESTED")
        for key in (
            "signature_verified",
            "production_authorization",
            "ranking_authorized",
        ):
            self.assertIs(report[key], False)
        self.assertNotIn("signature_verified", report["source_identity"])

    def test_pathlike_repo_names_and_bad_gpu_uuid_rejected(self):
        for name in ("../model", "owner/..", "./model", "owner/."):
            with self.subTest(name=name):
                battery, energy = examples()
                battery["model"] = name
                self.held(battery, energy, "MODEL_ID_INVALID")
        for gpu in ("GPU----------", "GPU-gggggggggggg", "gpu-000000000000"):
            with self.subTest(gpu=gpu):
                battery, energy = examples()
                energy["gpu_uuid"] = gpu
                self.held(battery, energy, "GPU_UUID_UNAVAILABLE")

    def test_unsigned_64bit_counter_bounds_and_huge_inputs(self):
        battery, energy = examples()
        energy["rows"][-1]["counter_end_mj"] = 1 << 4000
        self.held(battery, energy, "COUNTER_RANGE_INVALID")
        battery, energy = examples()
        shift = frontier.MAX_COUNTER_MJ - 10_000
        for row in energy["rows"]:
            row["counter_start_mj"] += shift
            row["counter_end_mj"] += shift
        result = frontier.assess(battery, energy)
        self.assertEqual(result["status"], "DERIVED_UNATTESTED")
        self.assertEqual(result["metric"]["measured_device_joules"], 6.0)

    def test_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.json"
            for content, code in (
                ('{"model":"a","model":"b"}', "DUPLICATE_JSON_KEY"),
                ('{"x":NaN}', "NONFINITE_JSON"),
                ('{"x":Infinity}', "NONFINITE_JSON"),
                ('{"x":1e999}', "NONFINITE_JSON"),
            ):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(frontier.Hold, code):
                        frontier.read_input(path)

    def test_missing_and_oversized_local_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            huge = root / "huge.json"
            huge.write_bytes(b" " * (frontier.MAX_BYTES + 1))
            with self.assertRaisesRegex(frontier.Hold, "INPUT_BYTE_LIMIT"):
                frontier.read_input(huge)
            with self.assertRaisesRegex(frontier.Hold, "INPUT_UNAVAILABLE_OR_INVALID"):
                frontier.read_input(root / "missing.json")

    def test_symlink_never_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "link.json"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("host lacks symlink creation privilege")
            with self.assertRaisesRegex(frontier.Hold, "INPUT_NOT_REGULAR"):
                frontier.read_input(link)

    def test_cli_hold_redacts_untrusted_text(self):
        battery, energy = examples()
        battery["run_id"] = "sensitive-local-opaque-token"
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = Path(d) / "battery.json", Path(d) / "energy.json"
            p1.write_text(json.dumps(battery), encoding="utf-8")
            p2.write_text(json.dumps(energy), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = frontier.main(["--battery", str(p1), "--energy", str(p2)])
            self.assertEqual(status, 3)
            payload = stdout.getvalue()
            self.assertEqual(json.loads(payload)["status"], "HOLD")
            self.assertNotIn("sensitive-local-opaque-token", payload)

    def test_cli_deterministic_and_has_no_effectors(self):
        battery, energy = examples()
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = Path(d) / "battery.json", Path(d) / "energy.json"
            p1.write_text(json.dumps(battery), encoding="utf-8")
            p2.write_text(json.dumps(energy), encoding="utf-8")
            outputs = []
            for _ in range(2):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(
                        frontier.main(["--battery", str(p1), "--energy", str(p2)]), 0
                    )
                outputs.append(out.getvalue())
            self.assertEqual(outputs[0], outputs[1])
            self.assertIs(json.loads(outputs[0])["external_writes"], False)


if __name__ == "__main__":
    unittest.main()
