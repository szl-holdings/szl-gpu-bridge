"""Hermetic negative controls for reported terminal events, no provider SDK."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.salad import terminal_evidence_shadow as s


def sample(**changed):
    row = {
        "schema": s.SCHEMA,
        "surface": "EDGE_UI",
        "api_state": "DENIED_401",
        "group_status": "RUNNING",
        "desired_replicas": 1,
        "actual_counts": {"allocating": 1, "creating": 0, "running": 0, "stopping": 0},
        "instance_get_confirmed": False,
        "events": ["EXIT_0", "ALLOCATED", "EXIT_0", "STARTING"],
        "trainer_markers": ["ALREADY_DONE", "FRONTIER_EXIT"],
        "claimed_evidence": {key: False for key in s.CLAIMS},
    }
    row.update(changed)
    return row


class TerminalEvidenceTests(unittest.TestCase):
    def test_exit0_cycle_suspected_not_completed(self):
        value = s.evaluate(sample())
        self.assertEqual(value["risk"], "RESTART_LOOP_SUSPECTED_NOT_PROVEN")
        self.assertEqual(value["reported_exit0_events"], 2)
        self.assertIn("RUNNING_LABEL_ZERO_ACTUAL_REPORTED", value["reason_codes"])
        self.assertIn(
            "APPROVED_API_ACCESS_DENIED_OR_RATE_LIMITED", value["reason_codes"]
        )
        self.assertFalse(value["training_completed"])
        self.assertFalse(value["start_stop_scale_authorized"])

    def test_single_exit_is_not_loop(self):
        value = s.evaluate(sample(events=["EXIT_0"]))
        self.assertNotIn("REPEATED_EXIT0_SUSPECTED", value["reason_codes"])

    def test_stopped_and_zero_replicas_not_completion(self):
        row = sample(
            group_status="STOPPED",
            desired_replicas=0,
            actual_counts={"allocating": 0, "creating": 0, "running": 0, "stopping": 0},
        )
        value = s.evaluate(row)
        self.assertFalse(value["training_completed"])
        self.assertNotIn("REPEATED_EXIT0_SUSPECTED", value["reason_codes"])

    def test_probe_readiness_does_not_prove_training(self):
        row = sample(
            surface="APPROVED_SDK",
            api_state="AUTHORIZED_READ",
            instance_get_confirmed=True,
            events=["PROBE_READY", "INSTANCE_RUNNING"],
            trainer_markers=["STEP_PROGRESS_REPORTED"],
        )
        value = s.evaluate(row)
        self.assertIn("PROVIDER_PROBE_NOT_TRAINER_PROGRESS", value["reason_codes"])
        self.assertFalse(value["training_steps_verified"])

    def test_all_claims_true_still_unverified(self):
        row = sample(claimed_evidence={key: True for key in s.CLAIMS})
        value = s.evaluate(row)
        self.assertIn(
            "SELF_REPORTED_EVIDENCE_NOT_INDEPENDENTLY_VERIFIED", value["reason_codes"]
        )
        for name in (
            "training_completed",
            "final_artifact_verified",
            "independent_backup_verified",
            "flagship54_shutdown_authorized",
            "training_dispatch_authorized",
            "release_qualified",
        ):
            self.assertFalse(value[name])

    def test_unknown_actual_counts_not_zero(self):
        value = s.evaluate(sample(actual_counts=None))
        self.assertIn("REPLICA_COUNTS_PARTIALLY_UNOBSERVED", value["reason_codes"])

    def test_allocating_without_running_label_does_not_trigger_label_gap(self):
        value = s.evaluate(sample(group_status="ALLOCATING"))
        self.assertNotIn("RUNNING_LABEL_ZERO_ACTUAL_REPORTED", value["reason_codes"])

    def test_stopped_label_with_running_instance_is_conflicting_evidence(self):
        row = sample(
            surface="APPROVED_SDK",
            api_state="AUTHORIZED_READ",
            instance_get_confirmed=True,
            group_status="STOPPED",
            desired_replicas=1,
            actual_counts={"allocating": 0, "creating": 0, "running": 1, "stopping": 0},
            events=[],
            trainer_markers=[],
        )
        result = s.evaluate(row)
        self.assertIn("STOPPED_LABEL_RUNNING_ACTUAL_REPORTED", result["reason_codes"])
        self.assertEqual(result["risk"], "REPORTED_STATUS_INSTANCE_CONFLICT")
        self.assertEqual(result["decision"], "HOLD_NO_PROVIDER_ACTION")
        self.assertFalse(result["training_completed"])
        self.assertFalse(result["start_stop_scale_authorized"])

    def test_stopped_zero_running_is_not_a_status_conflict(self):
        row = sample(
            surface="APPROVED_SDK",
            api_state="AUTHORIZED_READ",
            instance_get_confirmed=True,
            group_status="STOPPED",
            desired_replicas=0,
            actual_counts={"allocating": 0, "creating": 0, "running": 0, "stopping": 0},
            events=[],
            trainer_markers=[],
        )
        result = s.evaluate(row)
        self.assertNotIn("STOPPED_LABEL_RUNNING_ACTUAL_REPORTED", result["reason_codes"])
        self.assertEqual(result["risk"], "TRAINING_EVIDENCE_UNVERIFIED")
        self.assertFalse(result["training_dispatch_authorized"])

    def test_restart_loop_warning_takes_precedence_over_stopped_conflict(self):
        row = sample(
            surface="APPROVED_SDK",
            api_state="AUTHORIZED_READ",
            instance_get_confirmed=True,
            group_status="STOPPED",
            desired_replicas=1,
            actual_counts={"allocating": 0, "creating": 0, "running": 1, "stopping": 0},
            events=["EXIT_0", "EXIT_0"],
            trainer_markers=[],
        )
        result = s.evaluate(row)
        self.assertIn("STOPPED_LABEL_RUNNING_ACTUAL_REPORTED", result["reason_codes"])
        self.assertIn("REPEATED_EXIT0_SUSPECTED", result["reason_codes"])
        self.assertEqual(result["risk"], "RESTART_LOOP_SUSPECTED_NOT_PROVEN")
        self.assertFalse(result["training_completed"])

    def test_nonatomic_counts_not_deployment_proof(self):
        row = sample(
            actual_counts={"allocating": 1, "creating": 1, "running": 1, "stopping": 0}
        )
        self.assertIn(
            "NONATOMIC_OR_CONFLICTING_INSTANCE_COUNTS", s.evaluate(row)["reason_codes"]
        )

    def test_final_filename_does_not_prove_artifact(self):
        value = s.evaluate(
            sample(trainer_markers=["FINAL_FILENAME", "CHECKPOINT_FILENAME"])
        )
        self.assertIn("FILENAME_NOT_ARTIFACT_INTEGRITY", value["reason_codes"])
        self.assertFalse(value["final_artifact_verified"])

    def test_input_is_not_mutated_and_output_is_deterministic(self):
        row = sample()
        original = copy.deepcopy(row)
        self.assertEqual(s.evaluate(row), s.evaluate(row))
        self.assertEqual(row, original)

    def test_denied_api_cannot_claim_independent_instance_get(self):
        row = sample(surface="APPROVED_SDK", instance_get_confirmed=True)
        with self.assertRaisesRegex(s.ShadowHold, "CONTRADICTORY_AUTH_CONTEXT"):
            s.evaluate(row)

    def test_approved_api_without_group_counts_cannot_claim_instance_get(self):
        row = sample(
            surface="APPROVED_SDK",
            api_state="AUTHORIZED_READ",
            actual_counts=None,
            instance_get_confirmed=True,
        )
        with self.assertRaisesRegex(s.ShadowHold, "CONTRADICTORY_AUTH_CONTEXT"):
            s.evaluate(row)

    def test_edge_ui_cannot_claim_sdk_instance_get(self):
        with self.assertRaises(s.ShadowHold):
            s.evaluate(sample(instance_get_confirmed=True))

    def test_bool_as_integer_is_rejected(self):
        for field in s.COUNT_KEYS:
            row = sample()
            row["actual_counts"][field] = True
            with self.subTest(field=field), self.assertRaises(s.ShadowHold):
                s.evaluate(row)

    def test_invalid_claim_type_is_rejected(self):
        row = sample()
        row["claimed_evidence"]["signer_verified"] = "true"
        with self.assertRaisesRegex(s.ShadowHold, "INVALID_EXTERNAL_CLAIM"):
            s.evaluate(row)

    def test_raw_identifiers_not_accepted(self):
        row = sample()
        row["private_group_name"] = "PRIVATE_NEVER_ECHO"
        with self.assertRaises(s.ShadowHold):
            s.evaluate(row)

    def test_raw_log_lines_not_accepted(self):
        for key in ("events", "trainer_markers"):
            row = sample()
            row[key] = ["PRIVATE_SECRET_EXAMPLE"]
            with self.subTest(key=key), self.assertRaises(s.ShadowHold):
                s.evaluate(row)

    def test_bad_schema_and_types_rejected(self):
        for field, value in (
            ("schema", "x"),
            ("api_state", "SUCCESS"),
            ("desired_replicas", False),
            ("group_status", []),
        ):
            row = sample()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(s.ShadowHold):
                s.evaluate(row)

    def test_duplicate_json_keys_are_rejected_in_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            path.write_text(
                '{"schema":"a","schema":"b","secret":"PRIVATE_NEVER_ECHO"}',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(s.main([str(path)]), 78)
            self.assertNotIn("PRIVATE_NEVER_ECHO", stdout.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["risk"], "INVALID_INPUT")

    def test_nonfinite_and_oversized_inputs_hold(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            for content in (b'{"value":NaN}', b"x" * (s.MAX_BYTES + 1)):
                path.write_bytes(content)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(s.main([str(path)]), 78)

    def test_cli_valid_observation_still_exits_hold(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.json"
            path.write_text(json.dumps(sample()), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(s.main([str(path)]), 78)
            doc = json.loads(stdout.getvalue())
            self.assertFalse(doc["training_dispatch_authorized"])
            self.assertEqual(doc["decision"], "HOLD_NO_PROVIDER_ACTION")
