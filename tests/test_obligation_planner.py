"""Adversarial stdlib-unittest fixtures for the offline evidence advisor.

This project runs unittest discovery in native CI; no pytest dependency.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.evidence import obligation_planner as advice


def request(*, authority="ALLOW", budget=15, obligations=None, observations=None):
    return {
        "schema": advice.REQUEST_SCHEMA,
        "operation_id": "fixture:promotion-review",
        "authority_state": authority,
        "budget_units": budget,
        "obligations": obligations
        if obligations is not None
        else [
            {"id": label, "state": "OPEN", "evidence_digest": None}
            for label in ("source", "device", "window")
        ],
        "observations": observations
        if observations is not None
        else [
            {
                "id": "inspect-source",
                "mode": "READ_ONLY",
                "admission_state": "ALLOW",
                "cost_units": 2,
                "may_inform": ["source"],
            },
            {
                "id": "inspect-device-window",
                "mode": "READ_ONLY",
                "admission_state": "ALLOW",
                "cost_units": 6,
                "may_inform": ["device", "window"],
            },
            {
                "id": "inspect-all",
                "mode": "READ_ONLY",
                "admission_state": "ALLOW",
                "cost_units": 11,
                "may_inform": ["source", "device", "window"],
            },
        ],
    }


class ObligationAdvisorTests(unittest.TestCase):
    def test_exact_low_cost_selection_and_noncompensation(self):
        result = advice.advise(request())
        self.assertEqual(result["state"], "ADVISORY_CANDIDATE")
        self.assertEqual(
            result["selected_read_observations"],
            ["inspect-device-window", "inspect-source"],
        )
        self.assertEqual(result["proposed_cost_units"], 8)
        self.assertIs(result["authority_granted"], False)
        self.assertEqual(result["requirements_satisfied_by_advice"], 0)
        self.assertIs(result["quality_or_promotion_qualified"], False)
        self.assertEqual(result["remote_effects_performed"], 0)

    def test_denied_and_unknown_operations_never_receive_a_plan(self):
        for state, expected in (
            ("DENY", "BLOCKED_POLICY"),
            ("UNKNOWN", "HOLD_AUTHORITY"),
        ):
            with self.subTest(state=state):
                result = advice.advise(request(authority=state))
                self.assertEqual(result["state"], expected)
                self.assertEqual(result["selected_read_observations"], [])
                self.assertIs(result["authority_granted"], False)

    def test_stale_obligation_does_not_become_satisfied(self):
        spec = request()
        spec["obligations"][0]["state"] = "STALE"
        result = advice.advise(spec)
        self.assertIn("source", result["evidence_obligations_open"])
        self.assertEqual(result["requirements_satisfied_by_advice"], 0)

    def test_caller_declared_observation_is_not_release_authority(self):
        result = advice.advise(
            request(
                obligations=[
                    {"id": "source", "state": "OBSERVED", "evidence_digest": "a" * 64}
                ],
                observations=[],
            )
        )
        self.assertEqual(result["state"], "NO_MISSING_INPUT_OBLIGATIONS")
        self.assertIs(result["authority_granted"], False)
        self.assertIs(result["quality_or_promotion_qualified"], False)
        self.assertEqual(result["selected_read_observations"], [])

    def test_observed_input_requires_digest_without_proving_it(self):
        spec = request()
        spec["obligations"][0]["state"] = "OBSERVED"
        with self.assertRaisesRegex(advice.EvidenceHold, "OBSERVED_DIGEST_UNAVAILABLE"):
            advice.advise(spec)

    def test_unadmitted_reads_and_writes_are_excluded(self):
        for mode, admission in (
            ("WRITE", "ALLOW"),
            ("READ_ONLY", "DENY"),
            ("READ_ONLY", "UNKNOWN"),
        ):
            with self.subTest(mode=mode, admission=admission):
                spec = request(
                    obligations=[
                        {"id": "source", "state": "OPEN", "evidence_digest": None}
                    ],
                    observations=[
                        {
                            "id": "candidate",
                            "mode": mode,
                            "admission_state": admission,
                            "cost_units": 0,
                            "may_inform": ["source"],
                        }
                    ],
                )
                result = advice.advise(spec)
                self.assertEqual(result["state"], "HOLD_NO_READ_COVERAGE")
                self.assertIn("candidate", result["excluded_observations"])
                self.assertEqual(result["selected_read_observations"], [])

    def test_zero_cost_is_not_physical_measurement(self):
        spec = request(
            budget=0,
            obligations=[{"id": "source", "state": "OPEN", "evidence_digest": None}],
            observations=[
                {
                    "id": "free-local-read",
                    "mode": "READ_ONLY",
                    "admission_state": "ALLOW",
                    "cost_units": 0,
                    "may_inform": ["source"],
                }
            ],
        )
        result = advice.advise(spec)
        self.assertEqual(result["proposed_cost_units"], 0)
        self.assertEqual(result["state"], "ADVISORY_CANDIDATE")
        self.assertEqual(result["signature"], "UNSIGNED")

    def test_denied_write_does_not_compensate_for_budget(self):
        spec = request(budget=5)
        spec["observations"].append(
            {
                "id": "unsafe-skip",
                "mode": "WRITE",
                "admission_state": "ALLOW",
                "cost_units": 0,
                "may_inform": ["device", "window", "source"],
            }
        )
        result = advice.advise(spec)
        self.assertEqual(result["state"], "HOLD_UNCOVERED_OR_OVER_BUDGET")
        self.assertEqual(result["selected_read_observations"], [])
        self.assertEqual(
            result["excluded_observations"]["unsafe-skip"],
            "NOT_A_READ_ONLY_OBSERVATION",
        )

    def test_equal_cost_tie_breaks_by_count_and_stable_identifier(self):
        spec = request(
            obligations=[{"id": "source", "state": "OPEN", "evidence_digest": None}],
            observations=[
                {
                    "id": name,
                    "mode": "READ_ONLY",
                    "admission_state": "ALLOW",
                    "cost_units": 3,
                    "may_inform": ["source"],
                }
                for name in ("z-last", "a-first")
            ],
        )
        self.assertEqual(advice.advise(spec)["selected_read_observations"], ["a-first"])

    def test_state_cap_abstains_instead_of_claiming_an_optimum(self):
        spec = request(
            budget=0,
            obligations=[
                {"id": "o" + str(i), "state": "OPEN", "evidence_digest": None}
                for i in range(16)
            ],
            observations=[
                {
                    "id": "read" + str(i),
                    "mode": "READ_ONLY",
                    "admission_state": "ALLOW",
                    "cost_units": 0,
                    "may_inform": ["o" + str(i)],
                }
                for i in range(16)
            ],
        )
        result = advice.advise(spec)
        self.assertEqual(result["state"], "ABSTAIN_SEARCH_BOUND")
        self.assertEqual(result["selected_read_observations"], [])
        self.assertIsNone(result["proposed_cost_units"])

    def test_cost_type_and_bounds_are_strict(self):
        for bad in (True, -1, 1.5, 10**20, "1", None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(advice.EvidenceHold, "INVALID_COST_UNITS"):
                    advice.advise(request(budget=bad))

    def test_duplicate_ids_and_unknown_targets_are_rejected(self):
        spec = request()
        spec["obligations"].append(spec["obligations"][0])
        with self.assertRaisesRegex(advice.EvidenceHold, "DUPLICATE_OBLIGATION"):
            advice.advise(spec)
        spec = request()
        spec["observations"][0]["may_inform"] = ["unknown"]
        with self.assertRaisesRegex(advice.EvidenceHold, "INVALID_OBSERVATION_TARGETS"):
            advice.advise(spec)

    def test_strict_json_refuses_duplicate_keys_and_nonfinite(self):
        for raw, reason in (
            (b'{"schema":1,"schema":2}', "DUPLICATE_JSON_MEMBER"),
            (b'{"x":NaN}', "NONFINITE_JSON"),
            (b'{"x":Infinity}', "NONFINITE_JSON"),
        ):
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(advice.EvidenceHold, reason):
                    advice.load_strict(raw)

    def test_cli_exclusive_output_and_unsigned_result(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "request.json"
            target = Path(directory) / "result.json"
            raw = json.dumps(request(), sort_keys=True).encode()
            source.write_bytes(raw)
            command = [
                sys.executable,
                "-B",
                advice.__file__,
                "--input",
                str(source),
                "--output",
                str(target),
            ]
            first = subprocess.run(
                command, capture_output=True, text=True, timeout=20, check=False
            )
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            result = json.loads(target.read_text())
            self.assertIsNotNone(result["input_bytes_sha256"])
            self.assertEqual(result["signature"], "UNSIGNED")
            self.assertIs(result["authority_granted"], False)
            self.assertEqual(source.read_bytes(), raw)
            second = subprocess.run(
                command, capture_output=True, text=True, timeout=20, check=False
            )
            self.assertEqual(second.returncode, 2)
            self.assertEqual(json.loads(second.stdout)["state"], "HOLD")
            self.assertEqual(
                target.read_text(), json.dumps(result, indent=2, allow_nan=False) + "\n"
            )

    def test_schema_and_field_injections_hold(self):
        spec = request()
        spec["admit_everything"] = True
        with self.assertRaisesRegex(
            advice.EvidenceHold, "QUERY_FIELDS_UNAVAILABLE_OR_UNKNOWN"
        ):
            advice.advise(spec)
        del spec["admit_everything"]
        spec["schema"] = "unsupported"
        with self.assertRaisesRegex(advice.EvidenceHold, "UNSUPPORTED_QUERY_SCHEMA"):
            advice.advise(spec)


if __name__ == "__main__":
    unittest.main()
