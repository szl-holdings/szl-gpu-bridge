"""Offline, credential-free regression fixtures for the one-shot watch consumer."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.salad import observation_claim_graph as g


SHA = "a" * 64
REV = "b" * 40


def group(i, *, running=0, creating=0, classification="TRAINING"):
    return {
        "name": "szl-train-" + str(i),
        "project": "default",
        "classification": classification,
        "classification_basis": "NAME_HEURISTIC",
        "model_repo": None,
        "configured_max_steps": None,
        "desired_replicas": 1,
        "group_status": "running",
        "counts": {
            "allocating_count": 0,
            "creating_count": creating,
            "running_count": running,
            "stopping_count": 0,
        },
    }


def candidate(step=5, target=10, *, repo="SZLHOLDINGS/synthetic-evidence-lane"):
    return {
        "id": repo,
        "status": "OBSERVED",
        "revision": REV,
        "latest_checkpoint": step,
        "trainer_state": {
            "global_step": step,
            "max_steps": target,
            "trainer_state_sha256": SHA,
            "integrity": "PINNED_OBJECT_BYTES_VERIFIED",
        },
        "final_adapter": step == target,
    }


def report(at="2026-11-17T23:00:00+00:00", returned=132, kept=120):
    return {
        "watch_version": g.WATCH,
        "observed_at_utc": at,
        "providers": {
            "salad": {
                "projects": ["default"],
                "project_listing": {
                    "status": "UNSUPPORTED_ENDPOINT",
                    "http_code": 404,
                    "truncated": True,
                },
                "coverage": [
                    {
                        "project": "default",
                        "list": "OBSERVED",
                        "pagination": {
                            "returned": returned,
                            "pages": 1,
                            "truncated": False,
                        },
                    }
                ],
                "groups": [group(i) for i in range(kept)],
            },
            "huggingface": {
                "datasets_count_returned": 150,
                "dataset_limit": 150,
                "training_candidates": [candidate()],
            },
        },
    }


class CoverageTests(unittest.TestCase):
    def test_synthetic_twelve_omissions_are_not_hidden(self):
        o = g.derive(report())
        self.assertEqual(o["provider_coverage"]["salad"]["returned_count"], 132)
        self.assertEqual(o["provider_coverage"]["salad"]["captured_count"], 120)
        self.assertEqual(o["provider_coverage"]["salad"]["omitted_count"], 12)
        self.assertFalse(o["provider_coverage"]["salad"]["coverage_complete"])
        self.assertIn("SALAD_GROUP_DETAIL_OMISSION", o["findings"])
        self.assertIn("ORG_PROJECT_DISCOVERY_UNOBSERVED", o["findings"])
        self.assertIn("HF_DATASET_LIST_CAPPED", o["findings"])
        self.assertFalse(o["authority"]["qualification_conferred"])

    def test_legacy_120_of_120_is_not_org_completeness(self):
        o = g.derive(report(returned=120, kept=120))
        self.assertEqual(o["provider_coverage"]["salad"]["omitted_count"], 0)
        self.assertFalse(o["provider_coverage"]["salad"]["coverage_complete"])

    def test_full_project_discovery_plus_all_groups_would_be_covered(self):
        src = report(returned=2, kept=2)
        src["providers"]["salad"]["project_listing"] = {
            "pages": 1,
            "returned": 1,
            "truncated": False,
        }
        o = g.derive(src)
        self.assertTrue(o["provider_coverage"]["salad"]["coverage_complete"])

    def test_error_status_with_complete_pagination_is_not_org_discovery(self):
        source = report(returned=1, kept=1)
        source["providers"]["salad"]["project_listing"] = {
            "status": "ERROR",
            "pages": 1,
            "returned": 1,
            "truncated": False,
        }
        result = g.derive(source)
        self.assertFalse(result["provider_coverage"]["salad"]["coverage_complete"])
        self.assertFalse(
            result["provider_coverage"]["salad"]["project_discovery_complete"]
        )

    def test_default_project_fallback_not_in_org_response_is_not_complete(self):
        source = report(returned=1, kept=1)
        source["providers"]["salad"]["project_listing"] = {
            "pages": 1,
            "returned": 0,
            "truncated": False,
        }
        result = g.derive(source)
        self.assertFalse(
            result["provider_coverage"]["salad"]["project_discovery_complete"]
        )

    def test_unobserved_hf_model_does_not_confer_final_filename_presence(self):
        source = report(returned=1, kept=1)
        source["providers"]["huggingface"]["training_candidates"][0]["status"] = (
            "UNOBSERVED"
        )
        source["providers"]["huggingface"]["training_candidates"][0][
            "final_adapter"
        ] = True
        item = g.derive(source)["model_claims"][0]
        self.assertFalse(item["final_adapter_filename_present"])
        self.assertEqual(item["trainer_state"], "UNVERIFIED")

    def test_unobserved_project_is_not_a_zero_count(self):
        src = report(returned=2, kept=2)
        src["providers"]["salad"]["coverage"][0]["list"] = "UNOBSERVED"
        src["providers"]["salad"]["coverage"][0].pop("pagination")
        o = g.derive(src)
        self.assertIsNone(o["provider_coverage"]["salad"]["returned_count"])
        self.assertIsNone(o["provider_coverage"]["salad"]["omitted_count"])
        self.assertFalse(o["provider_coverage"]["salad"]["coverage_complete"])

    def test_more_retained_than_returned_is_invalid(self):
        with self.assertRaisesRegex(g.CoverageHold, "INVALID_PAGINATION"):
            g.derive(report(returned=119, kept=120))

    def test_missing_declared_project_fails_census(self):
        src = report(returned=1, kept=1)
        src["providers"]["salad"]["projects"].append("other")
        o = g.derive(src)
        self.assertEqual(
            o["provider_coverage"]["salad"]["unobserved_projects"], ["other"]
        )
        self.assertFalse(o["provider_coverage"]["salad"]["coverage_complete"])

    def test_boolean_running_count_is_unknown_not_one(self):
        src = report(returned=1, kept=1)
        src["providers"]["salad"]["groups"][0]["counts"]["running_count"] = True
        o = g.derive(src)
        self.assertIsNone(
            o["provider_coverage"]["salad"]["provider_instance_counts"]["running_count"]
        )
        self.assertEqual(
            o["provider_coverage"]["salad"]["groups_with_unknown_counts"], 1
        )

    def test_duplicate_identifiers_fail_closed(self):
        src = report(returned=2, kept=2)
        src["providers"]["salad"]["groups"][1]["name"] = src["providers"]["salad"][
            "groups"
        ][0]["name"]
        with self.assertRaisesRegex(g.CoverageHold, "DUPLICATE_SALAD_GROUP"):
            g.derive(src)

    def test_bad_counts_are_unobserved_not_silently_zero(self):
        src = report(returned=1, kept=1)
        src["providers"]["salad"]["groups"][0].pop("counts")
        with self.assertRaisesRegex(g.CoverageHold, "INVALID_SALAD_COUNTS"):
            g.derive(src)

    def test_group_without_project_cannot_be_accepted(self):
        src = report(returned=1, kept=1)
        src["providers"]["salad"]["groups"][0]["project"] = "other"
        with self.assertRaisesRegex(g.CoverageHold, "GROUP_WITHOUT_PROJECT_COVERAGE"):
            g.derive(src)

    def test_unsupported_mode_or_report_version_rejected(self):
        src = report()
        src["watch_version"] = "invalid"
        with self.assertRaisesRegex(g.CoverageHold, "UNRECOGNIZED_WATCH_VERSION"):
            g.derive(src)

    def test_unauthorized_claim_values_always_false(self):
        src = report(returned=1, kept=1)
        src["providers"]["huggingface"]["training_candidates"] = [
            candidate(step=10, target=10)
        ]
        o = g.derive(src)
        row = o["model_claims"][0]
        self.assertTrue(row["observed_terminal_step"])
        self.assertTrue(row["final_adapter_filename_present"])
        self.assertFalse(any(row["gates"].values()))
        self.assertFalse(o["authority"]["public_publication"])

    def test_inference_is_not_training(self):
        src = report(returned=1, kept=1)
        src["providers"]["salad"]["groups"][0]["classification"] = "INFERENCE"
        o = g.derive(src)
        self.assertNotIn("ACTIVE_TRAINING_VERIFIED", o["findings"])
        self.assertFalse(o["model_claims"][0]["gates"]["training_execution_verified"])


class ProgressTests(unittest.TestCase):
    def test_pinned_progress_delta_without_liveness_claim(self):
        current = report(at="2026-11-17T23:00:00Z", returned=1, kept=1)
        previous = report(at="2026-11-17T22:00:00Z", returned=1, kept=1)
        current["providers"]["huggingface"]["training_candidates"] = [
            candidate(step=62, target=100)
        ]
        previous["providers"]["huggingface"]["training_candidates"] = [
            candidate(step=55, target=100)
        ]
        current["providers"]["salad"]["groups"][0].update(
            model_repo="SZLHOLDINGS/synthetic-evidence-lane",
            configured_max_steps=100,
            classification_basis="COMMAND_TARGET",
        )
        row = g.derive(current, previous)["model_claims"][0]
        self.assertEqual(row["checkpoint_step_delta"], 7)
        self.assertEqual(row["command_target_binding"], "DECLARED_COMMAND_TARGET_ONLY")
        self.assertFalse(row["mapped_provider_groups"][0]["process_liveness_verified"])
        self.assertFalse(row["trainer_run_continuity_verified"])
        self.assertFalse(row["trainer_state_independently_reverified_by_graph"])

    def test_changed_target_has_no_comparable_counter_delta(self):
        current = report(at="2026-11-17T23:00:00Z")
        previous = report(at="2026-11-17T22:00:00Z")
        current["providers"]["huggingface"]["training_candidates"] = [
            candidate(step=15, target=30)
        ]
        previous["providers"]["huggingface"]["training_candidates"] = [
            candidate(step=10, target=20)
        ]
        item = g.derive(current, previous)
        self.assertIsNone(item["model_claims"][0]["checkpoint_step_delta"])
        self.assertTrue(item["model_claims"][0]["target_changed_since_prior"])
        self.assertIn("TARGET_CHANGED_COMPARISON_UNAVAILABLE", item["findings"])

    def test_true_step_regression_is_unresolved_not_data_loss(self):
        curr = report(at="2026-11-17T23:00:00Z")
        prior = report(at="2026-11-17T22:00:00Z")
        curr["providers"]["huggingface"]["training_candidates"] = [candidate(step=3)]
        prior["providers"]["huggingface"]["training_candidates"] = [candidate(step=8)]
        o = g.derive(curr, prior)
        self.assertIn(
            "CHECKPOINT_COUNTER_REGRESSION_OR_REPLACEMENT_UNRESOLVED", o["findings"]
        )

    def test_filename_without_verified_step_is_not_completion(self):
        current = report(returned=1, kept=1)
        current["providers"]["huggingface"]["training_candidates"][0]["trainer_state"][
            "trainer_state_sha256"
        ] = "wrong"
        current["providers"]["huggingface"]["training_candidates"][0][
            "final_adapter"
        ] = True
        row = g.derive(current)["model_claims"][0]
        self.assertEqual(row["trainer_state"], "UNVERIFIED")
        self.assertFalse(row["observed_terminal_step"])

    def test_bad_binding_is_declared_not_execution(self):
        src = report(returned=1, kept=1)
        src["providers"]["salad"]["groups"][0].update(
            model_repo="SZLHOLDINGS/synthetic-evidence-lane",
            configured_max_steps=50,
            classification_basis="COMMAND_TARGET",
        )
        o = g.derive(src)
        self.assertEqual(
            o["model_claims"][0]["command_target_binding"], "TARGET_MISMATCH"
        )
        self.assertIn("DECLARED_TARGET_MISMATCH", o["findings"])

    def test_future_or_equal_prior_timestamp_is_not_silent(self):
        now = report(at="2026-11-17T23:00:00+00:00")
        prev = report(at="2026-11-17T23:00:00+00:00")
        with self.assertRaisesRegex(g.CoverageHold, "OBSERVATION_TIME_ORDER_INVALID"):
            g.derive(now, prev)


class InputTests(unittest.TestCase):
    def test_duplicate_json_keys_fail_closed(self):
        with self.assertRaisesRegex(g.CoverageHold, "DUPLICATE_JSON_KEY"):
            g.strict_json(b'{"a":1,"a":2}')

    def test_nonfinite_and_overflow_fail_closed(self):
        for raw in (b'{"x":NaN}', b'{"x":1e999}'):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(g.CoverageHold, "NONFINITE_JSON"):
                    g.strict_json(raw)

    def test_read_bounded_input(self):
        with self.assertRaisesRegex(g.CoverageHold, "INVALID_INPUT_BYTE_BUDGET"):
            g.strict_json(b" " * (g.MAX_BYTES + 1))

    def test_main_exclusive_output_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            src = p / "input.json"
            output = p / "output.json"
            original = json.dumps(report(returned=1, kept=1)).encode()
            src.write_bytes(original)
            with mock.patch("sys.stdout"):
                self.assertEqual(
                    g.main(["--current", str(src), "--output", str(output)]), 0
                )
            self.assertEqual(src.read_bytes(), original)
            evidence = json.loads(output.read_bytes())
            self.assertEqual(
                evidence["input_sha256"]["current"],
                hashlib.sha256(original).hexdigest(),
            )
            self.assertFalse(evidence["authority"]["signed"])
            with mock.patch("sys.stderr"):
                self.assertEqual(
                    g.main(["--current", str(src), "--output", str(output)]), 2
                )
            self.assertEqual(src.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
