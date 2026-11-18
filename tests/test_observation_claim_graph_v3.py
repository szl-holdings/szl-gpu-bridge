"""Credential-free v3 candidate evidence fixtures. Synthetic IDs and counts only."""

import copy
import unittest

from tools.salad import observation_claim_graph as g


MODEL = "SZLHOLDINGS/synthetic-evaluation-candidate"
GROUP = "szl-train-synthetic-lane"


def fixture():
    return {
        "watch_version": g.CANARY_WATCH,
        "observed_at_utc": "2026-11-18T00:15:00Z",
        "providers": {
            "salad": {
                "projects": ["default"],
                "project_listing": {
                    "status": "UNSUPPORTED_ENDPOINT",
                    "http_code": 404,
                    "truncated": True,
                },
                "groups": [
                    {
                        "project": "default",
                        "name": GROUP,
                        "classification": "TRAINING",
                        "classification_basis": "COMMAND_TRAINING_TARGET",
                        "model_repo": MODEL,
                        "configured_max_steps": 100,
                        "group_status": "running",
                        "desired_replicas": 1,
                        "counts": {
                            "allocating_count": 0,
                            "creating_count": 1,
                            "running_count": 0,
                            "stopping_count": 0,
                        },
                    }
                ],
                "coverage": [
                    {
                        "project": "default",
                        "list": "OBSERVED",
                        "pagination": {
                            "pages": 1,
                            "returned": 1,
                            "truncated": False,
                        },
                        "group_capture": {
                            "returned_count": 1,
                            "captured_count": 1,
                            "capture_limit": 256,
                            "omitted_count": 0,
                            "omitted_names": [],
                            "omitted_with_nonzero_provider_counts": [],
                            "omitted_with_unknown_counts": [],
                            "listing_complete": True,
                            "coverage_complete": True,
                        },
                    }
                ],
                "instance_scope": {
                    "eligible_count": 1,
                    "attempted_count": 1,
                    "limit": 48,
                    "eligible_names": [GROUP],
                    "unattempted_names": [],
                    "provider_denial_stop": False,
                },
                "instance_inspections": [
                    {
                        "project": "default",
                        "group": GROUP,
                        "status": "OBSERVED",
                        "instance_evidence": {
                            "returned_count": 1,
                            "captured_count": 1,
                            "coverage_complete": True,
                            "unobserved_status_count": 0,
                            "conflicting_status_fields": 0,
                        },
                        "group_instance_count_reconciliation": "PROVIDER_COUNT_DRIFT",
                    }
                ],
            },
            "huggingface": {
                "datasets_count_returned": 154,
                "dataset_limit": 300,
                "training_candidates": [
                    {
                        "id": MODEL,
                        "status": "OBSERVED",
                        "revision": "b" * 40,
                        "latest_checkpoint": 50,
                        "trainer_state": {
                            "global_step": 50,
                            "max_steps": 100,
                            "trainer_state_sha256": "a" * 64,
                            "integrity": "PINNED_OBJECT_BYTES_VERIFIED",
                        },
                        "final_adapter": False,
                    }
                ],
            },
        },
    }


class V3CanaryContracts(unittest.TestCase):
    def test_no_omissions_within_known_project_not_org_claim(self):
        result = g.derive(fixture())
        coverage = result["provider_coverage"]["salad"]
        self.assertEqual(coverage["returned_count"], 1)
        self.assertEqual(coverage["omitted_count"], 0)
        self.assertTrue(coverage["known_project_group_details_complete"])
        self.assertFalse(coverage["coverage_complete"])
        self.assertFalse(coverage["project_discovery_complete"])
        self.assertIn("CANDIDATE_RELEASE_NOT_ADMITTED", result["findings"])
        self.assertEqual(
            result["producer_admission"], "V3_CANDIDATE_UNADMITTED_MANUAL_CANARY_ONLY"
        )
        self.assertFalse(result["authority"]["candidate_release_deployed"])

    def test_complete_selected_instance_gets_are_not_process_liveness(self):
        r = g.derive(fixture())
        inst = r["provider_coverage"]["salad_instances"]
        self.assertEqual(
            (
                inst["eligible_count"],
                inst["attempted_count"],
                inst["observed_count"],
                inst["unattempted_count"],
            ),
            (1, 1, 1, 0),
        )
        self.assertTrue(inst["coverage_complete_of_selected_eligible"])
        self.assertFalse(inst["trainer_process_liveness_verified"])
        self.assertFalse(inst["temporal_atomic"])
        self.assertEqual(inst["provider_count_drift"], 1)
        self.assertIn("SEQUENTIAL_PROVIDER_COUNTS_DIFFER_NOT_ATOMIC", r["findings"])

    def test_official_downloading_state_is_known_but_not_group_counter(self):
        report = fixture()
        report["providers"]["salad"]["instance_inspections"][0][
            "group_instance_count_reconciliation"
        ] = "PROVIDER_COUNTS_NOT_COMPARABLE_DOWNLOADING"
        row = g.derive(report)
        scope = row["provider_coverage"]["salad_instances"]
        self.assertEqual(scope["provider_count_drift"], 0)
        self.assertEqual(scope["provider_count_not_comparable_downloading"], 1)
        self.assertTrue(scope["coverage_complete_of_selected_eligible"])
        self.assertFalse(scope["trainer_process_liveness_verified"])
        self.assertIn(
            "DOWNLOADING_INSTANCE_HAS_NO_GROUP_COUNT_COUNTER", row["findings"]
        )

    def test_command_mapping_is_declared_only(self):
        row = g.derive(fixture())["model_claims"][0]
        self.assertEqual(row["command_target_binding"], "DECLARED_COMMAND_TARGET_ONLY")
        self.assertFalse(row["trainer_run_continuity_verified"])
        self.assertFalse(any(row["gates"].values()))

    def test_v2_prior_and_v3_current_can_be_compared_without_run_attestation(self):
        current = fixture()
        previous = copy.deepcopy(current)
        previous["watch_version"] = g.WATCH
        previous["observed_at_utc"] = "2026-11-18T00:00:00Z"
        previous["providers"]["huggingface"]["training_candidates"][0]["trainer_state"][
            "global_step"
        ] = 40
        previous["providers"]["huggingface"]["training_candidates"][0][
            "latest_checkpoint"
        ] = 40
        result = g.derive(current, previous)
        self.assertEqual(result["previous_watch_version"], g.WATCH)
        self.assertEqual(result["model_claims"][0]["checkpoint_step_delta"], 10)
        self.assertFalse(result["model_claims"][0]["trainer_run_continuity_verified"])

    def test_v3_missing_group_capture_fails(self):
        src = fixture()
        src["providers"]["salad"]["coverage"][0].pop("group_capture")
        with self.assertRaisesRegex(g.CoverageHold, "V3_GROUP_CAPTURE_MISSING"):
            g.derive(src)

    def test_v3_false_zero_omissions_fails(self):
        src = fixture()
        src["providers"]["salad"]["coverage"][0]["pagination"]["returned"] = 2
        with self.assertRaisesRegex(g.CoverageHold, "V3_GROUP_CAPTURE_MISMATCH"):
            g.derive(src)

    def test_v3_bool_omission_is_not_integer(self):
        src = fixture()
        src["providers"]["salad"]["coverage"][0]["group_capture"]["omitted_count"] = (
            False
        )
        with self.assertRaisesRegex(g.CoverageHold, "V3_GROUP_CAPTURE_MISMATCH"):
            g.derive(src)

    def test_missing_v3_instance_scope_fails(self):
        src = fixture()
        src["providers"]["salad"].pop("instance_scope")
        with self.assertRaisesRegex(g.CoverageHold, "V3_INSTANCE_SCOPE_MISSING"):
            g.derive(src)

    def test_tampered_attempted_count_is_not_a_pass(self):
        src = fixture()
        src["providers"]["salad"]["instance_scope"]["attempted_count"] = 0
        with self.assertRaisesRegex(g.CoverageHold, "INVALID_INSTANCE_SCOPE"):
            g.derive(src)

    def test_omitted_eligible_instance_get_is_exposed(self):
        src = fixture()
        src["providers"]["salad"]["instance_scope"]["attempted_count"] = 0
        src["providers"]["salad"]["instance_scope"]["unattempted_names"] = [GROUP]
        src["providers"]["salad"]["instance_inspections"] = []
        r = g.derive(src)
        inst = r["provider_coverage"]["salad_instances"]
        self.assertEqual(inst["unattempted_count"], 1)
        self.assertFalse(inst["coverage_complete_of_selected_eligible"])
        self.assertIn("INSTANCE_GET_ELIGIBLE_OMISSIONS", r["findings"])

    def test_denial_stop_with_unobserved_request_is_not_success(self):
        src = fixture()
        src["providers"]["salad"]["instance_scope"]["provider_denial_stop"] = True
        src["providers"]["salad"]["instance_inspections"][0] = {
            "project": "default",
            "group": GROUP,
            "status": "UNOBSERVED",
        }
        r = g.derive(src)
        self.assertEqual(r["provider_coverage"]["salad_instances"]["observed_count"], 0)
        self.assertFalse(
            r["provider_coverage"]["salad_instances"][
                "coverage_complete_of_selected_eligible"
            ]
        )
        self.assertIn("INSTANCE_PROVIDER_DENIAL_STOP", r["findings"])

    def test_wrong_instance_identity_holds(self):
        src = fixture()
        src["providers"]["salad"]["instance_inspections"][0]["group"] = "szl-other"
        with self.assertRaisesRegex(g.CoverageHold, "INSTANCE_IDENTITY_MISMATCH"):
            g.derive(src)

    def test_incomplete_instance_page_is_not_full_coverage(self):
        src = fixture()
        src["providers"]["salad"]["instance_inspections"][0]["instance_evidence"][
            "coverage_complete"
        ] = False
        r = g.derive(src)
        self.assertFalse(
            r["provider_coverage"]["salad_instances"][
                "coverage_complete_of_selected_eligible"
            ]
        )
        self.assertIn("INSTANCE_GET_PARTIAL_OBSERVATIONS", r["findings"])

    def test_producer_not_reported_qualified_from_final_filename(self):
        src = fixture()
        item = src["providers"]["huggingface"]["training_candidates"][0]
        item["final_adapter"] = True
        item["trainer_state"]["global_step"] = 100
        item["latest_checkpoint"] = 100
        r = g.derive(src)
        self.assertTrue(r["model_claims"][0]["observed_terminal_step"])
        self.assertFalse(any(r["model_claims"][0]["gates"].values()))
        self.assertFalse(r["authority"]["qualification_conferred"])


if __name__ == "__main__":
    unittest.main()
