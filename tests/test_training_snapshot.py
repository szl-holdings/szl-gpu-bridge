import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

SRC = Path(__file__).resolve().parents[1] / "training_snapshot.py"
spec = importlib.util.spec_from_file_location("training_snapshot", SRC)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.dumps({"global_step": 250, "max_steps": 1000}).encode()
        self.entry = self.entry_for(self.raw)
        self.info = NS(
            sha="a" * 40, private=True, siblings=[self.entry], last_modified=None
        )
        self.api = Mock()
        self.api.model_info.return_value = self.info
        self.api.list_repo_files.side_effect = AssertionError("moving branch access")

    def entry_for(self, raw, lfs=False):
        return NS(
            rfilename="checkpoint-250/trainer_state.json",
            size=len(raw),
            lfs={"sha256": hashlib.sha256(raw).hexdigest()} if lfs else None,
            blob_id=hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw
            ).hexdigest(),
        )

    def test_one_snapshot_never_relists_main(self):
        _, row, entries = g.model_snapshot(self.api, "SZLHOLDINGS/test")
        self.api.model_info.assert_called_once_with(
            "SZLHOLDINGS/test", files_metadata=True, timeout=12
        )
        self.api.list_repo_files.assert_not_called()
        self.assertEqual(row["revision"], "a" * 40)
        self.assertEqual(row["latest_checkpoint"], 250)
        self.assertIn(self.entry.rfilename, entries)

    def test_false_final_cannot_arrive_from_moving_main(self):
        self.api.list_repo_files.side_effect = None
        self.api.list_repo_files.return_value = ["final/adapter_model.safetensors"]
        _, row, _ = g.model_snapshot(self.api, "SZLHOLDINGS/test")
        self.assertFalse(row["final_adapter"])
        self.api.list_repo_files.assert_not_called()

    def test_presence_does_not_assert_integrity_or_completion(self):
        self.info.siblings.append(NS(rfilename="final/adapter_model.safetensors"))
        _, row, _ = g.model_snapshot(self.api, "SZLHOLDINGS/test")
        self.assertTrue(row["final_adapter"])
        self.assertEqual(row["training_complete"], "UNVERIFIED")
        self.assertFalse(row["process_liveness_verified"])
        self.assertEqual(row["artifact_integrity"], "NOT_VERIFIED_BY_METADATA")

    def test_missing_pin(self):
        self.info.sha = "main"
        self.assertRaises(
            g.EvidenceHold, g.model_snapshot, self.api, "SZLHOLDINGS/test"
        )

    def test_unknown_file_list_is_not_empty(self):
        self.info.siblings = None
        self.assertRaises(
            g.EvidenceHold, g.model_snapshot, self.api, "SZLHOLDINGS/test"
        )

    def test_empty_list_is_observed_empty(self):
        self.info.siblings = []
        _, row, _ = g.model_snapshot(self.api, "SZLHOLDINGS/test")
        self.assertEqual(row["file_count"], 0)
        self.assertIsNone(row["latest_checkpoint"])

    def test_visibility_unknown(self):
        self.info.private = None
        self.assertRaises(
            g.EvidenceHold, g.model_snapshot, self.api, "SZLHOLDINGS/test"
        )

    def test_public_snapshot_is_not_changed(self):
        self.info.private = False
        _, row, _ = g.model_snapshot(self.api, "SZLHOLDINGS/test")
        self.assertFalse(row["private"])

    def test_duplicate_paths(self):
        self.info.siblings *= 2
        self.assertRaises(
            g.EvidenceHold, g.model_snapshot, self.api, "SZLHOLDINGS/test"
        )

    def test_path_traversal(self):
        self.entry.rfilename = "../trainer_state.json"
        self.assertRaises(
            g.EvidenceHold, g.model_snapshot, self.api, "SZLHOLDINGS/test"
        )

    def test_latest_partial_snapshot_disclosed(self):
        self.info.siblings.append(NS(rfilename="checkpoint-500/adapter_config.json"))
        _, row, _ = g.model_snapshot(self.api, "SZLHOLDINGS/test")
        self.assertEqual(row["latest_checkpoint"], 500)
        self.assertFalse(row["latest_trainer_state_present"])

    def test_intermediate_length_pin_rejected(self):
        self.info.sha = "a" * 41
        self.assertRaises(
            g.EvidenceHold, g.model_snapshot, self.api, "SZLHOLDINGS/test"
        )

    def test_float_overflow_json_rejected(self):
        raw = b'{"global_step":250,"max_steps":1000,"epoch":1e999}'
        self.assertRaises(
            g.EvidenceHold, g.verified_trainer_state, raw, self.entry_for(raw), 250
        )

    def test_git_object_integrity(self):
        value = g.verified_trainer_state(self.raw, self.entry, 250)
        self.assertEqual(value["global_step"], 250)
        self.assertEqual(value["integrity"], "PINNED_OBJECT_BYTES_VERIFIED")

    def test_lfs_object_integrity(self):
        value = g.verified_trainer_state(self.raw, self.entry_for(self.raw, True), 250)
        self.assertEqual(value["max_steps"], 1000)

    def test_corrupt_same_size_json(self):
        self.assertRaises(
            g.EvidenceHold,
            g.verified_trainer_state,
            self.raw.replace(b"1000", b"2000"),
            self.entry,
            250,
        )

    def test_size_mismatch(self):
        self.entry.size += 1
        self.assertRaises(
            g.EvidenceHold, g.verified_trainer_state, self.raw, self.entry, 250
        )

    def test_missing_object_hash(self):
        self.entry.blob_id = None
        self.assertRaises(
            g.EvidenceHold, g.verified_trainer_state, self.raw, self.entry, 250
        )

    def test_wrong_step(self):
        self.assertRaises(
            g.EvidenceHold, g.verified_trainer_state, self.raw, self.entry, 500
        )

    def test_boolean_step(self):
        raw = b'{"global_step":true,"max_steps":1000}'
        self.assertRaises(
            g.EvidenceHold, g.verified_trainer_state, raw, self.entry_for(raw), 1
        )

    def test_boolean_target(self):
        raw = b'{"global_step":250,"max_steps":true}'
        self.assertRaises(
            g.EvidenceHold, g.verified_trainer_state, raw, self.entry_for(raw), 250
        )

    def test_duplicate_json_keys(self):
        raw = b'{"global_step":250,"global_step":1000,"max_steps":1000}'
        self.assertRaises(
            g.EvidenceHold, g.verified_trainer_state, raw, self.entry_for(raw), 1000
        )

    def test_nonfinite_json(self):
        raw = b'{"global_step":250,"max_steps":1000,"epoch":NaN}'
        self.assertRaises(
            g.EvidenceHold, g.verified_trainer_state, raw, self.entry_for(raw), 250
        )

    def test_json_budget(self):
        self.assertRaises(
            g.EvidenceHold, g.verified_trainer_state, self.raw, self.entry, 250, 1
        )

    def test_array_is_not_state(self):
        raw = b"[]"
        self.assertRaises(
            g.EvidenceHold, g.verified_trainer_state, raw, self.entry_for(raw), 250
        )

    def test_mapped_repo_missing_from_catalog_remains_selected(self):
        selected, cov = g.select_training_repos([], ["SZLHOLDINGS/mapped-missing"])
        self.assertEqual(selected, ["SZLHOLDINGS/mapped-missing"])
        self.assertEqual(cov["omitted_ids"], [])

    def test_selection_deduplicates_and_reports_limit(self):
        selected, cov = g.select_training_repos(
            ["SZLHOLDINGS/abst-a", "SZLHOLDINGS/abst-b"],
            ["SZLHOLDINGS/mapped"],
            limit=2,
        )
        self.assertEqual(selected, ["SZLHOLDINGS/mapped", "SZLHOLDINGS/abst-a"])
        self.assertEqual(cov["omitted_ids"], ["SZLHOLDINGS/abst-b"])

    def test_absrl_successors_selected_without_salad_mapping(self):
        candidates = [
            "SZLHOLDINGS/absrl-synthetic-a",
            "SZLHOLDINGS/absrl-synthetic-b",
        ]
        selected, coverage = g.select_training_repos(candidates, [], limit=1)
        self.assertEqual(selected, ["SZLHOLDINGS/absrl-synthetic-a"])
        self.assertEqual(coverage["eligible_count"], 2)
        self.assertEqual(
            coverage["omitted_ids"],
            ["SZLHOLDINGS/absrl-synthetic-b"],
        )

    def test_external_mapping_refused(self):
        selected, _ = g.select_training_repos([], ["foreign/repo", "SZLHOLDINGS/../x"])
        self.assertEqual(selected, [])

    def test_partial_catalog_never_claims_disappearance(self):
        d = g.catalog_delta(["a", "b"], ["a"], current_complete=False)
        self.assertEqual(d["not_returned"], [])
        self.assertEqual(d["prior_ids_unassessed"], ["b"])

    def test_complete_catalog_delta_is_not_loss(self):
        d = g.catalog_delta(["a"], [], current_complete=True)
        self.assertEqual(d["not_returned"], ["a"])
        self.assertIn("not deletion", d["interpretation"])


class SaladCoverageContracts(unittest.TestCase):
    """Fixtures model the observed 132-group cap and provider state schemas."""

    def group(self, name, *, running=0, creating=0, state="running"):
        return {
            "name": name,
            "replicas": 1,
            "current_state": {
                "status": state,
                "instance_status_counts": {
                    "allocating_count": 0,
                    "creating_count": creating,
                    "running_count": running,
                    "stopping_count": 0,
                },
            },
        }

    def test_132_groups_no_default_120_loss(self):
        groups = [self.group(f"szl-group-{i}") for i in range(132)]
        item = g.salad_group_page_evidence(groups, listing_complete=True)
        self.assertEqual(item["returned_count"], 132)
        self.assertEqual(item["captured_count"], 132)
        self.assertEqual(item["omitted_count"], 0)
        self.assertTrue(item["coverage_complete"])

    def test_coverage_defaults_to_unverified_pagination(self):
        item = g.salad_group_page_evidence([self.group("szl-example")])
        self.assertFalse(item["listing_complete"])
        self.assertFalse(item["coverage_complete"])

    def test_explicit_120_cap_discloses_12_groups(self):
        groups = [self.group(f"szl-group-{i}") for i in range(132)]
        item = g.salad_group_page_evidence(groups, capture_limit=120)
        self.assertEqual(item["omitted_count"], 12)
        self.assertEqual(item["omitted_names"][0], "szl-group-120")
        self.assertEqual(item["omitted_names"][-1], "szl-group-131")
        self.assertFalse(item["coverage_complete"])

    def test_omitted_activity_is_not_lost(self):
        groups = [self.group("szl-a"), self.group("szl-b", creating=1)]
        item = g.salad_group_page_evidence(groups, capture_limit=1)
        self.assertEqual(item["omitted_with_nonzero_provider_counts"], ["szl-b"])

    def test_incomplete_transport_never_claims_full_coverage(self):
        item = g.salad_group_page_evidence(
            [self.group("szl-a")], listing_complete=False
        )
        self.assertFalse(item["coverage_complete"])

    def test_unknown_group_counts_not_silent_zero(self):
        item = g.salad_group_page_evidence(
            [self.group("szl-a"), {"name": "szl-b"}],
            capture_limit=1,
        )
        self.assertIn("szl-b", item["omitted_with_unknown_counts"])
        self.assertIsNone(
            g.salad_group_page_evidence([{"name": "szl-b"}])["captured"][0]["counts"][
                "running_count"
            ]
        )

    def test_duplicate_names_fail_closed(self):
        with self.assertRaises(g.EvidenceHold):
            g.salad_group_page_evidence([self.group("szl-a")] * 2)

    def test_boolean_group_replica_does_not_become_one(self):
        item = g.salad_group_page_evidence([{"name": "szl-x", "replicas": True}])
        self.assertIsNone(item["captured"][0]["desired_replicas"])

    def test_active_group_status_does_not_prove_process(self):
        item = g.salad_group_page_evidence([self.group("szl-x", running=0)])
        self.assertEqual(item["captured"][0]["group_status"], "running")
        self.assertEqual(item["captured"][0]["counts"]["running_count"], 0)
        self.assertIs(item["captured"][0]["process_liveness_verified"], False)

    def test_real_provider_state_string_creating_is_preserved(self):
        item = g.salad_instance_evidence([{"state": "creating", "update_time": "ts"}])
        self.assertEqual(item["state_counts"]["creating"], 1)
        self.assertEqual(item["state_counts"]["running"], 0)
        self.assertFalse(item["trainer_process_liveness_verified"])

    def test_nested_instance_state_is_supported(self):
        item = g.salad_instance_evidence(
            {"instances": [{"current_state": {"status": "running"}}]}
        )
        self.assertEqual(item["state_counts"]["running"], 1)

    def test_conflicting_fields_never_become_silent_success(self):
        item = g.salad_instance_evidence(
            [{"state": "creating", "current_state": {"status": "running"}}]
        )
        self.assertEqual(item["captured_statuses"], ["UNOBSERVED"])
        self.assertEqual(item["conflicting_status_fields"], 1)

    def test_instance_cap_and_pagination_are_reported(self):
        item = g.salad_instance_evidence(
            {"instances": [{"state": "running"}] * 3, "next": "cursor"},
            capture_limit=2,
        )
        self.assertEqual(item["captured_count"], 2)
        self.assertTrue(item["capture_truncated"])
        self.assertFalse(item["listing_complete"])

    def test_reconcile_group_count_drift(self):
        group = g.salad_group_page_evidence(
            [self.group("szl-synthetic-v3", running=1)]
        )["captured"][0]
        instances = g.salad_instance_evidence([{"state": "creating"}])
        self.assertEqual(
            g.reconcile_salad_group_instance_counts(group, instances),
            "PROVIDER_COUNT_DRIFT",
        )

    def test_reconcile_group_count_match_is_not_process_proof(self):
        group = g.salad_group_page_evidence(
            [self.group("szl-synthetic-v2", creating=1)]
        )["captured"][0]
        instances = g.salad_instance_evidence([{"state": "creating"}])
        self.assertEqual(
            g.reconcile_salad_group_instance_counts(group, instances),
            "PROVIDER_COUNTS_AGREE_NOT_TRAINER_LIVENESS",
        )

    def test_reconcile_unknown_status_never_asserts_agreement(self):
        group = g.salad_group_page_evidence([self.group("szl-x")])["captured"][0]
        item = g.salad_instance_evidence([{"state": "pending"}])
        self.assertEqual(
            g.reconcile_salad_group_instance_counts(group, item), "UNOBSERVED"
        )


if __name__ == "__main__":
    unittest.main()
