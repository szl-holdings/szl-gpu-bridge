import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

SRC = Path(__file__).resolve().parents[1] / "tools/salad/training_snapshot.py"
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


if __name__ == "__main__":
    unittest.main()
