"""Receipt-routing and observation regressions; synthetic, no network or GPU."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.salad import receipt_observer as M

SHA = "a" * 40


class ReceiptObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "result.json"
        self.path.write_text(json.dumps({"status": "OK"}), encoding="utf-8")

    def api(self, target, *, siblings=True):
        names = [target["path"], target.get("source_path", "other")]
        return Mock(
            repo_info=Mock(
                return_value=SimpleNamespace(
                    sha=SHA,
                    siblings=[SimpleNamespace(rfilename=n) for n in names]
                    if siblings
                    else None,
                )
            )
        )

    def test_heldout_contracts_query_models_not_chain_dataset(self):
        for name in ("heldout-v4", "heldout-v5"):
            with self.subTest(name=name):
                target = M.TARGETS[name]
                api = self.api(target)
                downloader = Mock(return_value=str(self.path))
                with patch.object(M, "audit", return_value={"status": "SYNTHETIC"}):
                    result = M.observe_receipt(api, downloader, name, target)
                self.assertEqual(target["repo_type"], "model")
                self.assertIn("salad-frontier", target["repo_id"])
                self.assertEqual(result["observation"], "PUBLISHED")
                self.assertEqual(result["repo_revision"], SHA)
                self.assertEqual(len(downloader.call_args_list), 2)
                for call in downloader.call_args_list:
                    self.assertEqual(call.kwargs["repo_type"], "model")
                    self.assertEqual(call.kwargs["repo_id"], target["repo_id"])
                    self.assertEqual(call.kwargs["revision"], SHA)
                self.assertEqual(
                    downloader.call_args_list[0].kwargs["filename"], target["path"]
                )

    def test_training_receipt_keeps_dataset_contract(self):
        target = M.TARGETS["frontier-v5-r7"]
        api, downloader = self.api(target), Mock(return_value=str(self.path))
        result = M.observe_receipt(api, downloader, "frontier-v5-r7", target)
        self.assertEqual(downloader.call_args.kwargs["repo_type"], "dataset")
        self.assertEqual(result["classification"], "PUBLISHED_RECEIPT")
        self.assertFalse(result["model_qualified"])

    def test_missing_path_is_scoped_not_global_absence(self):
        target = M.TARGETS["heldout-v4"]
        api = self.api(target)
        api.repo_info.return_value.siblings = []
        downloader = Mock()
        result = M.observe_receipt(api, downloader, "heldout-v4", target)
        self.assertEqual(result["observation"], "ABSENT_AT_OBSERVED_REVISION")
        self.assertEqual(result["repo_id"], target["repo_id"])
        self.assertEqual(result["repo_revision"], SHA)
        downloader.assert_not_called()

    def test_missing_inventory_is_unavailable_not_absent(self):
        target = M.TARGETS["heldout-v4"]
        with self.assertRaisesRegex(ValueError, "inventory unavailable"):
            M.observe_receipt(self.api(target, siblings=False), Mock(), "v4", target)

    def test_no_mutable_revision_fallback(self):
        target = M.TARGETS["heldout-v4"]
        api = self.api(target)
        api.repo_info.return_value.sha = "main"
        with self.assertRaisesRegex(ValueError, "immutable revision"):
            M.observe_receipt(api, Mock(), "v4", target)

    def test_error_receipt_is_preserved_but_not_qualified(self):
        result = M.receipt_summary(
            {
                "status": "ERROR",
                "error": "signed-url-must-not-be-logged",
                "completion_qualified": False,
            }
        )
        self.assertEqual(result["producer"]["status"], "ERROR")
        self.assertEqual(result["classification"], "PRODUCER_ERROR")
        self.assertTrue(result["producer_error_present"])
        self.assertNotIn("error", result["producer"])
        self.assertFalse(result["model_qualified"])

    def test_terminal_resume_is_not_new_training(self):
        for evidence in (
            {"new_steps_this_boot": 0},
            {"new_steps_this_boot": 0, "completion_qualified": False},
        ):
            with self.subTest(evidence=evidence):
                result = M.receipt_summary({"status": "OK", **evidence})
                self.assertEqual(result["classification"], "NO_NEW_TRAINING")
                self.assertFalse(result["current_process_liveness_verified"])

    def test_unqualified_completion_does_not_discard_positive_or_unknown_work(self):
        for evidence in ({"new_steps_this_boot": 20}, {}):
            with self.subTest(evidence=evidence):
                result = M.receipt_summary(
                    {
                        "status": "OK",
                        "completion_qualified": False,
                        **evidence,
                    }
                )
                self.assertEqual(result["classification"], "UNQUALIFIED")
                self.assertEqual(
                    result["producer"].get("new_steps_this_boot"),
                    evidence.get("new_steps_this_boot"),
                )

    def test_boolean_is_not_zero_training_steps(self):
        result = M.receipt_summary({"status": "OK", "new_steps_this_boot": False})
        self.assertNotEqual(result["classification"], "NO_NEW_TRAINING")

    def test_receipt_parse_rejects_duplicate_status(self):
        self.path.write_text('{"status":"ERROR","status":"OK"}', encoding="utf-8")
        target = M.TARGETS["frontier-v5-r7"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            M.observe_receipt(
                self.api(target), Mock(return_value=str(self.path)), "r7", target
            )

    def test_heartbeat_numeric_order_and_terminal_are_separate(self):
        prefix = "mesh/lanelogs/flagship54-v1"
        result = M.heartbeat_selection(
            {
                prefix + "-hb99.log",
                prefix + "-hb100.log",
                prefix + "-hb916.log",
                prefix + "-hbfinal.log",
                prefix + "-other-hb9999.log",
            },
            prefix,
        )
        self.assertEqual(result["highest_sequence"], 916)
        self.assertEqual(result["numbered_count"], 3)
        self.assertEqual(result["highest_numbered_path"], prefix + "-hb916.log")
        self.assertEqual(result["terminal_path"], prefix + "-hbfinal.log")
        self.assertFalse(result["current_process_liveness_verified"])

    def test_unknown_source_does_not_qualify_success_receipt(self):
        target = M.TARGETS["heldout-v4"]
        result = M.observe_receipt(
            self.api(target), Mock(return_value=str(self.path)), "v4", target
        )
        self.assertEqual(result["observation"], "PUBLISHED")
        self.assertEqual(result["legacy_audit"]["status"], "UNAVAILABLE")
        self.assertFalse(result["model_qualified"])


if __name__ == "__main__":
    unittest.main()
