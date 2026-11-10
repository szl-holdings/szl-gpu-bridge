"""Offline checks: unavailable metadata cannot grant licenses or erase bytes."""

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "hf-spaces/estate-gardener/app.py"


class GardenerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {"GARDENER_STATE": self.tmp.name + "/state.json"})
        env.start()
        self.addCleanup(env.stop)
        deps = patch.dict(sys.modules, {"fastapi": None, "huggingface_hub": None})
        deps.start()
        self.addCleanup(deps.stop)
        spec = importlib.util.spec_from_file_location("gardener_evidence", APP)
        self.g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.g)

    def test_unknown_license_does_not_become_apache(self):
        original = "# Dataset\nOriginal author retains all rights.\n"
        receipt, proposed = self.g.garden_readme("SZLHOLDINGS/test", original)
        suffix = proposed[len(original) :]
        self.assertNotIn("Apache-2.0", suffix)
        self.assertNotIn("Copyright", suffix)
        self.assertEqual(receipt["license_state"], "UNKNOWN_REVIEW_REQUIRED")
        self.assertFalse(receipt["write_performed"])

    def test_preserves_every_original_byte_including_blank_lines(self):
        for original in ["# Data\n\n\n\n", "   \n\n", "\r\n\r\n", ""]:
            with self.subTest(original=repr(original)):
                receipt, proposed = self.g.garden_readme("SZLHOLDINGS/test", original)
                self.assertTrue(proposed.startswith(original))
                self.assertTrue(receipt["preserved_prefix"])
                self.assertEqual(
                    receipt["before_sha256"],
                    hashlib.sha256(original.encode()).hexdigest(),
                )

    def test_repeated_review_proposal_is_idempotent(self):
        first, _ = self.g.append_license_block("# Data\n", False)
        second, action = self.g.append_license_block(first, False)
        self.assertEqual(first, second)
        self.assertEqual(action, "SKIP_REVIEW_ALREADY_PRESENT")

    def test_existing_license_tag_is_untouched(self):
        original = "---\nlicense: cc-by-4.0\n---\nOriginal attribution.\n\n"
        proposed, action = self.g.append_license_block(original, True)
        self.assertEqual(proposed, original)
        self.assertEqual(action, "SKIP_ALREADY_TAGGED")

    def test_dry_run_false_still_performs_no_write(self):
        receipt, _ = self.g.garden_readme("SZLHOLDINGS/test", "# Data", dry_run=False)
        self.assertFalse(receipt["write_performed"])
        self.assertTrue(receipt["proposal_only"])

    def test_failed_inventory_is_not_an_observed_empty_estate(self):
        for unavailable in [None, {}, "bad response"]:
            with self.subTest(value=unavailable):
                with patch.object(self.g, "get", return_value=unavailable):
                    receipt = self.g.garden_cycle({"cycle": 1})
                self.assertEqual(receipt["inventory_state"], "UNAVAILABLE")
                self.assertIsNone(receipt["datasets_scanned"])
                self.assertFalse(receipt["inventory_complete"])

    def test_observed_empty_page_remains_distinct(self):
        with patch.object(self.g, "get", return_value=[]):
            receipt = self.g.garden_cycle({"cycle": 1})
        self.assertEqual(receipt["datasets_scanned"], 0)
        self.assertEqual(receipt["datasets_observed"], 0)
        self.assertNotEqual(receipt["inventory_state"], "UNAVAILABLE")

    def test_bounded_scan_counts_actual_rows_and_retains_all_proposals(self):
        rows = [{"id": f"SZLHOLDINGS/row-{i}", "tags": []} for i in range(40)]
        with patch.object(self.g, "get", return_value=rows):
            with patch.object(
                self.g.urllib.request,
                "urlopen",
                side_effect=lambda *a, **k: io.BytesIO(b"# Data\n"),
            ) as fetch:
                receipt = self.g.garden_cycle({"cycle": 7})
        self.assertEqual(fetch.call_count, 30)
        self.assertEqual(receipt["datasets_scanned"], 30)
        self.assertEqual(receipt["datasets_observed"], 40)
        self.assertEqual(receipt["datasets_deferred"], 10)
        self.assertEqual(len(receipt["proposals"]), 30)
        self.assertFalse(receipt["inventory_complete"])
        self.assertTrue(
            all(p["action"] == "PROPOSE_LICENSE_REVIEW" for p in receipt["proposals"])
        )

    def test_cycle_reuses_byte_preserving_review_helper(self):
        original = "# Data\n\n\n\n"
        _, expected = self.g.garden_readme("SZLHOLDINGS/test", original)
        with patch.object(self.g, "get", return_value=[{"id": "SZLHOLDINGS/test"}]):
            with patch.object(
                self.g.urllib.request,
                "urlopen",
                return_value=io.BytesIO(original.encode()),
            ):
                receipt = self.g.garden_cycle({"cycle": 1})
        proposal = receipt["proposals"][0]
        self.assertEqual(
            proposal["after_sha256"], hashlib.sha256(expected.encode()).hexdigest()
        )
        self.assertEqual(proposal["license_state"], "UNKNOWN_REVIEW_REQUIRED")

    def test_invalid_utf8_is_unavailable_not_replacement_content(self):
        with patch.object(self.g, "get", return_value=[{"id": "SZLHOLDINGS/test"}]):
            with patch.object(
                self.g.urllib.request, "urlopen", return_value=io.BytesIO(b"\xff")
            ):
                receipt = self.g.garden_cycle({"cycle": 1})
        self.assertEqual(receipt["proposals"][0]["state"], "UNAVAILABLE")
        self.assertNotIn("after_sha256", receipt["proposals"][0])

    def test_malformed_namespace_never_reaches_transport(self):
        rows = [{"id": "OTHER/test"}, {"id": "SZLHOLDINGS/../other"}]
        with patch.object(self.g, "get", return_value=rows):
            with patch.object(self.g.urllib.request, "urlopen") as fetch:
                receipt = self.g.garden_cycle({"cycle": 1})
        fetch.assert_not_called()
        self.assertTrue(all(p["state"] == "UNAVAILABLE" for p in receipt["proposals"]))


if __name__ == "__main__":
    unittest.main()
