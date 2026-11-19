# SPDX-License-Identifier: Apache-2.0
"""Offline tests for tools/receipts_index/refresh_receipts_index.py (no network)."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refresh_receipts_index", ROOT / "tools/receipts_index/refresh_receipts_index.py"
)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def report(**kw):
    base = {
        "old_rows": 10,
        "new_rows": 10,
        "added": 0,
        "removed": 0,
        "changed_fields": {},
        "hash_status": {"VERIFIED": 8, "UNVERIFIED": 1, "NO_HASH": 1},
    }
    base.update(kw)
    return base


class RefreshTests(unittest.TestCase):
    def test_fresh_within_threshold(self):
        self.assertFalse(R.is_stale(report(added=R.DEFAULT_MAX_ADDED)))

    def test_stale_when_many_unindexed(self):
        self.assertTrue(R.is_stale(report(added=R.DEFAULT_MAX_ADDED + 1)))

    def test_stale_when_row_removed_or_changed(self):
        self.assertTrue(R.is_stale(report(removed=1)))
        self.assertTrue(R.is_stale(report(changed_fields={"sha256": 1})))

    def test_publish_refuses_removed_rows(self):
        self.assertTrue(R.publish_blockers(report(removed=2), []))

    def test_publish_refuses_invalid_json(self):
        rows = [{"hash_status": "INVALID_JSON"}]
        self.assertTrue(R.publish_blockers(report(), rows))
        self.assertTrue(R.publish_blockers(report(hash_status={"INVALID_JSON": 1}), []))

    def test_publish_allows_additions_only(self):
        self.assertEqual(
            R.publish_blockers(report(added=5), [{"hash_status": "VERIFIED"}]), []
        )

    def test_commit_message_cites_builder_revision_and_counts(self):
        title, body = R.commit_message(report(new_rows=12), "abc123", "6c5a0dc7ffff")
        self.assertIn("6c5a0dc7", title)
        self.assertIn("abc123", body)
        self.assertIn("10 -> 12", body)
        self.assertIn("VERIFIED 8", body)
        self.assertIn("Only this file is changed", body)

    def test_report_states_check_is_read_only(self):
        md = R.markdown_report(report(added=99), "abc", True, 50)
        self.assertIn("Stale: **yes**", md)
        self.assertIn("never writes to the Hub", md)


if __name__ == "__main__":
    unittest.main()
