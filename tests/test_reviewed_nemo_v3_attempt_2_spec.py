from __future__ import annotations

import json
import pathlib
import re
import sys
import unittest
from datetime import datetime, timedelta, timezone


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "laptop"))

import nemo_v3_status  # noqa: E402
from nemo_v3_contract import validate_nemo_v3_spec  # noqa: E402


PREDECESSOR_PATH = ROOT / "jobspecs" / "nemo-v3-20260722-reviewed.json"
PREREGISTERED_PATH = ROOT / "jobspecs" / "nemo-v3-20260729-successor-2-reviewed.json"
ATTEMPT_2_PATH = ROOT / "jobspecs" / "nemo-v3-20260729-attempt-2-reviewed.json"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")


class ReviewedNemoV3Attempt2SpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predecessor = json.loads(PREDECESSOR_PATH.read_text(encoding="utf-8"))
        self.preregistered = json.loads(PREREGISTERED_PATH.read_text(encoding="utf-8"))
        self.attempt = json.loads(ATTEMPT_2_PATH.read_text(encoding="utf-8"))

    def test_attempt_passes_contract_with_new_dispatch_identity(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt), self.attempt)
        self.assertEqual(
            self.attempt["jobId"],
            "job-2026-nemo-v3-governed-attempt-2",
        )
        self.assertRegex(
            self.attempt["jobId"],
            r"^job-[0-9]{4}-nemo-v3-governed-attempt-(?:[2-9]|[1-9][0-9]+)$",
        )
        self.assertNotEqual(self.attempt["jobId"], self.predecessor["jobId"])
        self.assertNotEqual(
            self.attempt["outputs"]["candidateId"],
            self.preregistered["outputs"]["candidateId"],
        )

    def test_exact_settled_source_and_owner_dispatch_are_bound(self) -> None:
        self.assertEqual(
            self.attempt["source"],
            {
                "repoId": "szl-holdings/a11oy",
                "revision": "b21b8fb65400e7eb39595365c5f54c80ed78aa67",
                "licenseId": "apache-2.0",
            },
        )
        self.assertEqual(
            self.attempt["ownerDispatch"],
            {
                "workflowIdentity": (
                    "szl-holdings/a11oy/.github/workflows/"
                    "nemo-v3-isolated-owner-dispatch.yml@refs/heads/main"
                ),
                "workflowBlob": "7e08ffc8aa87b78d0fa1618d7d3c3e68cb81ca33",
                "workflowVersion": "nemo-v3-owner-dispatch.v2",
                "trainingImage": (
                    "unsloth/unsloth@sha256:"
                    "9cc97606fc386b4b13455285eb7bd2668f51530988a9c2578707fe6cdfc46123"
                ),
                "candidateUpload": False,
                "modelCardUpload": False,
                "datasetUpload": False,
                "receiptsRepoId": "SZLHOLDINGS/szl-training-receipts",
            },
        )

    def test_frozen_science_and_receipt_only_outputs_are_unchanged(self) -> None:
        for field in ("base", "dataset", "recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt[field], self.preregistered[field])
        self.assertEqual(
            {
                key: self.attempt["outputs"][key]
                for key in ("receiptsRepoId", "private", "publishCandidate")
            },
            {
                "receiptsRepoId": "SZLHOLDINGS/szl-training-receipts",
                "private": True,
                "publishCandidate": False,
            },
        )

    def test_lineage_binds_the_permanently_quarantined_attempt(self) -> None:
        self.assertEqual(self.attempt["lineage"], self.preregistered["lineage"])
        self.assertEqual(
            self.attempt["lineage"]["predecessorJobId"],
            "job-2026-nemo-v3-governed-attempt-1",
        )
        self.assertFalse(self.attempt["lineage"]["automaticRetry"])
        self.assertFalse(self.attempt["lineage"]["trainingStarted"])
        self.assertFalse(self.attempt["lineage"]["candidateProduced"])
        self.assertFalse(self.attempt["lineage"]["receiptIntentProduced"])
        self.assertFalse(self.attempt["lineage"]["terminalLedgerWritten"])

    def test_expiry_is_live_and_bounded_to_fourteen_days(self) -> None:
        created = datetime.fromisoformat(
            self.attempt["createdAt"].replace("Z", "+00:00")
        )
        expires = datetime.fromisoformat(
            self.attempt["expiresAt"].replace("Z", "+00:00")
        )
        self.assertEqual(expires - created, timedelta(days=14))
        self.assertGreater(expires, datetime(2026, 7, 29, tzinfo=timezone.utc))

    def test_attempt_remains_plaintext_only_without_fresh_signature(self) -> None:
        queue_path = ROOT / "queue" / "pending" / f"{self.attempt['jobId']}.json"
        self.assertFalse(queue_path.exists())
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_2_PATH,
            now=datetime(2026, 7, 29, 23, 10, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "AWAITING_ENGINE_SIGNATURE")
        self.assertFalse(report["terminal"])
        self.assertEqual(
            report["reviewed_spec"]["path"],
            "jobspecs/nemo-v3-20260729-attempt-2-reviewed.json",
        )

    def test_status_workflow_tracks_attempt_2_without_dispatching_it(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260729-attempt-2-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-2.json",
            "tests/test_reviewed_nemo_v3_attempt_2_spec.py",
            "attempt_id: attempt-2-b21",
            "spec_path: jobspecs/nemo-v3-20260729-attempt-2-reviewed.json",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIsNone(
            re.search(r"repository_dispatch|workflow_call", STATUS_WORKFLOW)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
