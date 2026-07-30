from __future__ import annotations

import hashlib
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
from frontier_contract import ContractError  # noqa: E402
from nemo_v3_contract import (  # noqa: E402
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)


PREDECESSOR_PATH = ROOT / "jobspecs" / "nemo-v3-20260722-reviewed.json"
PREREGISTERED_PATH = ROOT / "jobspecs" / "nemo-v3-20260729-successor-2-reviewed.json"
ATTEMPT_2_PATH = ROOT / "jobspecs" / "nemo-v3-20260729-attempt-2-reviewed.json"
EXPECTED_QUEUE_PAYLOAD_SHA256 = (
    "84a808615ba1693935eee8cc9fa1a4c5a83d119b79ad7e9437380ec73756b90d"
)
EXPECTED_ENGINE_KEY_ID = "5c6cf59741ade920"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
GIT_ATTRIBUTES = (ROOT / ".gitattributes").read_text(encoding="utf-8")


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

    def test_attempt_signature_is_preserved_under_exact_quarantine(self) -> None:
        queue_path = ROOT / "queue" / "pending" / f"{self.attempt['jobId']}.json"
        self.assertTrue(
            queue_path.is_file(),
            "attempt 2 must use the independently signed DSSE queue envelope",
        )
        envelope = json.loads(queue_path.read_text(encoding="utf-8"))
        self.assertNotEqual(envelope, self.attempt)
        self.assertEqual(len(envelope.get("signatures", [])), 1)

        evidence = nemo_v3_status.verify_queue(self.attempt, ROOT)
        self.assertTrue(evidence.present)
        self.assertTrue(evidence.valid, evidence.error)
        self.assertEqual(evidence.payload_sha256, EXPECTED_QUEUE_PAYLOAD_SHA256)
        self.assertEqual(evidence.engine_key_id, EXPECTED_ENGINE_KEY_ID)
        self.assertEqual(
            evidence.path,
            "queue/pending/job-2026-nemo-v3-governed-attempt-2.json",
        )

        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_2_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 29, 23, 10, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUARANTINED_NEVER_DISPATCH")
        self.assertTrue(report["terminal"])
        self.assertTrue(report["quarantine"]["valid"], report["quarantine"]["error"])
        self.assertEqual(
            report["quarantine"]["statuses"],
            ("STALE_SOURCE", "RETIRED_KEY", "NEVER_DISPATCH"),
        )
        self.assertEqual(
            hashlib.sha256(queue_path.read_bytes()).hexdigest(),
            "e74ecaea040c2abb52a5613c32e0648994f96ff39910c70e1fcc3e23fc053724",
        )
        self.assertEqual(
            report["reviewed_spec"]["path"],
            "jobspecs/nemo-v3-20260729-attempt-2-reviewed.json",
        )
        with self.assertRaisesRegex(ContractError, "NEVER_DISPATCH"):
            require_nemo_v3_dispatchable(self.attempt)

    def test_status_workflow_tracks_attempt_2_without_dispatching_it(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260729-attempt-2-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-2.json",
            "queue/quarantine/job-2026-nemo-v3-governed-attempt-2.json",
            "tests/test_reviewed_nemo_v3_attempt_2_spec.py",
            "attempt_id: attempt-2-b21",
            "spec_path: jobspecs/nemo-v3-20260729-attempt-2-reviewed.json",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIsNone(
            re.search(r"repository_dispatch|workflow_call", STATUS_WORKFLOW)
        )

    def test_signed_queue_bytes_are_stable_on_windows_checkout(self) -> None:
        self.assertIn(
            "queue/pending/*.json text eol=lf",
            GIT_ATTRIBUTES.splitlines(),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
