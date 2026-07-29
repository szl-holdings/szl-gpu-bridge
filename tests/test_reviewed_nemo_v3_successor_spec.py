from __future__ import annotations

import copy
import json
import pathlib
import sys
import unittest
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "laptop"))

import nemo_v3_status  # noqa: E402
from nemo_v3_contract import validate_nemo_v3_spec  # noqa: E402


PREDECESSOR_PATH = ROOT / "jobspecs" / "nemo-v3-20260722-reviewed.json"
SUCCESSOR_PATH = ROOT / "jobspecs" / "nemo-v3-20260729-successor-2-reviewed.json"


class ReviewedNemoV3SuccessorSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predecessor = json.loads(PREDECESSOR_PATH.read_text(encoding="utf-8"))
        self.successor = json.loads(SUCCESSOR_PATH.read_text(encoding="utf-8"))

    def test_successor_passes_contract_with_distinct_identity(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.successor), self.successor)
        self.assertNotEqual(self.successor["jobId"], self.predecessor["jobId"])
        self.assertNotEqual(
            self.successor["outputs"]["candidateId"],
            self.predecessor["outputs"]["candidateId"],
        )

    def test_science_contract_is_unchanged(self) -> None:
        predecessor = copy.deepcopy(self.predecessor)
        successor = copy.deepcopy(self.successor)
        for value in (predecessor, successor):
            value.pop("jobId")
            value.pop("createdAt")
            value.pop("expiresAt")
            value.pop("notes")
            value.pop("lineage", None)
            value["outputs"].pop("candidateId")
        self.assertEqual(successor, predecessor)

    def test_lineage_binds_exact_quarantined_predecessor(self) -> None:
        self.assertEqual(
            self.successor["lineage"],
            {
                "predecessorJobId": "job-2026-nemo-v3-governed-attempt-1",
                "predecessorClaimSha256": (
                    "77fd63583bf11f1d7416cea7e6e0c02b230973d4773f9c409ce18aa83140f10b"
                ),
                "predecessorEnvelopeSha256": (
                    "09187c0a724c8caf8a11dcd492d3f284af8a18791adac7e1a98b9a21bf81591b"
                ),
                "predecessorBridgeRevision": (
                    "114c3030763291009d665ae88cb3d6537fccacef"
                ),
                "predecessorImageId": (
                    "sha256:537e4a25a503d202ec75dbb9035bd9688"
                    "ba2ae8d8a7555466840e581d5109f28"
                ),
                "predecessorClaimedAt": "2026-07-29T16:41:34.8842570+00:00",
                "incidentUrl": (
                    "https://github.com/szl-holdings/szl-gpu-bridge/"
                    "issues/4#issuecomment-5120817312"
                ),
                "failurePhase": "PRE_TRAINING_RUNTIME_SOURCE_PARSE",
                "successorGeneration": 2,
                "automaticRetry": False,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": False,
                "candidateProduced": False,
                "receiptIntentProduced": False,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            },
        )

    def test_successor_remains_non_executable_without_fresh_signature(self) -> None:
        queue_path = ROOT / "queue" / "pending" / f"{self.successor['jobId']}.json"
        self.assertFalse(queue_path.exists())
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=SUCCESSOR_PATH,
            now=datetime(2026, 7, 29, 19, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "AWAITING_ENGINE_SIGNATURE")
        self.assertFalse(report["terminal"])
        self.assertEqual(
            report["reviewed_spec"]["path"],
            "jobspecs/nemo-v3-20260729-successor-2-reviewed.json",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
