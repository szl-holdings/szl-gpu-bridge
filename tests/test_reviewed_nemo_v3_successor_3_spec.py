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


SUCCESSOR_2_PATH = ROOT / "jobspecs" / "nemo-v3-20260729-successor-2-reviewed.json"
SUCCESSOR_3_PATH = ROOT / "jobspecs" / "nemo-v3-20260730-successor-3-reviewed.json"


class ReviewedNemoV3Successor3SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.successor_2 = json.loads(SUCCESSOR_2_PATH.read_text(encoding="utf-8"))
        cls.successor_3 = json.loads(SUCCESSOR_3_PATH.read_text(encoding="utf-8"))

    def test_recovery_generation_passes_contract_with_new_identity(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.successor_3), self.successor_3)
        self.assertEqual(
            self.successor_3["jobId"],
            "job-2026-nemo-v3-governed-successor-3",
        )
        self.assertNotEqual(
            self.successor_3["outputs"]["candidateId"],
            self.successor_2["outputs"]["candidateId"],
        )

    def test_science_contract_is_unchanged(self) -> None:
        earlier = copy.deepcopy(self.successor_2)
        recovery = copy.deepcopy(self.successor_3)
        for value in (earlier, recovery):
            value.pop("jobId")
            value.pop("createdAt")
            value.pop("expiresAt")
            value.pop("notes")
            value.pop("lineage")
            value.pop("authorization", None)
            value["outputs"].pop("candidateId")
        self.assertEqual(recovery, earlier)

    def test_recovery_authorization_is_exact_and_old_key_is_verify_only(self) -> None:
        self.assertEqual(
            self.successor_3["authorization"],
            {
                "engineKeyId": "815714c8d4ae3e4d",
                "previousEngineKeyId": "5c6cf59741ade920",
                "recoveryIssueUrl": (
                    "https://github.com/szl-holdings/szl-gpu-bridge/issues/25"
                ),
                "rotationMode": "LOST_PRIVATE_KEY_NEW_GENERATION",
                "oldKeyStatus": "VERIFY_ONLY",
                "decisionAt": "2026-07-30T15:38:47Z",
            },
        )
        self.assertEqual(self.successor_3["lineage"]["successorGeneration"], 3)
        self.assertFalse(self.successor_3["lineage"]["automaticRetry"])

    def test_signed_queue_uses_exact_enrolled_recovery_key(self) -> None:
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=SUCCESSOR_3_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 30, 16, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUEUED_AWAITING_GPU_RECEIPT")
        self.assertFalse(report["terminal"])
        self.assertTrue(report["queue"]["valid"])
        self.assertEqual(
            report["queue"]["engine_key_id"],
            "815714c8d4ae3e4d",
        )
        self.assertEqual(
            report["queue"]["payload_sha256"],
            "f20bf865dca5413262e5fd3733df112486aec72bb9b47932083ffecb2470a415",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
