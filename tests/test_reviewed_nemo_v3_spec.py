from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import unittest
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "laptop"))

import nemo_v3_status  # noqa: E402
from nemo_v3_contract import (  # noqa: E402
    ContractError,
    quarantine_policy,
    record_ids_sha256,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

SPEC_PATH = ROOT / "jobspecs" / "nemo-v3-20260722-reviewed.json"
SUCCESSOR_PATH = ROOT / "jobspecs" / "nemo-v3-20260729-successor-2-reviewed.json"
QUEUE_PATH = ROOT / "queue" / "pending" / "job-2026-nemo-v3-governed-attempt-1.json"
EVIDENCE_PATH = ROOT / "queue" / "evidence" / "job-2026-nemo-v3-governed-attempt-1.json"
QUARANTINE_PATH = (
    ROOT / "queue" / "quarantine" / "job-2026-nemo-v3-governed-attempt-1.json"
)
EXPECTED_QUEUE_PAYLOAD_SHA256 = (
    "8a5c2e3f99711be84e45371824ca737d480e587ff61c55cc3d30ad96d2c62055"
)
EXPECTED_ENGINE_KEY_ID = "5c6cf59741ade920"
EXPECTED_QUEUE_FILE_SHA256 = (
    "0686889c3abcf54e3f6b2151bc60155176e1eccb25af7b01d9f1fbf05080d80d"
)
EXPECTED_SUCCESSOR_BLOB_SHA256 = (
    "9d58f752c26ac37ae7fa4999e33a6f136d060e97704124df26f0ee7948a11746"
)
EXPECTED_EVIDENCE_SHA256 = (
    "d3f28fd63ee4c84ecf7aa72300a7fe55a29033953906356a83fdf089f47aaed6"
)
EXPECTED_QUARANTINE_STATUSES = (
    "PRE_TRAINING_RUNTIME_SOURCE_PARSE",
    "POST_CLAIM",
    "NEVER_DISPATCH",
    "NEVER_RESEND",
    "NEVER_RESIGN",
)


class ReviewedNemoV3SpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def test_reviewed_spec_passes_the_execution_contract(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.spec), self.spec)
        created = datetime.fromisoformat(self.spec["createdAt"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(self.spec["expiresAt"].replace("Z", "+00:00"))
        self.assertGreater(expires, created)
        self.assertGreater(expires, datetime(2026, 7, 22, tzinfo=timezone.utc))

    def test_source_base_and_rights_are_immutable(self) -> None:
        self.assertEqual(
            self.spec["source"],
            {
                "repoId": "szl-holdings/a11oy",
                "revision": "a5351c8e37a7cfe54e0c3cf53c8bbd460a16c11c",
                "licenseId": "apache-2.0",
            },
        )
        self.assertEqual(
            self.spec["base"]["repoId"],
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        )
        self.assertEqual(
            self.spec["base"]["revision"],
            "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        )
        self.assertEqual(self.spec["base"]["licenseId"], "nvidia-open-model-license")
        self.assertEqual(
            self.spec["dataset"]["rightsBasis"], "PROJECT_AUTHORED_SCENARIOS"
        )

    def test_all_frozen_file_digests_and_record_orders_are_bound(self) -> None:
        self.assertEqual(
            self.spec["dataset"]["train"],
            {
                "path": "model_release/szl-nemo-v3/train.jsonl",
                "sha256": "a81e5742d8146dfb67a0754e45b578765b5c6212ff6725b8157035b49c0e1c9a",
                "bytes": 28968,
            },
        )
        expected = [
            (
                "original-v2",
                "caeb07c94929c24a47fd12f35cbc9021523308dc9fcc684bd444ffcf4a367b0d",
                5967,
            ),
            (
                "shadow-v2",
                "1b8578051b7b829595493615bdf11e24d01a1837a23d6191c8e88e21cce990ac",
                8261,
            ),
            (
                "challenge-v3",
                "1b23d20406eb96d3c58741ac57c802eb723a19802c7f83b8500b99a71d15c35f",
                9903,
            ),
        ]
        observed = [
            (item["name"], item["sha256"], item["bytes"])
            for item in self.spec["dataset"]["holdouts"]
        ]
        self.assertEqual(observed, expected)
        for item in self.spec["dataset"]["holdouts"]:
            self.assertEqual(
                item["recordIdsSha256"], record_ids_sha256(item["recordIds"])
            )
        self.assertEqual(
            self.spec["dataset"]["preregistration"],
            {
                "path": "model_release/szl-nemo-v3/preregistration.json",
                "sha256": "41a27921ff3a377442e2cf7b4ffe569324de73cedb56b2485e50a6e8057cfacd",
                "bytes": 3076,
            },
        )

    def test_training_and_evaluation_gates_are_not_weakened(self) -> None:
        self.assertGreaterEqual(self.spec["gates"]["minFreeVramGb"], 6.5)
        self.assertGreaterEqual(self.spec["gates"]["minFreeDiskGb"], 50)
        self.assertLessEqual(self.spec["gates"]["maxTemperatureC"], 78)
        self.assertLessEqual(self.spec["gates"]["maxUtilizationPct"], 15)
        self.assertEqual(self.spec["evaluation"]["requiredPassRate"], 1.0)
        self.assertEqual(self.spec["evaluation"]["maxDegenerateRate"], 0.0)
        self.assertTrue(self.spec["evaluation"]["requireExactRecordOrder"])
        self.assertFalse(self.spec["outputs"]["publishCandidate"])
        self.assertTrue(self.spec["outputs"]["private"])

    def test_reviewed_plaintext_spec_is_distinct_from_exact_signed_queue(self) -> None:
        queued = ROOT / "queue" / "pending" / f"{self.spec['jobId']}.json"
        self.assertTrue(
            queued.is_file(),
            "the exact reviewed attempt must have an engine-signed queue envelope",
        )
        envelope = json.loads(queued.read_text(encoding="utf-8"))
        self.assertNotEqual(
            envelope,
            self.spec,
            "the reviewed plaintext spec must never be substituted for a DSSE envelope",
        )
        self.assertEqual(len(envelope.get("signatures", [])), 1)

        evidence = nemo_v3_status.verify_queue(self.spec, ROOT)
        self.assertTrue(evidence.present)
        self.assertTrue(evidence.valid, evidence.error)
        self.assertEqual(evidence.payload_sha256, EXPECTED_QUEUE_PAYLOAD_SHA256)
        self.assertEqual(evidence.engine_key_id, EXPECTED_ENGINE_KEY_ID)
        self.assertEqual(
            evidence.path,
            f"queue/pending/{self.spec['jobId']}.json",
        )

    def test_consumed_predecessor_is_immutable_terminal_quarantine(self) -> None:
        successor = json.loads(SUCCESSOR_PATH.read_text(encoding="utf-8"))
        evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        quarantine = json.loads(QUARANTINE_PATH.read_text(encoding="utf-8"))
        policy = quarantine_policy(self.spec)
        self.assertIsNotNone(policy)
        self.assertEqual(tuple(policy["statuses"]), EXPECTED_QUARANTINE_STATUSES)
        self.assertEqual(evidence["executionEvidence"], successor["lineage"])
        self.assertEqual(
            hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest(),
            EXPECTED_EVIDENCE_SHA256,
        )
        successor_blob = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:jobspecs/nemo-v3-20260729-successor-2-reviewed.json",
            ],
            cwd=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(successor_blob).hexdigest(),
            EXPECTED_SUCCESSOR_BLOB_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(QUEUE_PATH.read_bytes()).hexdigest(),
            EXPECTED_QUEUE_FILE_SHA256,
        )
        self.assertEqual(quarantine["status"], list(EXPECTED_QUARANTINE_STATUSES))
        self.assertEqual(quarantine["replacement"]["reviewedJobId"], successor["jobId"])
        self.assertEqual(
            quarantine["replacement"]["reviewedSpecSha256"],
            EXPECTED_SUCCESSOR_BLOB_SHA256,
        )
        self.assertTrue(quarantine["preserveEnvelope"])
        self.assertFalse(quarantine["dispatchAuthorized"])

        def unexpected_receipt_loader(_spec: dict[str, object], _token: str) -> None:
            self.fail("terminal quarantine must not inspect the receipt repository")

        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=SPEC_PATH,
            receipt_loader=unexpected_receipt_loader,
            now=datetime(2026, 8, 30, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUARANTINED_NEVER_DISPATCH")
        self.assertTrue(report["terminal"])
        self.assertTrue(report["queue"]["valid"], report["queue"]["error"])
        self.assertTrue(report["quarantine"]["valid"], report["quarantine"]["error"])
        self.assertEqual(report["quarantine"]["statuses"], EXPECTED_QUARANTINE_STATUSES)
        self.assertFalse(report["receipt"]["present"])
        with self.assertRaisesRegex(ContractError, "NEVER_DISPATCH"):
            require_nemo_v3_dispatchable(self.spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
