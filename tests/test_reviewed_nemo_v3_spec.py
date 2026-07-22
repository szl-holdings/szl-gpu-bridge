from __future__ import annotations

import json
import pathlib
import sys
import unittest
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

from nemo_v3_contract import record_ids_sha256, validate_nemo_v3_spec  # noqa: E402

SPEC_PATH = ROOT / "jobspecs" / "nemo-v3-20260722-reviewed.json"


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
        self.assertEqual(
            self.spec["base"]["licenseId"], "nvidia-open-model-license"
        )
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

    def test_unsigned_reviewed_spec_is_not_a_queue_job(self) -> None:
        queued = ROOT / "queue" / "pending" / f"{self.spec['jobId']}.json"
        self.assertFalse(
            queued.exists(),
            "the reviewed plaintext spec must never be treated as a signed queue envelope",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
