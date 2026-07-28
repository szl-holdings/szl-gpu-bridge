from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import frontier_job  # noqa: E402


class ReceiptOutboxTests(unittest.TestCase):
    def test_isolated_executor_emits_unsigned_intent_without_key(self) -> None:
        receipt = {
            "kind": "szl-frontier-training-blocked",
            "v": 2,
            "jobId": "job-test",
            "verdict": "BLOCKED",
            "stage": "gate:test",
            "reason": "negative control",
            "at": "2026-07-28T17:00:00Z",
        }
        spec = {"jobId": "job-test"}

        with tempfile.TemporaryDirectory() as temporary:
            bridge_root = pathlib.Path(temporary)
            with (
                mock.patch.object(frontier_job, "ROOT", bridge_root),
                mock.patch.dict(
                    os.environ,
                    {
                        frontier_job.RECEIPT_TRANSPORT_ENV: (
                            frontier_job.UNSIGNED_OUTBOX_TRANSPORT
                        )
                    },
                    clear=False,
                ),
                mock.patch.object(frontier_job, "sign_receipt") as sign,
                mock.patch.object(frontier_job, "upload_receipt") as upload,
            ):
                exit_code = frontier_job.deliver_receipt(
                    receipt, "blocked_receipt.signed.json", spec
                )

            intent_path = (
                bridge_root
                / "jobs"
                / "job-test"
                / "receipt-outbox"
                / "blocked_receipt.intent.json"
            )
            intent = json.loads(intent_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, frontier_job.RECEIPT_PENDING_FINALIZATION_EXIT_CODE)
        self.assertEqual(intent["receipt"], receipt)
        self.assertEqual(intent["transport"], frontier_job.UNSIGNED_OUTBOX_TRANSPORT)
        sign.assert_not_called()
        upload.assert_not_called()

    def test_unknown_transport_fails_closed(self) -> None:
        with mock.patch.dict(
            os.environ,
            {frontier_job.RECEIPT_TRANSPORT_ENV: "typo"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "unsupported receipt transport"):
                frontier_job.deliver_receipt(
                    {"jobId": "job-test"},
                    "blocked_receipt.signed.json",
                    {"jobId": "job-test"},
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
