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

    def test_isolated_executor_cannot_replace_existing_intent(self) -> None:
        original_receipt = {
            "kind": "szl-frontier-training-blocked",
            "v": 2,
            "jobId": "job-test",
            "verdict": "BLOCKED",
            "stage": "gate:first",
            "reason": "first immutable result",
            "at": "2026-07-28T17:00:00Z",
        }
        replacement_receipt = {
            **original_receipt,
            "stage": "gate:replacement",
            "reason": "must not replace earlier evidence",
        }
        spec = {"jobId": "job-test"}

        with tempfile.TemporaryDirectory() as temporary:
            bridge_root = pathlib.Path(temporary)
            environment = {
                frontier_job.RECEIPT_TRANSPORT_ENV: (
                    frontier_job.UNSIGNED_OUTBOX_TRANSPORT
                )
            }
            with (
                mock.patch.object(frontier_job, "ROOT", bridge_root),
                mock.patch.dict(os.environ, environment, clear=False),
            ):
                first_exit = frontier_job.deliver_receipt(
                    original_receipt,
                    "blocked_receipt.signed.json",
                    spec,
                )
                intent_path = (
                    bridge_root
                    / "jobs"
                    / "job-test"
                    / "receipt-outbox"
                    / "blocked_receipt.intent.json"
                )
                original_bytes = intent_path.read_bytes()
                with self.assertRaisesRegex(
                    RuntimeError, "immutable JSON already exists"
                ):
                    frontier_job.deliver_receipt(
                        replacement_receipt,
                        "blocked_receipt.signed.json",
                        spec,
                    )

            self.assertEqual(
                first_exit,
                frontier_job.RECEIPT_PENDING_FINALIZATION_EXIT_CODE,
            )
            self.assertEqual(intent_path.read_bytes(), original_bytes)
            self.assertEqual(
                json.loads(original_bytes)["receipt"],
                original_receipt,
            )
            self.assertEqual(list(intent_path.parent.glob(".*.tmp")), [])

    def test_new_json_flushes_before_exclusive_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary) / "receipt.intent.json"
            events: list[str] = []
            real_fsync = frontier_job.os.fsync
            publication_name = "rename" if frontier_job.os.name == "nt" else "link"
            real_publish = getattr(frontier_job.os, publication_name)

            def recording_fsync(descriptor: int) -> None:
                events.append("fsync")
                real_fsync(descriptor)

            def recording_publish(source: object, destination: object) -> None:
                events.append("publish")
                real_publish(source, destination)

            with (
                mock.patch.object(
                    frontier_job.os, "fsync", side_effect=recording_fsync
                ),
                mock.patch.object(
                    frontier_job.os,
                    publication_name,
                    side_effect=recording_publish,
                ),
            ):
                frontier_job.write_new_json(target, {"verdict": "BLOCKED"})

            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"verdict": "BLOCKED"},
            )
            self.assertEqual(events, ["fsync", "publish"])
            self.assertEqual(list(target.parent.glob(".*.tmp")), [])

    def test_new_json_preserves_competitor_winning_publish_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary) / "receipt.intent.json"
            competitor_bytes = b'{"writer":"competitor"}\n'
            publication_name = "rename" if frontier_job.os.name == "nt" else "link"
            real_publish = getattr(frontier_job.os, publication_name)

            def publish_after_competitor(source: object, destination: object) -> None:
                pathlib.Path(destination).write_bytes(competitor_bytes)
                real_publish(source, destination)

            with mock.patch.object(
                frontier_job.os,
                publication_name,
                side_effect=publish_after_competitor,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "immutable JSON already exists"
                ):
                    frontier_job.write_new_json(target, {"writer": "candidate"})

            self.assertEqual(target.read_bytes(), competitor_bytes)
            self.assertEqual(list(target.parent.glob(".*.tmp")), [])

    def test_new_json_fails_closed_when_no_overwrite_publish_is_unsupported(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary) / "receipt.intent.json"
            publication_name = "rename" if frontier_job.os.name == "nt" else "link"

            with mock.patch.object(
                frontier_job.os,
                publication_name,
                side_effect=OSError("operation unsupported by mounted filesystem"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "filesystem cannot publish immutable JSON without overwrite",
                ):
                    frontier_job.write_new_json(target, {"verdict": "BLOCKED"})

            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.glob(".*.tmp")), [])

    def test_new_json_preserves_preexisting_publication_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "receipt.intent.json"
            lock = root / ".receipt.intent.json.lock"
            lock.mkdir()

            with self.assertRaisesRegex(RuntimeError, "publication is locked"):
                frontier_job.write_new_json(target, {"verdict": "BLOCKED"})

            self.assertTrue(lock.is_dir())
            self.assertFalse(target.exists())

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
