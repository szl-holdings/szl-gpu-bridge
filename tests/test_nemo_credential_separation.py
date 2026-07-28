from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import finalize_nemo_v3_receipt  # noqa: E402
import prefetch_nemo_v3  # noqa: E402


class NemoCredentialSeparationTests(unittest.TestCase):
    def test_prefetch_reuses_only_byte_verified_input(self) -> None:
        content = b'{"record_id":"train:1"}\n'
        descriptor = {
            "path": "model_release/szl-nemo-v3/train.jsonl",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        spec = {
            "source": {
                "repoId": "szl-holdings/a11oy",
                "revision": "a" * 40,
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            cache = pathlib.Path(temporary)
            cached = cache / descriptor["path"]
            cached.parent.mkdir(parents=True)
            cached.write_bytes(content)
            evidence = prefetch_nemo_v3.fetch_descriptor(spec, descriptor, cache)

        self.assertEqual(evidence["sha256"], descriptor["sha256"])
        self.assertEqual(evidence["bytes"], descriptor["bytes"])

    def test_trusted_finalizer_accepts_fresh_blocked_intent(self) -> None:
        now = datetime.now(timezone.utc)
        receipt = {
            "kind": "szl-frontier-training-blocked",
            "v": 2,
            "jobId": "job-test",
            "verdict": "BLOCKED",
            "stage": "gate:test",
            "reason": "negative control",
            "at": now.isoformat().replace("+00:00", "Z"),
        }
        intent = {
            "kind": "szl-receipt-signing-intent",
            "v": 1,
            "jobId": "job-test",
            "requestedReceiptName": "blocked_receipt.signed.json",
            "receipt": receipt,
            "transport": "local-unsigned-outbox",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "blocked_receipt.intent.json"
            path.write_text(json.dumps(intent), encoding="utf-8")
            observed, state = finalize_nemo_v3_receipt.validate_intent(
                path,
                {"jobId": "job-test"},
                b"signed-job",
                now - timedelta(seconds=1),
                root,
            )

        self.assertEqual(observed, receipt)
        self.assertEqual(state, "BLOCKED")

    def test_trusted_finalizer_rejects_stale_intent(self) -> None:
        now = datetime.now(timezone.utc)
        intent = {
            "kind": "szl-receipt-signing-intent",
            "v": 1,
            "jobId": "job-test",
            "requestedReceiptName": "blocked_receipt.signed.json",
            "receipt": {
                "kind": "szl-frontier-training-blocked",
                "v": 2,
                "jobId": "job-test",
                "verdict": "BLOCKED",
                "stage": "gate:test",
                "reason": "negative control",
                "at": (now - timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
            },
            "transport": "local-unsigned-outbox",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "blocked_receipt.intent.json"
            path.write_text(json.dumps(intent), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "predates"):
                finalize_nemo_v3_receipt.validate_intent(
                    path,
                    {"jobId": "job-test"},
                    b"signed-job",
                    now,
                    root,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
