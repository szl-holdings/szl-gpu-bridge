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
            observed, state, requested_name = finalize_nemo_v3_receipt.validate_intent(
                path,
                {"jobId": "job-test"},
                b"signed-job",
                now - timedelta(seconds=1),
                root,
            )

        self.assertEqual(observed, receipt)
        self.assertEqual(state, "BLOCKED")
        self.assertEqual(requested_name, "blocked_receipt.signed.json")

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

    def test_trusted_finalizer_uploads_only_the_validated_requested_name(self) -> None:
        source = (ROOT / "laptop" / "finalize_nemo_v3_receipt.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "frontier_job.upload_receipt(signed, requested_name, spec)",
            source,
        )
        self.assertNotIn("args.intent.name.replace", source)

    def test_attempt_claim_binds_exact_signed_envelope_and_execution(self) -> None:
        now = datetime.now(timezone.utc)
        spec = {"jobId": "job-test"}
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec_path = root / "job.json"
            spec_path.write_bytes(b'{"signed":"envelope"}\n')
            claim_path = root / "claim.json"
            claim = {
                "kind": "szl-nemo-v3-attempt-claim",
                "v": 1,
                "jobId": "job-test",
                "jobEnvelopeSha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                "bridgeRevision": "a" * 40,
                "trainingImage": "unsloth/unsloth@sha256:" + "b" * 64,
                "githubRunId": "123",
                "claimedAt": now.isoformat().replace("+00:00", "Z"),
            }
            claim_path.write_text(json.dumps(claim), encoding="utf-8")

            observed = finalize_nemo_v3_receipt.validate_attempt_claim(
                claim_path,
                spec_path,
                spec,
                now,
            )

        self.assertEqual(observed, claim)

    def test_attempt_claim_rejects_a_different_signed_envelope(self) -> None:
        now = datetime.now(timezone.utc)
        spec = {"jobId": "job-test"}
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec_path = root / "job.json"
            spec_path.write_bytes(b'{"signed":"changed"}\n')
            claim_path = root / "claim.json"
            claim_path.write_text(
                json.dumps(
                    {
                        "kind": "szl-nemo-v3-attempt-claim",
                        "v": 1,
                        "jobId": "job-test",
                        "jobEnvelopeSha256": "0" * 64,
                        "bridgeRevision": "a" * 40,
                        "trainingImage": "unsloth/unsloth@sha256:" + "b" * 64,
                        "claimedAt": now.isoformat().replace("+00:00", "Z"),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "signed job envelope"):
                finalize_nemo_v3_receipt.validate_attempt_claim(
                    claim_path,
                    spec_path,
                    spec,
                    now,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
