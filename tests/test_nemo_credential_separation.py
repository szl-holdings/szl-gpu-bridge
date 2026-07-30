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


def valid_attempt_claim(
    spec_path: pathlib.Path,
    now: datetime,
) -> dict[str, object]:
    digest = "b" * 64
    return {
        "kind": "szl-nemo-v3-attempt-claim",
        "v": 3,
        "jobId": "job-test",
        "jobEnvelopeSha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "bridgeRevision": "a" * 40,
        "envelopeRevision": "c" * 40,
        "executionBridgeRevision": "a" * 40,
        "launcherSha256": "f" * 64,
        "trainingImage": f"unsloth/unsloth@sha256:{digest}",
        "observedImageId": f"sha256:{digest}",
        "environmentProbeSha256": "d" * 64,
        "githubRunId": "123",
        "claimedAt": now.isoformat().replace("+00:00", "Z"),
    }


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
            claim = valid_attempt_claim(spec_path, now)
            claim_path.write_text(json.dumps(claim), encoding="utf-8")

            observed = finalize_nemo_v3_receipt.validate_attempt_claim(
                claim_path,
                spec_path,
                spec,
                now,
            )
            self.assertEqual(observed, claim)

            claim["launcherSha256"] = "mutable"
            claim_path.write_text(json.dumps(claim), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "launcher binding"):
                finalize_nemo_v3_receipt.validate_attempt_claim(
                    claim_path,
                    spec_path,
                    spec,
                    now,
                )

    def test_attempt_claim_rejects_unapproved_registry_digest(self) -> None:
        now = datetime.now(timezone.utc)
        spec = {"jobId": "job-test"}
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec_path = root / "job.json"
            spec_path.write_bytes(b'{"signed":"envelope"}\n')
            claim_path = root / "claim.json"
            claim = valid_attempt_claim(spec_path, now)
            claim["trainingImage"] = "registry.example/image@sha256:" + "b" * 64
            claim_path.write_text(json.dumps(claim), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "immutable execution identity"):
                finalize_nemo_v3_receipt.validate_attempt_claim(
                    claim_path,
                    spec_path,
                    spec,
                    now,
                )

    def test_attempt_claim_rejects_same_envelope_and_execution_revision(self) -> None:
        now = datetime.now(timezone.utc)
        spec = {"jobId": "job-test"}
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec_path = root / "job.json"
            spec_path.write_bytes(b'{"signed":"envelope"}\n')
            claim_path = root / "claim.json"
            claim = valid_attempt_claim(spec_path, now)
            claim["envelopeRevision"] = claim["executionBridgeRevision"]
            claim_path.write_text(json.dumps(claim), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "immutable execution identity"):
                finalize_nemo_v3_receipt.validate_attempt_claim(
                    claim_path,
                    spec_path,
                    spec,
                    now,
                )

    def test_trusted_finalizer_binds_receipt_stack_to_exact_claim(self) -> None:
        now = datetime.now(timezone.utc)
        exact_payload = b"signed-job"
        spec = {
            "jobId": "job-test",
            "outputs": {"candidateId": "candidate-test"},
            "source": {"repoId": "szl-holdings/a11oy", "revision": "a" * 40},
            "base": {
                "repoId": "nvidia/model",
                "revision": "b" * 40,
                "licenseId": "test-license",
            },
            "dataset": {"rightsBasis": "project-authored"},
            "evaluation": {"requiredPassRate": 1.0},
        }
        claim = {
            "launcherSha256": "a" * 64,
            "trainingImage": "unsloth/unsloth@sha256:" + "c" * 64,
            "observedImageId": "sha256:" + "c" * 64,
            "environmentProbeSha256": "e" * 64,
            "envelopeRevision": "d" * 40,
            "executionBridgeRevision": "f" * 40,
        }
        container_image = finalize_nemo_v3_receipt.claim_container_image(claim)
        receipt = {
            "kind": "szl-nemo-v3-governed-training",
            "v": 1,
            "jobId": spec["jobId"],
            "candidateId": spec["outputs"]["candidateId"],
            "state": "EVALUATION_FAILED_NOT_PROMOTED_NOT_SIGNED",
            "at": now.isoformat().replace("+00:00", "Z"),
            "source": spec["source"],
            "base": spec["base"],
            "signed_job_payload_sha256": hashlib.sha256(exact_payload).hexdigest(),
            "training_rights_basis": spec["dataset"]["rightsBasis"],
            "evaluation": {
                "state": "FAIL",
                "stack": {
                    "containerImage": container_image,
                    "bridgeExecution": (
                        finalize_nemo_v3_receipt.claim_bridge_execution(claim)
                    ),
                    "launcherSha256": claim["launcherSha256"],
                },
            },
            "effects": {
                "candidate_uploaded": False,
                "published": False,
                "deployed": False,
                "promoted": False,
            },
            "decision": "TERMINAL_FAILURE_NO_AUTOMATIC_RETRY",
        }
        intent = {
            "kind": "szl-receipt-signing-intent",
            "v": 1,
            "jobId": spec["jobId"],
            "requestedReceiptName": "nemo-v3-terminal.signed.json",
            "receipt": receipt,
            "transport": "local-unsigned-outbox",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "terminal.intent.json"
            path.write_text(json.dumps(intent), encoding="utf-8")
            observed, state, requested_name = finalize_nemo_v3_receipt.validate_intent(
                path,
                spec,
                exact_payload,
                now - timedelta(seconds=1),
                root,
                claim,
            )
            self.assertEqual(observed, receipt)
            self.assertEqual(state, "EVALUATION_FAILED_NOT_PROMOTED_NOT_SIGNED")
            self.assertEqual(requested_name, "nemo-v3-terminal.signed.json")

            receipt["evaluation"]["stack"]["containerImage"]["id"] = (
                "sha256:" + "0" * 64
            )
            path.write_text(json.dumps(intent), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trusted claim"):
                finalize_nemo_v3_receipt.validate_intent(
                    path,
                    spec,
                    exact_payload,
                    now - timedelta(seconds=1),
                    root,
                    claim,
                )

            receipt["evaluation"]["stack"]["containerImage"] = (
                finalize_nemo_v3_receipt.claim_container_image(claim)
            )
            receipt["evaluation"]["stack"]["bridgeExecution"][
                "executionBridgeRevision"
            ] = "0" * 40
            path.write_text(json.dumps(intent), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Bridge revisions"):
                finalize_nemo_v3_receipt.validate_intent(
                    path,
                    spec,
                    exact_payload,
                    now - timedelta(seconds=1),
                    root,
                    claim,
                )

            receipt["evaluation"]["stack"]["bridgeExecution"] = (
                finalize_nemo_v3_receipt.claim_bridge_execution(claim)
            )
            receipt["evaluation"]["stack"]["launcherSha256"] = "0" * 64
            path.write_text(json.dumps(intent), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "receipt launcher"):
                finalize_nemo_v3_receipt.validate_intent(
                    path,
                    spec,
                    exact_payload,
                    now - timedelta(seconds=1),
                    root,
                    claim,
                )

    def test_attempt_claim_rejects_a_mutable_training_image(self) -> None:
        now = datetime.now(timezone.utc)
        spec = {"jobId": "job-test"}
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec_path = root / "job.json"
            spec_path.write_bytes(b'{"signed":"envelope"}\n')
            claim_path = root / "claim.json"
            claim = valid_attempt_claim(spec_path, now)
            claim["trainingImage"] = "szl-nemo-v3:local"
            claim_path.write_text(json.dumps(claim), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "immutable execution identity"):
                finalize_nemo_v3_receipt.validate_attempt_claim(
                    claim_path,
                    spec_path,
                    spec,
                    now,
                )

    def test_attempt_claim_rejects_a_different_signed_envelope(self) -> None:
        now = datetime.now(timezone.utc)
        spec = {"jobId": "job-test"}
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            spec_path = root / "job.json"
            spec_path.write_bytes(b'{"signed":"changed"}\n')
            claim_path = root / "claim.json"
            claim = valid_attempt_claim(spec_path, now)
            claim["jobEnvelopeSha256"] = "0" * 64
            claim_path.write_text(json.dumps(claim), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "signed job envelope"):
                finalize_nemo_v3_receipt.validate_attempt_claim(
                    claim_path,
                    spec_path,
                    spec,
                    now,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
