from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "laptop"))

from frontier_contract import canonicalize, derive_key_id, pae  # noqa: E402
from nemo_v3_contract import NEMO_V3_PAYLOAD_TYPE  # noqa: E402
import nemo_v3_status  # noqa: E402


class NemoV3StatusTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from nacl.signing import SigningKey
        except ImportError as exc:  # pragma: no cover - CI installs PyNaCl
            self.skipTest(f"PyNaCl unavailable: {exc}")
        self.SigningKey = SigningKey
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "jobspecs").mkdir()
        (self.root / "keys").mkdir()
        (self.root / "queue" / "pending").mkdir(parents=True)
        self.spec = json.loads(
            (ROOT / "jobspecs" / "nemo-v3-20260722-reviewed.json").read_text(
                encoding="utf-8"
            )
        )
        (self.root / "jobspecs" / "nemo-v3-20260722-reviewed.json").write_text(
            json.dumps(self.spec, indent=2) + "\n", encoding="utf-8"
        )
        self.engine = self.SigningKey.generate()
        self.engine_spki = (
            b"\x30\x2a\x30\x05\x06\x03\x2b\x65\x70\x03\x21\x00"
            + bytes(self.engine.verify_key)
        )
        self.engine_key_id = derive_key_id(self.engine_spki)
        (self.root / "keys" / "engine_pubkey.json").write_text(
            json.dumps(
                {
                    "keyId": self.engine_key_id,
                    "publicKeySpkiBase64": base64.b64encode(self.engine_spki).decode(),
                }
            ),
            encoding="utf-8",
        )
        self.laptop = self.SigningKey.generate()
        self.laptop_spki = (
            b"\x30\x2a\x30\x05\x06\x03\x2b\x65\x70\x03\x21\x00"
            + bytes(self.laptop.verify_key)
        )
        self.laptop_key_id = derive_key_id(self.laptop_spki)
        self.now = datetime(2026, 7, 23, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def enqueue(self) -> str:
        payload = canonicalize(self.spec).encode("utf-8")
        envelope = {
            "payloadType": NEMO_V3_PAYLOAD_TYPE,
            "payload": base64.b64encode(payload).decode(),
            "signatures": [
                {
                    "keyid": self.engine_key_id,
                    "sig": base64.b64encode(
                        self.engine.sign(pae(NEMO_V3_PAYLOAD_TYPE, payload)).signature
                    ).decode(),
                }
            ],
            "publicKeySpkiBase64": base64.b64encode(self.engine_spki).decode(),
        }
        path = self.root / "queue" / "pending" / f"{self.spec['jobId']}.json"
        path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
        return hashlib.sha256(payload).hexdigest()

    def signed_receipt(self, *, state: str, payload_sha: str) -> dict[str, object]:
        receipt = {
            "kind": "szl-nemo-v3-governed-training",
            "v": 1,
            "jobId": self.spec["jobId"],
            "candidateId": self.spec["outputs"]["candidateId"],
            "state": state,
            "source": self.spec["source"],
            "base": {
                key: self.spec["base"][key]
                for key in ("repoId", "revision", "licenseId")
            },
            "signed_job_payload_sha256": payload_sha,
            "training_rights_basis": "PROJECT_AUTHORED_SCENARIOS",
            "evaluation": (
                {
                    "state": "PASS",
                    "rows": 30,
                    "passes": 30,
                    "pass_rate": 1.0,
                    "degenerate": 0,
                }
                if state == "QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW"
                else {
                    "state": "FAIL",
                    "rows": 30,
                    "passes": 29,
                    "pass_rate": 29 / 30,
                    "degenerate": 0,
                }
            ),
            "effects": {
                "candidate_uploaded": False,
                "published": False,
                "deployed": False,
                "promoted": False,
            },
            "decision": (
                "SEPARATE_PROMOTION_REVIEW_REQUIRED"
                if state == "QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW"
                else "TERMINAL_FAILURE_NO_AUTOMATIC_RETRY"
            ),
        }
        body = canonicalize(receipt).encode("utf-8")
        return {
            "receipt": receipt,
            "bodyBase64": base64.b64encode(body).decode(),
            "signatureBase64": base64.b64encode(
                self.laptop.sign(body).signature
            ).decode(),
            "publicKeySpkiBase64": base64.b64encode(self.laptop_spki).decode(),
            "keyId": self.laptop_key_id,
            "scheme": "ed25519-over-exact-bytes-v2",
        }

    def test_plaintext_spec_is_waiting_not_executable(self) -> None:
        report = nemo_v3_status.evaluate(root=self.root, now=self.now)
        self.assertEqual(report["status"], "AWAITING_ENGINE_SIGNATURE")
        self.assertFalse(report["queue"]["present"])
        self.assertFalse(report["terminal"])

    def test_valid_queue_waits_for_gpu_receipt(self) -> None:
        self.enqueue()
        report = nemo_v3_status.evaluate(
            root=self.root,
            receipt_loader=lambda _spec, _token: None,
            now=self.now,
        )
        self.assertEqual(report["status"], "QUEUED_AWAITING_GPU_RECEIPT")
        self.assertTrue(report["queue"]["valid"])
        self.assertEqual(report["queue"]["engine_key_id"], self.engine_key_id)

    def test_valid_but_unenrolled_laptop_receipt_is_not_trusted(self) -> None:
        payload_sha = self.enqueue()
        signed = self.signed_receipt(
            state="QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW",
            payload_sha=payload_sha,
        )
        report = nemo_v3_status.evaluate(
            root=self.root,
            receipt_loader=lambda _spec, _token: (
                f"{self.spec['jobId']}/nemo-v3-qualified.signed.json",
                signed,
            ),
            now=self.now,
        )
        self.assertEqual(
            report["status"], "AWAITING_LAPTOP_RECEIPT_KEY_ENROLLMENT"
        )
        self.assertEqual(report["receipt"]["observed_key_id"], self.laptop_key_id)
        self.assertFalse(report["receipt"]["valid"])

    def test_pinned_all_pass_receipt_qualifies_only_for_separate_review(self) -> None:
        payload_sha = self.enqueue()
        signed = self.signed_receipt(
            state="QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW",
            payload_sha=payload_sha,
        )
        report = nemo_v3_status.evaluate(
            root=self.root,
            expected_laptop_key_id=self.laptop_key_id,
            receipt_loader=lambda _spec, _token: (
                f"{self.spec['jobId']}/nemo-v3-qualified.signed.json",
                signed,
            ),
            now=self.now,
        )
        self.assertEqual(
            report["status"], "QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW"
        )
        self.assertTrue(report["terminal"])
        self.assertTrue(report["receipt"]["identity_pinned"])
        self.assertEqual(
            report["receipt"]["payload_binding"], "EXACT_SIGNED_PAYLOAD_SHA256"
        )
        self.assertFalse(report["reviewed_spec"]["candidate_publication_enabled"])

    def test_receipt_for_different_queue_payload_fails_closed(self) -> None:
        self.enqueue()
        signed = self.signed_receipt(
            state="EVALUATION_FAILED_NOT_PROMOTED_NOT_SIGNED",
            payload_sha="f" * 64,
        )
        report = nemo_v3_status.evaluate(
            root=self.root,
            expected_laptop_key_id=self.laptop_key_id,
            receipt_loader=lambda _spec, _token: (
                f"{self.spec['jobId']}/nemo-v3-terminal.signed.json",
                signed,
            ),
            now=self.now,
        )
        self.assertEqual(report["status"], "INVALID_TERMINAL_RECEIPT")
        self.assertIn("exact signed queue payload", report["receipt"]["error"])

    def test_pinned_blocked_receipt_is_an_honest_terminal_failure(self) -> None:
        self.enqueue()
        receipt = {
            "kind": "szl-frontier-training-blocked",
            "v": 2,
            "jobId": self.spec["jobId"],
            "verdict": "BLOCKED",
            "stage": "gate:initial:vram",
            "reason": "measured free VRAM below the signed gate",
            "extra": {},
            "doctrine": {"failClosed": True},
        }
        body = canonicalize(receipt).encode("utf-8")
        signed = {
            "receipt": receipt,
            "bodyBase64": base64.b64encode(body).decode(),
            "signatureBase64": base64.b64encode(
                self.laptop.sign(body).signature
            ).decode(),
            "publicKeySpkiBase64": base64.b64encode(self.laptop_spki).decode(),
            "keyId": self.laptop_key_id,
            "scheme": "ed25519-over-exact-bytes-v2",
        }
        report = nemo_v3_status.evaluate(
            root=self.root,
            expected_laptop_key_id=self.laptop_key_id,
            receipt_loader=lambda _spec, _token: (
                f"{self.spec['jobId']}/blocked_receipt.signed.json",
                signed,
            ),
            now=self.now,
        )
        self.assertEqual(report["status"], "TERMINAL_FAILURE")
        self.assertEqual(report["receipt"]["payload_binding"], "JOB_ID_ONLY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
