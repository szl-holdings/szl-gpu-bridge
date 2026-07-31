from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "laptop"))

import nemo_v3_status  # noqa: E402
from frontier_contract import ContractError  # noqa: E402
from nemo_v3_contract import (  # noqa: E402
    ATTEMPT_12_CORRECTED_BRIDGE_REVISION,
    ATTEMPT_12_REVIEWED_JOB_ID,
    ATTEMPT_13_REVIEWED_JOB_ID,
    ATTEMPT_14_REVIEWED_JOB_ID,
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
    EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

ATTEMPT_13_CORRECTED_BRIDGE_REVISION = "2783b3518abcec9f38d3f6504c06e305a4723801"
ATTEMPT_12_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-12-reviewed.json"
ATTEMPT_13_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-13-reviewed.json"
ATTEMPT_13_QUEUE = ROOT / "queue" / "pending" / f"{ATTEMPT_13_REVIEWED_JOB_ID}.json"
ATTEMPT_13_QUARANTINE = (
    ROOT / "queue" / "quarantine" / f"{ATTEMPT_13_REVIEWED_JOB_ID}.json"
)
ATTEMPT_13_EVIDENCE = ROOT / "queue" / "evidence" / f"{ATTEMPT_13_REVIEWED_JOB_ID}.json"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt13SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_12 = json.loads(ATTEMPT_12_PATH.read_text(encoding="utf-8"))
        cls.attempt_13 = json.loads(ATTEMPT_13_PATH.read_text(encoding="utf-8"))

    def test_exact_source_workflow_runtime_key_and_license(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_13), self.attempt_13)
        self.assertEqual(self.attempt_13["jobId"], ATTEMPT_13_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_13["source"]["revision"],
            EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            self.attempt_13["ownerDispatch"]["workflowBlob"],
            EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_13["authorization"]["settledA11oyRelockRunUrl"],
            EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
        )
        self.assertEqual(
            self.attempt_13["authorization"]["correctedBridgeRevision"],
            ATTEMPT_13_CORRECTED_BRIDGE_REVISION,
        )
        self.assertEqual(
            self.attempt_13["authorization"]["engineKeyId"],
            COORDINATED_ENGINE_KEY_ID,
        )
        self.assertEqual(
            self.attempt_13["authorization"]["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            self.attempt_13["base"]["licenseId"],
            "nvidia-nemotron-open-model-license",
        )
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_13,
                expected_execution_bridge_revision=ATTEMPT_13_CORRECTED_BRIDGE_REVISION,
            )

    def test_exact_attempt_12_zero_effect_lineage(self) -> None:
        self.assertEqual(
            self.attempt_13["lineage"],
            {
                "predecessorJobId": ATTEMPT_12_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "a1c9f3d909b120d3675efe2cee0ba06b1c92c950f3a9ed4cc4e5b242971ed70f"
                ),
                "predecessorPayloadSha256": (
                    "a5e04951412bb0c4d085e567e4e869d52bdf6987546b16ffcd6d2bcb72768ce8"
                ),
                "predecessorEnvelopeRevision": (
                    "6b21684b64bf01971f3c3aac71493bba8078e532"
                ),
                "predecessorExecutionBridgeRevision": (
                    ATTEMPT_12_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30626533443"
                ),
                "failurePhase": ("PRE_CLAIM_AUTHENTICATED_PREFETCH_RUNTIME_BINDING"),
                "successorGeneration": 13,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": False,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": False,
                "candidateProduced": False,
                "receiptIntentProduced": False,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            },
        )

    def test_science_license_and_upload_boundaries_remain_frozen(self) -> None:
        self.assertEqual(self.attempt_13["base"], self.attempt_12["base"])
        self.assertEqual(self.attempt_13["dataset"], self.attempt_12["dataset"])
        for field in ("recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt_13[field], self.attempt_12[field])
        self.assertFalse(self.attempt_13["outputs"]["publishCandidate"])
        self.assertTrue(self.attempt_13["outputs"]["private"])
        for field in ("candidateUpload", "modelCardUpload", "datasetUpload"):
            self.assertFalse(self.attempt_13["ownerDispatch"][field])

    def test_attempt_12_signed_and_quarantine_bytes_are_preserved(self) -> None:
        expected = {
            "jobspecs/nemo-v3-20260731-attempt-12-reviewed.json": (
                "9170e853b7e63fd3069953fda47f31bd10065803280332418da20dc8d72bb8bd"
            ),
            "queue/pending/job-2026-nemo-v3-governed-attempt-12.json": (
                "a1c9f3d909b120d3675efe2cee0ba06b1c92c950f3a9ed4cc4e5b242971ed70f"
            ),
            "queue/quarantine/job-2026-nemo-v3-governed-attempt-12.json": (
                "6201a3c64b6bc41748ed6b00f61db0646aeb40cb62cf79c8bb4f3e49fa11a288"
            ),
            "queue/evidence/job-2026-nemo-v3-governed-attempt-12.json": (
                "0d5caab31736bf00fd8e6457aa437edc8d2a86466c5f6c8bb31fe67d63274215"
            ),
        }
        for path, digest in expected.items():
            with self.subTest(path=path):
                content = subprocess.check_output(
                    ["git", "cat-file", "blob", f"HEAD:{path}"],
                    cwd=ROOT,
                )
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)

    def test_attempt_13_spec_and_envelope_bytes_are_immutable(self) -> None:
        expected = {
            "jobspecs/nemo-v3-20260731-attempt-13-reviewed.json": (
                "bd394cbb68f60ac181333156cb53d9c0074b234352843aa976533021f5f396e5"
            ),
            "queue/pending/job-2026-nemo-v3-governed-attempt-13.json": (
                "de31cbb574cdeeaaf611a25fe1e40616b7fe8d4f6e2e138b66697474f5d800b0"
            ),
        }
        for path, digest in expected.items():
            with self.subTest(path=path):
                content = subprocess.check_output(
                    ["git", "cat-file", "blob", f"HEAD:{path}"],
                    cwd=ROOT,
                )
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)

    def test_signed_queue_is_quarantined_with_exact_blocked_receipt_truth(self) -> None:
        self.assertTrue(ATTEMPT_13_QUEUE.is_file())
        self.assertEqual(
            hashlib.sha256(ATTEMPT_13_QUEUE.read_bytes()).hexdigest(),
            "de31cbb574cdeeaaf611a25fe1e40616b7fe8d4f6e2e138b66697474f5d800b0",
        )
        payload_sha256 = hashlib.sha256(
            nemo_v3_status.signer_canonicalize(self.attempt_13).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            payload_sha256,
            "82f619eb1fff6a7617b5761358d2f5c1d8ca62a306eb7cb1bf2570e096b2b9fc",
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_13_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 31, 11, 40, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUARANTINED_NEVER_DISPATCH")
        self.assertTrue(report["terminal"])
        self.assertTrue(report["queue"]["present"])
        self.assertTrue(report["queue"]["valid"], report["queue"]["error"])
        self.assertEqual(report["queue"]["payload_sha256"], payload_sha256)
        self.assertEqual(report["queue"]["engine_key_id"], COORDINATED_ENGINE_KEY_ID)
        self.assertFalse(report["receipt"]["present"])
        self.assertFalse(report["reviewed_spec"]["candidate_publication_enabled"])
        self.assertTrue(report["quarantine"]["present"])
        self.assertTrue(report["quarantine"]["valid"], report["quarantine"]["error"])

        quarantine = json.loads(ATTEMPT_13_QUARANTINE.read_text(encoding="utf-8"))
        self.assertEqual(
            quarantine["status"],
            [
                "SFTCONFIG_STRATEGY_KEY_BLOCKED",
                "POST_CLAIM",
                "PRE_TRAINING",
                "SIGNED_BLOCKED_RECEIPT",
                "NEVER_DISPATCH",
            ],
        )
        self.assertTrue(quarantine["preserveEnvelope"])
        self.assertFalse(quarantine["dispatchAuthorized"])
        self.assertEqual(
            quarantine["replacement"]["reviewedJobId"],
            ATTEMPT_14_REVIEWED_JOB_ID,
        )

        evidence_record = json.loads(ATTEMPT_13_EVIDENCE.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(ATTEMPT_13_EVIDENCE.read_bytes()).hexdigest(),
            "8d9c7e4b37138a2b61de9e15f3c622dc1291f2c38f909a7a9f48115385831c4a",
        )
        evidence = evidence_record["executionEvidence"]
        self.assertEqual(evidence["workflowRunId"], "30629929196")
        self.assertEqual(evidence["workflowJobId"], "91153664576")
        self.assertEqual(
            evidence["attemptClaimSha256"],
            "bb1fd12fb73289864503d5f8d65aacb4b34d0db0d0ba2fcce73a975c71364293",
        )
        self.assertEqual(
            evidence["receiptRevision"],
            "ac219fe87da9acf57141ff24ffbd330216584f7c",
        )
        self.assertEqual(
            evidence["receiptFileSha256"],
            "384e64b0ebd43fcfd2f52a3b1139cf1bca04f23c43ccfd9738af3a1fdfe46d02",
        )
        self.assertEqual(
            evidence["receiptBodySha256"],
            "ec5f8b173f3e8f13c252bf9c7eb52625210b3bf936c7dec88fc640e032275876",
        )
        self.assertEqual(evidence["receiptKeyId"], "167c14fbddbe97cc")
        self.assertEqual(evidence["receiptVerdict"], "BLOCKED")
        self.assertTrue(evidence["claimCreated"])
        self.assertTrue(evidence["receiptIntentProduced"])
        self.assertTrue(evidence["receiptUploaded"])
        for field in (
            "trainingStarted",
            "candidateUploaded",
            "adapterUploaded",
            "modelCardUploaded",
            "datasetUploaded",
            "deployed",
            "promoted",
        ):
            self.assertFalse(evidence[field])

    def test_standalone_execution_evidence_hash_fails_closed(self) -> None:
        policy = copy.deepcopy(nemo_v3_status.quarantine_policy(self.attempt_13))
        assert policy is not None
        policy["execution_evidence_sha256"] = "0" * 64
        with mock.patch.object(
            nemo_v3_status,
            "quarantine_policy",
            return_value=policy,
        ):
            report = nemo_v3_status.evaluate(
                root=ROOT,
                spec_path=ATTEMPT_13_PATH,
                receipt_loader=lambda _spec, _token: None,
                now=datetime(2026, 7, 31, 11, 40, tzinfo=timezone.utc),
            )
        self.assertEqual(report["status"], "INVALID_QUARANTINE_RECORD")
        self.assertFalse(report["terminal"])
        self.assertTrue(report["quarantine"]["present"])
        self.assertFalse(report["quarantine"]["valid"])

    def test_runtime_bound_attempt_14_has_exact_attempt_13_lineage(self) -> None:
        attempt_14 = copy.deepcopy(self.attempt_13)
        attempt_14["jobId"] = ATTEMPT_14_REVIEWED_JOB_ID
        attempt_14["createdAt"] = "2026-07-31T12:40:00Z"
        attempt_14["expiresAt"] = "2026-08-14T12:40:00Z"
        attempt_14["outputs"]["candidateId"] = (
            "SZL-Nemo-v3-Nemotron-4B-Adapter-Attempt-14-SFTConfigRecovery"
        )
        attempt_14["authorization"]["decisionAt"] = "2026-07-31T12:40:00Z"
        attempt_14["authorization"]["correctedBridgeRevision"] = "c" * 40
        attempt_14["lineage"] = {
            "predecessorJobId": ATTEMPT_13_REVIEWED_JOB_ID,
            "predecessorEnvelopeSha256": (
                "de31cbb574cdeeaaf611a25fe1e40616b7fe8d4f6e2e138b66697474f5d800b0"
            ),
            "predecessorPayloadSha256": (
                "82f619eb1fff6a7617b5761358d2f5c1d8ca62a306eb7cb1bf2570e096b2b9fc"
            ),
            "predecessorEnvelopeRevision": ("b929bae4230ffe39ee63b34b8e9f9974cffc66ca"),
            "predecessorExecutionBridgeRevision": ATTEMPT_13_CORRECTED_BRIDGE_REVISION,
            "transportEvidenceUrl": (
                "https://github.com/szl-holdings/a11oy/actions/runs/30629929196"
            ),
            "failurePhase": "POST_CLAIM_SFTCONFIG_STRATEGY_COMPATIBILITY",
            "successorGeneration": 14,
            "automaticRetry": False,
            "eventCreated": True,
            "workflowRunCreated": True,
            "claimCreated": True,
            "trainingStarted": False,
            "modelRepositoryCodeImported": True,
            "holdoutsAccessed": True,
            "candidateProduced": False,
            "receiptIntentProduced": True,
            "terminalLedgerWritten": True,
            "scienceInputsReused": True,
        }
        self.assertIs(validate_nemo_v3_spec(attempt_14), attempt_14)
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                attempt_14,
                expected_execution_bridge_revision="c" * 40,
            )

        for field, value in (
            ("predecessorJobId", ATTEMPT_12_REVIEWED_JOB_ID),
            ("predecessorEnvelopeSha256", "0" * 64),
            ("failurePhase", "POST_CLAIM_UNKNOWN"),
            ("claimCreated", False),
            ("trainingStarted", True),
            ("receiptIntentProduced", False),
            ("successorGeneration", 13),
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(attempt_14)
                mutated["lineage"][field] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

    def test_exact_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            (("source", "revision"), "0" * 40),
            (("base", "licenseId"), "nvidia-open-model-license"),
            (("ownerDispatch", "workflowBlob"), "1" * 40),
            (("authorization", "correctedBridgeRevision"), "2" * 39),
            (("lineage", "predecessorEnvelopeSha256"), "3" * 64),
            (("lineage", "predecessorJobId"), ATTEMPT_12_REVIEWED_JOB_ID + "-old"),
            (("lineage", "claimCreated"), True),
            (("lineage", "eventCreated"), False),
            (("lineage", "successorGeneration"), 12),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.attempt_13)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_13,
                expected_execution_bridge_revision="4" * 40,
            )

        replay = copy.deepcopy(self.attempt_12)
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                replay,
                expected_execution_bridge_revision=ATTEMPT_12_CORRECTED_BRIDGE_REVISION,
            )

    def test_signer_and_status_track_plaintext_attempt_13(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260731-attempt-13-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-13.json",
            "queue/quarantine/job-2026-nemo-v3-governed-attempt-13.json",
            "queue/evidence/job-2026-nemo-v3-governed-attempt-13.json",
            "tests/test_reviewed_nemo_v3_attempt_13_spec.py",
            "tests/test_nemo_v3_sftconfig_compat.py",
            "attempt_id: attempt-13-runtime-binding-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIn(
            "signer is locked to ${ATTEMPT_16_REVIEWED_JOB_ID}",
            SIGNER_SOURCE,
        )
        self.assertIn("'job-2026-nemo-v3-governed-attempt-12',", SIGNER_SOURCE)
        self.assertIn("'job-2026-nemo-v3-governed-attempt-13',", SIGNER_SOURCE)
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
