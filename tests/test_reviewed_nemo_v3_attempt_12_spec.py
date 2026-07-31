from __future__ import annotations

import copy
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
from frontier_contract import ContractError  # noqa: E402
from nemo_v3_contract import (  # noqa: E402
    ATTEMPT_11_CORRECTED_BRIDGE_REVISION,
    ATTEMPT_11_REVIEWED_JOB_ID,
    ATTEMPT_12_CORRECTED_BRIDGE_REVISION,
    ATTEMPT_12_REVIEWED_JOB_ID,
    ATTEMPT_13_REVIEWED_JOB_ID,
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
    EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

ATTEMPT_11_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-11-reviewed.json"
ATTEMPT_12_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-12-reviewed.json"
ATTEMPT_12_QUEUE = ROOT / "queue" / "pending" / f"{ATTEMPT_12_REVIEWED_JOB_ID}.json"
ATTEMPT_12_QUARANTINE = (
    ROOT / "queue" / "quarantine" / f"{ATTEMPT_12_REVIEWED_JOB_ID}.json"
)
ATTEMPT_12_EVIDENCE = ROOT / "queue" / "evidence" / f"{ATTEMPT_12_REVIEWED_JOB_ID}.json"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt12SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_11 = json.loads(ATTEMPT_11_PATH.read_text(encoding="utf-8"))
        cls.attempt_12 = json.loads(ATTEMPT_12_PATH.read_text(encoding="utf-8"))

    def test_exact_source_workflow_runtime_key_and_license(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_12), self.attempt_12)
        self.assertEqual(self.attempt_12["jobId"], ATTEMPT_12_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_12["source"]["revision"],
            EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            self.attempt_12["ownerDispatch"]["workflowBlob"],
            EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_12["authorization"]["settledA11oyRelockRunUrl"],
            EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
        )
        self.assertEqual(
            self.attempt_12["authorization"]["correctedBridgeRevision"],
            ATTEMPT_12_CORRECTED_BRIDGE_REVISION,
        )
        self.assertEqual(
            self.attempt_12["authorization"]["engineKeyId"],
            COORDINATED_ENGINE_KEY_ID,
        )
        self.assertEqual(
            self.attempt_12["authorization"]["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            self.attempt_12["base"]["licenseId"],
            "nvidia-nemotron-open-model-license",
        )
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_12,
                expected_execution_bridge_revision=(
                    ATTEMPT_12_CORRECTED_BRIDGE_REVISION
                ),
            )

    def test_exact_attempt_11_signed_blocked_lineage(self) -> None:
        self.assertEqual(
            self.attempt_12["lineage"],
            {
                "predecessorJobId": ATTEMPT_11_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "7b9af824b529fa80ec51e060cd0fa14f1af8acc8ded5fff5b10f159acb861918"
                ),
                "predecessorPayloadSha256": (
                    "85f08bc171370b25606915008d1b96ff50f670d09e20eb631b4c1ebeb108d994"
                ),
                "predecessorEnvelopeRevision": (
                    "61bb29bdad1e6b76bf3d818428c1d81149a6e72f"
                ),
                "predecessorExecutionBridgeRevision": (
                    ATTEMPT_11_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30620232291"
                ),
                "failurePhase": "POST_CLAIM_TOKENIZER_LOAD",
                "successorGeneration": 12,
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
            },
        )

    def test_science_license_and_upload_boundaries_remain_frozen(self) -> None:
        self.assertEqual(self.attempt_12["base"], self.attempt_11["base"])
        self.assertEqual(self.attempt_12["dataset"], self.attempt_11["dataset"])
        for field in ("recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt_12[field], self.attempt_11[field])
        self.assertFalse(self.attempt_12["outputs"]["publishCandidate"])
        self.assertTrue(self.attempt_12["outputs"]["private"])
        for field in ("candidateUpload", "modelCardUpload", "datasetUpload"):
            self.assertFalse(self.attempt_12["ownerDispatch"][field])

    def test_attempt_11_signed_bytes_are_preserved_and_evidence_is_exact(self) -> None:
        expected = {
            "jobspecs/nemo-v3-20260731-attempt-11-reviewed.json": (
                "a9ec46dcbd9e011c6bddd7513e48a076e1755f06290ec8061caafe4411cad9ca"
            ),
            "queue/pending/job-2026-nemo-v3-governed-attempt-11.json": (
                "7b9af824b529fa80ec51e060cd0fa14f1af8acc8ded5fff5b10f159acb861918"
            ),
        }
        for path, digest in expected.items():
            with self.subTest(path=path):
                content = subprocess.check_output(
                    ["git", "cat-file", "blob", f"HEAD:{path}"],
                    cwd=ROOT,
                )
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)
        evidence_path = (
            ROOT / "queue" / "evidence" / f"{ATTEMPT_11_REVIEWED_JOB_ID}.json"
        )
        self.assertEqual(
            hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "ab8876488cb198718b576c53db427242b85f5152628bae2c0d040ce8f82a4908",
        )

    def test_attempt_12_is_exact_pre_claim_never_dispatch_evidence(self) -> None:
        self.assertTrue(ATTEMPT_12_QUEUE.is_file())
        for path, digest in {
            "jobspecs/nemo-v3-20260731-attempt-12-reviewed.json": (
                "9170e853b7e63fd3069953fda47f31bd10065803280332418da20dc8d72bb8bd"
            ),
            "queue/pending/job-2026-nemo-v3-governed-attempt-12.json": (
                "a1c9f3d909b120d3675efe2cee0ba06b1c92c950f3a9ed4cc4e5b242971ed70f"
            ),
        }.items():
            with self.subTest(path=path):
                content = subprocess.check_output(
                    ["git", "cat-file", "blob", f"HEAD:{path}"],
                    cwd=ROOT,
                )
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)
        self.assertEqual(
            hashlib.sha256(ATTEMPT_12_QUEUE.read_bytes()).hexdigest(),
            "a1c9f3d909b120d3675efe2cee0ba06b1c92c950f3a9ed4cc4e5b242971ed70f",
        )
        payload_sha256 = hashlib.sha256(
            nemo_v3_status.signer_canonicalize(self.attempt_12).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            payload_sha256,
            "a5e04951412bb0c4d085e567e4e869d52bdf6987546b16ffcd6d2bcb72768ce8",
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_12_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 31, 11, 20, tzinfo=timezone.utc),
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
        record = json.loads(ATTEMPT_12_QUARANTINE.read_text(encoding="utf-8"))
        self.assertEqual(
            record["status"],
            [
                "RUNTIME_JOB_BINDING_REJECTED",
                "PRE_CLAIM",
                "NEVER_DISPATCH",
            ],
        )
        self.assertFalse(record["dispatchAuthorized"])
        self.assertTrue(record["preserveEnvelope"])
        self.assertEqual(
            record["replacement"]["reviewedJobId"],
            ATTEMPT_13_REVIEWED_JOB_ID,
        )
        evidence_record = json.loads(ATTEMPT_12_EVIDENCE.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(ATTEMPT_12_EVIDENCE.read_bytes()).hexdigest(),
            "0d5caab31736bf00fd8e6457aa437edc8d2a86466c5f6c8bb31fe67d63274215",
        )
        evidence = evidence_record["executionEvidence"]
        self.assertEqual(evidence["workflowRunId"], "30626533443")
        self.assertEqual(evidence["workflowJobId"], "91142994672")
        self.assertEqual(
            evidence["error"],
            "coordinated authorization requires an exact reviewed job binding",
        )
        for field in (
            "claimCreated",
            "jobDirectoryCreated",
            "prefetchReceiptCreated",
            "trainingStarted",
            "receiptIntentProduced",
            "receiptUploaded",
            "candidateUploaded",
            "adapterUploaded",
            "modelCardUploaded",
            "datasetUploaded",
            "deployed",
            "promoted",
        ):
            self.assertFalse(evidence[field])

    def test_runtime_bound_attempt_13_has_no_commit_self_reference(self) -> None:
        attempt_13 = copy.deepcopy(self.attempt_12)
        attempt_13["jobId"] = ATTEMPT_13_REVIEWED_JOB_ID
        attempt_13["createdAt"] = "2026-07-31T11:25:00Z"
        attempt_13["expiresAt"] = "2026-08-14T11:25:00Z"
        attempt_13["outputs"]["candidateId"] = (
            "SZL-Nemo-v3-Nemotron-4B-Adapter-Attempt-13-RuntimeBindingRecovery"
        )
        attempt_13["authorization"]["decisionAt"] = "2026-07-31T11:25:00Z"
        attempt_13["authorization"]["correctedBridgeRevision"] = "c" * 40
        attempt_13["lineage"] = {
            "predecessorJobId": ATTEMPT_12_REVIEWED_JOB_ID,
            "predecessorEnvelopeSha256": (
                "a1c9f3d909b120d3675efe2cee0ba06b1c92c950f3a9ed4cc4e5b242971ed70f"
            ),
            "predecessorPayloadSha256": (
                "a5e04951412bb0c4d085e567e4e869d52bdf6987546b16ffcd6d2bcb72768ce8"
            ),
            "predecessorEnvelopeRevision": ("6b21684b64bf01971f3c3aac71493bba8078e532"),
            "predecessorExecutionBridgeRevision": (
                ATTEMPT_12_CORRECTED_BRIDGE_REVISION
            ),
            "transportEvidenceUrl": (
                "https://github.com/szl-holdings/a11oy/actions/runs/30626533443"
            ),
            "failurePhase": "PRE_CLAIM_AUTHENTICATED_PREFETCH_RUNTIME_BINDING",
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
        }
        self.assertIs(validate_nemo_v3_spec(attempt_13), attempt_13)
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                attempt_13,
                expected_execution_bridge_revision="c" * 40,
            )

        skipped = copy.deepcopy(attempt_13)
        skipped["lineage"]["predecessorJobId"] = ATTEMPT_11_REVIEWED_JOB_ID
        with self.assertRaisesRegex(ContractError, "lineage"):
            validate_nemo_v3_spec(skipped)

        unknown = copy.deepcopy(attempt_13)
        unknown["jobId"] = "job-2026-nemo-v3-governed-attempt-16"
        with self.assertRaisesRegex(ContractError, "exact reviewed job binding"):
            validate_nemo_v3_spec(unknown)

    def test_exact_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            (("source", "revision"), "0" * 40),
            (("base", "licenseId"), "nvidia-open-model-license"),
            (("ownerDispatch", "workflowBlob"), "1" * 40),
            (("authorization", "correctedBridgeRevision"), "2" * 40),
            (("lineage", "predecessorEnvelopeSha256"), "3" * 64),
            (("lineage", "predecessorJobId"), ATTEMPT_11_REVIEWED_JOB_ID + "-old"),
            (("lineage", "claimCreated"), False),
            (("lineage", "modelRepositoryCodeImported"), False),
            (("lineage", "holdoutsAccessed"), False),
            (("lineage", "receiptIntentProduced"), False),
            (("lineage", "terminalLedgerWritten"), False),
            (("lineage", "successorGeneration"), 11),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.attempt_12)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_12,
                expected_execution_bridge_revision="4" * 40,
            )

    def test_signer_and_status_track_plaintext_attempt_12(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260731-attempt-12-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-12.json",
            "queue/quarantine/job-2026-nemo-v3-governed-attempt-12.json",
            "queue/evidence/job-2026-nemo-v3-governed-attempt-12.json",
            "tests/test_reviewed_nemo_v3_attempt_12_spec.py",
            "attempt_id: attempt-12-tokenizer-load-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIn(
            "signer is locked to ${ATTEMPT_16_REVIEWED_JOB_ID}",
            SIGNER_SOURCE,
        )
        self.assertIn("'job-2026-nemo-v3-governed-attempt-11',", SIGNER_SOURCE)
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
