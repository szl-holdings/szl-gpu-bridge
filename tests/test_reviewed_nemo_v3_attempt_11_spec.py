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
    ATTEMPT_10_CORRECTED_BRIDGE_REVISION,
    ATTEMPT_10_REVIEWED_JOB_ID,
    ATTEMPT_11_CORRECTED_BRIDGE_REVISION,
    ATTEMPT_11_REVIEWED_JOB_ID,
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
    EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    canonical_json,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

ATTEMPT_10_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-10-reviewed.json"
ATTEMPT_11_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-11-reviewed.json"
ATTEMPT_10_QUEUE = ROOT / "queue" / "pending" / f"{ATTEMPT_10_REVIEWED_JOB_ID}.json"
ATTEMPT_11_QUEUE = ROOT / "queue" / "pending" / f"{ATTEMPT_11_REVIEWED_JOB_ID}.json"
ATTEMPT_11_EVIDENCE = ROOT / "queue" / "evidence" / f"{ATTEMPT_11_REVIEWED_JOB_ID}.json"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt11SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_10 = json.loads(ATTEMPT_10_PATH.read_text(encoding="utf-8"))
        cls.attempt_11 = json.loads(ATTEMPT_11_PATH.read_text(encoding="utf-8"))

    def test_exact_source_workflow_runtime_key_and_license(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_11), self.attempt_11)
        self.assertEqual(self.attempt_11["jobId"], ATTEMPT_11_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_11["source"]["revision"],
            EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            self.attempt_11["ownerDispatch"]["workflowBlob"],
            EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_11["authorization"]["settledA11oyRelockRunUrl"],
            EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
        )
        self.assertEqual(
            self.attempt_11["authorization"]["correctedBridgeRevision"],
            ATTEMPT_11_CORRECTED_BRIDGE_REVISION,
        )
        self.assertEqual(
            self.attempt_11["authorization"]["engineKeyId"],
            COORDINATED_ENGINE_KEY_ID,
        )
        self.assertEqual(
            self.attempt_11["authorization"]["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            self.attempt_11["base"]["licenseId"],
            "nvidia-nemotron-open-model-license",
        )
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_11,
                expected_execution_bridge_revision=ATTEMPT_11_CORRECTED_BRIDGE_REVISION,
            )

    def test_exact_attempt_10_zero_effect_lineage(self) -> None:
        self.assertEqual(
            self.attempt_11["lineage"],
            {
                "predecessorJobId": ATTEMPT_10_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "b354d34dcc6487e311b2d40413de4920ef8646d3f40e9d7442d366152aac901b"
                ),
                "predecessorPayloadSha256": (
                    "2287b1be69239ec0f577ee6e712e0093345e46640485dc6fefa88e8104d727c9"
                ),
                "predecessorEnvelopeRevision": (
                    "5c0aa8e9949b1cf2593acc269eb3fefffeaa36e1"
                ),
                "predecessorExecutionBridgeRevision": (
                    ATTEMPT_10_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30612658302"
                ),
                "failurePhase": ("PRE_CLAIM_IMMUTABLE_RUNTIME_JOB_BINDING_VALIDATION"),
                "successorGeneration": 11,
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

    def test_science_and_upload_boundaries_remain_frozen(self) -> None:
        self.assertEqual(self.attempt_11["base"], self.attempt_10["base"])
        self.assertEqual(
            self.attempt_11["dataset"]["rightsBasis"],
            self.attempt_10["dataset"]["rightsBasis"],
        )
        for field in ("train", "holdouts", "preregistration"):
            self.assertEqual(
                self.attempt_11["dataset"][field],
                self.attempt_10["dataset"][field],
            )
        for field in ("recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt_11[field], self.attempt_10[field])
        self.assertFalse(self.attempt_11["outputs"]["publishCandidate"])
        self.assertTrue(self.attempt_11["outputs"]["private"])
        for field in ("candidateUpload", "modelCardUpload", "datasetUpload"):
            self.assertFalse(self.attempt_11["ownerDispatch"][field])

    def test_attempt_10_spec_and_envelope_bytes_are_preserved(self) -> None:
        spec_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:jobspecs/nemo-v3-20260731-attempt-10-reviewed.json",
            ],
            cwd=ROOT,
        )
        envelope_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:queue/pending/job-2026-nemo-v3-governed-attempt-10.json",
            ],
            cwd=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(spec_bytes).hexdigest(),
            "6cd898fd1094eb63e7c993a3efe1346e870f698ec9b7cc5706f90002902fe84a",
        )
        self.assertEqual(
            hashlib.sha256(envelope_bytes).hexdigest(),
            "b354d34dcc6487e311b2d40413de4920ef8646d3f40e9d7442d366152aac901b",
        )

    def test_signed_blocked_receipt_is_quarantined_and_never_retried(self) -> None:
        self.assertTrue(ATTEMPT_11_QUEUE.is_file())
        self.assertEqual(
            hashlib.sha256(ATTEMPT_11_QUEUE.read_bytes()).hexdigest(),
            "7b9af824b529fa80ec51e060cd0fa14f1af8acc8ded5fff5b10f159acb861918",
        )
        self.assertEqual(
            hashlib.sha256(canonical_json(self.attempt_11).encode("utf-8")).hexdigest(),
            "ffe92c2833d37f3c5d58805c397c2ed11010293da1354102df825d8d94eab98a",
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_11_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 31, 8, 6, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUARANTINED_NEVER_DISPATCH")
        self.assertTrue(report["terminal"])
        self.assertTrue(report["queue"]["present"])
        self.assertTrue(report["queue"]["valid"], report["queue"]["error"])
        self.assertEqual(
            report["queue"]["payload_sha256"],
            "85f08bc171370b25606915008d1b96ff50f670d09e20eb631b4c1ebeb108d994",
        )
        self.assertEqual(report["queue"]["engine_key_id"], COORDINATED_ENGINE_KEY_ID)
        self.assertFalse(report["receipt"]["present"])
        self.assertFalse(report["reviewed_spec"]["candidate_publication_enabled"])
        quarantine = json.loads(
            (
                ROOT / "queue" / "quarantine" / f"{ATTEMPT_11_REVIEWED_JOB_ID}.json"
            ).read_text(encoding="utf-8")
        )
        self.assertNotIn("executionEvidence", quarantine)
        evidence_record = json.loads(ATTEMPT_11_EVIDENCE.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(ATTEMPT_11_EVIDENCE.read_bytes()).hexdigest(),
            "ab8876488cb198718b576c53db427242b85f5152628bae2c0d040ce8f82a4908",
        )
        self.assertEqual(
            set(evidence_record),
            {"kind", "v", "jobId", "executionEvidence"},
        )
        self.assertEqual(evidence_record["kind"], "szl-nemo-v3-execution-evidence")
        self.assertEqual(evidence_record["v"], 1)
        self.assertEqual(evidence_record["jobId"], ATTEMPT_11_REVIEWED_JOB_ID)
        evidence = evidence_record["executionEvidence"]
        self.assertEqual(evidence["workflowRunId"], "30620232291")
        self.assertEqual(
            evidence["runtimeClaimSha256"],
            "f73c18a970d5b99ea8f567ff682eb9c8b7e1ba9f1e769b8c3f6ce4ad93765cc2",
        )
        self.assertEqual(
            evidence["attemptClaimSha256"],
            "3b0caf335622a1034d5e5ce31dd81d4b66819f520805c3cfe1f10c634a7d1f80",
        )
        self.assertEqual(
            evidence["receiptRevision"],
            "1a74ad3f5fc2682e6bbdd034a68399dee7e79525",
        )
        self.assertEqual(
            evidence["receiptFileSha256"],
            "f6f1c5af7c8a47c4c4a4ce35ccb9d2859cf3177c06c439bd529c901308aeb9e3",
        )
        self.assertEqual(evidence["receiptVerdict"], "BLOCKED")
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
        policy = copy.deepcopy(nemo_v3_status.quarantine_policy(self.attempt_11))
        assert policy is not None
        policy["execution_evidence_sha256"] = "0" * 64
        with mock.patch.object(
            nemo_v3_status,
            "quarantine_policy",
            return_value=policy,
        ):
            report = nemo_v3_status.evaluate(
                root=ROOT,
                spec_path=ATTEMPT_11_PATH,
                receipt_loader=lambda _spec, _token: None,
                now=datetime(2026, 7, 31, 8, 6, tzinfo=timezone.utc),
            )
        self.assertEqual(report["status"], "INVALID_QUARANTINE_RECORD")
        self.assertFalse(report["terminal"])
        self.assertTrue(report["quarantine"]["present"])
        self.assertFalse(report["quarantine"]["valid"])

    def test_exact_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            (("source", "revision"), "0" * 40),
            (("base", "licenseId"), "nvidia-open-model-license"),
            (("ownerDispatch", "workflowBlob"), "1" * 40),
            (("authorization", "correctedBridgeRevision"), "2" * 40),
            (("lineage", "predecessorEnvelopeSha256"), "3" * 64),
            (("lineage", "predecessorJobId"), ATTEMPT_10_REVIEWED_JOB_ID + "-old"),
            (("lineage", "eventCreated"), False),
            (("lineage", "claimCreated"), True),
            (("lineage", "holdoutsAccessed"), True),
            (("lineage", "receiptIntentProduced"), True),
            (("lineage", "successorGeneration"), 10),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.attempt_11)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_11,
                expected_execution_bridge_revision="4" * 40,
            )

    def test_signer_and_status_track_plaintext_attempt_11(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260731-attempt-11-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-11.json",
            "queue/evidence/job-2026-nemo-v3-governed-attempt-11.json",
            "tests/test_reviewed_nemo_v3_attempt_11_spec.py",
            "attempt_id: attempt-11-runtime-admission-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIn(
            "signer is locked to ${ATTEMPT_16_REVIEWED_JOB_ID}",
            SIGNER_SOURCE,
        )
        self.assertIn("'job-2026-nemo-v3-governed-attempt-10',", SIGNER_SOURCE)
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
