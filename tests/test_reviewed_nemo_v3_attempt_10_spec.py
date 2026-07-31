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
    ATTEMPT_10_CORRECTED_BRIDGE_REVISION,
    ATTEMPT_10_REVIEWED_JOB_ID,
    ATTEMPT_11_CORRECTED_BRIDGE_REVISION,
    ATTEMPT_11_REVIEWED_JOB_ID,
    ATTEMPT_9_CORRECTED_BRIDGE_REVISION,
    ATTEMPT_9_REVIEWED_JOB_ID,
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
    EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    RECOVERY_A11OY_RELOCK_RUN_URL,
    RECOVERY_A11OY_SOURCE_REVISION,
    RECOVERY_OWNER_WORKFLOW_BLOB,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

ATTEMPT_9_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-9-reviewed.json"
ATTEMPT_10_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-10-reviewed.json"
ATTEMPT_9_QUEUE = ROOT / "queue" / "pending" / f"{ATTEMPT_9_REVIEWED_JOB_ID}.json"
ATTEMPT_10_QUEUE = ROOT / "queue" / "pending" / f"{ATTEMPT_10_REVIEWED_JOB_ID}.json"
ATTEMPT_10_QUARANTINE = (
    ROOT / "queue" / "quarantine" / f"{ATTEMPT_10_REVIEWED_JOB_ID}.json"
)
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt10SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_9 = json.loads(ATTEMPT_9_PATH.read_text(encoding="utf-8"))
        cls.attempt_10 = json.loads(ATTEMPT_10_PATH.read_text(encoding="utf-8"))

    def test_exact_source_workflow_runtime_key_and_license(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_10), self.attempt_10)
        self.assertEqual(self.attempt_10["jobId"], ATTEMPT_10_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_10["source"]["revision"], RECOVERY_A11OY_SOURCE_REVISION
        )
        self.assertEqual(
            self.attempt_10["ownerDispatch"]["workflowBlob"],
            RECOVERY_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_10["authorization"]["settledA11oyRelockRunUrl"],
            RECOVERY_A11OY_RELOCK_RUN_URL,
        )
        self.assertEqual(
            self.attempt_10["authorization"]["correctedBridgeRevision"],
            ATTEMPT_10_CORRECTED_BRIDGE_REVISION,
        )
        self.assertEqual(
            self.attempt_10["authorization"]["engineKeyId"],
            COORDINATED_ENGINE_KEY_ID,
        )
        self.assertEqual(
            self.attempt_10["authorization"]["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            self.attempt_10["base"]["licenseId"],
            "nvidia-nemotron-open-model-license",
        )

    def test_exact_attempt_9_post_claim_lineage(self) -> None:
        self.assertEqual(
            self.attempt_10["lineage"],
            {
                "predecessorJobId": ATTEMPT_9_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "a7b67f1245137b3422d6e2ce5cf379aa9adb193e1f1d9db0dec8abf92bf5fa49"
                ),
                "predecessorPayloadSha256": (
                    "f8ec93b0a2967e548ba2222cbf8a69abbe89987c98e695688c39c0e0d3827c5b"
                ),
                "predecessorEnvelopeRevision": (
                    "4fa21a298e9b8f8dd6827f6dd0406ba6de02421e"
                ),
                "predecessorExecutionBridgeRevision": (
                    ATTEMPT_9_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30609977388"
                ),
                "failurePhase": "POST_CLAIM_CACHE_LICENSE_AND_FINALIZER_BINDING",
                "successorGeneration": 10,
                "automaticRetry": False,
                "eventCreated": True,
                "workflowRunCreated": True,
                "claimCreated": True,
                "trainingStarted": False,
                "modelRepositoryCodeImported": False,
                "holdoutsAccessed": True,
                "candidateProduced": False,
                "receiptIntentProduced": True,
                "terminalLedgerWritten": False,
                "scienceInputsReused": True,
            },
        )

    def test_science_and_upload_boundaries_remain_frozen(self) -> None:
        for field in ("dataset", "recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt_10[field], self.attempt_9[field])
        old_base = copy.deepcopy(self.attempt_9["base"])
        new_base = copy.deepcopy(self.attempt_10["base"])
        self.assertEqual(old_base.pop("licenseId"), "nvidia-open-model-license")
        self.assertEqual(
            new_base.pop("licenseId"), "nvidia-nemotron-open-model-license"
        )
        self.assertEqual(new_base, old_base)
        self.assertFalse(self.attempt_10["outputs"]["publishCandidate"])
        self.assertTrue(self.attempt_10["outputs"]["private"])
        for field in ("candidateUpload", "modelCardUpload", "datasetUpload"):
            self.assertFalse(self.attempt_10["ownerDispatch"][field])

    def test_attempt_9_spec_and_envelope_bytes_are_preserved(self) -> None:
        spec_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:jobspecs/nemo-v3-20260731-attempt-9-reviewed.json",
            ],
            cwd=ROOT,
        )
        envelope_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:queue/pending/job-2026-nemo-v3-governed-attempt-9.json",
            ],
            cwd=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(spec_bytes).hexdigest(),
            "462444ac636da559b610fc0fb8cde5d802a13c926364f669967b0a05f978dc35",
        )
        self.assertEqual(
            hashlib.sha256(envelope_bytes).hexdigest(),
            "a7b67f1245137b3422d6e2ce5cf379aa9adb193e1f1d9db0dec8abf92bf5fa49",
        )

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

    def test_signed_status_is_terminal_quarantine_without_publication(self) -> None:
        self.assertTrue(ATTEMPT_10_QUEUE.is_file())
        self.assertEqual(
            hashlib.sha256(ATTEMPT_10_QUEUE.read_bytes()).hexdigest(),
            "b354d34dcc6487e311b2d40413de4920ef8646d3f40e9d7442d366152aac901b",
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_10_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 31, 7, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUARANTINED_NEVER_DISPATCH")
        self.assertTrue(report["terminal"])
        self.assertTrue(report["queue"]["present"])
        self.assertTrue(report["queue"]["valid"], report["queue"]["error"])
        self.assertEqual(
            report["queue"]["payload_sha256"],
            "2287b1be69239ec0f577ee6e712e0093345e46640485dc6fefa88e8104d727c9",
        )
        self.assertEqual(report["queue"]["engine_key_id"], COORDINATED_ENGINE_KEY_ID)
        self.assertTrue(report["quarantine"]["present"])
        self.assertTrue(report["quarantine"]["valid"], report["quarantine"]["error"])
        self.assertFalse(report["receipt"]["present"])

    def test_attempt_10_quarantine_binds_zero_effect_pre_claim_failure(self) -> None:
        record = json.loads(ATTEMPT_10_QUARANTINE.read_text(encoding="utf-8"))
        self.assertEqual(
            record["status"],
            [
                "IMMUTABLE_RUNTIME_JOB_BINDING_REJECTED",
                "PRE_CLAIM",
                "NEVER_DISPATCH",
            ],
        )
        self.assertEqual(
            record["queueFileSha256"],
            "b354d34dcc6487e311b2d40413de4920ef8646d3f40e9d7442d366152aac901b",
        )
        self.assertEqual(
            record["signedPayloadSha256"],
            "2287b1be69239ec0f577ee6e712e0093345e46640485dc6fefa88e8104d727c9",
        )
        self.assertTrue(record["preserveEnvelope"])
        self.assertFalse(record["dispatchAuthorized"])
        self.assertEqual(
            record["replacement"],
            {
                "sourceRevision": EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
                "workflowBlob": EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
                "engineKeyId": COORDINATED_ENGINE_KEY_ID,
                "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
                "reviewedJobId": ATTEMPT_11_REVIEWED_JOB_ID,
            },
        )
        for prohibited in (
            "No claim",
            "prefetch receipt",
            "signed receipt",
            "candidate",
            "Hugging Face artifact publication",
        ):
            self.assertIn(prohibited, record["reason"])

    def test_runtime_bound_attempt_11_requires_explicit_execution_revision(
        self,
    ) -> None:
        attempt_11 = copy.deepcopy(self.attempt_10)
        attempt_11["jobId"] = ATTEMPT_11_REVIEWED_JOB_ID
        attempt_11["source"]["revision"] = EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION
        attempt_11["ownerDispatch"]["workflowBlob"] = (
            EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB
        )
        attempt_11["authorization"]["settledA11oyRelockRunUrl"] = (
            EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL
        )
        attempt_11["authorization"]["correctedBridgeRevision"] = (
            ATTEMPT_11_CORRECTED_BRIDGE_REVISION
        )
        attempt_11["lineage"] = {
            "predecessorJobId": ATTEMPT_10_REVIEWED_JOB_ID,
            "predecessorEnvelopeSha256": (
                "b354d34dcc6487e311b2d40413de4920ef8646d3f40e9d7442d366152aac901b"
            ),
            "predecessorPayloadSha256": (
                "2287b1be69239ec0f577ee6e712e0093345e46640485dc6fefa88e8104d727c9"
            ),
            "predecessorEnvelopeRevision": ("5c0aa8e9949b1cf2593acc269eb3fefffeaa36e1"),
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
        }
        self.assertIs(validate_nemo_v3_spec(attempt_11), attempt_11)
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                attempt_11,
                expected_execution_bridge_revision=ATTEMPT_11_CORRECTED_BRIDGE_REVISION,
            )

    def test_exact_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            (("source", "revision"), "0" * 40),
            (("base", "licenseId"), "nvidia-open-model-license"),
            (("ownerDispatch", "workflowBlob"), "1" * 40),
            (("authorization", "correctedBridgeRevision"), "2" * 40),
            (("lineage", "predecessorEnvelopeSha256"), "3" * 64),
            (("lineage", "predecessorJobId"), ATTEMPT_9_REVIEWED_JOB_ID + "-old"),
            (("lineage", "claimCreated"), False),
            (("lineage", "holdoutsAccessed"), False),
            (("lineage", "receiptIntentProduced"), False),
            (("lineage", "trainingStarted"), True),
            (("lineage", "successorGeneration"), 9),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.attempt_10)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

    def test_status_preserves_attempt_10_while_signer_advances(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260731-attempt-10-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-10.json",
            "queue/quarantine/job-2026-nemo-v3-governed-attempt-10.json",
            "tests/test_reviewed_nemo_v3_attempt_10_spec.py",
            "attempt_id: attempt-10-cache-license-finalizer-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIn(
            "signer is locked to ${ATTEMPT_14_REVIEWED_JOB_ID}", SIGNER_SOURCE
        )
        self.assertIn("'job-2026-nemo-v3-governed-attempt-9',", SIGNER_SOURCE)
        self.assertIn("'job-2026-nemo-v3-governed-attempt-10',", SIGNER_SOURCE)
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
