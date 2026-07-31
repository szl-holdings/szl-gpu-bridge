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
    ATTEMPT_9_CORRECTED_BRIDGE_REVISION,
    ATTEMPT_9_REVIEWED_JOB_ID,
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    FUTURE_REVIEWED_JOB_ID,
    NEXT_RUNTIME_CORRECTED_BRIDGE_REVISION,
    NEXT_RUNTIME_REVIEWED_JOB_ID,
    RECOVERY_A11OY_RELOCK_RUN_URL,
    RECOVERY_A11OY_SOURCE_REVISION,
    RECOVERY_OWNER_WORKFLOW_BLOB,
    SUCCESSOR_A11OY_RELOCK_RUN_URL,
    SUCCESSOR_A11OY_SOURCE_REVISION,
    SUCCESSOR_CORRECTED_BRIDGE_REVISION,
    SUCCESSOR_OWNER_WORKFLOW_BLOB,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

ATTEMPT_7_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-7-reviewed.json"
ATTEMPT_8_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-8-reviewed.json"
ATTEMPT_8_QUEUE = ROOT / "queue" / "pending" / f"{NEXT_RUNTIME_REVIEWED_JOB_ID}.json"
ATTEMPT_8_QUARANTINE = (
    ROOT / "queue" / "quarantine" / f"{NEXT_RUNTIME_REVIEWED_JOB_ID}.json"
)
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt8SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_7 = json.loads(ATTEMPT_7_PATH.read_text(encoding="utf-8"))
        cls.attempt_8 = json.loads(ATTEMPT_8_PATH.read_text(encoding="utf-8"))

    def test_contract_binds_exact_source_workflow_runtime_and_key(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_8), self.attempt_8)
        self.assertEqual(self.attempt_8["jobId"], NEXT_RUNTIME_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_8["source"]["revision"],
            SUCCESSOR_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            self.attempt_8["ownerDispatch"]["workflowBlob"],
            SUCCESSOR_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_8["ownerDispatch"]["workflowVersion"],
            "nemo-v3-owner-dispatch.v4",
        )
        authorization = self.attempt_8["authorization"]
        self.assertEqual(authorization["engineKeyId"], COORDINATED_ENGINE_KEY_ID)
        self.assertEqual(
            authorization["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            authorization["settledA11oyRelockRunUrl"],
            SUCCESSOR_A11OY_RELOCK_RUN_URL,
        )
        self.assertEqual(
            authorization["correctedBridgeRevision"],
            NEXT_RUNTIME_CORRECTED_BRIDGE_REVISION,
        )

    def test_lineage_binds_the_exact_preclaim_runtime_rejection(self) -> None:
        self.assertEqual(
            self.attempt_8["lineage"],
            {
                "predecessorJobId": FUTURE_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "8c1e333f797a8de634217b19cd140994a1d4f3920afebdf6f658dcc984188a96"
                ),
                "predecessorPayloadSha256": (
                    "0fa239d3e14f0644d26b76c0e605ea8068b305cd4d96ea41385cad38fbdfbde7"
                ),
                "predecessorEnvelopeRevision": (
                    "21553a898db76dddba3227e91518835185b55a6f"
                ),
                "predecessorExecutionBridgeRevision": (
                    SUCCESSOR_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30605081533"
                ),
                "failurePhase": "PRE_CLAIM_RUNTIME_CONTRACT_VALIDATION",
                "successorGeneration": 8,
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

    def test_science_inputs_and_receipt_only_outputs_remain_frozen(self) -> None:
        for field in ("base", "recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt_8[field], self.attempt_7[field])
        predecessor_dataset = copy.deepcopy(self.attempt_7["dataset"])
        current_dataset = copy.deepcopy(self.attempt_8["dataset"])
        predecessor_dataset.pop("provenance")
        current_dataset.pop("provenance")
        self.assertEqual(current_dataset, predecessor_dataset)
        self.assertFalse(self.attempt_8["outputs"]["publishCandidate"])
        self.assertTrue(self.attempt_8["outputs"]["private"])
        for field in ("candidateUpload", "modelCardUpload", "datasetUpload"):
            self.assertFalse(self.attempt_8["ownerDispatch"][field])

    def test_attempt_7_spec_and_envelope_bytes_are_preserved(self) -> None:
        spec_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:jobspecs/nemo-v3-20260731-attempt-7-reviewed.json",
            ],
            cwd=ROOT,
        )
        envelope_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:queue/pending/job-2026-nemo-v3-governed-attempt-7.json",
            ],
            cwd=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(spec_bytes).hexdigest(),
            "be20e5deaea656c3927754c67e5cd54d781ce0c9863a678f9cd937916333307b",
        )
        self.assertEqual(
            hashlib.sha256(envelope_bytes).hexdigest(),
            "8c1e333f797a8de634217b19cd140994a1d4f3920afebdf6f658dcc984188a96",
        )

    def test_attempt_8_spec_and_envelope_bytes_are_preserved(self) -> None:
        spec_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:jobspecs/nemo-v3-20260731-attempt-8-reviewed.json",
            ],
            cwd=ROOT,
        )
        envelope_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:queue/pending/job-2026-nemo-v3-governed-attempt-8.json",
            ],
            cwd=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(spec_bytes).hexdigest(),
            "cae580780583dad8ef8c50db2a538d17275fddb834e8f9b5581dd5fcc977c224",
        )
        self.assertEqual(
            hashlib.sha256(envelope_bytes).hexdigest(),
            "b2db463661ab9e16bf24267c82ee104cf25344e7b4addbd2e9867e7e33be3719",
        )

    def test_signed_status_is_quarantined_after_dirty_checkout_rejection(
        self,
    ) -> None:
        self.assertTrue(ATTEMPT_8_QUEUE.is_file())
        self.assertTrue(ATTEMPT_8_QUARANTINE.is_file())
        self.assertEqual(
            hashlib.sha256(ATTEMPT_8_QUEUE.read_bytes()).hexdigest(),
            "b2db463661ab9e16bf24267c82ee104cf25344e7b4addbd2e9867e7e33be3719",
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_8_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 31, 5, 51, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUARANTINED_NEVER_DISPATCH")
        self.assertTrue(report["terminal"])
        self.assertTrue(report["queue"]["present"])
        self.assertTrue(report["queue"]["valid"], report["queue"]["error"])
        self.assertEqual(
            report["queue"]["payload_sha256"],
            "3372fff9c21a73ee140598c152b728b4d7694fb0a066c80e8b55e09832a0769d",
        )
        self.assertEqual(
            report["queue"]["engine_key_id"],
            COORDINATED_ENGINE_KEY_ID,
        )
        self.assertFalse(report["receipt"]["present"])
        self.assertTrue(report["quarantine"]["present"])
        self.assertTrue(
            report["quarantine"]["valid"],
            report["quarantine"]["error"],
        )
        self.assertEqual(
            report["quarantine"]["statuses"],
            (
                "TRUSTED_PREFETCH_DIRTIED_EXECUTION_CHECKOUT",
                "PRE_CLAIM",
                "NEVER_DISPATCH",
            ),
        )
        with self.assertRaisesRegex(ContractError, "NEVER_DISPATCH"):
            require_nemo_v3_dispatchable(self.attempt_8)

    def test_attempt_9_requires_the_exact_reviewed_execution_revision(
        self,
    ) -> None:
        attempt_9 = copy.deepcopy(self.attempt_8)
        attempt_9["jobId"] = ATTEMPT_9_REVIEWED_JOB_ID
        attempt_9["source"]["revision"] = RECOVERY_A11OY_SOURCE_REVISION
        attempt_9["ownerDispatch"]["workflowBlob"] = RECOVERY_OWNER_WORKFLOW_BLOB
        attempt_9["authorization"]["settledA11oyRelockRunUrl"] = (
            RECOVERY_A11OY_RELOCK_RUN_URL
        )
        attempt_9["authorization"]["correctedBridgeRevision"] = (
            ATTEMPT_9_CORRECTED_BRIDGE_REVISION
        )
        attempt_9["lineage"] = {
            "predecessorJobId": NEXT_RUNTIME_REVIEWED_JOB_ID,
            "predecessorEnvelopeSha256": (
                "b2db463661ab9e16bf24267c82ee104cf25344e7b4addbd2e9867e7e33be3719"
            ),
            "predecessorPayloadSha256": (
                "3372fff9c21a73ee140598c152b728b4d7694fb0a066c80e8b55e09832a0769d"
            ),
            "predecessorEnvelopeRevision": ("08b1bd8bc0659b939d3d6d08c2ee7c670f82cd09"),
            "predecessorExecutionBridgeRevision": (
                NEXT_RUNTIME_CORRECTED_BRIDGE_REVISION
            ),
            "transportEvidenceUrl": (
                "https://github.com/szl-holdings/a11oy/actions/runs/30606664591"
            ),
            "failurePhase": "PRE_CLAIM_DIRTY_EXECUTION_CHECKOUT",
            "successorGeneration": 9,
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
        self.assertIs(validate_nemo_v3_spec(attempt_9), attempt_9)
        with self.assertRaisesRegex(ContractError, "does not match"):
            require_nemo_v3_dispatchable(
                attempt_9,
                expected_execution_bridge_revision="e" * 40,
            )
        require_nemo_v3_dispatchable(
            attempt_9,
            expected_execution_bridge_revision=ATTEMPT_9_CORRECTED_BRIDGE_REVISION,
        )

    def test_exact_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            (("source", "revision"), "0" * 40),
            (("ownerDispatch", "workflowBlob"), "1" * 40),
            (
                ("authorization", "settledA11oyRelockRunUrl"),
                "https://github.com/szl-holdings/a11oy/actions/runs/1",
            ),
            (("authorization", "correctedBridgeRevision"), "2" * 40),
            (("lineage", "predecessorEnvelopeSha256"), "3" * 64),
            (
                ("lineage", "predecessorJobId"),
                "job-2026-nemo-v3-governed-attempt-6",
            ),
            (("lineage", "claimCreated"), True),
            (("lineage", "successorGeneration"), 7),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.attempt_8)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

    def test_status_preserves_attempt_8_while_signer_advances_to_attempt_9(
        self,
    ) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260731-attempt-8-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-8.json",
            "tests/test_reviewed_nemo_v3_attempt_8_spec.py",
            "attempt_id: attempt-8-runtime-binding-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIn(
            "const NEXT_RUNTIME_REVIEWED_JOB_ID = 'job-2026-nemo-v3-governed-attempt-8';",
            SIGNER_SOURCE,
        )
        self.assertIn(
            "signer is locked to ${ATTEMPT_9_REVIEWED_JOB_ID}",
            SIGNER_SOURCE,
        )
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
