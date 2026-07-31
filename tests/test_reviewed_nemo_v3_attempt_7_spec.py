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
    ATTEMPT_6_REVIEWED_JOB_ID,
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    FUTURE_REVIEWED_JOB_ID,
    NEXT_RUNTIME_CORRECTED_BRIDGE_REVISION,
    NEXT_RUNTIME_REVIEWED_JOB_ID,
    SUCCESSOR_A11OY_RELOCK_RUN_URL,
    SUCCESSOR_A11OY_SOURCE_REVISION,
    SUCCESSOR_CORRECTED_BRIDGE_REVISION,
    SUCCESSOR_OWNER_WORKFLOW_BLOB,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

ATTEMPT_6_PATH = ROOT / "jobspecs" / "nemo-v3-20260730-attempt-6-reviewed.json"
ATTEMPT_7_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-7-reviewed.json"
ATTEMPT_7_QUEUE = ROOT / "queue" / "pending" / f"{FUTURE_REVIEWED_JOB_ID}.json"
ATTEMPT_7_QUARANTINE = ROOT / "queue" / "quarantine" / f"{FUTURE_REVIEWED_JOB_ID}.json"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")
PREFETCH_SOURCE = (ROOT / "laptop" / "prefetch_nemo_v3.py").read_text(encoding="utf-8")
DISPATCHER_SOURCE = (ROOT / "laptop" / "dispatcher.py").read_text(encoding="utf-8")
RUNNER_SOURCE = (ROOT / "laptop" / "runjob_nemo_v3.py").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt7SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_6 = json.loads(ATTEMPT_6_PATH.read_text(encoding="utf-8"))
        cls.attempt_7 = json.loads(ATTEMPT_7_PATH.read_text(encoding="utf-8"))

    def test_contract_binds_relocked_source_workflow_runtime_and_key(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_7), self.attempt_7)
        self.assertEqual(self.attempt_7["jobId"], FUTURE_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_7["source"]["revision"],
            SUCCESSOR_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            self.attempt_7["ownerDispatch"]["workflowBlob"],
            SUCCESSOR_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_7["ownerDispatch"]["workflowVersion"],
            "nemo-v3-owner-dispatch.v4",
        )
        authorization = self.attempt_7["authorization"]
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
            SUCCESSOR_CORRECTED_BRIDGE_REVISION,
        )

    def test_lineage_binds_the_zero_event_validator_rejection(self) -> None:
        self.assertEqual(
            self.attempt_7["lineage"],
            {
                "predecessorJobId": ATTEMPT_6_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "c68e1ecf380d7023c27439e9988ca182ebd9b2446dc769269d4de1c48d507d70"
                ),
                "predecessorPayloadSha256": (
                    "d0fa9bd15f8e576411b643858d650470b6f1d5ddd56003cd53eda28d83dd914d"
                ),
                "predecessorEnvelopeRevision": (
                    "72f9bf650b081fec0a016825f2cb7f962c52242d"
                ),
                "predecessorExecutionBridgeRevision": (
                    "69a097d2eb0619506d673464353f1aea7174cf05"
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/szl-gpu-bridge/issues/41"
                ),
                "failurePhase": "PRE_DISPATCH_VALIDATOR_REJECTION",
                "successorGeneration": 7,
                "automaticRetry": False,
                "eventCreated": False,
                "workflowRunCreated": False,
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
            self.assertEqual(self.attempt_7[field], self.attempt_6[field])
        predecessor_dataset = copy.deepcopy(self.attempt_6["dataset"])
        current_dataset = copy.deepcopy(self.attempt_7["dataset"])
        predecessor_dataset.pop("provenance")
        current_dataset.pop("provenance")
        self.assertEqual(current_dataset, predecessor_dataset)
        self.assertEqual(
            {
                key: self.attempt_7["outputs"][key]
                for key in ("receiptsRepoId", "private", "publishCandidate")
            },
            {
                "receiptsRepoId": "SZLHOLDINGS/szl-training-receipts",
                "private": True,
                "publishCandidate": False,
            },
        )
        self.assertEqual(
            {
                key: self.attempt_7["ownerDispatch"][key]
                for key in ("candidateUpload", "modelCardUpload", "datasetUpload")
            },
            {
                "candidateUpload": False,
                "modelCardUpload": False,
                "datasetUpload": False,
            },
        )

    def test_attempt_6_spec_and_envelope_bytes_are_preserved(self) -> None:
        spec_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:jobspecs/nemo-v3-20260730-attempt-6-reviewed.json",
            ],
            cwd=ROOT,
        )
        envelope_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:queue/pending/job-2026-nemo-v3-governed-attempt-6.json",
            ],
            cwd=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(spec_bytes).hexdigest(),
            "fa6be2b8580692075dab7a8cf97f9321d94b655f22bb030af29b18d169939e9e",
        )
        self.assertEqual(
            hashlib.sha256(envelope_bytes).hexdigest(),
            "c68e1ecf380d7023c27439e9988ca182ebd9b2446dc769269d4de1c48d507d70",
        )

    def test_signed_envelope_is_quarantined_after_runtime_rejection(self) -> None:
        self.assertTrue(ATTEMPT_7_QUEUE.is_file())
        self.assertTrue(ATTEMPT_7_QUARANTINE.is_file())
        self.assertEqual(
            hashlib.sha256(ATTEMPT_7_QUEUE.read_bytes()).hexdigest(),
            "8c1e333f797a8de634217b19cd140994a1d4f3920afebdf6f658dcc984188a96",
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_7_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 31, 4, 53, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUARANTINED_NEVER_DISPATCH")
        self.assertTrue(report["terminal"])
        self.assertTrue(report["queue"]["present"])
        self.assertTrue(report["queue"]["valid"], report["queue"]["error"])
        self.assertEqual(
            report["queue"]["payload_sha256"],
            "0fa239d3e14f0644d26b76c0e605ea8068b305cd4d96ea41385cad38fbdfbde7",
        )
        self.assertEqual(
            report["queue"]["engine_key_id"],
            COORDINATED_ENGINE_KEY_ID,
        )
        self.assertTrue(report["quarantine"]["present"])
        self.assertTrue(
            report["quarantine"]["valid"],
            report["quarantine"]["error"],
        )
        self.assertEqual(
            report["quarantine"]["statuses"],
            (
                "RUNTIME_CONTRACT_BINDING_REJECTED",
                "PRE_CLAIM",
                "NEVER_DISPATCH",
            ),
        )
        self.assertFalse(report["receipt"]["present"])
        with self.assertRaisesRegex(ContractError, "NEVER_DISPATCH"):
            require_nemo_v3_dispatchable(self.attempt_7)

    def test_runtime_bound_attempt_8_requires_the_exact_execution_revision(
        self,
    ) -> None:
        attempt_8 = copy.deepcopy(self.attempt_7)
        attempt_8["jobId"] = NEXT_RUNTIME_REVIEWED_JOB_ID
        attempt_8["authorization"]["correctedBridgeRevision"] = (
            NEXT_RUNTIME_CORRECTED_BRIDGE_REVISION
        )
        attempt_8["lineage"] = {
            "predecessorJobId": FUTURE_REVIEWED_JOB_ID,
            "predecessorEnvelopeSha256": (
                "8c1e333f797a8de634217b19cd140994a1d4f3920afebdf6f658dcc984188a96"
            ),
            "predecessorPayloadSha256": (
                "0fa239d3e14f0644d26b76c0e605ea8068b305cd4d96ea41385cad38fbdfbde7"
            ),
            "predecessorEnvelopeRevision": ("21553a898db76dddba3227e91518835185b55a6f"),
            "predecessorExecutionBridgeRevision": (
                "2f33607d8fcbec76fe98290258ec3dfa728fb509"
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
        }
        self.assertIs(validate_nemo_v3_spec(attempt_8), attempt_8)
        with self.assertRaisesRegex(ContractError, "execution Bridge revision"):
            require_nemo_v3_dispatchable(attempt_8)
        with self.assertRaisesRegex(ContractError, "does not match"):
            require_nemo_v3_dispatchable(
                attempt_8,
                expected_execution_bridge_revision="e" * 40,
            )
        require_nemo_v3_dispatchable(
            attempt_8,
            expected_execution_bridge_revision=NEXT_RUNTIME_CORRECTED_BRIDGE_REVISION,
        )
        self.assertIn('"EXECUTION_BRIDGE_REVISION"', PREFETCH_SOURCE)
        self.assertIn('"SZL_EXECUTION_BRIDGE_REVISION"', DISPATCHER_SOURCE)
        self.assertIn('"SZL_EXECUTION_BRIDGE_REVISION"', RUNNER_SOURCE)

    def test_attempt_7_exact_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            ("source", ("source", "revision"), "0" * 40),
            ("workflow", ("ownerDispatch", "workflowBlob"), "1" * 40),
            (
                "relock",
                ("authorization", "settledA11oyRelockRunUrl"),
                "https://github.com/szl-holdings/a11oy/actions/runs/1",
            ),
            (
                "runtime",
                ("authorization", "correctedBridgeRevision"),
                "2" * 40,
            ),
            (
                "predecessor envelope",
                ("lineage", "predecessorEnvelopeSha256"),
                "3" * 64,
            ),
            (
                "predecessor job",
                ("lineage", "predecessorJobId"),
                "job-2026-nemo-v3-governed-attempt-5",
            ),
            ("event side effect", ("lineage", "eventCreated"), True),
            ("claim side effect", ("lineage", "claimCreated"), True),
            ("generation", ("lineage", "successorGeneration"), 6),
        )
        for label, path, value in mutations:
            with self.subTest(label=label):
                mutated = copy.deepcopy(self.attempt_7)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

    def test_signer_and_status_workflow_track_plaintext_attempt_7(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260731-attempt-7-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-7.json",
            "queue/quarantine/job-2026-nemo-v3-governed-attempt-7.json",
            "tests/test_reviewed_nemo_v3_attempt_7_spec.py",
            "attempt_id: attempt-7-validator-lineage-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIn(
            "const FUTURE_REVIEWED_JOB_ID = 'job-2026-nemo-v3-governed-attempt-7';",
            SIGNER_SOURCE,
        )
        self.assertIn(
            "const NEXT_RUNTIME_REVIEWED_JOB_ID = 'job-2026-nemo-v3-governed-attempt-8';",
            SIGNER_SOURCE,
        )
        self.assertIn(
            "signer is locked to ${NEXT_RUNTIME_REVIEWED_JOB_ID}", SIGNER_SOURCE
        )
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
