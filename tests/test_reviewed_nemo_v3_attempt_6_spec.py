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
    ATTEMPT_5_REVIEWED_JOB_ID,
    ATTEMPT_6_REVIEWED_JOB_ID,
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    EXECUTION_A11OY_RELOCK_RUN_URL,
    EXECUTION_A11OY_SOURCE_REVISION,
    EXECUTION_OWNER_WORKFLOW_BLOB,
    FINAL_CORRECTED_BRIDGE_REVISION,
    FUTURE_REVIEWED_JOB_ID,
    SUCCESSOR_A11OY_RELOCK_RUN_URL,
    SUCCESSOR_A11OY_SOURCE_REVISION,
    SUCCESSOR_OWNER_WORKFLOW_BLOB,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

ATTEMPT_5_PATH = ROOT / "jobspecs" / "nemo-v3-20260730-attempt-5-reviewed.json"
ATTEMPT_6_PATH = ROOT / "jobspecs" / "nemo-v3-20260730-attempt-6-reviewed.json"
ATTEMPT_6_QUEUE = ROOT / "queue" / "pending" / f"{ATTEMPT_6_REVIEWED_JOB_ID}.json"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt6SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_5 = json.loads(ATTEMPT_5_PATH.read_text(encoding="utf-8"))
        cls.attempt_6 = json.loads(ATTEMPT_6_PATH.read_text(encoding="utf-8"))

    def test_contract_binds_relocked_source_workflow_runtime_and_key(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_6), self.attempt_6)
        self.assertEqual(self.attempt_6["jobId"], ATTEMPT_6_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_6["source"]["revision"],
            EXECUTION_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            self.attempt_6["ownerDispatch"]["workflowBlob"],
            EXECUTION_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_6["ownerDispatch"]["workflowVersion"],
            "nemo-v3-owner-dispatch.v4",
        )
        authorization = self.attempt_6["authorization"]
        self.assertEqual(authorization["engineKeyId"], COORDINATED_ENGINE_KEY_ID)
        self.assertEqual(
            authorization["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            authorization["settledA11oyRelockRunUrl"],
            EXECUTION_A11OY_RELOCK_RUN_URL,
        )
        self.assertEqual(
            authorization["correctedBridgeRevision"],
            "69a097d2eb0619506d673464353f1aea7174cf05",
        )

    def test_lineage_binds_the_single_pre_admission_failure(self) -> None:
        self.assertEqual(
            self.attempt_6["lineage"],
            {
                "predecessorJobId": ATTEMPT_5_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "30549fc522238193b4985dbf96a690518bad2ae8c399dc3ee78fb9dd7f551009"
                ),
                "predecessorPayloadSha256": (
                    "374901dec6923e0c28688407e581d374827d76f7567970d8ec481b6bf140c67b"
                ),
                "predecessorEnvelopeRevision": (
                    "d127d7bcd734235fba83e786de923787ab90c51b"
                ),
                "predecessorExecutionBridgeRevision": FINAL_CORRECTED_BRIDGE_REVISION,
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30591897165"
                ),
                "failurePhase": "PRE_ADMISSION_HOST_EXECUTION_POLICY",
                "successorGeneration": 6,
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
            self.assertEqual(self.attempt_6[field], self.attempt_5[field])
        predecessor_dataset = copy.deepcopy(self.attempt_5["dataset"])
        current_dataset = copy.deepcopy(self.attempt_6["dataset"])
        predecessor_dataset.pop("provenance")
        current_dataset.pop("provenance")
        self.assertEqual(current_dataset, predecessor_dataset)
        self.assertEqual(
            {
                key: self.attempt_6["outputs"][key]
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
                key: self.attempt_6["ownerDispatch"][key]
                for key in ("candidateUpload", "modelCardUpload", "datasetUpload")
            },
            {
                "candidateUpload": False,
                "modelCardUpload": False,
                "datasetUpload": False,
            },
        )

    def test_predecessor_spec_and_envelope_bytes_are_preserved(self) -> None:
        spec_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:jobspecs/nemo-v3-20260730-attempt-5-reviewed.json",
            ],
            cwd=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(spec_bytes).hexdigest(),
            "f78bbbbe95a9c77d92825dd6021a5bc656e719461d12126afde3c52bd49ae594",
        )
        envelope_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:queue/pending/job-2026-nemo-v3-governed-attempt-5.json",
            ],
            cwd=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(envelope_bytes).hexdigest(),
            "30549fc522238193b4985dbf96a690518bad2ae8c399dc3ee78fb9dd7f551009",
        )

    def test_signed_envelope_is_valid_quarantined_without_side_effects(self) -> None:
        self.assertTrue(ATTEMPT_6_QUEUE.is_file())
        self.assertEqual(
            hashlib.sha256(ATTEMPT_6_QUEUE.read_bytes()).hexdigest(),
            "c68e1ecf380d7023c27439e9988ca182ebd9b2446dc769269d4de1c48d507d70",
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_6_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 31, 0, 31, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUARANTINED_NEVER_DISPATCH")
        self.assertTrue(report["terminal"])
        self.assertTrue(report["queue"]["present"])
        self.assertTrue(report["queue"]["valid"], report["queue"]["error"])
        self.assertEqual(report["queue"]["engine_key_id"], COORDINATED_ENGINE_KEY_ID)
        self.assertEqual(
            report["queue"]["payload_sha256"],
            "d0fa9bd15f8e576411b643858d650470b6f1d5ddd56003cd53eda28d83dd914d",
        )
        self.assertTrue(report["quarantine"]["present"])
        self.assertTrue(report["quarantine"]["valid"], report["quarantine"]["error"])
        self.assertEqual(
            report["quarantine"]["statuses"],
            (
                "STALE_SOURCE",
                "PRE_DISPATCH_VALIDATOR_REJECTED",
                "PRE_EVENT",
                "NEVER_DISPATCH",
            ),
        )
        self.assertFalse(report["receipt"]["present"])
        with self.assertRaisesRegex(ContractError, "NEVER_DISPATCH"):
            require_nemo_v3_dispatchable(self.attempt_6)

    def test_attempt_6_bytes_and_replacement_binding_are_exact(self) -> None:
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
        record = json.loads(
            (
                ROOT / "queue" / "quarantine" / f"{ATTEMPT_6_REVIEWED_JOB_ID}.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            record["replacement"],
            {
                "sourceRevision": SUCCESSOR_A11OY_SOURCE_REVISION,
                "workflowBlob": SUCCESSOR_OWNER_WORKFLOW_BLOB,
                "engineKeyId": COORDINATED_ENGINE_KEY_ID,
                "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
                "reviewedJobId": FUTURE_REVIEWED_JOB_ID,
            },
        )
        self.assertEqual(
            SUCCESSOR_A11OY_RELOCK_RUN_URL,
            "https://github.com/szl-holdings/a11oy/actions/runs/30601635066",
        )

    def test_attempt_6_exact_bindings_fail_closed_on_drift(self) -> None:
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
            ("event side effect", ("lineage", "eventCreated"), False),
            ("claim side effect", ("lineage", "claimCreated"), True),
            ("generation", ("lineage", "successorGeneration"), 5),
        )
        for label, path, value in mutations:
            with self.subTest(label=label):
                mutated = copy.deepcopy(self.attempt_6)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

    def test_signer_and_status_workflow_preserve_attempt_6_as_quarantined(
        self,
    ) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260730-attempt-6-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-6.json",
            "queue/quarantine/job-2026-nemo-v3-governed-attempt-6.json",
            "tests/test_reviewed_nemo_v3_attempt_6_spec.py",
            "attempt_id: attempt-6-host-policy-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIn(
            "const ATTEMPT_6_REVIEWED_JOB_ID = 'job-2026-nemo-v3-governed-attempt-6';",
            SIGNER_SOURCE,
        )
        self.assertIn(
            "const FUTURE_REVIEWED_JOB_ID = 'job-2026-nemo-v3-governed-attempt-7';",
            SIGNER_SOURCE,
        )
        self.assertIn("'job-2026-nemo-v3-governed-attempt-6',", SIGNER_SOURCE)
        self.assertIn("'job-2026-nemo-v3-governed-attempt-5',", SIGNER_SOURCE)
        self.assertIn("is quarantined and marked NEVER_DISPATCH", SIGNER_SOURCE)
        self.assertIn(
            "signer is locked to ${ATTEMPT_14_REVIEWED_JOB_ID}",
            SIGNER_SOURCE,
        )
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
