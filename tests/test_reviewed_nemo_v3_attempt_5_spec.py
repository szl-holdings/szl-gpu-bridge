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
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    FINAL_A11OY_RELOCK_RUN_URL,
    FINAL_A11OY_SOURCE_REVISION,
    FINAL_CORRECTED_BRIDGE_REVISION,
    FINAL_OWNER_WORKFLOW_BLOB,
    FUTURE_REVIEWED_JOB_ID,
    NEXT_REVIEWED_JOB_ID,
    validate_nemo_v3_spec,
)

ATTEMPT_4_PATH = ROOT / "jobspecs" / "nemo-v3-20260730-attempt-4-reviewed.json"
ATTEMPT_5_PATH = ROOT / "jobspecs" / "nemo-v3-20260730-attempt-5-reviewed.json"
ATTEMPT_5_QUEUE = ROOT / "queue" / "pending" / f"{FUTURE_REVIEWED_JOB_ID}.json"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt5SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_4 = json.loads(ATTEMPT_4_PATH.read_text(encoding="utf-8"))
        cls.attempt_5 = json.loads(ATTEMPT_5_PATH.read_text(encoding="utf-8"))

    def test_contract_binds_final_source_workflow_runtime_and_key(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_5), self.attempt_5)
        self.assertEqual(self.attempt_5["jobId"], FUTURE_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_5["source"]["revision"], FINAL_A11OY_SOURCE_REVISION
        )
        self.assertEqual(
            self.attempt_5["ownerDispatch"]["workflowBlob"],
            FINAL_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_5["ownerDispatch"]["workflowVersion"],
            "nemo-v3-owner-dispatch.v4",
        )
        authorization = self.attempt_5["authorization"]
        self.assertEqual(authorization["engineKeyId"], COORDINATED_ENGINE_KEY_ID)
        self.assertEqual(
            authorization["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            authorization["settledA11oyRelockRunUrl"],
            FINAL_A11OY_RELOCK_RUN_URL,
        )
        self.assertEqual(
            authorization["correctedBridgeRevision"],
            FINAL_CORRECTED_BRIDGE_REVISION,
        )

    def test_transport_lineage_records_zero_side_effect_predecessor(self) -> None:
        self.assertEqual(
            self.attempt_5["lineage"],
            {
                "predecessorJobId": NEXT_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "e240a176849b1f6c0d453ac55277cd7732b3a302ea9679db78d3c612501f27f2"
                ),
                "predecessorPayloadSha256": (
                    "14441cf982b177c1b613e56e63eae8be3e589ae35444826b40731c32312268e5"
                ),
                "predecessorEnvelopeRevision": (
                    "7045fe223703ba8fb2d710a59989f971080e7702"
                ),
                "predecessorExecutionBridgeRevision": (
                    "2237bb3f36663343ace29d98cda6c32e165450a0"
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/szl-gpu-bridge/issues/32"
                ),
                "failurePhase": "PRE_EVENT_TRANSPORT_VALIDATION",
                "successorGeneration": 5,
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

    def test_science_inputs_and_receipt_only_effects_remain_frozen(self) -> None:
        for field in ("base", "recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt_5[field], self.attempt_4[field])
        earlier_dataset = copy.deepcopy(self.attempt_4["dataset"])
        current_dataset = copy.deepcopy(self.attempt_5["dataset"])
        earlier_dataset.pop("provenance")
        current_dataset.pop("provenance")
        self.assertEqual(current_dataset, earlier_dataset)
        self.assertEqual(
            {
                key: self.attempt_5["outputs"][key]
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
                key: self.attempt_5["ownerDispatch"][key]
                for key in ("candidateUpload", "modelCardUpload", "datasetUpload")
            },
            {
                "candidateUpload": False,
                "modelCardUpload": False,
                "datasetUpload": False,
            },
        )

    def test_attempt_4_evidence_is_byte_preserved(self) -> None:
        spec_bytes = subprocess.check_output(
            [
                "git",
                "show",
                "HEAD:jobspecs/nemo-v3-20260730-attempt-4-reviewed.json",
            ],
            cwd=ROOT,
        )
        envelope_bytes = subprocess.check_output(
            [
                "git",
                "show",
                f"HEAD:queue/pending/{NEXT_REVIEWED_JOB_ID}.json",
            ],
            cwd=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(spec_bytes).hexdigest(),
            "10e4659c9414fd183f7ccdf9534d40735e33f9188ddba0a7b4fc135c439086df",
        )
        self.assertEqual(
            hashlib.sha256(envelope_bytes).hexdigest(),
            "e240a176849b1f6c0d453ac55277cd7732b3a302ea9679db78d3c612501f27f2",
        )

    def test_signed_envelope_is_valid_and_receipt_is_absent(self) -> None:
        self.assertTrue(ATTEMPT_5_QUEUE.is_file())
        self.assertEqual(
            hashlib.sha256(ATTEMPT_5_QUEUE.read_bytes()).hexdigest(),
            "30549fc522238193b4985dbf96a690518bad2ae8c399dc3ee78fb9dd7f551009",
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_5_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 30, 23, 31, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUEUED_AWAITING_GPU_RECEIPT")
        self.assertFalse(report["terminal"])
        self.assertTrue(report["queue"]["present"])
        self.assertTrue(report["queue"]["valid"], report["queue"]["error"])
        self.assertEqual(report["queue"]["engine_key_id"], COORDINATED_ENGINE_KEY_ID)
        self.assertEqual(
            report["queue"]["payload_sha256"],
            "374901dec6923e0c28688407e581d374827d76f7567970d8ec481b6bf140c67b",
        )
        self.assertFalse(report["receipt"]["present"])

    def test_attempt_5_exact_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            ("source", ("source", "revision"), "0" * 40),
            ("workflow", ("ownerDispatch", "workflowBlob"), "1" * 40),
            ("workflow version", ("ownerDispatch", "workflowVersion"), "v2"),
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
            ("event side effect", ("lineage", "eventCreated"), True),
            ("claim side effect", ("lineage", "claimCreated"), True),
            ("generation", ("lineage", "successorGeneration"), 4),
        )
        for label, path, value in mutations:
            with self.subTest(label=label):
                mutated = copy.deepcopy(self.attempt_5)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

    def test_signer_and_status_workflow_track_attempt_5(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260730-attempt-5-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-5.json",
            "tests/test_reviewed_nemo_v3_attempt_5_spec.py",
            "attempt_id: attempt-5-transport-v3-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIn(
            "const FUTURE_REVIEWED_JOB_ID = 'job-2026-nemo-v3-governed-attempt-5';",
            SIGNER_SOURCE,
        )
        self.assertIn("signer is locked to ${FUTURE_REVIEWED_JOB_ID}", SIGNER_SOURCE)
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
