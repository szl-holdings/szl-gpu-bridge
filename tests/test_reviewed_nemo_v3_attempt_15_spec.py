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
    ATTEMPT_14_REVIEWED_JOB_ID,
    ATTEMPT_15_REVIEWED_JOB_ID,
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
    EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

ATTEMPT_14_CORRECTED_BRIDGE_REVISION = "e150711a6ba6a0c29109a00da7fc82af2967f588"
ATTEMPT_15_CORRECTED_BRIDGE_REVISION = "60b9894efe9e0e782999aaa4ee5b0d668e7a9b63"
ATTEMPT_14_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-14-reviewed.json"
ATTEMPT_15_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-15-reviewed.json"
ATTEMPT_15_QUEUE = ROOT / "queue" / "pending" / f"{ATTEMPT_15_REVIEWED_JOB_ID}.json"
ATTEMPT_15_QUARANTINE = (
    ROOT / "queue" / "quarantine" / f"{ATTEMPT_15_REVIEWED_JOB_ID}.json"
)
ATTEMPT_15_EVIDENCE = ROOT / "queue" / "evidence" / f"{ATTEMPT_15_REVIEWED_JOB_ID}.json"
SCHEMA_PATH = ROOT / "schema" / "nemo-v3-jobspec.v1.json"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt15SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_14 = json.loads(ATTEMPT_14_PATH.read_text(encoding="utf-8"))
        cls.attempt_15 = json.loads(ATTEMPT_15_PATH.read_text(encoding="utf-8"))

    def test_exact_source_workflow_runtime_key_and_license(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_15), self.attempt_15)
        self.assertEqual(self.attempt_15["jobId"], ATTEMPT_15_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_15["source"]["revision"],
            EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            self.attempt_15["ownerDispatch"]["workflowBlob"],
            EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_15["authorization"]["settledA11oyRelockRunUrl"],
            EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
        )
        self.assertEqual(
            self.attempt_15["authorization"]["correctedBridgeRevision"],
            ATTEMPT_15_CORRECTED_BRIDGE_REVISION,
        )
        self.assertEqual(
            self.attempt_15["authorization"]["engineKeyId"],
            COORDINATED_ENGINE_KEY_ID,
        )
        self.assertEqual(
            self.attempt_15["authorization"]["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            self.attempt_15["base"]["licenseId"],
            "nvidia-nemotron-open-model-license",
        )
        with mock.patch("nemo_v3_contract.quarantine_policy", return_value=None):
            require_nemo_v3_dispatchable(
                self.attempt_15,
                expected_execution_bridge_revision=(
                    ATTEMPT_15_CORRECTED_BRIDGE_REVISION
                ),
            )

    def test_json_schema_admits_only_the_exact_attempt_15_binding(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        authorization_schema = schema["properties"]["authorization"]["oneOf"][1]
        self.assertIn(
            ATTEMPT_15_CORRECTED_BRIDGE_REVISION,
            authorization_schema["properties"]["correctedBridgeRevision"]["enum"],
        )
        attempt_15_rule = next(
            rule
            for rule in schema["allOf"]
            if rule["if"]["properties"]["jobId"].get("const")
            == ATTEMPT_15_REVIEWED_JOB_ID
        )
        properties = attempt_15_rule["then"]["properties"]
        self.assertEqual(
            properties["authorization"]["properties"]["correctedBridgeRevision"][
                "const"
            ],
            ATTEMPT_15_CORRECTED_BRIDGE_REVISION,
        )
        exact_lineage = properties["lineage"]["allOf"][1]["properties"]
        for field in (
            "predecessorJobId",
            "predecessorEnvelopeSha256",
            "predecessorPayloadSha256",
            "predecessorEnvelopeRevision",
            "predecessorExecutionBridgeRevision",
            "transportEvidenceUrl",
            "failurePhase",
            "successorGeneration",
        ):
            self.assertEqual(
                exact_lineage[field]["const"], self.attempt_15["lineage"][field]
            )

        terminal_lineage = schema["$defs"]["terminalReceiptFailureLineage"][
            "properties"
        ]
        self.assertIn(
            ATTEMPT_14_REVIEWED_JOB_ID,
            terminal_lineage["predecessorJobId"]["enum"],
        )
        self.assertIn(
            "POST_CLAIM_TRAINER_META_TENSOR",
            terminal_lineage["failurePhase"]["enum"],
        )

    def test_exact_attempt_14_signed_blocked_lineage(self) -> None:
        self.assertEqual(
            self.attempt_15["lineage"],
            {
                "predecessorJobId": ATTEMPT_14_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "207f0c58525f042d31a748404d0acb678f5fd83722d2a3eacf8399e4e34c9f82"
                ),
                "predecessorPayloadSha256": (
                    "162354602784e8a1cbcecbbfc8a5d7cc9af6be2dd58c66fae442d4f5a292f1da"
                ),
                "predecessorEnvelopeRevision": (
                    "fd97065eb2aa9fc3299706c531597538a65eb735"
                ),
                "predecessorExecutionBridgeRevision": (
                    ATTEMPT_14_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30634484969"
                ),
                "failurePhase": "POST_CLAIM_TRAINER_META_TENSOR",
                "successorGeneration": 15,
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
        for field in ("source", "base", "dataset", "recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt_15[field], self.attempt_14[field])
        self.assertFalse(self.attempt_15["outputs"]["publishCandidate"])
        self.assertTrue(self.attempt_15["outputs"]["private"])
        for field in ("candidateUpload", "modelCardUpload", "datasetUpload"):
            self.assertFalse(self.attempt_15["ownerDispatch"][field])

    def test_attempt_14_signed_and_evidence_bytes_are_preserved(self) -> None:
        expected = {
            "jobspecs/nemo-v3-20260731-attempt-14-reviewed.json": (
                "99e293ab4c2dd4282bd39a5f741b8359652792c68215c0e7100114a77bbacdf6"
            ),
            "queue/pending/job-2026-nemo-v3-governed-attempt-14.json": (
                "207f0c58525f042d31a748404d0acb678f5fd83722d2a3eacf8399e4e34c9f82"
            ),
            "queue/evidence/job-2026-nemo-v3-governed-attempt-14.json": (
                "430aa2494b6b1bbcae45f99409075cfbe525ab628582806e3be1c8ae18204bc4"
            ),
            "queue/quarantine/job-2026-nemo-v3-governed-attempt-14.json": (
                "ad8d0813ac4746e6e6284862aa4f16097ef59713ee307bfe9b7bfcee9a7d7ebf"
            ),
        }
        for path, digest in expected.items():
            with self.subTest(path=path):
                content = subprocess.check_output(
                    ["git", "cat-file", "blob", f"HEAD:{path}"], cwd=ROOT
                )
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)

    def test_signed_attempt_15_is_quarantined_with_zero_effect_evidence(self) -> None:
        self.assertTrue(ATTEMPT_15_QUEUE.exists())
        self.assertEqual(
            hashlib.sha256(ATTEMPT_15_QUEUE.read_bytes()).hexdigest(),
            "93d5effe94740af9135c3ffa379c85df1aa88e6ad5717bc6421266d21bb9dbe7",
        )
        payload_sha256 = hashlib.sha256(
            nemo_v3_status.signer_canonicalize(self.attempt_15).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            payload_sha256,
            "9c55b95627b93e522eaebec5cb9e837b46d8e368065470aa45f55f488aeff873",
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_15_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 31, 14, 50, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUARANTINED_NEVER_DISPATCH")
        self.assertTrue(report["terminal"])
        self.assertTrue(report["queue"]["present"])
        self.assertTrue(report["queue"]["valid"], report["queue"]["error"])
        self.assertEqual(report["queue"]["engine_key_id"], COORDINATED_ENGINE_KEY_ID)
        self.assertEqual(report["queue"]["payload_sha256"], payload_sha256)
        self.assertFalse(report["receipt"]["present"])
        self.assertFalse(report["reviewed_spec"]["candidate_publication_enabled"])
        self.assertTrue(report["quarantine"]["present"])
        self.assertTrue(report["quarantine"]["valid"], report["quarantine"]["error"])

        quarantine = json.loads(ATTEMPT_15_QUARANTINE.read_text(encoding="utf-8"))
        self.assertEqual(
            quarantine["status"],
            ["RUNTIME_JOB_BINDING_REJECTED", "PRE_CLAIM", "NEVER_DISPATCH"],
        )
        self.assertFalse(quarantine["dispatchAuthorized"])
        self.assertEqual(
            quarantine["replacement"]["reviewedJobId"],
            "job-2026-nemo-v3-governed-attempt-16",
        )
        self.assertEqual(quarantine["replacement"]["successorGeneration"], 16)

        evidence = json.loads(ATTEMPT_15_EVIDENCE.read_text(encoding="utf-8"))[
            "executionEvidence"
        ]
        self.assertEqual(evidence["workflowRunId"], "30641766033")
        self.assertEqual(evidence["workflowJobId"], "91193214499")
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

    def test_exact_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            (("source", "revision"), "0" * 40),
            (("base", "licenseId"), "nvidia-open-model-license"),
            (("ownerDispatch", "workflowBlob"), "1" * 40),
            (("lineage", "predecessorEnvelopeSha256"), "3" * 64),
            (("lineage", "claimCreated"), False),
            (("lineage", "trainingStarted"), True),
            (("lineage", "receiptIntentProduced"), False),
            (("lineage", "successorGeneration"), 14),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.attempt_15)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

        with mock.patch("nemo_v3_contract.quarantine_policy", return_value=None):
            with self.assertRaisesRegex(ContractError, "runtime-bound successor"):
                require_nemo_v3_dispatchable(
                    self.attempt_15,
                    expected_execution_bridge_revision="4" * 40,
                )
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_15,
                expected_execution_bridge_revision=(
                    ATTEMPT_15_CORRECTED_BRIDGE_REVISION
                ),
            )
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_14,
                expected_execution_bridge_revision=ATTEMPT_14_CORRECTED_BRIDGE_REVISION,
            )

    def test_signer_and_status_are_locked_to_attempt_15(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260731-attempt-15-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-15.json",
            "tests/test_reviewed_nemo_v3_attempt_15_spec.py",
            "attempt_id: attempt-15-meta-tensor-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIn(
            "signer is locked to ${ATTEMPT_15_REVIEWED_JOB_ID}",
            SIGNER_SOURCE,
        )
        self.assertIn("'job-2026-nemo-v3-governed-attempt-14',", SIGNER_SOURCE)
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
