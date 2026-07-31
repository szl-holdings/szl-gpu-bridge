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
    ATTEMPT_13_REVIEWED_JOB_ID,
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

ATTEMPT_13_CORRECTED_BRIDGE_REVISION = "2783b3518abcec9f38d3f6504c06e305a4723801"
ATTEMPT_14_CORRECTED_BRIDGE_REVISION = "e150711a6ba6a0c29109a00da7fc82af2967f588"
ATTEMPT_13_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-13-reviewed.json"
ATTEMPT_14_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-14-reviewed.json"
ATTEMPT_14_QUEUE = ROOT / "queue" / "pending" / f"{ATTEMPT_14_REVIEWED_JOB_ID}.json"
ATTEMPT_14_QUARANTINE = (
    ROOT / "queue" / "quarantine" / f"{ATTEMPT_14_REVIEWED_JOB_ID}.json"
)
ATTEMPT_14_EVIDENCE = ROOT / "queue" / "evidence" / f"{ATTEMPT_14_REVIEWED_JOB_ID}.json"
NEMO_SCHEMA_PATH = ROOT / "schema" / "nemo-v3-jobspec.v1.json"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt14SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_13 = json.loads(ATTEMPT_13_PATH.read_text(encoding="utf-8"))
        cls.attempt_14 = json.loads(ATTEMPT_14_PATH.read_text(encoding="utf-8"))

    def test_exact_source_workflow_runtime_key_and_license(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_14), self.attempt_14)
        self.assertEqual(self.attempt_14["jobId"], ATTEMPT_14_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_14["source"]["revision"],
            EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            self.attempt_14["ownerDispatch"]["workflowBlob"],
            EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_14["authorization"]["settledA11oyRelockRunUrl"],
            EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
        )
        self.assertEqual(
            self.attempt_14["authorization"]["correctedBridgeRevision"],
            ATTEMPT_14_CORRECTED_BRIDGE_REVISION,
        )
        self.assertEqual(
            self.attempt_14["authorization"]["engineKeyId"],
            COORDINATED_ENGINE_KEY_ID,
        )
        self.assertEqual(
            self.attempt_14["authorization"]["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            self.attempt_14["base"]["licenseId"],
            "nvidia-nemotron-open-model-license",
        )

    def test_json_schema_carries_the_exact_attempt_14_binding(self) -> None:
        schema = json.loads(NEMO_SCHEMA_PATH.read_text(encoding="utf-8"))
        authorization_schema = schema["properties"]["authorization"]["oneOf"][1]
        self.assertIn(
            ATTEMPT_14_CORRECTED_BRIDGE_REVISION,
            authorization_schema["properties"]["correctedBridgeRevision"]["enum"],
        )
        attempt_14_rule = next(
            rule
            for rule in schema["allOf"]
            if rule["if"]["properties"]["jobId"].get("const")
            == ATTEMPT_14_REVIEWED_JOB_ID
        )
        properties = attempt_14_rule["then"]["properties"]
        self.assertEqual(
            properties["authorization"]["properties"]["correctedBridgeRevision"][
                "const"
            ],
            ATTEMPT_14_CORRECTED_BRIDGE_REVISION,
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
                exact_lineage[field]["const"], self.attempt_14["lineage"][field]
            )

        terminal_lineage = schema["$defs"]["terminalReceiptFailureLineage"][
            "properties"
        ]
        self.assertIn(
            ATTEMPT_13_REVIEWED_JOB_ID,
            terminal_lineage["predecessorJobId"]["enum"],
        )
        self.assertIn(
            "POST_CLAIM_SFTCONFIG_STRATEGY_COMPATIBILITY",
            terminal_lineage["failurePhase"]["enum"],
        )

    def test_exact_attempt_13_signed_blocked_lineage(self) -> None:
        self.assertEqual(
            self.attempt_14["lineage"],
            {
                "predecessorJobId": ATTEMPT_13_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "de31cbb574cdeeaaf611a25fe1e40616b7fe8d4f6e2e138b66697474f5d800b0"
                ),
                "predecessorPayloadSha256": (
                    "82f619eb1fff6a7617b5761358d2f5c1d8ca62a306eb7cb1bf2570e096b2b9fc"
                ),
                "predecessorEnvelopeRevision": (
                    "b929bae4230ffe39ee63b34b8e9f9974cffc66ca"
                ),
                "predecessorExecutionBridgeRevision": (
                    ATTEMPT_13_CORRECTED_BRIDGE_REVISION
                ),
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
            },
        )

    def test_science_license_and_upload_boundaries_remain_frozen(self) -> None:
        for field in ("source", "base", "dataset", "recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt_14[field], self.attempt_13[field])
        self.assertFalse(self.attempt_14["outputs"]["publishCandidate"])
        self.assertTrue(self.attempt_14["outputs"]["private"])
        for field in ("candidateUpload", "modelCardUpload", "datasetUpload"):
            self.assertFalse(self.attempt_14["ownerDispatch"][field])

    def test_attempt_13_and_14_signed_bytes_are_preserved(self) -> None:
        expected = {
            "jobspecs/nemo-v3-20260731-attempt-13-reviewed.json": (
                "bd394cbb68f60ac181333156cb53d9c0074b234352843aa976533021f5f396e5"
            ),
            "queue/pending/job-2026-nemo-v3-governed-attempt-13.json": (
                "de31cbb574cdeeaaf611a25fe1e40616b7fe8d4f6e2e138b66697474f5d800b0"
            ),
            "queue/evidence/job-2026-nemo-v3-governed-attempt-13.json": (
                "8d9c7e4b37138a2b61de9e15f3c622dc1291f2c38f909a7a9f48115385831c4a"
            ),
            "jobspecs/nemo-v3-20260731-attempt-14-reviewed.json": (
                "99e293ab4c2dd4282bd39a5f741b8359652792c68215c0e7100114a77bbacdf6"
            ),
            "queue/pending/job-2026-nemo-v3-governed-attempt-14.json": (
                "207f0c58525f042d31a748404d0acb678f5fd83722d2a3eacf8399e4e34c9f82"
            ),
        }
        for path, digest in expected.items():
            with self.subTest(path=path):
                content = subprocess.check_output(
                    ["git", "cat-file", "blob", f"HEAD:{path}"], cwd=ROOT
                )
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)

    def test_signed_attempt_14_is_quarantined_with_exact_blocked_truth(self) -> None:
        self.assertTrue(ATTEMPT_14_QUEUE.exists())
        self.assertEqual(
            hashlib.sha256(ATTEMPT_14_QUEUE.read_bytes()).hexdigest(),
            "207f0c58525f042d31a748404d0acb678f5fd83722d2a3eacf8399e4e34c9f82",
        )
        payload_sha256 = hashlib.sha256(
            nemo_v3_status.signer_canonicalize(self.attempt_14).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            payload_sha256,
            "162354602784e8a1cbcecbbfc8a5d7cc9af6be2dd58c66fae442d4f5a292f1da",
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_14_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 31, 12, 52, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUARANTINED_NEVER_DISPATCH")
        self.assertTrue(report["terminal"])
        self.assertTrue(report["queue"]["present"])
        self.assertTrue(report["queue"]["valid"])
        self.assertEqual(report["queue"]["engine_key_id"], COORDINATED_ENGINE_KEY_ID)
        self.assertEqual(report["queue"]["payload_sha256"], payload_sha256)
        self.assertFalse(report["receipt"]["present"])
        self.assertFalse(report["reviewed_spec"]["candidate_publication_enabled"])
        self.assertTrue(report["quarantine"]["present"])
        self.assertTrue(report["quarantine"]["valid"], report["quarantine"]["error"])

        quarantine = json.loads(ATTEMPT_14_QUARANTINE.read_text(encoding="utf-8"))
        self.assertEqual(
            quarantine["status"],
            [
                "META_TENSOR_MATERIALIZATION_BLOCKED",
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
            ATTEMPT_15_REVIEWED_JOB_ID,
        )

        evidence_record = json.loads(ATTEMPT_14_EVIDENCE.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(ATTEMPT_14_EVIDENCE.read_bytes()).hexdigest(),
            "430aa2494b6b1bbcae45f99409075cfbe525ab628582806e3be1c8ae18204bc4",
        )
        evidence = evidence_record["executionEvidence"]
        self.assertEqual(evidence["workflowRunId"], "30634484969")
        self.assertEqual(evidence["workflowJobId"], "91168515330")
        self.assertEqual(
            evidence["attemptClaimSha256"],
            "fc93d880beb0ff183e7da4f7a9a42f0fd075addfc07056e9d260539d9f1dfd92",
        )
        self.assertEqual(
            evidence["prefetchReceiptSha256"],
            "bd315a4a97356451781ceee3e390b847c559381c69eaf50d5efed8c191d2e28c",
        )
        self.assertEqual(
            evidence["receiptRevision"],
            "8c504d466d6b1b3fb0a755768341a34e58b82c11",
        )
        self.assertEqual(
            evidence["receiptFileSha256"],
            "f45c7b319f5f762d03b100149732a4287dfda0d7c91046f21d580fc6f7684ecd",
        )
        self.assertEqual(
            evidence["receiptBodySha256"],
            "cb4dc5cce83797f5d39f86f1c7078230344dc176c854dd3f07988177cafd2500",
        )
        self.assertEqual(evidence["receiptKeyId"], "167c14fbddbe97cc")
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

    def test_exact_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            (("source", "revision"), "0" * 40),
            (("base", "licenseId"), "nvidia-open-model-license"),
            (("ownerDispatch", "workflowBlob"), "1" * 40),
            (("lineage", "predecessorEnvelopeSha256"), "3" * 64),
            (("lineage", "predecessorJobId"), ATTEMPT_13_REVIEWED_JOB_ID + "-old"),
            (("lineage", "claimCreated"), False),
            (("lineage", "trainingStarted"), True),
            (("lineage", "receiptIntentProduced"), False),
            (("lineage", "successorGeneration"), 13),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.attempt_14)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

        runtime_drift = copy.deepcopy(self.attempt_14)
        runtime_drift["authorization"]["correctedBridgeRevision"] = "2" * 40
        self.assertIs(validate_nemo_v3_spec(runtime_drift), runtime_drift)
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                runtime_drift,
                expected_execution_bridge_revision=ATTEMPT_14_CORRECTED_BRIDGE_REVISION,
            )

        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_14,
                expected_execution_bridge_revision="4" * 40,
            )

        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_13,
                expected_execution_bridge_revision=ATTEMPT_13_CORRECTED_BRIDGE_REVISION,
            )
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_14,
                expected_execution_bridge_revision=ATTEMPT_14_CORRECTED_BRIDGE_REVISION,
            )

    def test_signer_and_status_are_locked_to_attempt_14(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260731-attempt-14-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-14.json",
            "tests/test_reviewed_nemo_v3_attempt_14_spec.py",
            "attempt_id: attempt-14-sftconfig-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIn(
            "signer is locked to ${ATTEMPT_14_REVIEWED_JOB_ID}",
            SIGNER_SOURCE,
        )
        self.assertIn("'job-2026-nemo-v3-governed-attempt-13',", SIGNER_SOURCE)
        self.assertIn("'job-2026-nemo-v3-governed-attempt-14',", SIGNER_SOURCE)
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
