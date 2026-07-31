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
    ATTEMPT_15_REVIEWED_JOB_ID,
    ATTEMPT_16_REVIEWED_JOB_ID,
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
    EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
    EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

ATTEMPT_15_CORRECTED_BRIDGE_REVISION = "60b9894efe9e0e782999aaa4ee5b0d668e7a9b63"
ATTEMPT_16_CORRECTED_BRIDGE_REVISION = "b99f37260bcabf7f5c98cddbc5988a3ba87b766e"
ATTEMPT_15_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-15-reviewed.json"
ATTEMPT_16_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-16-reviewed.json"
ATTEMPT_16_QUEUE = ROOT / "queue" / "pending" / f"{ATTEMPT_16_REVIEWED_JOB_ID}.json"
SCHEMA_PATH = ROOT / "schema" / "nemo-v3-jobspec.v1.json"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt16SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_15 = json.loads(ATTEMPT_15_PATH.read_text(encoding="utf-8"))
        cls.attempt_16 = json.loads(ATTEMPT_16_PATH.read_text(encoding="utf-8"))

    def test_exact_source_workflow_runtime_key_and_license(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_16), self.attempt_16)
        self.assertEqual(self.attempt_16["jobId"], ATTEMPT_16_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_16["source"]["revision"],
            EXPLICIT_RUNTIME_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            self.attempt_16["ownerDispatch"]["workflowBlob"],
            EXPLICIT_RUNTIME_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_16["authorization"]["settledA11oyRelockRunUrl"],
            EXPLICIT_RUNTIME_A11OY_RELOCK_RUN_URL,
        )
        self.assertEqual(
            self.attempt_16["authorization"]["correctedBridgeRevision"],
            ATTEMPT_16_CORRECTED_BRIDGE_REVISION,
        )
        self.assertEqual(
            self.attempt_16["authorization"]["engineKeyId"],
            COORDINATED_ENGINE_KEY_ID,
        )
        self.assertEqual(
            self.attempt_16["authorization"]["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            self.attempt_16["base"]["licenseId"],
            "nvidia-nemotron-open-model-license",
        )
        require_nemo_v3_dispatchable(
            self.attempt_16,
            expected_execution_bridge_revision=ATTEMPT_16_CORRECTED_BRIDGE_REVISION,
        )

    def test_json_schema_admits_only_the_exact_attempt_16_binding(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        authorization_schema = schema["properties"]["authorization"]["oneOf"][1]
        self.assertIn(
            ATTEMPT_16_CORRECTED_BRIDGE_REVISION,
            authorization_schema["properties"]["correctedBridgeRevision"]["enum"],
        )
        attempt_16_rule = next(
            rule
            for rule in schema["allOf"]
            if rule["if"]["properties"]["jobId"].get("const")
            == ATTEMPT_16_REVIEWED_JOB_ID
        )
        properties = attempt_16_rule["then"]["properties"]
        self.assertEqual(
            properties["authorization"]["properties"]["correctedBridgeRevision"][
                "const"
            ],
            ATTEMPT_16_CORRECTED_BRIDGE_REVISION,
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
            "eventCreated",
            "workflowRunCreated",
            "claimCreated",
            "trainingStarted",
            "modelRepositoryCodeImported",
            "holdoutsAccessed",
            "candidateProduced",
            "receiptIntentProduced",
            "terminalLedgerWritten",
        ):
            self.assertEqual(
                exact_lineage[field]["const"], self.attempt_16["lineage"][field]
            )

        transport_lineage = schema["$defs"]["transportFailureLineage"]["properties"]
        self.assertIn(
            self.attempt_16["lineage"]["transportEvidenceUrl"],
            transport_lineage["transportEvidenceUrl"]["enum"],
        )
        self.assertIn(
            self.attempt_16["lineage"]["failurePhase"],
            transport_lineage["failurePhase"]["enum"],
        )

    def test_exact_attempt_15_zero_effect_lineage(self) -> None:
        self.assertEqual(
            self.attempt_16["lineage"],
            {
                "predecessorJobId": ATTEMPT_15_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "93d5effe94740af9135c3ffa379c85df1aa88e6ad5717bc6421266d21bb9dbe7"
                ),
                "predecessorPayloadSha256": (
                    "9c55b95627b93e522eaebec5cb9e837b46d8e368065470aa45f55f488aeff873"
                ),
                "predecessorEnvelopeRevision": (
                    "7f42bad2cb7c762f8eb771922a0ba6e94c96e908"
                ),
                "predecessorExecutionBridgeRevision": (
                    ATTEMPT_15_CORRECTED_BRIDGE_REVISION
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/actions/runs/30641766033"
                ),
                "failurePhase": ("PRE_CLAIM_AUTHENTICATED_PREFETCH_RUNTIME_BINDING"),
                "successorGeneration": 16,
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
        for field in ("source", "base", "dataset", "recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt_16[field], self.attempt_15[field])
        self.assertFalse(self.attempt_16["outputs"]["publishCandidate"])
        self.assertTrue(self.attempt_16["outputs"]["private"])
        for field in ("candidateUpload", "modelCardUpload", "datasetUpload"):
            self.assertFalse(self.attempt_16["ownerDispatch"][field])

    def test_attempt_15_signed_and_evidence_bytes_are_preserved(self) -> None:
        expected = {
            "jobspecs/nemo-v3-20260731-attempt-15-reviewed.json": (
                "6fd61348cb0cba5fdf338935574deaec827da9ee1f827d8a43e6382993519198"
            ),
            "queue/pending/job-2026-nemo-v3-governed-attempt-15.json": (
                "93d5effe94740af9135c3ffa379c85df1aa88e6ad5717bc6421266d21bb9dbe7"
            ),
            "queue/evidence/job-2026-nemo-v3-governed-attempt-15.json": (
                "a5af132a89fdf26f2857c06891711e56843c6708d5db14d1f6bf20fc3cf81779"
            ),
            "queue/quarantine/job-2026-nemo-v3-governed-attempt-15.json": (
                "4023fe3234de29211bbdb20b628616b8e3b4654cecbc304b67c3bbb9694d742f"
            ),
        }
        for path, digest in expected.items():
            with self.subTest(path=path):
                content = subprocess.check_output(
                    ["git", "cat-file", "blob", f"HEAD:{path}"], cwd=ROOT
                )
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)

    def test_plaintext_attempt_16_waits_for_separate_engine_signature(self) -> None:
        self.assertFalse(ATTEMPT_16_QUEUE.exists())
        reviewed_bytes = subprocess.check_output(
            [
                "git",
                "cat-file",
                "blob",
                "HEAD:jobspecs/nemo-v3-20260731-attempt-16-reviewed.json",
            ],
            cwd=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(reviewed_bytes).hexdigest(),
            "1daa8ea3a30a1d497f60431f9f4a33a9edd5d286236f3e8bf44240ef8630c5da",
        )
        payload_sha256 = hashlib.sha256(
            nemo_v3_status.signer_canonicalize(self.attempt_16).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            payload_sha256,
            "0b80bc0e42edd75de9e63f9f74f53df1d10c328d89b84c8481834a27fa4111f8",
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_16_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 31, 16, 5, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "AWAITING_ENGINE_SIGNATURE")
        self.assertFalse(report["terminal"])
        self.assertFalse(report["queue"]["present"])
        self.assertFalse(report["receipt"]["present"])
        self.assertFalse(report["reviewed_spec"]["candidate_publication_enabled"])

    def test_exact_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            (("source", "revision"), "0" * 40),
            (("base", "licenseId"), "nvidia-open-model-license"),
            (("ownerDispatch", "workflowBlob"), "1" * 40),
            (("lineage", "predecessorEnvelopeSha256"), "3" * 64),
            (("lineage", "claimCreated"), True),
            (("lineage", "trainingStarted"), True),
            (("lineage", "receiptIntentProduced"), True),
            (("lineage", "successorGeneration"), 15),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.attempt_16)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

        with self.assertRaisesRegex(ContractError, "runtime-bound successor"):
            require_nemo_v3_dispatchable(
                self.attempt_16,
                expected_execution_bridge_revision="4" * 40,
            )
        with self.assertRaisesRegex(ContractError, "quarantined"):
            require_nemo_v3_dispatchable(
                self.attempt_15,
                expected_execution_bridge_revision=(
                    ATTEMPT_15_CORRECTED_BRIDGE_REVISION
                ),
            )

    def test_signer_and_status_are_locked_to_attempt_16(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260731-attempt-16-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-16.json",
            "tests/test_reviewed_nemo_v3_attempt_16_spec.py",
            "attempt_id: attempt-16-runtime-binding-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertIn(
            "signer is locked to ${ATTEMPT_16_REVIEWED_JOB_ID}",
            SIGNER_SOURCE,
        )
        self.assertIn("'job-2026-nemo-v3-governed-attempt-15',", SIGNER_SOURCE)
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
