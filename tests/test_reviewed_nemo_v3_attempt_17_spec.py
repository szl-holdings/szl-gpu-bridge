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
GIT_CAT_FILE = [
    "git",
    "-c",
    f"safe.directory={ROOT.as_posix()}",
    "cat-file",
    "blob",
]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "laptop"))

import nemo_v3_status  # noqa: E402
from frontier_contract import ContractError  # noqa: E402
from nemo_v3_contract import (  # noqa: E402
    ATTEMPT_16_REVIEWED_JOB_ID,
    ATTEMPT_17_A11OY_RELOCK_RUN_URL,
    ATTEMPT_17_A11OY_SOURCE_REVISION,
    ATTEMPT_17_CORRECTED_BRIDGE_REVISION,
    ATTEMPT_17_OWNER_WORKFLOW_BLOB,
    ATTEMPT_17_REVIEWED_JOB_ID,
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

ATTEMPT_16_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-16-reviewed.json"
ATTEMPT_17_PATH = ROOT / "jobspecs" / "nemo-v3-20260801-attempt-17-reviewed.json"
ATTEMPT_17_QUEUE = ROOT / "queue" / "pending" / f"{ATTEMPT_17_REVIEWED_JOB_ID}.json"
SCHEMA_PATH = ROOT / "schema" / "nemo-v3-jobspec.v1.json"
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")
SIGNER_SOURCE = (ROOT / "cloud" / "sign-nemo-v3-job.mjs").read_text(encoding="utf-8")


class ReviewedNemoV3Attempt17SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_16 = json.loads(ATTEMPT_16_PATH.read_text(encoding="utf-8"))
        cls.attempt_17 = json.loads(ATTEMPT_17_PATH.read_text(encoding="utf-8"))

    def test_exact_settled_source_workflow_runtime_key_and_license(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_17), self.attempt_17)
        self.assertEqual(self.attempt_17["jobId"], ATTEMPT_17_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_17["source"]["revision"],
            ATTEMPT_17_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            self.attempt_17["ownerDispatch"]["workflowBlob"],
            ATTEMPT_17_OWNER_WORKFLOW_BLOB,
        )
        self.assertEqual(
            self.attempt_17["authorization"]["settledA11oyRelockRunUrl"],
            ATTEMPT_17_A11OY_RELOCK_RUN_URL,
        )
        self.assertEqual(
            self.attempt_17["authorization"]["correctedBridgeRevision"],
            ATTEMPT_17_CORRECTED_BRIDGE_REVISION,
        )
        self.assertEqual(
            self.attempt_17["authorization"]["engineKeyId"],
            COORDINATED_ENGINE_KEY_ID,
        )
        self.assertEqual(
            self.attempt_17["authorization"]["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            self.attempt_17["base"]["licenseId"],
            "nvidia-nemotron-open-model-license",
        )
        require_nemo_v3_dispatchable(
            self.attempt_17,
            expected_execution_bridge_revision=ATTEMPT_17_CORRECTED_BRIDGE_REVISION,
        )

    def test_json_schema_pins_the_exact_attempt_17_binding(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        authorization_schema = schema["properties"]["authorization"]["oneOf"][1]
        properties = authorization_schema["properties"]
        self.assertIn(
            ATTEMPT_17_A11OY_RELOCK_RUN_URL,
            properties["settledA11oyRelockRunUrl"]["enum"],
        )
        self.assertIn(
            ATTEMPT_17_CORRECTED_BRIDGE_REVISION,
            properties["correctedBridgeRevision"]["enum"],
        )
        attempt_17_rule = next(
            rule
            for rule in schema["allOf"]
            if rule["if"]["properties"]["jobId"].get("const")
            == ATTEMPT_17_REVIEWED_JOB_ID
        )
        exact = attempt_17_rule["then"]["properties"]
        self.assertEqual(
            exact["source"]["properties"]["revision"]["const"],
            ATTEMPT_17_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            exact["authorization"]["properties"]["correctedBridgeRevision"]["const"],
            ATTEMPT_17_CORRECTED_BRIDGE_REVISION,
        )
        exact_lineage = exact["lineage"]["allOf"][1]["properties"]
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
                exact_lineage[field]["const"], self.attempt_17["lineage"][field]
            )

    def test_exact_attempt_16_zero_event_lineage(self) -> None:
        self.assertEqual(
            self.attempt_17["lineage"],
            {
                "predecessorJobId": ATTEMPT_16_REVIEWED_JOB_ID,
                "predecessorEnvelopeSha256": (
                    "5f657aebb650c6a9c19b4b52e710236220fe7ab89e6a50488ee270017a78f756"
                ),
                "predecessorPayloadSha256": (
                    "0b80bc0e42edd75de9e63f9f74f53df1d10c328d89b84c8481834a27fa4111f8"
                ),
                "predecessorEnvelopeRevision": (
                    "0939008a73fa8b1912c842a304c5d0204a5b9d57"
                ),
                "predecessorExecutionBridgeRevision": (
                    "b99f37260bcabf7f5c98cddbc5988a3ba87b766e"
                ),
                "transportEvidenceUrl": (
                    "https://github.com/szl-holdings/a11oy/pull/1217"
                ),
                "failurePhase": "PRE_DISPATCH_VALIDATOR_REJECTED",
                "successorGeneration": 17,
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

    def test_science_license_and_publication_boundaries_remain_frozen(self) -> None:
        for field in ("base", "dataset", "recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt_17[field], self.attempt_16[field])
        self.assertFalse(self.attempt_17["outputs"]["publishCandidate"])
        self.assertTrue(self.attempt_17["outputs"]["private"])
        for field in ("candidateUpload", "modelCardUpload", "datasetUpload"):
            self.assertFalse(self.attempt_17["ownerDispatch"][field])

    def test_attempt_16_terminal_evidence_bytes_are_immutable(self) -> None:
        expected = {
            "jobspecs/nemo-v3-20260731-attempt-16-reviewed.json": (
                "1daa8ea3a30a1d497f60431f9f4a33a9edd5d286236f3e8bf44240ef8630c5da"
            ),
            "queue/pending/job-2026-nemo-v3-governed-attempt-16.json": (
                "5f657aebb650c6a9c19b4b52e710236220fe7ab89e6a50488ee270017a78f756"
            ),
            "queue/evidence/job-2026-nemo-v3-governed-attempt-16.json": (
                "efff8d60590b317a873772e72e401165300331daf5431136ec18a1ddcab85389"
            ),
            "queue/quarantine/job-2026-nemo-v3-governed-attempt-16.json": (
                "07217325859fdef6db30f915d82eca18dd0dfdabc8dc104fab9d3a994a666990"
            ),
        }
        for path, digest in expected.items():
            with self.subTest(path=path):
                content = subprocess.check_output(
                    [*GIT_CAT_FILE, f"HEAD:{path}"], cwd=ROOT
                )
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)

    def test_plaintext_phase_waits_for_separate_exclusive_signature(self) -> None:
        self.assertFalse(ATTEMPT_17_QUEUE.exists())
        self.assertIn(
            "signer is locked to ${ATTEMPT_17_REVIEWED_JOB_ID}", SIGNER_SOURCE
        )
        self.assertIn("'job-2026-nemo-v3-governed-attempt-16',", SIGNER_SOURCE)
        self.assertNotIn("'job-2026-nemo-v3-governed-attempt-17',\n]);", SIGNER_SOURCE)
        self.assertIn("{ flag: 'wx' }", SIGNER_SOURCE)

        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_17_PATH,
            receipt_loader=lambda _spec, _token: self.fail(
                "unsigned plaintext must not query a receipt"
            ),
            now=datetime(2026, 8, 1, 17, 10, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "AWAITING_ENGINE_SIGNATURE")
        self.assertFalse(report["terminal"])
        self.assertFalse(report["queue"]["present"])

    def test_exact_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            (("source", "revision"), "0" * 40),
            (("base", "licenseId"), "nvidia-open-model-license"),
            (("ownerDispatch", "workflowBlob"), "1" * 40),
            (("authorization", "settledA11oyRelockRunUrl"), "https://example.com"),
            (("authorization", "correctedBridgeRevision"), "2" * 40),
            (("lineage", "predecessorEnvelopeSha256"), "3" * 64),
            (("lineage", "eventCreated"), True),
            (("lineage", "workflowRunCreated"), True),
            (("lineage", "successorGeneration"), 18),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.attempt_17)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

        with self.assertRaisesRegex(ContractError, "runtime-bound successor"):
            require_nemo_v3_dispatchable(
                self.attempt_17,
                expected_execution_bridge_revision="4" * 40,
            )

    def test_status_workflow_tracks_attempt_17_without_dispatch(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260801-attempt-17-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-17.json",
            "tests/test_reviewed_nemo_v3_attempt_17_spec.py",
            "attempt_id: attempt-17-settled-runtime-successor",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
