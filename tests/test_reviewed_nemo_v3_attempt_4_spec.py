from __future__ import annotations

import base64
import copy
import hashlib
import json
import pathlib
import sys
import unittest
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "laptop"))

import nemo_v3_status  # noqa: E402
from frontier_contract import ContractError, derive_key_id  # noqa: E402
from nemo_v3_contract import (  # noqa: E402
    COORDINATED_ENGINE_KEY_ID,
    COORDINATED_ENGINE_SPKI_SHA256,
    CORRECTED_BRIDGE_REVISION,
    FINAL_A11OY_SOURCE_REVISION,
    FINAL_OWNER_WORKFLOW_BLOB,
    FUTURE_REVIEWED_JOB_ID,
    NEXT_REVIEWED_JOB_ID,
    PROVISIONAL_ENGINE_KEY_ID,
    SETTLED_A11OY_RELOCK_RUN_URL,
    SETTLED_A11OY_SOURCE_REVISION,
    SETTLED_OWNER_WORKFLOW_BLOB,
    require_nemo_v3_dispatchable,
    validate_nemo_v3_spec,
)

SUCCESSOR_3_PATH = ROOT / "jobspecs" / "nemo-v3-20260730-successor-3-reviewed.json"
ATTEMPT_4_PATH = ROOT / "jobspecs" / "nemo-v3-20260730-attempt-4-reviewed.json"
ATTEMPT_4_QUEUE = ROOT / "queue" / "pending" / f"{NEXT_REVIEWED_JOB_ID}.json"
EXPECTED_PAYLOAD_SHA256 = (
    "14441cf982b177c1b613e56e63eae8be3e589ae35444826b40731c32312268e5"
)
EXPECTED_ENVELOPE_FILE_SHA256 = (
    "e240a176849b1f6c0d453ac55277cd7732b3a302ea9679db78d3c612501f27f2"
)
STATUS_WORKFLOW = (
    ROOT / ".github" / "workflows" / "nemo-v3-attempt-status.yml"
).read_text(encoding="utf-8")


class ReviewedNemoV3Attempt4SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.successor_3 = json.loads(SUCCESSOR_3_PATH.read_text(encoding="utf-8"))
        cls.attempt_4 = json.loads(ATTEMPT_4_PATH.read_text(encoding="utf-8"))

    def test_contract_binds_final_source_workflow_key_and_correction(self) -> None:
        self.assertIs(validate_nemo_v3_spec(self.attempt_4), self.attempt_4)
        self.assertEqual(self.attempt_4["jobId"], NEXT_REVIEWED_JOB_ID)
        self.assertEqual(
            self.attempt_4["source"]["revision"],
            SETTLED_A11OY_SOURCE_REVISION,
        )
        self.assertEqual(
            self.attempt_4["ownerDispatch"]["workflowBlob"],
            SETTLED_OWNER_WORKFLOW_BLOB,
        )
        authorization = self.attempt_4["authorization"]
        self.assertEqual(authorization["engineKeyId"], COORDINATED_ENGINE_KEY_ID)
        self.assertEqual(
            authorization["enginePublicKeySpkiSha256"],
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        self.assertEqual(
            authorization["provisionalEngineKeyId"],
            PROVISIONAL_ENGINE_KEY_ID,
        )
        self.assertEqual(authorization["provisionalKeyStatus"], "VERIFY_ONLY")
        self.assertEqual(authorization["coordinationMode"], "FINAL_ACTIVE_TRUST_ROOT")
        self.assertEqual(
            authorization["settledA11oyRelockRunUrl"],
            SETTLED_A11OY_RELOCK_RUN_URL,
        )
        self.assertIs(authorization["cryptographicContinuityClaimed"], False)
        self.assertEqual(
            authorization["correctedBridgeRevision"],
            CORRECTED_BRIDGE_REVISION,
        )
        self.assertEqual(self.attempt_4["lineage"]["successorGeneration"], 4)

    def test_science_inputs_and_receipt_only_effects_remain_frozen(self) -> None:
        for field in ("base", "recipe", "gates", "evaluation"):
            self.assertEqual(self.attempt_4[field], self.successor_3[field])
        earlier_dataset = copy.deepcopy(self.successor_3["dataset"])
        settled_dataset = copy.deepcopy(self.attempt_4["dataset"])
        earlier_dataset.pop("provenance")
        settled_dataset.pop("provenance")
        self.assertEqual(settled_dataset, earlier_dataset)
        self.assertEqual(
            {
                key: self.attempt_4["outputs"][key]
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
                key: self.attempt_4["ownerDispatch"][key]
                for key in ("candidateUpload", "modelCardUpload", "datasetUpload")
            },
            {
                "candidateUpload": False,
                "modelCardUpload": False,
                "datasetUpload": False,
            },
        )

    def test_quarantine_records_bind_the_exact_superseded_lineage(self) -> None:
        attempt_4_replacement = {
            "sourceRevision": SETTLED_A11OY_SOURCE_REVISION,
            "workflowBlob": SETTLED_OWNER_WORKFLOW_BLOB,
            "engineKeyId": COORDINATED_ENGINE_KEY_ID,
            "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
            "reviewedJobId": NEXT_REVIEWED_JOB_ID,
        }
        attempt_5_replacement = {
            "sourceRevision": FINAL_A11OY_SOURCE_REVISION,
            "workflowBlob": FINAL_OWNER_WORKFLOW_BLOB,
            "engineKeyId": COORDINATED_ENGINE_KEY_ID,
            "enginePublicKeySpkiSha256": COORDINATED_ENGINE_SPKI_SHA256,
            "reviewedJobId": FUTURE_REVIEWED_JOB_ID,
        }
        expected = {
            "job-2026-nemo-v3-governed-attempt-2": (
                ["STALE_SOURCE", "RETIRED_KEY", "NEVER_DISPATCH"],
                "e74ecaea040c2abb52a5613c32e0648994f96ff39910c70e1fcc3e23fc053724",
                "84a808615ba1693935eee8cc9fa1a4c5a83d119b79ad7e9437380ec73756b90d",
                attempt_4_replacement,
            ),
            "job-2026-nemo-v3-governed-successor-3": (
                [
                    "UNAUTHORIZED_PROVISIONAL_KEY",
                    "STALE_SOURCE",
                    "NEVER_DISPATCH",
                ],
                "bb624d301f23552617566f57167a12360bbba27afebee086a8262b1be7ee6eaa",
                "f20bf865dca5413262e5fd3733df112486aec72bb9b47932083ffecb2470a415",
                attempt_4_replacement,
            ),
            "job-2026-nemo-v3-governed-attempt-4": (
                [
                    "STALE_SOURCE",
                    "TRANSPORT_UNREPRESENTABLE",
                    "NEVER_DISPATCH",
                ],
                EXPECTED_ENVELOPE_FILE_SHA256,
                EXPECTED_PAYLOAD_SHA256,
                attempt_5_replacement,
            ),
        }
        for job_id, (
            statuses,
            envelope_sha,
            payload_sha,
            replacement,
        ) in expected.items():
            with self.subTest(job_id=job_id):
                record = json.loads(
                    (ROOT / "queue" / "quarantine" / f"{job_id}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(record["status"], statuses)
                self.assertEqual(record["queueFileSha256"], envelope_sha)
                self.assertEqual(record["signedPayloadSha256"], payload_sha)
                self.assertEqual(record["replacement"], replacement)

    def test_public_pin_derives_the_exact_active_identity(self) -> None:
        pin = json.loads(
            (ROOT / "keys" / "engine_pubkey_b8041281c81c4caa.json").read_text(
                encoding="utf-8"
            )
        )
        spki = base64.b64decode(pin["publicKeySpkiBase64"], validate=True)
        self.assertEqual(derive_key_id(spki), COORDINATED_ENGINE_KEY_ID)
        self.assertEqual(
            hashlib.sha256(spki).hexdigest(),
            COORDINATED_ENGINE_SPKI_SHA256,
        )
        keyring = json.loads(
            (ROOT / "keys" / "engine_keyring.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            keyring["keys"][PROVISIONAL_ENGINE_KEY_ID]["status"],
            "VERIFY_ONLY",
        )
        self.assertEqual(
            keyring["keys"][COORDINATED_ENGINE_KEY_ID]["status"],
            "ACTIVE",
        )

    def test_signed_queue_is_exact_quarantined_evidence_without_receipt(self) -> None:
        self.assertTrue(ATTEMPT_4_QUEUE.is_file())
        self.assertEqual(
            hashlib.sha256(ATTEMPT_4_QUEUE.read_bytes()).hexdigest(),
            EXPECTED_ENVELOPE_FILE_SHA256,
        )
        report = nemo_v3_status.evaluate(
            root=ROOT,
            spec_path=ATTEMPT_4_PATH,
            receipt_loader=lambda _spec, _token: None,
            now=datetime(2026, 7, 30, 19, 30, tzinfo=timezone.utc),
        )
        self.assertEqual(report["status"], "QUARANTINED_NEVER_DISPATCH")
        self.assertTrue(report["terminal"])
        self.assertTrue(report["queue"]["present"])
        self.assertTrue(report["queue"]["valid"], report["queue"]["error"])
        self.assertEqual(
            report["queue"]["path"],
            "queue/pending/job-2026-nemo-v3-governed-attempt-4.json",
        )
        self.assertEqual(
            report["queue"]["payload_sha256"],
            EXPECTED_PAYLOAD_SHA256,
        )
        self.assertEqual(
            report["queue"]["engine_key_id"],
            COORDINATED_ENGINE_KEY_ID,
        )
        self.assertTrue(report["quarantine"]["present"])
        self.assertTrue(report["quarantine"]["valid"], report["quarantine"]["error"])
        self.assertFalse(report["receipt"]["present"])
        self.assertEqual(
            report["reviewed_spec"]["sha256"],
            EXPECTED_PAYLOAD_SHA256,
        )
        with self.assertRaisesRegex(ContractError, "NEVER_DISPATCH"):
            require_nemo_v3_dispatchable(self.attempt_4)

    def test_coordinated_bindings_fail_closed_on_drift(self) -> None:
        mutations = (
            ("job identity", ("jobId",), "job-2026-nemo-v3-governed-attempt-5"),
            ("source revision", ("source", "revision"), "0" * 40),
            ("workflow blob", ("ownerDispatch", "workflowBlob"), "1" * 40),
            (
                "engine SPKI",
                ("authorization", "enginePublicKeySpkiSha256"),
                "2" * 64,
            ),
            (
                "provisional status",
                ("authorization", "provisionalKeyStatus"),
                "ACTIVE",
            ),
            (
                "continuity claim",
                ("authorization", "cryptographicContinuityClaimed"),
                True,
            ),
            (
                "bridge correction",
                ("authorization", "correctedBridgeRevision"),
                "3" * 40,
            ),
            ("generation", ("lineage", "successorGeneration"), 3),
        )
        for label, path, value in mutations:
            with self.subTest(label=label):
                mutated = copy.deepcopy(self.attempt_4)
                if len(path) == 1:
                    mutated[path[0]] = value
                else:
                    mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    validate_nemo_v3_spec(mutated)

    def test_status_workflow_tracks_signed_queue_without_dispatch(self) -> None:
        for expected in (
            "jobspecs/nemo-v3-20260730-attempt-4-reviewed.json",
            "queue/pending/job-2026-nemo-v3-governed-attempt-4.json",
            "queue/quarantine/job-2026-nemo-v3-governed-attempt-4.json",
            "tests/test_reviewed_nemo_v3_attempt_4_spec.py",
            "attempt_id: attempt-4-coordinated-recovery",
        ):
            self.assertIn(expected, STATUS_WORKFLOW)
        self.assertNotIn("repository_dispatch", STATUS_WORKFLOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
