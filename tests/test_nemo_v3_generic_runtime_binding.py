from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import nemo_v3_contract  # noqa: E402
import finalize_nemo_v3_receipt  # noqa: E402
import prefetch_nemo_v3  # noqa: E402
import runjob_nemo_v3  # noqa: E402
from frontier_contract import ContractError  # noqa: E402

ATTEMPT_15_PATH = ROOT / "jobspecs" / "nemo-v3-20260731-attempt-15-reviewed.json"
ATTEMPT_15_RUNTIME = "60b9894efe9e0e782999aaa4ee5b0d668e7a9b63"
ATTEMPT_16_RUNTIME = "a" * 40
ATTEMPT_17_RUNTIME = "b" * 40


def attempt_16_spec() -> dict:
    spec = json.loads(ATTEMPT_15_PATH.read_text(encoding="utf-8"))
    spec["jobId"] = nemo_v3_contract.ATTEMPT_16_REVIEWED_JOB_ID
    spec["authorization"]["correctedBridgeRevision"] = ATTEMPT_16_RUNTIME
    spec["lineage"] = {
        "predecessorJobId": nemo_v3_contract.ATTEMPT_15_REVIEWED_JOB_ID,
        "predecessorEnvelopeSha256": (
            "93d5effe94740af9135c3ffa379c85df1aa88e6ad5717bc6421266d21bb9dbe7"
        ),
        "predecessorPayloadSha256": (
            "9c55b95627b93e522eaebec5cb9e837b46d8e368065470aa45f55f488aeff873"
        ),
        "predecessorEnvelopeRevision": ("7f42bad2cb7c762f8eb771922a0ba6e94c96e908"),
        "predecessorExecutionBridgeRevision": ATTEMPT_15_RUNTIME,
        "transportEvidenceUrl": (
            "https://github.com/szl-holdings/a11oy/actions/runs/30641766033"
        ),
        "failurePhase": "PRE_CLAIM_AUTHENTICATED_PREFETCH_RUNTIME_BINDING",
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
    }
    return spec


def attempt_17_spec() -> dict:
    spec = attempt_16_spec()
    spec["jobId"] = nemo_v3_contract.ATTEMPT_17_REVIEWED_JOB_ID
    spec["source"]["revision"] = nemo_v3_contract.ATTEMPT_17_A11OY_SOURCE_REVISION
    spec["ownerDispatch"]["workflowBlob"] = (
        nemo_v3_contract.ATTEMPT_17_OWNER_WORKFLOW_BLOB
    )
    spec["authorization"]["settledA11oyRelockRunUrl"] = (
        nemo_v3_contract.ATTEMPT_17_A11OY_RELOCK_RUN_URL
    )
    spec["authorization"]["correctedBridgeRevision"] = ATTEMPT_17_RUNTIME
    spec["lineage"] = {
        "predecessorJobId": nemo_v3_contract.ATTEMPT_16_REVIEWED_JOB_ID,
        "predecessorEnvelopeSha256": (
            "5f657aebb650c6a9c19b4b52e710236220fe7ab89e6a50488ee270017a78f756"
        ),
        "predecessorPayloadSha256": (
            "0b80bc0e42edd75de9e63f9f74f53df1d10c328d89b84c8481834a27fa4111f8"
        ),
        "predecessorEnvelopeRevision": "0939008a73fa8b1912c842a304c5d0204a5b9d57",
        "predecessorExecutionBridgeRevision": (
            "b99f37260bcabf7f5c98cddbc5988a3ba87b766e"
        ),
        "transportEvidenceUrl": "https://github.com/szl-holdings/a11oy/pull/1217",
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
    }
    return spec


class GenericRuntimeBindingTests(unittest.TestCase):
    def test_execution_revision_propagates_prefetch_claim_runner_finalizer(
        self,
    ) -> None:
        spec = attempt_16_spec()
        launcher = (ROOT / "laptop" / "run_nemo_v3_isolated.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("executionBridgeRevision = $BridgeRevision", launcher)
        self.assertIn(
            '"--env", "SZL_EXECUTION_BRIDGE_REVISION=$BridgeRevision"', launcher
        )

        with tempfile.TemporaryDirectory() as temporary:
            envelope = pathlib.Path(temporary) / "attempt16.json"
            envelope.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(
                    nemo_v3_contract, "quarantine_policy", return_value=None
                ),
                mock.patch.object(
                    prefetch_nemo_v3,
                    "load_pin",
                    return_value={"keyId": nemo_v3_contract.COORDINATED_ENGINE_KEY_ID},
                ),
                mock.patch.object(
                    prefetch_nemo_v3,
                    "verify_envelope",
                    return_value=(
                        spec,
                        b"attempt-16-payload",
                        nemo_v3_contract.NEMO_V3_PAYLOAD_TYPE,
                    ),
                ),
            ):
                observed, _ = prefetch_nemo_v3.load_verified_job(
                    envelope,
                    pathlib.Path(temporary) / "engine.json",
                    expected_job_id=spec["jobId"],
                    expected_source_revision=spec["source"]["revision"],
                    expected_workflow_blob=spec["ownerDispatch"]["workflowBlob"],
                    expected_execution_bridge_revision=ATTEMPT_16_RUNTIME,
                )
                self.assertIs(observed, spec)
                with self.assertRaisesRegex(ContractError, "runtime-bound successor"):
                    prefetch_nemo_v3.load_verified_job(
                        envelope,
                        pathlib.Path(temporary) / "engine.json",
                        expected_job_id=spec["jobId"],
                        expected_source_revision=spec["source"]["revision"],
                        expected_workflow_blob=spec["ownerDispatch"]["workflowBlob"],
                        expected_execution_bridge_revision="b" * 40,
                    )

            with (
                mock.patch.object(
                    nemo_v3_contract, "quarantine_policy", return_value=None
                ),
                mock.patch.object(
                    runjob_nemo_v3,
                    "load_engine_pin_for_envelope",
                    return_value={"keyId": nemo_v3_contract.COORDINATED_ENGINE_KEY_ID},
                ),
                mock.patch.object(
                    runjob_nemo_v3,
                    "verify_envelope",
                    return_value=(
                        spec,
                        b"attempt-16-payload",
                        nemo_v3_contract.NEMO_V3_PAYLOAD_TYPE,
                    ),
                ),
                mock.patch.object(
                    runjob_nemo_v3,
                    "_require_remote_code_isolation",
                    side_effect=RuntimeError("stop after exact contract"),
                ),
                mock.patch.dict(
                    "os.environ",
                    {"SZL_EXECUTION_BRIDGE_REVISION": ATTEMPT_16_RUNTIME},
                    clear=False,
                ),
            ):
                self.assertEqual(runjob_nemo_v3.main(str(envelope)), 4)

            claim = {"executionBridgeRevision": ATTEMPT_16_RUNTIME}
            with mock.patch.object(
                nemo_v3_contract, "quarantine_policy", return_value=None
            ):
                finalize_nemo_v3_receipt.require_claim_bound_dispatchable(
                    spec,
                    claim,
                    ATTEMPT_16_RUNTIME,
                )
                with self.assertRaisesRegex(
                    ValueError, "explicit execution Bridge revision"
                ):
                    finalize_nemo_v3_receipt.require_claim_bound_dispatchable(
                        spec,
                        claim,
                        "b" * 40,
                    )

    def test_attempt_15_uses_generic_predecessor_binding(self) -> None:
        spec = json.loads(ATTEMPT_15_PATH.read_text(encoding="utf-8"))
        self.assertNotIn(
            nemo_v3_contract.ATTEMPT_15_REVIEWED_JOB_ID,
            nemo_v3_contract._COORDINATED_JOB_BINDINGS,
        )
        self.assertIs(nemo_v3_contract.validate_nemo_v3_spec(spec), spec)
        binding = nemo_v3_contract._coordinated_job_binding(spec)
        self.assertTrue(binding["runtimeBound"])
        self.assertEqual(binding["successorGeneration"], 15)
        self.assertEqual(
            binding["predecessorJobId"],
            nemo_v3_contract.ATTEMPT_14_REVIEWED_JOB_ID,
        )
        with mock.patch.object(
            nemo_v3_contract, "quarantine_policy", return_value=None
        ):
            nemo_v3_contract.require_nemo_v3_dispatchable(
                spec,
                expected_execution_bridge_revision=ATTEMPT_15_RUNTIME,
            )

    def test_attempt_16_requires_exact_protected_attempt_15_replacement(self) -> None:
        spec = attempt_16_spec()
        self.assertIs(nemo_v3_contract.validate_nemo_v3_spec(spec), spec)
        with self.assertRaisesRegex(ContractError, "quarantined"):
            nemo_v3_contract.require_nemo_v3_dispatchable(
                spec,
                expected_execution_bridge_revision=ATTEMPT_16_RUNTIME,
            )

        policy = nemo_v3_contract.QUARANTINE_POLICIES.pop(
            nemo_v3_contract.ATTEMPT_15_REVIEWED_JOB_ID
        )
        try:
            with self.assertRaisesRegex(ContractError, "predecessor quarantine"):
                nemo_v3_contract.validate_nemo_v3_spec(spec)
        finally:
            nemo_v3_contract.QUARANTINE_POLICIES[
                nemo_v3_contract.ATTEMPT_15_REVIEWED_JOB_ID
            ] = policy

    def test_replacement_and_runtime_mismatches_fail_closed(self) -> None:
        base_policy = nemo_v3_contract.QUARANTINE_POLICIES[
            nemo_v3_contract.ATTEMPT_15_REVIEWED_JOB_ID
        ]
        mutations = {
            "reviewed job": ("reviewedJobId", "job-2026-nemo-v3-governed-attempt-17"),
            "source": ("sourceRevision", "1" * 40),
            "workflow": ("workflowBlob", "2" * 40),
            "version": ("workflowVersion", "nemo-v3-owner-dispatch.mutable"),
            "relock": (
                "settledA11oyRelockRunUrl",
                "https://github.com/szl-holdings/a11oy/actions/runs/1",
            ),
            "key": ("engineKeyId", "0" * 16),
            "spki": ("enginePublicKeySpkiSha256", "0" * 64),
            "generation": ("successorGeneration", 17),
        }
        for name, (field, value) in mutations.items():
            with self.subTest(name=name):
                policy = copy.deepcopy(base_policy)
                policy["replacement"][field] = value
                with mock.patch.dict(
                    nemo_v3_contract.QUARANTINE_POLICIES,
                    {nemo_v3_contract.ATTEMPT_15_REVIEWED_JOB_ID: policy},
                ):
                    with self.assertRaises(ContractError):
                        nemo_v3_contract.validate_nemo_v3_spec(attempt_16_spec())

        with mock.patch.object(
            nemo_v3_contract, "quarantine_policy", return_value=None
        ):
            with self.assertRaisesRegex(ContractError, "runtime-bound successor"):
                nemo_v3_contract.require_nemo_v3_dispatchable(
                    attempt_16_spec(),
                    expected_execution_bridge_revision="b" * 40,
                )
            with self.assertRaises(ContractError):
                nemo_v3_contract.require_nemo_v3_dispatchable(
                    attempt_16_spec(),
                    expected_execution_bridge_revision="main",
                )

    def test_unknown_skipped_replayed_and_path_anomaly_are_rejected(self) -> None:
        unknown = attempt_17_spec()
        unknown["jobId"] = "job-2026-nemo-v3-governed-attempt-18"
        unknown["lineage"]["predecessorJobId"] = (
            nemo_v3_contract.ATTEMPT_17_REVIEWED_JOB_ID
        )
        unknown["lineage"]["successorGeneration"] = 18
        with self.assertRaisesRegex(ContractError, "predecessor quarantine"):
            nemo_v3_contract.validate_nemo_v3_spec(unknown)

        skipped = copy.deepcopy(unknown)
        skipped["lineage"]["predecessorJobId"] = (
            nemo_v3_contract.ATTEMPT_15_REVIEWED_JOB_ID
        )
        with self.assertRaisesRegex(ContractError, "skip a generation"):
            nemo_v3_contract.validate_nemo_v3_spec(skipped)

        with self.assertRaisesRegex(ContractError, "quarantined"):
            nemo_v3_contract.require_nemo_v3_dispatchable(
                json.loads(ATTEMPT_15_PATH.read_text(encoding="utf-8")),
                expected_execution_bridge_revision=ATTEMPT_15_RUNTIME,
            )

        malformed = attempt_16_spec()
        malformed["jobId"] = "job-2026-nemo-v3-governed-attempt-16/../escape"
        with self.assertRaisesRegex(ContractError, "idempotency pattern"):
            nemo_v3_contract.validate_nemo_v3_spec(malformed)

    def test_attempt_16_quarantine_admits_only_exact_attempt_17(self) -> None:
        successor = attempt_17_spec()
        self.assertIs(nemo_v3_contract.validate_nemo_v3_spec(successor), successor)
        nemo_v3_contract.require_nemo_v3_dispatchable(
            successor,
            expected_execution_bridge_revision=ATTEMPT_17_RUNTIME,
        )

        for path, value in (
            (("source", "revision"), "1" * 40),
            (("ownerDispatch", "workflowBlob"), "2" * 40),
            (
                ("authorization", "settledA11oyRelockRunUrl"),
                "https://github.com/szl-holdings/a11oy/actions/runs/1",
            ),
            (("lineage", "eventCreated"), True),
            (("lineage", "workflowRunCreated"), True),
            (("lineage", "transportEvidenceUrl"), "https://example.com/fake"),
        ):
            with self.subTest(path=path):
                mutated = copy.deepcopy(successor)
                mutated[path[0]][path[1]] = value
                with self.assertRaises(ContractError):
                    nemo_v3_contract.validate_nemo_v3_spec(mutated)

        policy = copy.deepcopy(
            nemo_v3_contract.QUARANTINE_POLICIES[
                nemo_v3_contract.ATTEMPT_16_REVIEWED_JOB_ID
            ]
        )
        policy["pre_event_evidence"]["receiptUploaded"] = True
        with mock.patch.dict(
            nemo_v3_contract.QUARANTINE_POLICIES,
            {nemo_v3_contract.ATTEMPT_16_REVIEWED_JOB_ID: policy},
        ):
            with self.assertRaisesRegex(ContractError, "zero-event boundary"):
                nemo_v3_contract.validate_nemo_v3_spec(successor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
