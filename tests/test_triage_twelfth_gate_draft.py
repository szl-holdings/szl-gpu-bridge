"""Offline source-binding tests; never dispatch or qualify a study workload."""

import json
import unittest
from pathlib import Path


DRAFT = (
    Path(__file__).resolve().parents[1]
    / "jobspecs/drafts/triage-twelfth-gate-study.draft.json"
)


class TwelfthGateDraftTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(DRAFT.read_text(encoding="utf-8"))

    def test_admitted_source_and_actual_runner_are_immutable(self):
        source = self.spec["source"]
        self.assertEqual(source["revision"], "d14b18cf7c07e6ff72abf734008004c3e09022e9")
        self.assertEqual(
            source["treeGitSha"], "95e682a84720d2743cd81f7fcb01576392063fa5"
        )
        self.assertEqual(source["entrypoint"], "scripts/run_twelfth_gate_v110.ps1")
        self.assertEqual(
            source["entrypointGitBlob"], "56d28f475b9b1f7a74c74dd1ae2d4d6b1c6f4d9b"
        )
        self.assertEqual(
            source["verdictScriptGitBlob"], "1d663c476cc0ade8d998766edf28c606d13d8889"
        )
        self.assertEqual(
            source["contractTestGitBlob"], "b197ac0f6c76a2bde746dfb3cf98e30d56dcc20d"
        )
        self.assertEqual(
            source["runtimeGitBlobs"],
            {
                "scripts/five_seed_eval.py": "1e98463248bd746d9a4b87e89fc1360274717ee4",
                "scripts/train_eval_publish.py": "590408b6084bd2b48ec6cfe3d972eda6500a2d0b",
                "scripts/triage_text.py": "bc6daf3409035c4fae49e46dba8a73bfc1d189b1",
                "scripts/twelfth_gate_contract.py": "5f9a38df82af5851ed2269b01b4f43ce11fc44f3",
                "scripts/study_process_guard.py": "e9076ad5a691bd38fe87e9eba0fc19113e5717e4",
                "scripts/study_windows_job.py": "1133ba4cf49a8167738206c506619f8bc479ca22",
                "src/szl_triage/study_evidence.py": "a86ff4a29c06bee1cae38119cb7e2c8f974d955e",
            },
        )

    def test_source_admission_is_not_job_or_provider_authority(self):
        auth = self.spec["authorization"]
        for field in (
            "executable",
            "signingAuthorized",
            "dispatchAuthorized",
            "uploadAuthorized",
        ):
            self.assertIs(auth[field], False)
        self.assertEqual(auth["requiredSignerKeyId"], "b8041281c81c4caa")
        self.assertEqual(
            auth["providerDistributionAdmission"],
            "BLOCKED_EMPTY_APPROVED_SOURCE_DIGEST_MAP",
        )
        self.assertNotIn("signature", self.spec)
        self.assertIn("not a queue-envelope signature", " ".join(self.spec["notes"]))

    def test_expiry_and_dataset_restrictions_are_preserved(self):
        self.assertEqual(self.spec["expiresAt"], "2026-10-09T00:00:00Z")
        dataset = self.spec["dataset"]
        self.assertEqual(
            dataset["frozenTrainSha256"],
            "c498b3d4be6d5b04b89b537941bddd35155b275b0643060899b736730a22e1ad",
        )
        self.assertEqual(
            dataset["frozenHeldSha256"],
            "04c37f596ffeb89b3da0522812e9ae1cff78d5a4dbe684e9ca2104a81482fd4d",
        )
        self.assertEqual(
            dataset["augmentationSha256"],
            "47f5ae0cff5c8db929063075eb36885f9bda3c69ec19a5708664c7421a3a5378",
        )
        self.assertEqual(
            dataset["challengeSha256"],
            "8746e84319d9649bdcdfb9a0b3995bb90c2e2101fab06bc91c8f8f3d3b038509",
        )
        self.assertEqual(
            dataset["challengeWindowsCheckoutSha256"],
            "847a176f9100858219f4b9fd823276f38800da1dd08d5fa191a6374356f3f905",
        )
        self.assertEqual(
            self.spec["gates"]["sealedThresholdsSha256"],
            "bf940e78c3ffe99eecf98654cfa14080f5a4b9e3619a4884dd9bcd8d93ac7dc4",
        )
        self.assertEqual(
            dataset["evaluationOnly"], ["policies/redteam_probes.verified.jsonl"]
        )
        self.assertIn("12/12 typed REVIEW refusals", self.spec["gates"]["twelfthGate"])
        self.assertIn("30 paraphrase rows", self.spec["gates"]["twelfthGate"])

    def test_only_source_preflight_command_is_qualified(self):
        self.assertEqual(
            self.spec["recipe"]["sourceOnlyPreflightArgs"],
            ["-NoPublish", "-PreflightOnly", "-SkipGpu"],
        )
        command = self.spec["authorization"]["sourceOnlyPreflightCommand"]
        self.assertIn("-Python .venv/Scripts/python.exe", command)
        self.assertTrue(command.endswith("-NoPublish -PreflightOnly -SkipGpu"))
        self.assertFalse(self.spec["gates"]["proposedHostControls"]["runtimeQualified"])
        self.assertEqual(
            self.spec["gates"]["proposedHostControls"]["maxWallclockMinutes"], 180
        )
        self.assertEqual(
            self.spec["gates"]["proposedHostControls"]["thermalGuardCelsius"], 78
        )
        self.assertEqual(
            self.spec["base"]["revision"], "2fc06364715b967f1860aea9cf38778875588b17"
        )
        self.assertIs(self.spec["base"]["trustRemoteCode"], False)
        self.assertIn("DISABLED_FAIL_CLOSED", self.spec["outputs"]["publish"])


if __name__ == "__main__":
    unittest.main()
