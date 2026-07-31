from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "laptop" / "run_nemo_v3_isolated.ps1"


class IsolatedNemoLauncherContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_container_is_offline_read_only_and_unprivileged(self) -> None:
        for fragment in (
            '"--network", "none"',
            '"--read-only"',
            '"--cap-drop", "ALL"',
            '"--security-opt", "no-new-privileges:true"',
            '"--pids-limit", "2048"',
        ):
            self.assertIn(fragment, self.source)

    def test_only_public_engine_key_enters_sandbox(self) -> None:
        self.assertIn('"keys\\engine_pubkey.json"', self.source)
        self.assertIn('"engine_keyring.json"', self.source)
        self.assertIn("$Envelope.signatures[0].keyid", self.source)
        self.assertIn("$EngineKey", self.source)
        self.assertNotIn("laptop_key.pem", self.source)
        self.assertNotIn("laptop_pubkey.json", self.source)

    def test_sandbox_can_write_only_the_selected_job_state(self) -> None:
        self.assertIn(
            '"type=bind,src=$JobRoot,dst=/bridge/jobs/$JobId"',
            self.source,
        )
        self.assertNotIn('"type=bind,src=$Jobs,dst=/bridge/jobs"', self.source)

    def test_offline_hub_cache_is_readable_without_a_root_profile_or_token(
        self,
    ) -> None:
        for fragment in (
            '"type=bind,src=$HfCache,dst=/hf-cache,readonly"',
            '"HF_HOME=/tmp/huggingface"',
            '"HF_HUB_CACHE=/hf-cache"',
            '"HF_HUB_DISABLE_IMPLICIT_TOKEN=1"',
            "$Prefetch.model.license.expected",
            "$Prefetch.model.license.readmeSha256",
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn(
            "dst=/root/.cache/huggingface/hub",
            self.source,
        )

    def test_image_and_source_are_immutable_identifiers(self) -> None:
        self.assertIn(
            "[ValidatePattern('^unsloth/unsloth@sha256:[0-9a-f]{64}$')]",
            self.source,
        )
        self.assertIn("[ValidatePattern('^[0-9a-f]{40}$')]", self.source)
        self.assertIn("$ObservedRevision -ne $BridgeRevision", self.source)
        self.assertIn("$PSCommandPath", self.source)
        self.assertIn("$InvokedLauncherSha256", self.source)
        self.assertIn("$ApprovedLauncherSha256", self.source)
        self.assertIn(
            "$InvokedLauncherSha256 -ne $ApprovedLauncherSha256",
            self.source,
        )
        self.assertIn(
            "$ImageMetadataText = & $Docker image inspect $Image", self.source
        )
        self.assertIn("$ImageMetadataText | ConvertFrom-Json", self.source)
        self.assertIn("$ImageMetadata.Count -ne 1", self.source)
        self.assertIn('$ExpectedImageId = ($Image -split "@", 2)[1]', self.source)
        self.assertIn("$ObservedImageId -ne $ExpectedImageId", self.source)
        self.assertIn("$ProbeProgram | & $Docker @ProbeArguments", self.source)
        self.assertIn('"SZL_NEMO_IMAGE_PROBE_JSON="', self.source)
        self.assertIn('"--workdir", "/tmp"', self.source)
        self.assertIn('"--workdir", "/workspace"', self.source)
        self.assertIn("$EnvironmentProbeSha256", self.source)
        self.assertIn("launcherSha256 = $ApprovedLauncherSha256", self.source)
        self.assertIn("observedImageId = $ObservedImageId", self.source)
        self.assertIn('"SZL_CONTAINER_IMAGE_ID=$ObservedImageId"', self.source)
        self.assertIn(
            '"SZL_CONTAINER_ENVIRONMENT_PROBE_SHA256=$EnvironmentProbeSha256"',
            self.source,
        )
        self.assertIn(
            '"SZL_LAUNCHER_SHA256=$ApprovedLauncherSha256"',
            self.source,
        )

    def test_image_build_is_digest_pinned_and_cuda_smoked(self) -> None:
        dockerfile = (ROOT / "laptop" / "Dockerfile.nemo-v3").read_text(
            encoding="utf-8"
        )
        build = (ROOT / "laptop" / "build_nemo_v3_image.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "pytorch/pytorch@sha256:"
            "417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385",
            dockerfile,
        )
        for package in (
            "unsloth==2026.7.4",
            "unsloth-zoo==2026.7.4",
            "bitsandbytes==0.50.0",
            "xformers==0.0.32.post2",
        ):
            self.assertIn(package, dockerfile)
        self.assertIn('"--gpus", "all"', build)
        self.assertIn('"--network", "none"', build)
        self.assertIn('"--interactive"', build)
        self.assertIn("torch.cuda.is_available()", build)
        self.assertIn("$ObservedImageId", build)
        self.assertIn("$ImageMetadataText | ConvertFrom-Json", build)
        self.assertIn(
            "Config.Labels.'org.opencontainers.image.revision'",
            build,
        )
        self.assertNotIn("{{index .Config.Labels", build)
        self.assertIn("$SmokeProgram | & $Docker @SmokeArguments", build)
        self.assertIn('$ObservedImageId,\n  "-"', build)
        self.assertIn('"SZL_NEMO_IMAGE_SMOKE_JSON=" + receipt', build)
        self.assertIn("$SmokeLines.Count -ne 1", build)
        self.assertIn("$SmokeLines[0].Substring($SmokePrefix.Length)", build)
        self.assertIn("$ExpectedPackages", build)
        self.assertIn("$ObservedPackage -ne $ExpectedPackages[$Name]", build)
        self.assertLess(
            dockerfile.index("RUN python -m pip install"),
            dockerfile.index("ARG BRIDGE_REVISION"),
        )

    def test_training_receipt_requires_trusted_finalization(self) -> None:
        self.assertIn('"SZL_RECEIPT_TRANSPORT=local-unsigned-outbox"', self.source)
        self.assertIn("$ExitCode -ne 7", self.source)

    def test_attempt_is_atomically_claimed_before_docker_starts(self) -> None:
        self.assertIn("[System.IO.FileMode]::CreateNew", self.source)
        self.assertIn('"szl-nemo-v3-attempt-claim"', self.source)
        self.assertIn("v = 3", self.source)
        self.assertIn("jobEnvelopeSha256", self.source)
        self.assertIn("envelopeRevision = $EnvelopeRevision", self.source)
        self.assertIn("executionBridgeRevision = $BridgeRevision", self.source)
        self.assertLess(
            self.source.index("[System.IO.FileMode]::CreateNew"),
            self.source.index("& $Docker @Arguments"),
        )

    def test_quarantined_jobs_are_refused_before_any_claim(self) -> None:
        for fragment in (
            "job-2026-nemo-v3-governed-attempt-2",
            "job-2026-nemo-v3-governed-successor-3",
            "job-2026-nemo-v3-governed-attempt-4",
            "NEVER_DISPATCH",
        ):
            self.assertIn(fragment, self.source)
        self.assertLess(
            self.source.index("NEVER_DISPATCH"),
            self.source.index("[System.IO.FileMode]::CreateNew"),
        )

    def test_exact_container_compiles_source_before_attempt_claim(self) -> None:
        for fragment in (
            "$CompatibilityArguments = @(",
            '"--network", "none"',
            '"--read-only"',
            '"--cap-drop", "ALL"',
            '"--security-opt", "no-new-privileges:true"',
            '"PYTHONPYCACHEPREFIX=/tmp/pycache"',
            '"-m", "compileall", "-q", "-f", "/bridge"',
            "container-runtime source compatibility gate failed",
        ):
            self.assertIn(fragment, self.source)
        self.assertLess(
            self.source.index("& $Docker @CompatibilityArguments"),
            self.source.index("[System.IO.FileMode]::CreateNew"),
        )

    def test_envelope_is_separate_data_only_protected_history(self) -> None:
        for fragment in (
            "[string]$EnvelopePath",
            "$JobSpec = [System.IO.Path]::GetFullPath($EnvelopePath)",
            "$EnvelopeRevision = (& $Git -C $EnvelopeSource rev-parse HEAD).Trim()",
            "refs/remotes/origin/main",
            "merge-base --is-ancestor",
            "$BridgeRevision",
            "$EnvelopeRevision",
            "$Prefetch.signedJobPayloadSha256 -ne $SignedPayloadSha256",
        ):
            self.assertIn(fragment, self.source)

    def test_bootstrap_pins_resolvable_unsloth_dependencies(self) -> None:
        bootstrap = (ROOT / "laptop" / "bootstrap.ps1").read_text(encoding="utf-8")
        self.assertIn('"unsloth==2026.7.4"', bootstrap)
        self.assertIn('"datasets==4.3.0"', bootstrap)
        self.assertIn('"trl==0.24.0"', bootstrap)
        self.assertNotIn('"datasets==5.0.0"', bootstrap)
        self.assertNotIn('"trl==1.8.0"', bootstrap)

    def test_unsloth_is_imported_before_transformer_trainers(self) -> None:
        runner = (ROOT / "laptop" / "runjob_nemo_v3.py").read_text(encoding="utf-8")
        self.assertLess(runner.index("import unsloth"), runner.index("from trl import"))
        self.assertLess(
            runner.index("import unsloth"),
            runner.index("from transformers import"),
        )

    def test_container_identity_is_recorded_in_stack_evidence(self) -> None:
        runtime = (ROOT / "laptop" / "frontier_runtime.py").read_text(encoding="utf-8")
        self.assertIn('"SZL_CONTAINER_IMAGE_REFERENCE"', runtime)
        self.assertIn('"SZL_CONTAINER_IMAGE_ID"', runtime)
        self.assertIn('"SZL_CONTAINER_ENVIRONMENT_PROBE_SHA256"', runtime)
        self.assertIn('"SZL_ENVELOPE_REVISION"', runtime)
        self.assertIn('"SZL_EXECUTION_BRIDGE_REVISION"', runtime)
        self.assertIn('"SZL_LAUNCHER_SHA256"', runtime)
        self.assertIn('evidence["containerImage"]', runtime)
        self.assertIn('evidence["bridgeExecution"]', runtime)
        self.assertIn('evidence["launcherSha256"]', runtime)


if __name__ == "__main__":
    unittest.main(verbosity=2)
