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
        self.assertNotIn("laptop_key.pem", self.source)
        self.assertNotIn("laptop_pubkey.json", self.source)

    def test_sandbox_can_write_only_the_selected_job_state(self) -> None:
        self.assertIn(
            '"type=bind,src=$JobRoot,dst=/bridge/jobs/$JobId"',
            self.source,
        )
        self.assertNotIn('"type=bind,src=$Jobs,dst=/bridge/jobs"', self.source)

    def test_image_and_source_are_immutable_identifiers(self) -> None:
        self.assertIn(
            "[ValidatePattern('^(?:[^@\\s]+@)?sha256:[0-9a-f]{64}$')]",
            self.source,
        )
        self.assertIn("[ValidatePattern('^[0-9a-f]{40}$')]", self.source)
        self.assertIn("$ObservedRevision -ne $BridgeRevision", self.source)
        self.assertIn(
            'image inspect --format "{{.Id}}" $Image',
            self.source,
        )
        self.assertIn("$ObservedImageId -ne $Image", self.source)
        self.assertIn("$ObservedRevisionLabel -ne $BridgeRevision", self.source)
        self.assertIn("$BuildReceipt.imageId -ne $ObservedImageId", self.source)
        self.assertIn(
            "$BuildReceipt.dockerfileSha256 -ne $ImageDockerfileSha256",
            self.source,
        )
        self.assertIn("imageBuildReceiptSha256 = $ImageBuildReceiptSha256", self.source)
        self.assertIn("observedImageId = $ObservedImageId", self.source)
        self.assertIn('"SZL_CONTAINER_IMAGE_ID=$ObservedImageId"', self.source)
        self.assertIn(
            '"SZL_CONTAINER_IMAGE_BUILD_RECEIPT_SHA256=$ImageBuildReceiptSha256"',
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
        self.assertIn("torch.cuda.is_available()", build)
        self.assertIn("$ObservedImageId", build)
        self.assertIn('"SZL_NEMO_IMAGE_SMOKE_JSON=" + receipt', build)
        self.assertIn("$SmokeLines.Count -ne 1", build)
        self.assertIn("$SmokeLines[0].Substring($SmokePrefix.Length)", build)
        self.assertIn("$ExpectedPackages", build)
        self.assertIn("$ObservedPackage -ne $ExpectedPackages[$Name]", build)

    def test_training_receipt_requires_trusted_finalization(self) -> None:
        self.assertIn('"SZL_RECEIPT_TRANSPORT=local-unsigned-outbox"', self.source)
        self.assertIn("$ExitCode -ne 7", self.source)

    def test_attempt_is_atomically_claimed_before_docker_starts(self) -> None:
        self.assertIn("[System.IO.FileMode]::CreateNew", self.source)
        self.assertIn('"szl-nemo-v3-attempt-claim"', self.source)
        self.assertIn("jobEnvelopeSha256", self.source)
        self.assertLess(
            self.source.index("[System.IO.FileMode]::CreateNew"),
            self.source.index("& $Docker @Arguments"),
        )

    def test_bootstrap_pins_resolvable_unsloth_dependencies(self) -> None:
        bootstrap = (ROOT / "laptop" / "bootstrap.ps1").read_text(encoding="utf-8")
        self.assertIn('"unsloth==2026.7.4"', bootstrap)
        self.assertIn('"datasets==4.3.0"', bootstrap)
        self.assertIn('"trl==0.24.0"', bootstrap)
        self.assertNotIn('"datasets==5.0.0"', bootstrap)
        self.assertNotIn('"trl==1.8.0"', bootstrap)

    def test_container_identity_is_recorded_in_stack_evidence(self) -> None:
        runtime = (ROOT / "laptop" / "frontier_runtime.py").read_text(encoding="utf-8")
        self.assertIn('"SZL_CONTAINER_IMAGE_REFERENCE"', runtime)
        self.assertIn('"SZL_CONTAINER_IMAGE_ID"', runtime)
        self.assertIn('evidence["containerImage"]', runtime)


if __name__ == "__main__":
    unittest.main(verbosity=2)
