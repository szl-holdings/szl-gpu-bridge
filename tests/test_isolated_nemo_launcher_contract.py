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

    def test_image_and_source_are_immutable_identifiers(self) -> None:
        self.assertIn(
            "[ValidatePattern('^[^@\\s]+@sha256:[0-9a-f]{64}$')]",
            self.source,
        )
        self.assertIn("[ValidatePattern('^[0-9a-f]{40}$')]", self.source)
        self.assertIn("$ObservedRevision -ne $BridgeRevision", self.source)

    def test_training_receipt_requires_trusted_finalization(self) -> None:
        self.assertIn('"SZL_RECEIPT_TRANSPORT=local-unsigned-outbox"', self.source)
        self.assertIn("$ExitCode -ne 7", self.source)

    def test_bootstrap_pins_resolvable_unsloth_dependencies(self) -> None:
        bootstrap = (ROOT / "laptop" / "bootstrap.ps1").read_text(encoding="utf-8")
        self.assertIn('"unsloth==2026.7.4"', bootstrap)
        self.assertIn('"datasets==4.3.0"', bootstrap)
        self.assertIn('"trl==0.24.0"', bootstrap)
        self.assertNotIn('"datasets==5.0.0"', bootstrap)
        self.assertNotIn('"trl==1.8.0"', bootstrap)


if __name__ == "__main__":
    unittest.main(verbosity=2)
