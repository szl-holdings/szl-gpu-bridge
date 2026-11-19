"""Contract tests: bridge bootstrap must not create a second owner.

Pure source inspection. Tests never install packages, register tasks or train.
"""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "laptop" / "bootstrap.ps1"


class BootstrapOwnerAdmissionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = BOOTSTRAP.read_text(encoding="utf-8-sig")

    def test_requires_exact_commit_not_moving_main(self) -> None:
        source = self.source
        self.assertIn("[Parameter(Mandatory = $true)]", source)
        self.assertIn("ValidatePattern('^[0-9a-fA-F]{40}$')", source)
        self.assertIn("& git -C $RepoRoot rev-parse HEAD", source)
        self.assertIn("$Head -cne $SourceRevision", source)
        self.assertIn("status --porcelain --untracked-files=no", source)
        self.assertNotIn("raw.githubusercontent.com/szl-holdings/szl-gpu-bridge/main", source)
        self.assertNotIn(" | iex", source)

    def test_owner_and_running_task_gate_precede_any_writes(self) -> None:
        source = self.source
        owner = source.index("Get-ScheduledTask -TaskName 'SZL-GPU-Bridge'")
        daemon = source.index("$ExistingDaemon = Test-Path")
        refuse = source.index("refusing a second dispatcher")
        active = source.index("refusing to modify active dispatch")
        principal = source.index("Existing owner principal must remain user scoped")
        mkdir = source.index("New-Item -ItemType Directory -Force -Path $Root")
        self.assertLess(owner, daemon)
        self.assertLess(daemon, refuse)
        self.assertLess(refuse, active)
        self.assertLess(active, principal)
        self.assertLess(principal, mkdir)
        self.assertIn("-UserId $TaskOwner -LogonType S4U", source)
        self.assertNotIn('-UserId "$env:USERDOMAIN', source)

    def test_checkout_sources_prechecked_and_copied_without_network(self) -> None:
        source = self.source
        self.assertEqual(source.count("$Files = @("), 1)
        self.assertIn('@("laptop/storage_preflight.py", "$Root\\storage_preflight.py")', source)
        precheck = source.index("Reviewed checkout missing required bridge source")
        first_mutation = source.index("New-Item -ItemType Directory -Force -Path $Root")
        copy = source.index("Copy-Item -LiteralPath $source -Destination $Pair[1] -Force")
        verify = source.index("Installed file does not match the pinned local checkout")
        register = source.index("Register-ScheduledTask -TaskName $TaskName")
        self.assertLess(precheck, first_mutation)
        self.assertLess(first_mutation, copy)
        self.assertLess(copy, verify)
        self.assertLess(verify, register)
        self.assertNotIn('Invoke-WebRequest "$Repo/', source)

    def test_existing_receipt_identity_is_not_regenerated(self) -> None:
        source = self.source
        self.assertIn('if (-not (Test-Path "$Root\\keys\\laptop_key.pem"))', source)
        self.assertIn("Register-ScheduledTask -TaskName $TaskName", source)


if __name__ == "__main__":
    unittest.main()
