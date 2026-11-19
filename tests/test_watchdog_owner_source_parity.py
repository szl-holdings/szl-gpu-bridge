"""Source-only regression gates for the owner-bound Windows watchdog workflow."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "windows-watchdog-install.yml"


class WindowsOwnerSourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = WORKFLOW.read_text(encoding="utf-8")

    def test_source_parity_precedes_any_task_installation(self) -> None:
        s = self.source
        bind = s.index("- name: Bind execution to an existing SZL bridge owner node")
        guard = s.index("- name: Refuse stale owner daemon")
        install = s.index("- name: Install the exact local watchdog package")
        reconcile = s.index("- name: Run one bounded reconciliation cycle")
        self.assertLess(bind, guard)
        self.assertLess(guard, install)
        self.assertLess(install, reconcile)
        block = s[guard:install]
        self.assertIn("@('daemon.ps1','storage_preflight.py')", block)
        self.assertIn("Get-FileHash -LiteralPath $installed", block)
        self.assertIn("Get-FileHash -LiteralPath $expected", block)
        self.assertIn("Installed owner source parity mismatch", block)
        self.assertIn("Missing reviewed owner prerequisite", block)
        for forbidden in (
            "Register-ScheduledTask",
            "Start-ScheduledTask",
            "Copy-Item",
            "Invoke-WebRequest",
        ):
            self.assertNotIn(forbidden, block)

    def test_proof_includes_preflight_artifact(self) -> None:
        s = self.source
        self.assertIn(
            "@('daemon.ps1','watchdog.ps1',"
            "'watchdog-config.json','storage_preflight.py')",
            s,
        )
        self.assertIn(
            "if (@($proof.installed_files).Count -ne 4)", s
        )
        self.assertIn(
            "@('daemon.ps1','watchdog.ps1','storage_preflight.py')", s
        )
        self.assertIn("Installed owner proof source drift", s)

    def test_protected_sha_and_existing_owner_checks_remain(self) -> None:
        s = self.source
        for anchor in (
            "ref: ${{ github.sha }}",
            "persist-credentials: false",
            "runs-on: [self-hosted, Windows, X64]",
            "Get-ScheduledTask -TaskName 'SZL-GPU-Bridge'",
            "This machine is not bound to the existing SZL GPU bridge.",
            "git rev-parse HEAD",
            "if ($proof.principal_recorded -ne $false)",
            "if ($proof.credential_values_recorded -ne $false)",
            "if ($proof.inbound_ports_opened -ne $false)",
        ):
            self.assertIn(anchor, s)


if __name__ == "__main__":
    unittest.main()
