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
        self.assertIn("source_parity.ps1", block)
        self.assertIn("-ExpectedSource $expected -InstalledSource $installed", block)
        self.assertIn("SOURCE_PARITY_VERIFIED", block)
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
        self.assertIn("if (@($proof.installed_files).Count -ne 4)", s)
        self.assertIn("@('daemon.ps1','watchdog.ps1','storage_preflight.py')", s)
        self.assertIn("Installed owner proof source drift", s)
        self.assertIn("Installed owner proof changed since readback", s)
        self.assertIn('      - "laptop/storage_preflight.py"', s)
        self.assertIn('      - "laptop/source_parity.ps1"', s)

    def test_protected_sha_and_existing_owner_checks_remain(self) -> None:
        s = self.source
        for anchor in (
            "ref: ${{ github.sha }}",
            "persist-credentials: false",
            "runs-on: [self-hosted, Windows, X64]",
            "Get-ScheduledTask -TaskName 'SZL-GPU-Bridge'",
            "Existing owner prerequisites unavailable: both primary task and installed daemon are required.",
            "git rev-parse HEAD",
            "if ($proof.principal_recorded -ne $false)",
            "if ($proof.credential_values_recorded -ne $false)",
            "if ($proof.inbound_ports_opened -ne $false)",
        ):
            self.assertIn(anchor, s)

    def test_primary_task_and_daemon_both_required_before_install(self) -> None:
        """Missing primary task cannot be compensated by a copied daemon file."""
        source = self.source
        bind = source.index(
            "- name: Bind execution to an existing SZL bridge owner node"
        )
        guard = source.index("- name: Refuse stale owner daemon")
        install = source.index("- name: Install the exact local watchdog package")
        block = source[bind:guard]
        self.assertLess(bind, guard)
        self.assertLess(guard, install)
        self.assertIn("Get-ScheduledTask -TaskName 'SZL-GPU-Bridge'", block)
        self.assertIn("Test-Path -LiteralPath", block)
        self.assertIn("if ($null -eq $task -or -not $daemon) {", block)
        self.assertNotIn("if ($null -eq $task -and -not $daemon) {", block)
        self.assertIn("both primary task and installed daemon are required", block)
        for forbidden in ("Register-ScheduledTask", "Start-ScheduledTask", "Copy-Item"):
            self.assertNotIn(forbidden, block)

    def test_real_owner_gate_expression_in_isolated_powershell(self) -> None:
        """Run only the extracted predicate with fake values; no Windows task APIs."""
        import shutil
        import subprocess

        binary = shutil.which("powershell.exe") or shutil.which("pwsh")
        if binary is None:
            self.skipTest(
                "PowerShell is unavailable; static source contract still runs"
            )
        block = self.source.split(
            "- name: Bind execution to an existing SZL bridge owner node", 1
        )[1].split("- name: Refuse stale owner daemon", 1)[0]
        marker = "if ($null -eq $task -or -not $daemon) {"
        predicate = marker + block.split(marker, 1)[1].split("}", 1)[0] + "}"
        for task_present, daemon_present, accepted in (
            (False, False, False),
            (False, True, False),
            (True, False, False),
            (True, True, True),
        ):
            with self.subTest(task_present=task_present, daemon_present=daemon_present):
                script = (
                    "$ErrorActionPreference = 'Stop';"
                    + (
                        "$task = [pscustomobject]@{};"
                        if task_present
                        else "$task = $null;"
                    )
                    + ("$daemon = $true;" if daemon_present else "$daemon = $false;")
                    + "try { "
                    + predicate
                    + "; 'ACCEPTED' } catch { 'REFUSED' }"
                )
                outcome = subprocess.run(
                    [binary, "-NoProfile", "-NonInteractive", "-Command", script],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(outcome.returncode, 0, outcome.stderr)
                self.assertEqual(
                    outcome.stdout.strip(), "ACCEPTED" if accepted else "REFUSED"
                )

    def test_primary_action_identity_must_be_verified_before_install(self) -> None:
        """A task name alone cannot authorize an action-bearing watchdog installer."""
        s = self.source
        bind = s.index("- name: Bind execution to an existing SZL bridge owner node")
        verify = s.index("function Test-ReviewedPrimaryTaskAction")
        installed = s.index("- name: Install the exact local watchdog package")
        self.assertLess(bind, verify)
        self.assertLess(verify, installed)
        block = s[bind:installed]
        self.assertIn("Test-ReviewedPrimaryTaskAction -Task $task", block)
        self.assertIn("Existing primary task action is unverified", block)
        for required in (
            "@($Task.Actions).Count -ne 1",
            "[string]$Task.TaskPath -cne",
            "[string]$Task.Principal.UserId",
            "[string]$Task.Principal.LogonType",
            "StringComparison]::OrdinalIgnoreCase",
            "IsNullOrWhiteSpace([string]$action.WorkingDirectory)",
        ):
            self.assertIn(required, block)
        for forbidden in (
            "Register-ScheduledTask",
            "Start-ScheduledTask",
            "Copy-Item",
            "Remove-Item",
            "Set-ScheduledTask",
        ):
            self.assertNotIn(forbidden, block)

    def test_reviewed_action_predicate_12_synthetic_cases(self) -> None:
        """Run the real source predicate on fake task objects, never Task Scheduler."""
        import shutil
        import subprocess

        powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
        if powershell is None:
            self.skipTest("PowerShell unavailable; source guard remains enforced")
        after = self.source.split("function Test-ReviewedPrimaryTaskAction {", 1)[1]
        terminator = "if (-not (Test-ReviewedPrimaryTaskAction -Task $task))"
        function_source = "function Test-ReviewedPrimaryTaskAction {"
        function_source += after.split(terminator, 1)[0]
        synthetic = r"""
$ErrorActionPreference = 'Stop'
$exe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$argsValue = '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "C:\szl-bridge\daemon.ps1"'
function MakeFixture {
  param([string]$Executable,[string]$Arguments,[string]$TaskPath='\',[string]$WorkingDirectory='',[string]$User='TEST\Owner',[string]$LogonType='Interactive')
  if ($null -eq $Executable) { $Executable = $exe }
  if ($null -eq $Arguments) { $Arguments = $argsValue }
  [pscustomobject]@{TaskPath=$TaskPath;Actions=@([pscustomobject]@{Execute=$Executable;Arguments=$Arguments;WorkingDirectory=$WorkingDirectory});Principal=[pscustomobject]@{UserId=$User;LogonType=$LogonType}}
}
$good = MakeFixture -Executable $exe -Arguments $argsValue
$noActions = MakeFixture -Executable $exe -Arguments $argsValue; $noActions.Actions=@()
$nullActions = MakeFixture -Executable $exe -Arguments $argsValue; $nullActions.Actions=$null
$twoActions = MakeFixture -Executable $exe -Arguments $argsValue; $twoActions.Actions=@($good.Actions[0],[pscustomobject]@{Execute='cmd.exe';Arguments='/c calc';WorkingDirectory=''})
$noPrincipal = MakeFixture -Executable $exe -Arguments $argsValue; $noPrincipal.Principal=$null
$cases = @(
  [pscustomobject]@{name='valid';task=$good;want=$true},
  [pscustomobject]@{name='noTask';task=$null;want=$false},
  [pscustomobject]@{name='zeroActions';task=$noActions;want=$false},
  [pscustomobject]@{name='nullActions';task=$nullActions;want=$false},
  [pscustomobject]@{name='extraAction';task=$twoActions;want=$false},
  [pscustomobject]@{name='wrongExe';task=(MakeFixture -Executable 'C:\Windows\System32\cmd.exe' -Arguments $argsValue);want=$false},
  [pscustomobject]@{name='injectedArgs';task=(MakeFixture -Executable $exe -Arguments ($argsValue+' -EncodedCommand abc'));want=$false},
  [pscustomobject]@{name='wrongTaskPath';task=(MakeFixture -Executable $exe -Arguments $argsValue -TaskPath '\Custom\');want=$false},
  [pscustomobject]@{name='workingDirectory';task=(MakeFixture -Executable $exe -Arguments $argsValue -WorkingDirectory 'C:\Other');want=$false},
  [pscustomobject]@{name='serviceAccount';task=(MakeFixture -Executable $exe -Arguments $argsValue -User 'NT AUTHORITY\SYSTEM');want=$false},
  [pscustomobject]@{name='badLogon';task=(MakeFixture -Executable $exe -Arguments $argsValue -LogonType 'ServiceAccount');want=$false},
  [pscustomobject]@{name='noPrincipal';task=$noPrincipal;want=$false}
)
$failures = @()
foreach ($case in $cases) {
  $observed = Test-ReviewedPrimaryTaskAction -Task $case.task
  if ($observed -ne $case.want) { $failures += $case.name }
}
if ($cases.Count -ne 12 -or $failures.Count -ne 0) {
  throw ('SYNTHETIC_OWNER_TASK_GUARD_FAILED:' + ($failures -join ','))
}
'PASS=12'
"""
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                function_source + synthetic,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS=12", result.stdout)


if __name__ == "__main__":
    unittest.main()
