from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = ROOT / "laptop" / "watchdog.ps1"
INSTALLER = ROOT / "laptop" / "install_watchdog.ps1"
DAEMON = ROOT / "laptop" / "daemon.ps1"
WINDOWS_POWERSHELL = (
    Path(os.environ.get("SystemRoot", r"C:\Windows"))
    / "System32"
    / "WindowsPowerShell"
    / "v1.0"
    / "powershell.exe"
)


class WindowsWatchdogContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.watchdog = WATCHDOG.read_text(encoding="utf-8")
        cls.installer = INSTALLER.read_text(encoding="utf-8")
        cls.daemon = DAEMON.read_text(encoding="utf-8-sig")
        cls.combined = "\n".join((cls.watchdog, cls.installer))

    def run_windows_powershell(
        self, script: str, *, extra_env: dict[str, str] | None = None
    ) -> str:
        env = os.environ.copy()
        env["SZL_WATCHDOG_SOURCE"] = str(WATCHDOG)
        if extra_env:
            env.update(extra_env)
        completed = subprocess.run(
            [
                str(WINDOWS_POWERSHELL),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        return completed.stdout

    def test_expected_files_exist(self) -> None:
        for path in (WATCHDOG, INSTALLER, DAEMON):
            self.assertTrue(path.is_file(), path)

    def test_exact_task_allowlist_is_present(self) -> None:
        expected = {
            "SZL-GPU-Bridge",
            "SZL-GPU-Bridge-Watchdog",
            "SZL-GPU-Bridge-Guardian",
        }
        observed = set(
            re.findall(r"SZL-GPU-Bridge(?:-Watchdog|-Guardian)?", self.combined)
        )
        self.assertEqual(observed, expected)
        self.assertIn("Task is outside the fixed allowlist", self.combined)

    def test_redundant_cadence_and_single_flight_are_fixed(self) -> None:
        for marker in (
            "New-TimeSpan -Minutes 5",
            "New-TimeSpan -Minutes 15",
            "New-ScheduledTaskTrigger -AtStartup",
            "-MultipleInstances IgnoreNew",
            "-StartWhenAvailable",
            "Global\\SZLGpuBridgeWatchdog",
        ):
            self.assertIn(marker, self.combined)

    def test_primary_daemon_retains_long_running_semantics(self) -> None:
        self.assertIn("New-TimeSpan -Hours 26", self.watchdog)
        self.assertIn("New-TimeSpan -Hours 26", self.installer)
        self.assertIn("daemon.ps1", self.installer)
        self.assertIn("Test-DaemonLogFreshness", self.watchdog)
        self.assertIn("if (Test-Path $Lock)", self.daemon)
        self.assertIn("$age.TotalHours -lt 26", self.daemon)
        self.assertIn("New-Item -ItemType File -Path $Lock -Force", self.daemon)
        self.assertIn("finally {", self.daemon)
        self.assertIn(
            "Remove-Item $Lock -Force -ErrorAction SilentlyContinue", self.daemon
        )

    def test_state_reset_does_not_pollute_the_success_pipeline(self) -> None:
        self.assertIn(
            "Write-Log -Level 'WARN' -Message ('State rejected and reset: {0}' -f $_.Exception.GetType().Name) | Out-Null",
            self.watchdog,
        )

    def test_enabled_tasks_are_reconciled_against_the_fixed_definition(self) -> None:
        for marker in (
            "function Get-TaskDefinitionFingerprint",
            "function Test-TaskDefinition",
            "function Register-FixedTask",
            "Test-TaskDefinition -Task $task -Definition $definition",
            "definition drift replaced from fixed local files",
        ):
            self.assertIn(marker, self.watchdog)

    @unittest.skipUnless(
        WINDOWS_POWERSHELL.is_file(), "Windows PowerShell 5.1 is required"
    )
    def test_task_fingerprint_rejects_behavioral_scheduler_drift(self) -> None:
        output = self.run_windows_powershell(
            r"""
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:SZL_WATCHDOG_SOURCE,
    [ref]$tokens,
    [ref]$errors
)
if ($errors.Count -ne 0) { throw 'watchdog source did not parse' }
$functions = @($ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]
}, $true))
foreach ($name in @('Get-DynamicProperty', 'Get-TaskDefinitionFingerprint')) {
    $function = $functions | Where-Object { $_.Name -eq $name } | Select-Object -First 1
    if ($null -eq $function) { throw ('missing function: {0}' -f $name) }
    Invoke-Expression $function.Extent.Text
}

$at = (Get-Date).AddMinutes(2)
$action = New-ScheduledTaskAction `
    -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
    -Argument '-NoLogo'
$workingDirectoryAction = New-ScheduledTaskAction `
    -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
    -Argument '-NoLogo' `
    -WorkingDirectory 'C:\Windows'
$principal = New-ScheduledTaskPrincipal `
    -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType S4U `
    -RunLevel Highest
$limitedPrincipal = New-ScheduledTaskPrincipal `
    -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType S4U `
    -RunLevel Limited
$baseTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At $at `
    -RepetitionInterval (New-TimeSpan -Minutes 5)
$nearRegistrationSkewTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At $at.AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes 5)
$activePastTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At $at.AddDays(-30) `
    -RepetitionInterval (New-TimeSpan -Minutes 5)
$farFutureTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At $at.AddYears(10) `
    -RepetitionInterval (New-TimeSpan -Minutes 5)
$beyondToleranceTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At $at.AddMinutes(4) `
    -RepetitionInterval (New-TimeSpan -Minutes 5)
$malformedTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At $at `
    -RepetitionInterval (New-TimeSpan -Minutes 5)
$malformedTrigger.StartBoundary = 'not-a-date'
$delayedTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At $at `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RandomDelay (New-TimeSpan -Hours 12)
$baseSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$idleSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -RunOnlyIfIdle

$base = Get-TaskDefinitionFingerprint `
    -Actions @($action) `
    -Triggers @($baseTrigger) `
    -Principal $principal `
    -Settings $baseSettings
$nearRegistrationSkew = Get-TaskDefinitionFingerprint `
    -Actions @($action) `
    -Triggers @($nearRegistrationSkewTrigger) `
    -Principal $principal `
    -Settings $baseSettings
$activePast = Get-TaskDefinitionFingerprint `
    -Actions @($action) `
    -Triggers @($activePastTrigger) `
    -Principal $principal `
    -Settings $baseSettings
$farFuture = Get-TaskDefinitionFingerprint `
    -Actions @($action) `
    -Triggers @($farFutureTrigger) `
    -Principal $principal `
    -Settings $baseSettings
$beyondTolerance = Get-TaskDefinitionFingerprint `
    -Actions @($action) `
    -Triggers @($beyondToleranceTrigger) `
    -Principal $principal `
    -Settings $baseSettings
$malformed = Get-TaskDefinitionFingerprint `
    -Actions @($action) `
    -Triggers @($malformedTrigger) `
    -Principal $principal `
    -Settings $baseSettings
$delayed = Get-TaskDefinitionFingerprint `
    -Actions @($action) `
    -Triggers @($delayedTrigger) `
    -Principal $principal `
    -Settings $baseSettings
$idleOnly = Get-TaskDefinitionFingerprint `
    -Actions @($action) `
    -Triggers @($baseTrigger) `
    -Principal $principal `
    -Settings $idleSettings
$actionDrift = Get-TaskDefinitionFingerprint `
    -Actions @($workingDirectoryAction) `
    -Triggers @($baseTrigger) `
    -Principal $principal `
    -Settings $baseSettings
$principalDrift = Get-TaskDefinitionFingerprint `
    -Actions @($action) `
    -Triggers @($baseTrigger) `
    -Principal $limitedPrincipal `
    -Settings $baseSettings

if ($base -cne $nearRegistrationSkew) { throw 'small registration-time StartBoundary skew was not normalized' }
if ($base -cne $activePast) { throw 'active repeating StartBoundary caused perpetual drift' }
if ($base -ceq $farFuture) { throw 'far-future StartBoundary drift was accepted' }
if ($base -ceq $beyondTolerance) { throw 'StartBoundary beyond the five-minute tolerance was accepted' }
if ($base -ceq $malformed) { throw 'malformed StartBoundary was accepted' }
if ($base -ceq $delayed) { throw 'RandomDelay drift was accepted' }
if ($base -ceq $idleOnly) { throw 'RunOnlyIfIdle drift was accepted' }
if ($base -ceq $actionDrift) { throw 'action working-directory drift was accepted' }
if ($base -ceq $principalDrift) { throw 'principal run-level drift was accepted' }
Write-Output 'SCHEDULER_FINGERPRINT_BEHAVIOR_OK'
"""
        )
        self.assertIn("SCHEDULER_FINGERPRINT_BEHAVIOR_OK", output)

    @unittest.skipUnless(
        WINDOWS_POWERSHELL.is_file(), "Windows PowerShell 5.1 is required"
    )
    def test_malformed_state_returns_one_default_state_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            state_path = temporary / "watchdog-state.json"
            log_path = temporary / "watchdog.log"
            state_path.write_text("{malformed", encoding="utf-8")
            output = self.run_windows_powershell(
                r"""
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:SZL_WATCHDOG_SOURCE,
    [ref]$tokens,
    [ref]$errors
)
if ($errors.Count -ne 0) { throw 'watchdog source did not parse' }
$functions = @($ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]
}, $true))
foreach ($name in @('Write-Log', 'New-DefaultState', 'Read-State')) {
    $function = $functions | Where-Object { $_.Name -eq $name } | Select-Object -First 1
    if ($null -eq $function) { throw ('missing function: {0}' -f $name) }
    Invoke-Expression $function.Extent.Text
}
$Mode = 'Watchdog'
$StatePath = $env:SZL_WATCHDOG_STATE_PATH
$LogPath = $env:SZL_WATCHDOG_LOG_PATH
$result = @(Read-State)
if ($result.Count -ne 1) {
    throw ('Read-State returned {0} pipeline objects' -f $result.Count)
}
if ($result[0].schema -ne 'szl.gpu-bridge-watchdog-state/v1') {
    throw 'Read-State did not return the default state object'
}
if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
    throw 'malformed state warning was not logged'
}
if ((Get-Content -LiteralPath $LogPath -Raw) -notmatch 'State rejected and reset') {
    throw 'malformed state warning was missing from the log'
}
Write-Output 'MALFORMED_STATE_PIPELINE_OK'
""",
                extra_env={
                    "SZL_WATCHDOG_STATE_PATH": str(state_path),
                    "SZL_WATCHDOG_LOG_PATH": str(log_path),
                },
            )
        self.assertIn("MALFORMED_STATE_PIPELINE_OK", output)

    def test_restart_budget_is_small_and_rolling(self) -> None:
        for marker in (
            "window_minutes = 30",
            "max_starts_per_window = 3",
            "daemon_stale_minutes = 35",
        ):
            self.assertIn(marker, self.installer)
        self.assertIn("rolling start budget exhausted", self.watchdog)
        self.assertIn("Test-StartBudget", self.watchdog)
        self.assertIn("Record-Start", self.watchdog)

    def test_user_scoped_authentication_boundary_is_preserved(self) -> None:
        self.assertIn("Resolve-TaskUser", self.installer)
        self.assertIn("existing.Principal.UserId", self.installer)
        self.assertIn("-LogonType S4U", self.installer)
        for forbidden_principal in (
            "NT AUTHORITY\\SYSTEM",
            "LOCAL SERVICE",
            "NETWORK SERVICE",
        ):
            self.assertIn(forbidden_principal, self.installer)
        self.assertIn("user-scoped bridge principal is required", self.installer)
        self.assertNotIn("-LogonType ServiceAccount", self.combined)

    def test_state_and_receipts_are_fail_visible(self) -> None:
        for marker in (
            "watchdog-state.json",
            "watchdog-receipts.ndjson",
            "watchdog-receipt-tip.txt",
            "previous_sha256",
            "szl.gpu-bridge-watchdog-receipt/v1",
            "Get-Sha256Hex",
            "Move-Item -LiteralPath $temporary -Destination $StatePath -Force",
            "Recovered an abandoned watchdog mutex",
        ):
            self.assertIn(marker, self.watchdog)

    def test_expected_recovery_cycles_do_not_trigger_scheduler_storms(self) -> None:
        self.assertIn(
            "$status = if ($healthy) { 'HEALTHY' } else { 'DEGRADED' }", self.watchdog
        )
        self.assertRegex(
            self.watchdog,
            r"Write-Receipt -Status \$status[\s\S]{0,700}exit 0",
        )
        self.assertIn("exit 1", self.watchdog)

    def test_no_dynamic_remote_execution_or_download_path(self) -> None:
        forbidden = (
            "Invoke-Expression",
            "DownloadString(",
            "FromBase64String",
            "-EncodedCommand",
            "cmd.exe /c",
            "scriptblock::Create",
            "Invoke-WebRequest",
            "Invoke-RestMethod",
            "Start-BitsTransfer",
            "curl.exe",
            "wget.exe",
        )
        for marker in forbidden:
            self.assertNotIn(marker.lower(), self.combined.lower(), marker)
        self.assertNotRegex(
            self.combined,
            r"(?i)(?:iex|invoke-expression)\s+\$",
        )

    def test_no_credential_or_private_network_value_is_committed(self) -> None:
        for marker in (
            "credential_values_read = $false",
            "remote_commands_accepted = $false",
            "inbound_ports_opened = $false",
        ):
            self.assertIn(marker, self.combined)
        self.assertIsNone(
            re.search(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", self.combined)
        )
        self.assertIsNone(re.search(r"\b192\.168\.\d{1,3}\.\d{1,3}\b", self.combined))
        self.assertIsNone(
            re.search(
                r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b",
                self.combined,
            )
        )
        self.assertIsNone(
            re.search(
                r"(?i)(?:token|secret|password)\s*=\s*['\"][A-Za-z0-9_.-]{20,}",
                self.combined,
            )
        )

    def test_installer_emits_file_hashes_and_sanitized_authority_flags(self) -> None:
        self.assertIn("Get-FileHash", self.installer)
        self.assertIn("szl.gpu-bridge-watchdog-install/v1", self.installer)
        self.assertIn("installed_files", self.installer)
        self.assertIn("credential_values_read = $false", self.installer)
        self.assertIn("remote_commands_accepted = $false", self.installer)
        self.assertIn("inbound_ports_opened = $false", self.installer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
