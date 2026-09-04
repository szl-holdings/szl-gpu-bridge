from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = ROOT / "laptop" / "watchdog.ps1"
INSTALLER = ROOT / "laptop" / "install_watchdog.ps1"
DAEMON = ROOT / "laptop" / "daemon.ps1"


class WindowsWatchdogContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.watchdog = WATCHDOG.read_text(encoding="utf-8")
        cls.installer = INSTALLER.read_text(encoding="utf-8")
        cls.daemon = DAEMON.read_text(encoding="utf-8-sig")
        cls.combined = "\n".join((cls.watchdog, cls.installer))

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

    def test_enabled_task_definition_is_reconciled(self) -> None:
        self.assertIn("Test-TaskDefinitionMatches", self.watchdog)
        self.assertIn("fixed task definition reconciled", self.watchdog)

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
