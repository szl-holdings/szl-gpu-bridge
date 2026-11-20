"""Offline tests for PowerShell daemon failure-exit semantics.

Never executes the real daemon, reads job ledgers or calls a provider.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DAEMON = ROOT / "laptop" / "daemon.ps1"
POWERSHELL = shutil.which("powershell.exe") if os.name == "nt" else None


class DaemonFailureExitContractTests(unittest.TestCase):
    def test_unexpected_exception_exit_is_after_lock_disposal(self) -> None:
        source = DAEMON.read_text(encoding="utf-8")
        catch = source.rfind("} catch {")
        finally_block = source.rfind("} finally {")
        post_finally_exit = source.rfind(
            "if ($DaemonFailureExit -ne 0) { exit $DaemonFailureExit }"
        )
        self.assertGreater(catch, 0)
        self.assertLess(catch, finally_block)
        self.assertLess(finally_block, post_finally_exit)
        self.assertIn("$DaemonFailureExit = 0", source)
        self.assertIn("$DaemonFailureExit = 70", source[catch:finally_block])
        self.assertIn("$LockHandle.Dispose()", source[finally_block:post_finally_exit])
        self.assertIn("reconcile per-job ledger and receipts", source)
        self.assertIn(
            '[Console]::Error.WriteLine("SZL_DAEMON_ERROR_LOG_UNAVAILABLE; exit=70")',
            source[catch:finally_block],
        )
        self.assertNotIn("no job ledger mutation", source)
        self.assertIn("exit 78", source)  # lock denial remains distinct
        self.assertNotIn("Remove-Item $Lock", source)

    @unittest.skipUnless(POWERSHELL, "Windows PowerShell is required")
    def test_only_extracted_catch_finally_in_isolated_fixture(self) -> None:
        source = DAEMON.read_text(encoding="utf-8")
        tail = source[source.rfind("} catch {") :]
        self.assertIn("if ($DaemonFailureExit -ne 0)", tail)
        self.assertIn("$LockHandle.Dispose()", tail)
        fixture = tail.replace(
            "$LockHandle.Dispose()",
            '$LockHandle.Dispose(); Write-Output "FIXTURE_FINALLY"',
            1,
        )
        prefix = (
            '$ErrorActionPreference="Stop"; '
            "$Lock=$env:SZL_TEST_LOCK_PATH; "
            'function Log($m) { Write-Output "FIXTURE_CAUGHT" }; '
            "$LockHandle=[System.IO.File]::Open($Lock, "
            "[System.IO.FileMode]::OpenOrCreate, "
            "[System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None); "
            '$DaemonFailureExit=0; try { throw "SYNTHETIC_POST_LOCK_FAILURE" '
        )
        with tempfile.TemporaryDirectory() as dirname:
            env = os.environ.copy()
            env["SZL_TEST_LOCK_PATH"] = str(Path(dirname) / "fixture.lock")
            result = subprocess.run(
                [
                    POWERSHELL,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    prefix + fixture + '\nWrite-Output "FALSE_SUCCESS"',
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=25,
                check=False,
            )
            self.assertEqual(result.returncode, 70, result.stderr)
            self.assertIn("FIXTURE_CAUGHT", result.stdout)
            self.assertIn("FIXTURE_FINALLY", result.stdout)
            self.assertNotIn("FALSE_SUCCESS", result.stdout)

    @unittest.skipUnless(POWERSHELL, "Windows PowerShell is required")
    def test_log_failure_still_exits_70_after_finally(self) -> None:
        source = DAEMON.read_text(encoding="utf-8")
        tail = source[source.rfind("} catch {") :]
        self.assertIn("SZL_DAEMON_ERROR_LOG_UNAVAILABLE", tail)
        fixture = tail.replace(
            "$LockHandle.Dispose()",
            '$LockHandle.Dispose(); Write-Output "FIXTURE_FINALLY"',
            1,
        )
        prefix = (
            '$ErrorActionPreference="Stop"; '
            "$Lock=$env:SZL_TEST_LOCK_PATH; "
            'function Log($m) { throw "ISOLATED_LOG_FAILURE" }; '
            "$LockHandle=[System.IO.File]::Open($Lock, "
            "[System.IO.FileMode]::OpenOrCreate, "
            "[System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None); "
            '$DaemonFailureExit=0; try { throw "SYNTHETIC_POST_LOCK_FAILURE" '
        )
        with tempfile.TemporaryDirectory() as dirname:
            env = os.environ.copy()
            env["SZL_TEST_LOCK_PATH"] = str(Path(dirname) / "fixture.lock")
            result = subprocess.run(
                [
                    POWERSHELL,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    prefix + fixture + '\nWrite-Output "FALSE_SUCCESS"',
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=25,
                check=False,
            )
            self.assertEqual(result.returncode, 70, result.stderr)
            self.assertIn("SZL_DAEMON_ERROR_LOG_UNAVAILABLE", result.stderr)
            self.assertIn("FIXTURE_FINALLY", result.stdout)
            self.assertNotIn("FALSE_SUCCESS", result.stdout)


if __name__ == "__main__":
    unittest.main()
