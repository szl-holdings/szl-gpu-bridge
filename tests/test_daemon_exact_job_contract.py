from __future__ import annotations

import base64
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON = ROOT / "laptop" / "daemon.ps1"


class DaemonExactJobContractTests(unittest.TestCase):
    def test_exact_git_bytes_are_ps51_encoding_safe(self) -> None:
        raw = subprocess.check_output(
            ["git", "show", ":laptop/daemon.ps1"],
            cwd=ROOT,
        )

        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        raw.decode("ascii")

        powershell = shutil.which("powershell.exe")
        if powershell is None:
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            materialized = pathlib.Path(temp_dir) / "daemon.ps1"
            materialized.write_bytes(raw)
            command = (
                "$tokens = $null; $errors = $null; "
                "[System.Management.Automation.Language.Parser]::ParseFile("
                f"'{materialized}', [ref]$tokens, [ref]$errors) | Out-Null; "
                "if ($errors.Count -gt 0) { "
                "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
            )
            encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
            completed = subprocess.run(
                [
                    powershell,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-EncodedCommand",
                    encoded,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_optional_exact_job_filter_precedes_dispatch(self) -> None:
        source = DAEMON.read_text(encoding="utf-8")

        parameter = "[string]$OnlyJobId"
        filter_expression = (
            '$files = @($allFiles | Where-Object { $_.name -eq "${OnlyJobId}.json" })'
        )
        dispatcher = '& $Py "$Root\\dispatcher.py" $specPath'

        self.assertIn(
            "[ValidatePattern('^job-[A-Za-z0-9][A-Za-z0-9._-]*$')]",
            source,
        )
        self.assertLess(source.index(parameter), source.index(filter_expression))
        self.assertLess(source.index(filter_expression), source.index(dispatcher))
        self.assertIn(
            "requested job is not present in the public queue: $OnlyJobId",
            source,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
