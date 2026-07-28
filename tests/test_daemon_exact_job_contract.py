from __future__ import annotations

import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON = ROOT / "laptop" / "daemon.ps1"


class DaemonExactJobContractTests(unittest.TestCase):
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
