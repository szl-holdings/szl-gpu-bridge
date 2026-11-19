"""The bootstrap must upgrade pip through the interpreter, never through pip.exe.

On Windows, pip.exe cannot replace itself: in a fresh environment,
``pip.exe install --upgrade pip`` exits 1 with "To modify pip, please run ...
-m pip", and the bootstrap stops with "pip toolchain upgrade failed". This was
observed on BETTERWITHAGE on 2026-11-19.
"""

from __future__ import annotations

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "laptop" / "bootstrap.ps1"


class BootstrapPipSelfUpgradeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = BOOTSTRAP.read_text(encoding="utf-8")

    def test_pip_is_upgraded_through_python_module(self) -> None:
        self.assertEqual(
            self.source.count("& $Py -m pip install --upgrade pip"),
            1,
        )

    def test_pip_executable_never_upgrades_itself(self) -> None:
        self.assertIsNone(
            re.search(r"&\s*\$Pip\s+install\s+--upgrade\s+pip", self.source)
        )

    def test_upgrade_precedes_torch_and_keeps_its_failure_message(self) -> None:
        upgrade = self.source.index("& $Py -m pip install --upgrade pip")
        failure = self.source.index('throw "pip toolchain upgrade failed"')
        torch = self.source.index("install torch --index-url")
        self.assertLess(upgrade, failure)
        self.assertLess(failure, torch)

    def test_interpreter_is_defined_before_use(self) -> None:
        definition = self.source.index(
            '$Py  = "$CondaBase\\envs\\szl-bridge\\python.exe"'
        )
        self.assertLess(
            definition, self.source.index("& $Py -m pip install --upgrade pip")
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
