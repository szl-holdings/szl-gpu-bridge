"""The bridge daemon must pass the fixed storage gate before every dispatch.

Start, restart and resume ordering: the preflight runs inside the dispatch
loop after the single-flight lock and before dispatcher.py, so each cycle
(including the first one after a reboot and every retry of a held job)
re-measures storage. A HOLD never mutates the job ledger.
"""

from __future__ import annotations

import importlib.util
import io
import pathlib
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON = ROOT / "laptop" / "daemon.ps1"
BOOTSTRAP = ROOT / "laptop" / "bootstrap.ps1"
PREFLIGHT = ROOT / "laptop" / "storage_preflight.py"

spec = importlib.util.spec_from_file_location(
    "szl_daemon_storage_preflight_test_module", PREFLIGHT
)
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)

INVOCATION_MARKER = "& $Py -I -B $Preflight "
DISPATCH = '& $Py "$Root\\dispatcher.py" $specPath'
LEDGER_WRITE = "Add-Content -Path $Ledger"


def _daemon() -> str:
    return DAEMON.read_text(encoding="utf-8")


def _invocation_line(source: str) -> str:
    lines = [line for line in source.splitlines() if INVOCATION_MARKER in line]
    if len(lines) != 1:
        raise AssertionError(f"expected one preflight invocation, found {len(lines)}")
    return lines[0].strip()


def _gate_block(source: str) -> str:
    start = source.index("if (-not (Test-Path -LiteralPath $Preflight")
    return source[start : source.index(DISPATCH)]


class DaemonStoragePreflightOrderingTests(unittest.TestCase):
    def test_gate_runs_after_lock_inside_loop_and_before_dispatch(self) -> None:
        source = _daemon()
        lock_check = source.index("$LockHandle = [System.IO.File]::Open(")
        loop = source.index("foreach ($f in $files)")
        seen_skip = source.index("if ($seen[$jobId]) { continue }")
        invocation = source.index(INVOCATION_MARKER)
        dispatch = source.index(DISPATCH)

        self.assertLess(lock_check, loop)
        self.assertLess(loop, seen_skip)
        self.assertLess(seen_skip, invocation)
        self.assertLess(invocation, dispatch)
        self.assertEqual(source.count(DISPATCH), 1)

    def test_exclusive_lease_is_os_owned_fail_closed_and_not_deleted(self) -> None:
        source = _daemon()
        lease = source.index("$LockHandle = [System.IO.File]::Open(")
        network = source.index("Invoke-RestMethod -Uri")
        dispatch = source.index(DISPATCH)
        self.assertLess(lease, network)
        self.assertLess(lease, dispatch)
        for required in (
            "[System.IO.FileMode]::OpenOrCreate",
            "[System.IO.FileAccess]::ReadWrite",
            "[System.IO.FileShare]::None",
            "$LockHandle.Dispose()",
            "exit 78",
        ):
            self.assertIn(required, source)
        self.assertNotIn("New-Item -ItemType File -Path $Lock -Force", source)
        self.assertNotIn("Remove-Item $Lock", source)
        self.assertNotIn("if (Test-Path $Lock)", source)

    def test_thresholds_are_fixed_constants_not_spec_values(self) -> None:
        line = _invocation_line(_daemon())
        for required in (
            "--root 'C:\\'",
            "--samples 5",
            "--interval-sec 2",
            "--min-free-gib 50",
            "--max-swing-gib 2",
            "--receipt $preflightReceipt",
        ):
            self.assertIn(required, line)
        for forbidden in ("$spec", "gates", "minFreeDiskGb", "$f.", "$env:"):
            self.assertNotIn(forbidden, line)

    def test_preflight_script_path_is_fixed_under_bridge_root(self) -> None:
        source = _daemon()
        self.assertIn('$Preflight = "$Root\\storage_preflight.py"', source)
        self.assertIn('$PreflightEvidence = "$Root\\logs\\storage-preflight"', source)
        self.assertEqual(source.count("$Preflight ="), 1)

    def test_hold_paths_stop_the_cycle_without_ledger_mutation(self) -> None:
        block = _gate_block(_daemon())
        self.assertNotIn(LEDGER_WRITE, block)
        self.assertNotIn("Remove-Item", block)
        self.assertNotIn("continue", block)
        self.assertEqual(block.count("break"), 2)
        self.assertIn("$preflightCode = $LASTEXITCODE", block)
        self.assertIn("if ($preflightCode -ne 0) {", block)

    def test_missing_preflight_script_fails_closed_before_invocation(self) -> None:
        block = _gate_block(_daemon())
        missing = block.index(
            "if (-not (Test-Path -LiteralPath $Preflight -PathType Leaf))"
        )
        refuse = block.index("break", missing)
        self.assertLess(refuse, block.index(INVOCATION_MARKER))

    def test_receipt_name_does_not_embed_the_queue_file_name(self) -> None:
        source = _daemon()
        receipt_lines = [
            line
            for line in source.splitlines()
            if line.strip().startswith("$preflightReceipt =")
        ]
        self.assertEqual(len(receipt_lines), 1)
        self.assertNotIn("$jobId", receipt_lines[0])
        self.assertNotIn("$f", receipt_lines[0])
        self.assertIn("$PreflightEvidence\\", receipt_lines[0])

    def test_daemon_stays_ascii(self) -> None:
        DAEMON.read_bytes().decode("ascii")


class BootstrapShipsPreflightTests(unittest.TestCase):
    def test_bootstrap_copies_pinned_preflight_and_compiles_it(self) -> None:
        source = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn(
            '@("laptop/storage_preflight.py", "$Root\\storage_preflight.py")', source
        )
        source_precheck = source.index(
            "Reviewed checkout missing required bridge source"
        )
        first_write = source.index("New-Item -ItemType Directory -Force -Path $Root")
        source_copy = source.index(
            "Copy-Item -LiteralPath $source -Destination $Pair[1] -Force"
        )
        copy_verify = source.index(
            "Installed file does not match the pinned local checkout"
        )
        compile_start = source.index("& $Py -m py_compile")
        compile_end = source.index(
            'throw "reviewed source bridge Python failed compilation"'
        )
        self.assertLess(source_precheck, first_write)
        self.assertLess(first_write, source_copy)
        self.assertLess(source_copy, copy_verify)
        self.assertLess(copy_verify, compile_start)
        self.assertIn(
            '"$Root\\storage_preflight.py"', source[compile_start:compile_end]
        )
        self.assertLess(compile_end, source.index("Register-ScheduledTask"))


class DaemonArgumentsMatchReviewedCliTests(unittest.TestCase):
    """Run the reviewed CLI with the daemon's exact argument vector."""

    def _argv(self, root: pathlib.Path, receipt: pathlib.Path) -> list[str]:
        tail = _invocation_line(_daemon()).split(INVOCATION_MARKER, 1)[1]
        tail = tail.split("*>>", 1)[0]
        argv = []
        for token in tail.split():
            if token == "'C:\\'":
                argv.append(str(root))
            elif token == "$preflightReceipt":
                argv.append(str(receipt))
            else:
                self.assertIsNone(re.search(r"[$'\"]", token), token)
                argv.append(token)
        return argv

    def _run(self, free_gib: list[float]) -> tuple[int, pathlib.Path]:
        values = iter(free_gib)
        total = 1024 * guard.GIB
        real_probe = guard.probe

        def usage(_root):
            free = int(next(values) * guard.GIB)
            return SimpleNamespace(total=total, free=free, used=total - free)

        def probe(root, **kwargs):
            return real_probe(root, usage=usage, sleeper=lambda _s: None, **kwargs)

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        base = pathlib.Path(temp.name)
        receipt = base / "logs" / "storage-preflight" / "r.json"
        with patch.object(guard, "probe", probe), redirect_stdout(io.StringIO()):
            code = guard.main(self._argv(base, receipt))
        return code, receipt

    def test_stable_capacity_above_floor_admits(self) -> None:
        code, receipt = self._run([60, 60.2, 60.1, 60.3, 60.2])
        self.assertEqual(code, 0)
        self.assertTrue(receipt.is_file())

    def test_capacity_below_floor_holds(self) -> None:
        code, receipt = self._run([31.3, 31.3, 31.2, 31.2, 31.2])
        self.assertEqual(code, guard.REFUSED_EXIT)
        self.assertTrue(receipt.is_file())

    def test_observed_volatility_holds_even_above_floor(self) -> None:
        code, _receipt = self._run([80, 61, 80, 61, 80])
        self.assertEqual(code, guard.REFUSED_EXIT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
