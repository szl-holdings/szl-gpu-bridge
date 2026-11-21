"""Untrusted disk-usage values must not become healthy readings by coercion.

These offline tests perform no I/O beyond private temporary directory creation,
make no network calls, and never dispatch training or remove any data.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from laptop import storage_preflight as gate


class CapacitySnapshotTypeTests(unittest.TestCase):
    def test_non_integer_disk_usage_values_fail_closed(self):
        baseline = {
            "total": 128 * gate.GIB,
            "free": 64 * gate.GIB,
            "used": 64 * gate.GIB,
        }
        invalid = (True, False, 64.5, "68719476736", None)
        with tempfile.TemporaryDirectory() as tmp:
            for field in ("total", "free", "used"):
                for value in invalid:
                    with self.subTest(field=field, value=repr(value)):
                        snapshot = dict(baseline)
                        snapshot[field] = value
                        result = gate.probe(
                            Path(tmp),
                            samples=3,
                            interval_sec=0,
                            usage=lambda _path, s=snapshot: SimpleNamespace(**s),
                            sleeper=lambda _: None,
                            clock=lambda: "2026-11-21T00:00:00+00:00",
                        )
                        self.assertEqual(
                            result["verdict"]["state"], "HOLD_UNAVAILABLE_STORAGE_PROBE"
                        )
                        self.assertEqual(result["observed_samples"], [])
                        self.assertEqual(result["observation_error_type"], "ValueError")
                        self.assertFalse(result["authorizes_flagship_dispatch"])

    def test_exact_native_integer_readings_keep_storage_only_pass(self):
        snapshot = SimpleNamespace(
            total=128 * gate.GIB,
            free=64 * gate.GIB,
            used=64 * gate.GIB,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = gate.probe(
                Path(tmp),
                samples=3,
                interval_sec=0,
                usage=lambda _path: snapshot,
                sleeper=lambda _: None,
                clock=lambda: "2026-11-21T00:00:00+00:00",
            )
        self.assertEqual(result["verdict"]["state"], "CAPACITY_GATE_PASS_ONLY")
        self.assertEqual(result["observed_samples"][0]["free_bytes"], 64 * gate.GIB)
        self.assertFalse(result["authorizes_flagship_dispatch"])


if __name__ == "__main__":
    unittest.main()
