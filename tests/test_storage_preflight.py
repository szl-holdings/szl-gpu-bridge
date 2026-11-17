"""Offline tests for fail-closed workstation storage gate (no disk cleanup)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "laptop" / "storage_preflight.py"
spec = importlib.util.spec_from_file_location(
    "szl_storage_preflight_test_module", SOURCE
)
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)
GIB = guard.GIB


def samples(*gib_values):
    return [
        guard.Reading(
            observed_utc=f"2026-11-17T15:00:{i:02d}+00:00",
            total_bytes=1024 * GIB,
            free_bytes=int(v * GIB),
            used_bytes=1024 * GIB - int(v * GIB),
        )
        for i, v in enumerate(gib_values)
    ]


class CapacityPolicyTests(unittest.TestCase):
    def test_stable_above_50gib_is_only_storage_pass(self):
        o = guard.classify(samples(60, 61, 59.5))
        self.assertEqual(o["state"], "CAPACITY_GATE_PASS_ONLY")
        self.assertTrue(o["storage_only_passed"])
        self.assertFalse(o["other_training_qualification_gates_checked"])
        self.assertEqual(o["sample_count"], 3)

    def test_low_free_space_refuses(self):
        o = guard.classify(samples(0.25, 2.2, 0.3))
        self.assertEqual(o["state"], "HOLD")
        self.assertIn("HOLD_LOW_MINIMUM_FREE_SPACE", o["reasons"])

    def test_large_swing_refuses_even_above_minimum(self):
        o = guard.classify(samples(70, 60, 70))
        self.assertFalse(o["storage_only_passed"])
        self.assertEqual(o["reasons"], ["HOLD_UNSTABLE_FREE_SPACE"])

    def test_both_low_and_unstable_refuse(self):
        o = guard.classify(samples(0.25, 10.6, 0.3))
        self.assertEqual(
            set(o["reasons"]),
            {"HOLD_LOW_MINIMUM_FREE_SPACE", "HOLD_UNSTABLE_FREE_SPACE"},
        )

    def test_threshold_boundary_and_exact_swing(self):
        self.assertTrue(guard.classify(samples(50, 52, 50))["storage_only_passed"])
        self.assertFalse(guard.classify(samples(49.999, 52, 50))["storage_only_passed"])

    def test_count_and_type_refuse(self):
        for seq in ([], samples(50, 50), samples(*([50] * 13))):
            with self.assertRaisesRegex(ValueError, "SAMPLE_COUNT"):
                guard.classify(seq)
        with self.assertRaisesRegex(ValueError, "BAD_READING"):
            guard.classify([1, 2, 3])

    def test_invalid_disk_report_fails(self):
        broken = guard.Reading(
            "2026-11-17Z", total_bytes=10, free_bytes=11, used_bytes=0
        )
        with self.assertRaisesRegex(ValueError, "INVALID_CAPACITY"):
            guard.classify([broken] * 3)
        changed = [*samples(50, 51, 52)]
        changed[2] = guard.Reading(
            changed[2].observed_utc,
            changed[2].total_bytes + 10,
            changed[2].free_bytes,
            changed[2].used_bytes,
        )
        with self.assertRaisesRegex(ValueError, "FILESYSTEM_IDENTITY"):
            guard.classify(changed)

    def test_unapproved_policy_rejected(self):
        for v in (0, -1, 2049, True):
            with self.assertRaisesRegex(ValueError, "POLICY"):
                guard.classify(samples(50, 51, 52), min_free_gib=v)

    def test_probe_reads_actual_usage_samples_without_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            values = [70, 69.5, 69]
            seen = []

            def usage(p):
                seen.append(str(p))
                v = values[len(seen) - 1]
                return SimpleNamespace(
                    total=1024 * GIB, free=int(v * GIB), used=int((1024 - v) * GIB)
                )

            report = guard.probe(
                Path(folder),
                samples=3,
                interval_sec=0,
                usage=usage,
                sleeper=lambda sec: None,
                clock=lambda: "2026-11-17T15:00:00+00:00",
            )
            self.assertEqual(len(seen), 3)
            self.assertTrue(report["verdict"]["storage_only_passed"])
            self.assertFalse(report["authorizes_flagship_dispatch"])
            self.assertFalse(report["files_removed"])
            self.assertFalse(report["watchdog_tasks_changed"])

    def test_missing_path_and_disk_failure_refuse(self):
        report = guard.probe(
            Path("/path-not-present-szl-capacity"),
            samples=3,
            interval_sec=0,
            sleeper=lambda _: None,
        )
        self.assertEqual(report["verdict"]["state"], "HOLD_UNAVAILABLE_STORAGE_PROBE")
        with tempfile.TemporaryDirectory() as folder:

            def dead(p):
                raise OSError("IO_failure")

            report = guard.probe(
                Path(folder),
                samples=3,
                interval_sec=0,
                usage=dead,
                sleeper=lambda _: None,
            )
            self.assertFalse(report["verdict"]["storage_only_passed"])
            self.assertEqual(report["observation_error_type"], "OSError")

    def test_bounded_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            for n in (0, 2, 13):
                with self.assertRaisesRegex(ValueError, "SAMPLE_COUNT"):
                    guard.probe(Path(folder), samples=n)
            with self.assertRaisesRegex(ValueError, "INVALID_INTERVAL"):
                guard.probe(Path(folder), interval_sec=10)

    def test_append_only_receipt_and_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "owner" / "receipt.json"
            data = {"schema": "synthetic-test", "state": "HOLD"}
            digest = guard.write_exclusive_receipt(target, data)
            self.assertEqual(len(digest), 64)
            self.assertEqual(json.loads(target.read_text()), data)
            with self.assertRaises(FileExistsError):
                guard.write_exclusive_receipt(target, {"state": "FAKE_PASS"})
            self.assertEqual(json.loads(target.read_text()), data)

    def test_cli_missing_path_hold_exit_78(self):
        with patch.object(
            sys,
            "argv",
            [
                "storage_preflight",
                "--root",
                "/not-a-drive-szl",
                "--samples",
                "3",
                "--interval-sec",
                "0",
            ],
        ):
            self.assertEqual(guard.main(), guard.REFUSED_EXIT)


if __name__ == "__main__":
    unittest.main()
