"""Known power curves, missing measurements, and incomplete benchmark receipts."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_power", ROOT / "laptop/benchmark_power.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PowerTests(unittest.TestCase):
    def make_sampler(self, values):
        self.now = 0.0
        values = iter(values)

        def read():
            value = next(values)
            if isinstance(value, Exception):
                raise value
            return value

        return module.PowerSampler(read, interval=1000, clock=lambda: self.now)

    def test_integrates_known_ramp_with_monotonic_time(self):
        sampler = self.make_sampler([10, 20])
        sampler.start()
        self.now = 2.0
        sampler.stop()
        r = sampler.report(10)
        self.assertEqual(r["joules_measured"], 30)
        self.assertEqual(r["joules_per_token"], 3)
        self.assertEqual(r["avg_watts"], 15)
        self.assertEqual(r["status"], "MEASURED_POWER_INTEGRATED")

    def test_read_error_does_not_become_zero_watts(self):
        sampler = self.make_sampler([10, RuntimeError("NVML failed")])
        sampler.start()
        self.now = 2.0
        sampler.stop()
        r = sampler.report(10)
        self.assertEqual(r["status"], "INCOMPLETE")
        self.assertEqual(r["sampling_errors"], 1)
        for key in [
            "joules_measured",
            "avg_watts",
            "kwh",
            "joules_per_token",
            "co2_kg_estimated_us_average",
        ]:
            self.assertIsNone(r[key])

    def test_missing_interval_is_not_interpolated(self):
        sampler = self.make_sampler([10, RuntimeError("missing"), 20])
        sampler.start()
        self.now = 1.0
        sampler.sample()
        self.now = 2.0
        sampler.stop()
        self.assertEqual(sampler.covered_seconds, 0)
        self.assertIsNone(sampler.report(10)["joules_measured"])

    def test_invalid_readings_are_incomplete(self):
        for bad in [float("nan"), float("inf"), -1]:
            sampler = self.make_sampler([10, bad])
            sampler.start()
            self.now = 1.0
            sampler.stop()
            self.assertEqual(sampler.report(10)["status"], "INCOMPLETE")

    def test_zero_power_is_valid_but_zero_duration_is_not(self):
        sampler = self.make_sampler([0, 0])
        sampler.start()
        self.now = 1
        sampler.stop()
        self.assertEqual(sampler.report(10)["joules_measured"], 0)
        sampler = self.make_sampler([0, 0])
        sampler.start()
        sampler.stop()
        self.assertIsNone(sampler.report(10)["joules_measured"])

    def test_restart_and_reporting_live_are_rejected(self):
        sampler = self.make_sampler([10, 10])
        sampler.start()
        try:
            with self.assertRaises(RuntimeError):
                sampler.start()
            with self.assertRaises(RuntimeError):
                sampler.report(10)
        finally:
            sampler.stop()


if __name__ == "__main__":
    unittest.main()
