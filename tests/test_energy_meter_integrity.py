"""Deterministic, CPU-only regression tests for complete energy windows."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "energy_integrity", ROOT / "scripts/energy_meter.py"
)
energy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(energy)


class Clock:
    now = 10.0
    wall = 1000.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.wall


def fake_nvml(values, counter=True):
    queue = iter(values)
    fake = types.ModuleType("pynvml")
    fake.nvmlInit = lambda: None
    fake.nvmlDeviceGetHandleByIndex = lambda index: index

    def sample(handle):
        result = next(queue)
        if isinstance(result, Exception):
            raise result
        return result

    def unsupported(handle):
        raise RuntimeError("synthetic unsupported energy counter")

    fake.nvmlDeviceGetTotalEnergyConsumption = sample if counter else unsupported
    fake.nvmlDeviceGetPowerUsage = unsupported if counter else sample
    return fake


class EnergyIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.addCleanup(patch.stopall)
        patch.object(energy.time, "time", self.clock.time).start()
        patch.object(energy.time, "monotonic", self.clock.monotonic).start()

    def meter(self, values, counter=True):
        patch.dict(sys.modules, {"pynvml": fake_nvml(values, counter)}).start()
        return energy.EnergyMeter()

    def assert_unavailable(self, result):
        self.assertEqual(result["energy_state"], "UNAVAILABLE")
        self.assertIsNone(result["energy_joules"])
        json.dumps(result, allow_nan=False)

    def test_partial_integral_after_read_failure_is_not_full_window(self):
        m = self.meter(
            [100000, 100000, 100000, RuntimeError("synthetic read failure"), 100000],
            False,
        )
        m.start()
        self.clock.now = 11.0
        self.clock.wall = 1001.0
        m.tick()
        self.clock.now = 12.0
        self.clock.wall = 1002.0
        m.tick()
        self.clock.now = 13.0
        self.clock.wall = 1003.0
        result = m.stop()
        self.assert_unavailable(result)
        self.assertEqual(result["samples_err"], 1)

    def test_failed_final_power_read_does_not_leave_measured_prefix(self):
        m = self.meter(
            [100000, 100000, 100000, RuntimeError("synthetic stop failure")], False
        )
        m.start()
        self.clock.now = 11.0
        self.clock.wall = 1001.0
        m.tick()
        self.clock.now = 12.0
        self.clock.wall = 1002.0
        self.assert_unavailable(m.stop())

    def test_invalid_counter_values_are_rejected_before_counting_success(self):
        for invalid in [float("inf"), float("nan"), -1, True, "1000", None]:
            with self.subTest(value=repr(invalid)):
                m = self.meter([1000, 2000, invalid])
                m.start()
                self.clock.now += 1
                result = m.stop()
                self.assert_unavailable(result)
                self.assertEqual(result["samples_ok"], 1)
                self.assertEqual(result["samples_err"], 1)

    def test_negative_start_counter_is_not_valid_baseline(self):
        m = self.meter([1000, -2000, 3500])
        m.start()
        self.assert_unavailable(m.stop())

    def test_invalid_power_values_are_rejected(self):
        for invalid in [float("inf"), float("nan"), -1, True, "100000", None]:
            with self.subTest(value=repr(invalid)):
                m = self.meter([100000, 100000, invalid], False)
                m.start()
                self.clock.now += 1
                self.clock.wall += 1
                result = m.stop()
                self.assert_unavailable(result)
                self.assertEqual(result["samples_err"], 1)

    def test_counter_reset_is_explicit_measurement_error(self):
        m = self.meter([1000, 2000, 500])
        m.start()
        result = m.stop()
        self.assert_unavailable(result)
        self.assertGreater(result["samples_err"], 0)

    def test_wall_clock_jump_does_not_distort_power_integration(self):
        m = self.meter([100000, 100000, 100000], False)
        m.start()
        self.clock.now = 12.0
        self.clock.wall = -10000.0
        result = m.stop()
        self.assertEqual(result["energy_state"], "MEASURED")
        self.assertEqual(result["energy_joules"], 200.0)
        self.assertEqual(result["wall_seconds"], 2.0)

    def test_zero_origin_monotonic_clock_records_duration(self):
        self.clock.now = self.clock.wall = 0.0
        m = self.meter([1000, 2000, 3000])
        m.start()
        self.clock.now = self.clock.wall = 2.0
        self.assertEqual(m.stop()["wall_seconds"], 2.0)

    def test_stop_is_idempotent_and_result_is_not_mutable_state(self):
        m = self.meter([1000, 2000, 3000, 5000])
        m.start()
        self.clock.now = self.clock.wall = 12.0
        first = m.stop()
        expected = copy.deepcopy(first)
        first["errors"].append("caller mutation")
        self.clock.now = self.clock.wall = 14.0
        self.assertEqual(m.stop(), expected)

    def test_restart_same_instance_is_rejected(self):
        m = self.meter([1000, 2000, 3000])
        m.start()
        with self.assertRaises(RuntimeError):
            m.start()

    def test_stop_before_start_is_rejected(self):
        m = self.meter([1000, 2000])
        with self.assertRaises(RuntimeError):
            m.stop()

    def test_successful_counter_delta_is_preserved(self):
        m = self.meter([1000, 2000, 3500])
        m.start()
        self.clock.now = 11.0
        result = m.stop()
        self.assertEqual(result["energy_joules"], 1.5)
        self.assertEqual(result["energy_mode"], "total-energy")
        self.assertEqual(result["samples_err"], 0)

    def test_genuine_zero_counter_delta_is_not_sampling_failure(self):
        m = self.meter([1000, 2000, 2000])
        m.start()
        self.assertEqual(m.stop()["energy_joules"], 0.0)

    def test_genuine_zero_power_interval_is_measured(self):
        m = self.meter([0, 0, 0], False)
        m.start()
        self.clock.now = self.clock.wall = 12.0
        result = m.stop()
        self.assertEqual(result["energy_state"], "MEASURED")
        self.assertEqual(result["energy_joules"], 0.0)

    def test_zero_counter_selftest_is_consistent(self):
        patch.dict(sys.modules, {"pynvml": fake_nvml([1000, 2000, 2000])}).start()
        self.assertEqual(energy.self_test()["self_test"], "PASS")


if __name__ == "__main__":
    unittest.main()
