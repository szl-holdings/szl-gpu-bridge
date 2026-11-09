"""Offline tests for the energy meter and officer fixes. No GPU required.

Runnable standalone (``python3 tests/test_energy_and_officers.py``) and by
CI (``python -m unittest discover -s tests``): importing this module has no
side effects.
"""

import hashlib
import hmac
import importlib.util
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


ENERGY_METER = ROOT / "scripts" / "energy_meter.py"
GARDENER_APP = ROOT / "hf-spaces" / "estate-gardener" / "app.py"
AUDITOR_APP = ROOT / "hf-spaces" / "eternal-auditor" / "app.py"


def make_fake_nvml(energy_values=None, power_watts=None, total_energy_broken=False):
    fake = types.ModuleType("pynvml")

    def nvmlInit():
        pass

    def nvmlDeviceGetHandleByIndex(i):
        return ("handle", i)

    fake.nvmlInit = nvmlInit
    fake.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
    if total_energy_broken:

        def broken(h):
            raise AttributeError("no such symbol")

        fake.nvmlDeviceGetTotalEnergyConsumption = broken
    elif energy_values is not None:
        queue = list(energy_values)

        def get_energy(h):
            return queue.pop(0) if queue else energy_values[-1]

        fake.nvmlDeviceGetTotalEnergyConsumption = get_energy
    if power_watts is not None:
        state = {"w": power_watts}

        def get_power(h):
            return state["w"]

        fake.nvmlDeviceGetPowerUsage = get_power
    return fake


class EnergyMeterTests(unittest.TestCase):
    em = None

    @classmethod
    def setUpClass(cls):
        cls.em = load_module("energy_meter_under_test", ENERGY_METER)

    def setUp(self):
        sys.modules.pop("pynvml", None)

    def tearDown(self):
        sys.modules.pop("pynvml", None)

    def test_no_gpu_reports_unavailable_never_zero(self):
        r = self.em.self_test()
        self.assertEqual(r["self_test"], "PASS")
        self.assertEqual(r["energy_state"], "UNAVAILABLE")
        self.assertIsNone(r["energy_joules"])
        self.assertIsNone(r["energy_mode"])

    def test_total_energy_mode_measures_joules(self):
        # mJ: probe, start, stop -> 2000, 3500 => 1.5 J
        sys.modules["pynvml"] = make_fake_nvml(energy_values=[1000.0, 2000.0, 3500.0])
        m = self.em.EnergyMeter()
        m.start()
        r = m.stop()
        self.assertEqual(r["energy_mode"], "total-energy")
        self.assertEqual(r["energy_joules"], 1.5)
        self.assertEqual(r["energy_state"], "MEASURED")
        self.assertEqual(r["samples_err"], 0)

    def test_power_integral_fallback(self):
        sys.modules["pynvml"] = make_fake_nvml(
            power_watts=300.0, total_energy_broken=True
        )
        m = self.em.EnergyMeter()
        m.start()
        for _ in range(5):
            time.sleep(0.02)
            m.tick()
        r = m.stop()
        self.assertEqual(r["energy_mode"], "power-integral")
        self.assertIsNotNone(r["energy_joules"])
        self.assertGreater(r["energy_joules"], 0)
        self.assertEqual(r["energy_state"], "MEASURED")

    def test_dead_meter_mid_run_is_unavailable_not_zero(self):
        def dead(h):
            raise RuntimeError("NVML not initialized")

        fake = make_fake_nvml(total_energy_broken=True)
        fake.nvmlDeviceGetPowerUsage = dead
        sys.modules["pynvml"] = fake
        m = self.em.EnergyMeter()
        m.start()
        m.tick()
        r = m.stop()
        self.assertEqual(r["energy_state"], "UNAVAILABLE")
        self.assertIsNone(r["energy_joules"])
        self.assertGreater(r["samples_err"], 0)
        self.assertTrue(r["errors"])


class GardenerTests(unittest.TestCase):
    def _gardener(self, tmpdir):
        import os

        os.environ["GARDENER_STATE"] = str(Path(tmpdir) / "state.json")
        return load_module(f"gardener_app_{id(tmpdir)}", GARDENER_APP)

    def test_append_only_preserves_content(self):
        with tempfile.TemporaryDirectory() as td:
            g = self._gardener(td)
            original = (
                "# My Dataset\n\nCurated by Dr. Sara Illion, CC-BY-4.0 requested.\n"
                "Real content here."
            )
            receipt, new = g.garden_readme("test/ds", original, has_license_tag=False)
            self.assertEqual(receipt["action"], "APPEND")
            self.assertTrue(new.startswith(original))
            self.assertGreater(len(new), len(original))
            self.assertTrue(receipt["preserved_prefix"])

    def test_tagged_readme_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            g = self._gardener(td)
            original = "# My Dataset\n\ncontent\n"
            _, action = g.append_license_block(original, True)
            self.assertEqual(action, "SKIP_ALREADY_TAGGED")
            receipt, _ = g.garden_readme("test/ds2", original, has_license_tag=True)
            self.assertEqual(receipt["action"], "SKIP_ALREADY_TAGGED")

    def test_empty_readme_gets_block_without_loss(self):
        with tempfile.TemporaryDirectory() as td:
            g = self._gardener(td)
            new, action = g.append_license_block("", False)
            self.assertEqual(action, "WRITE_EMPTY")
            self.assertTrue(new.strip())


class AuditorTests(unittest.TestCase):
    def _auditor(self, tmpdir):
        import os

        os.environ["AUDITOR_STATE"] = str(Path(tmpdir) / "audit-state.json")
        return load_module(f"auditor_app_{id(tmpdir)}", AUDITOR_APP)

    def test_full_signature_and_chain_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            a = self._auditor(td)
            finding = [{"target": "x", "check": "y", "result": "PASS"}]
            r1 = a.run_cycle(finding, secret="s3cret")
            self.assertEqual(len(r1["signature"]), 64)
            self.assertEqual(r1["signature_length"], 64)
            self.assertIsNone(r1["prev_receipt_sha256"])
            # simulate restart: reload module, state must persist
            a2 = self._auditor(td)
            r2 = a2.run_cycle(finding, secret="s3cret")
            self.assertEqual(r2["cycle"], 2)
            self.assertEqual(r2["prev_receipt_sha256"], r1["body_sha256"])
            self.assertEqual(r2["chain_state"], "CONTINUOUS")
            expect = hmac.new(
                b"s3cret", r2["body_sha256"].encode(), hashlib.sha256
            ).hexdigest()
            self.assertEqual(r2["signature"], expect)


if __name__ == "__main__":
    unittest.main(verbosity=2)
