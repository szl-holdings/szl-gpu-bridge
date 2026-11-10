"""Offline tests for eternal-auditor receipt verification (gallery must not
display unverified signatures as proof). No network, no GPU."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDITOR_APP = ROOT / "hf-spaces" / "eternal-auditor" / "app.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class AuditorVerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as td:
            import os
            os.environ["AUDITOR_STATE"] = str(Path(td) / "state.json")
            cls.app = load_module("auditor_app_verify", AUDITOR_APP)

    def test_run_cycle_receipt_verifies(self):
        r = self.app.run_cycle([{"check": "x", "verdict": "PASS"}], secret="s")
        self.assertTrue(self.app.verify_receipt(r, "s"))
        self.assertEqual(len(r["signature"]), 64)

    def test_tampered_and_wrong_secret_fail(self):
        r = self.app.run_cycle([], secret="s")
        r2 = dict(r)
        r2["cycle"] = 999
        self.assertFalse(self.app.verify_receipt(r2, "s"))
        self.assertFalse(self.app.verify_receipt(r, "wrong"))

    def test_truncated_signature_fails(self):
        r = self.app.run_cycle([], secret="s")
        r["signature"] = r["signature"][:48]
        self.assertFalse(self.app.verify_receipt(r, "s"))

    def test_gallery_keeps_full_receipts_for_verification(self):
        st = {"cycle": 0, "prev": None, "chain_state": "GENESIS"}
        r = self.app.run_cycle([], secret="s")
        self.app.remember(st, r)
        self.assertIn("findings", st["history"][-1])  # full body retained
        self.assertTrue(self.app.verify_receipt(st["history"][-1], "s"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
