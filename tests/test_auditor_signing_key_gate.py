"""Source-only fail-closed auditor signing regressions; no live keys or network."""

import importlib.util
import json
import os
import sys
import tempfile
from unittest import TestCase, main, mock
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "hf-spaces/eternal-auditor/app.py"
RETIRED_PUBLIC_FALLBACK = "szl-auditor-v2-default"


class SigningKeyGateTests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / "state.json"

    def load(self, secret):
        with (
            mock.patch.dict(
                os.environ,
                {
                    "AUDITOR_STATE": str(self.state),
                    "AUDITOR_SECRET": secret,
                    "HF_TOKEN": "",
                },
            ),
            mock.patch.dict(sys.modules, {"fastapi": None, "huggingface_hub": None}),
            mock.patch(
                "threading.Thread.start", side_effect=AssertionError("worker forbidden")
            ),
            mock.patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("network forbidden"),
            ),
        ):
            spec = importlib.util.spec_from_file_location(
                "auditor_key_gate_fixture", APP
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module

    def test_missing_key_cannot_mint_or_advance_state(self):
        app = self.load("")
        before = dict(app.state)
        self.assertFalse(app.signing_key_configured(app.SECRET))
        with self.assertRaises(ValueError):
            app.run_cycle([], app.SECRET)
        with self.assertRaises(ValueError):
            app.sign("a" * 64, app.SECRET)
        self.assertFalse(
            app.verify_receipt({"kind": "szl.auditor-receipt/v2"}, app.SECRET)
        )
        self.assertEqual(app.state, before)
        self.assertFalse(self.state.exists())

    def test_retired_public_fallback_is_rejected_even_if_explicit(self):
        app = self.load(RETIRED_PUBLIC_FALLBACK)
        self.assertFalse(app.signing_key_configured(app.SECRET))
        with self.assertRaises(ValueError):
            app.run_cycle([], app.SECRET)
        with self.assertRaises(ValueError):
            app.sign("b" * 64, RETIRED_PUBLIC_FALLBACK)
        self.assertFalse(self.state.exists())

    def test_loop_without_key_never_polls_or_publishes(self):
        app = self.load("")
        before = dict(app.state)
        with (
            mock.patch.object(
                app, "audit_cycle", side_effect=AssertionError("no poll")
            ),
            mock.patch.object(app.time, "sleep", side_effect=StopIteration),
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(StopIteration):
                app.loop()
        self.assertEqual(app.state, before)
        self.assertFalse(self.state.exists())

    def test_explicit_test_fixture_still_signs_and_verifies(self):
        app = self.load("offline-test-signing-fixture")
        self.assertTrue(app.signing_key_configured(app.SECRET))
        receipt = app.run_cycle([], app.SECRET)
        self.assertEqual(len(receipt["signature"]), 64)
        self.assertTrue(app.verify_receipt(receipt, app.SECRET))
        self.assertFalse(app.verify_receipt(receipt, ""))
        self.assertEqual(json.loads(self.state.read_text())["cycle"], 1)


class AuditorConsoleStatusTests(TestCase):
    """Reject unqualified public LIVE claims when signing is unavailable."""

    def setUp(self):
        self.html = (
            Path(__file__).resolve().parents[1]
            / "hf-spaces/eternal-auditor/static/index.html"
        ).read_text(encoding="utf-8")

    def test_signer_hold_is_visible_by_default(self):
        self.assertIn('id="livepill">SIGNING UNKNOWN — HOLD', self.html)
        self.assertNotIn('id="livepill">LIVE', self.html)
        self.assertIn("state === 'HOLD_UNAVAILABLE'", self.html)
        self.assertIn("renderLiveStatus(null)", self.html)
        self.assertIn("chain.className = 'warn'", self.html)

    def test_cycle_pulse_requires_configured_signer_and_new_cycle(self):
        self.assertIn("state === 'CONFIGURED_UNVERIFIED'", self.html)
        self.assertIn("'SIGNING CONFIGURED — UNVERIFIED'", self.html)
        self.assertIn("cycle > lastObservedCycle", self.html)
        self.assertIn("if (!configured) pulse = 0;", self.html)
        self.assertNotIn("setInterval(async()=>{try{const L=", self.html)


if __name__ == "__main__":
    main()
