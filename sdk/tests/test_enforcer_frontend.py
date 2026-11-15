"""Tests for the enforcer law engine and the stdlib HTTP frontend."""

import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from szl import enforcer, frontend, kamachikuna, openshell  # noqa: E402

SECRET = "test-secret"


class EnforcerTests(unittest.TestCase):
    def test_all_ten_laws_classified(self):
        engine = enforcer.LawEngine(secret=SECRET)
        status = engine.law_status()
        self.assertEqual(sorted(status), list(range(1, 11)))
        for n in (5, 6, 7, 8, 9):
            self.assertEqual(status[n]["status"], "ENFORCED")
        for n in (1, 2, 3, 4, 10):
            self.assertEqual(status[n]["status"], "DOCTRINE")
        self.assertEqual(
            set(enforcer.ENFORCED_LAWS) | set(enforcer.DOCTRINE_LAWS),
            set(kamachikuna.LAWS),
        )

    def test_allowed_decision_carries_signed_receipt(self):
        engine = enforcer.LawEngine(secret=SECRET)
        d = engine.evaluate("train.launch", True, receipt={"sha256": "ab" * 32})
        self.assertEqual(d["decision"], "ALLOWED")
        self.assertTrue(engine.verify_receipt(d["receipt"]))

    def test_blocked_when_unverified(self):
        engine = enforcer.LawEngine(secret=SECRET)
        d = engine.evaluate("train.launch", False)
        self.assertEqual(d["decision"], "BLOCKED")
        self.assertIn(6, d["laws"])
        self.assertTrue(engine.verify_receipt(d["receipt"]))  # denials receipted too (Law 5)

    def test_irreversible_is_human_only(self):
        engine = enforcer.LawEngine(secret=SECRET)
        d = engine.evaluate("payment.execute", True, receipt={"sha256": "cd" * 32})
        self.assertEqual(d["decision"], "BLOCKED")
        self.assertIn(7, d["laws"])

    def test_no_secret_means_nothing_verifies(self):
        engine = enforcer.LawEngine(secret=None)
        self.assertFalse(engine.verify_receipt({"anything": True}))
        self.assertFalse(engine.verify_chain([]))
        d = engine.evaluate("train.launch", True, receipt={"sha256": "ab" * 32})
        self.assertEqual(d["decision"], "ALLOWED")
        self.assertIsNone(d["receipt"])  # unsigned allow receipt never minted

    def test_decision_matches_openshell_gate(self):
        engine = enforcer.LawEngine(secret=SECRET)
        for verified in (True, False):
            d = engine.evaluate("eval.run", verified, receipt={"sha256": "ab" * 32})
            o = openshell.admission_gate("eval.run", verified, receipt={"sha256": "ab" * 32})
            self.assertEqual(d["decision"], o["decision"])
            self.assertEqual(d["laws"], o["laws"])


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = frontend.serve(port=0, secret=SECRET)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
            return r.status, json.loads(r.read())

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())

    def test_healthz(self):
        status, body = self._get("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_laws_endpoint(self):
        status, body = self._get("/laws")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["laws"]), 10)
        self.assertEqual(len(body["primePrinciples"]), 5)

    def test_policy_endpoint(self):
        status, body = self._get("/policy")
        self.assertEqual(status, 200)
        self.assertEqual(body["spec"]["default"], "deny")

    def test_gate_allowed_then_verify(self):
        status, d = self._post("/gate", {"action": "eval.run", "verified": True,
                                         "receipt": {"sha256": "ab" * 32}})
        self.assertEqual(status, 200)
        self.assertEqual(d["decision"], "ALLOWED")
        status, v = self._post("/verify", {"receipt": d["receipt"]})
        self.assertTrue(v["verified"])

    def test_gate_blocked_and_bad_action(self):
        _, d = self._post("/gate", {"action": "eval.run", "verified": False})
        self.assertEqual(d["decision"], "BLOCKED")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/gate", {"verified": True})
        self.assertEqual(cm.exception.code, 400)

    def test_verify_rejects_tampered_receipt(self):
        _, d = self._post("/gate", {"action": "eval.run", "verified": True,
                                    "receipt": {"sha256": "ab" * 32}})
        d["receipt"]["claim"]["decision"]["decision"] = "ALLOWED-FORGED"
        _, v = self._post("/verify", {"receipt": d["receipt"]})
        self.assertFalse(v["verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
