"""Offline receipt-history regressions; no live signer, network or worker."""

import copy
import hashlib
import hmac
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

APP = Path(__file__).resolve().parents[1] / "hf-spaces/eternal-auditor/app.py"
TEST_KEY = b"offline-history-test-fixture"
ENVELOPE_FIELDS = {"body_sha256", "signature", "signature_length"}


def body_digest(receipt):
    body = {key: value for key, value in receipt.items() if key not in ENVELOPE_FIELDS}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fixture(cycle=1, prev=None, **extra):
    body = {
        "kind": "szl.auditor-receipt/v2",
        "officer": "eternal-auditor",
        "cycle": cycle,
        "chain_state": "CONTINUOUS",
        "prev_receipt_sha256": prev,
        "findings": [{"check": "synthetic", "verdict": "UNKNOWN", "value": None}],
        "ts": "2026-11-10T00:00:00Z",
        **extra,
    }
    digest = body_digest(body)
    signature = hmac.new(TEST_KEY, digest.encode(), hashlib.sha256).hexdigest()
    return {
        **body,
        "body_sha256": digest,
        "signature": signature,
        "signature_length": len(signature),
    }


class AuditorHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_path = Path(self.tmp.name) / "state.json"
        self.env = mock.patch.dict(
            os.environ,
            {"AUDITOR_STATE": str(self.state_path), "HF_TOKEN": ""},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.optional = mock.patch.dict(
            sys.modules, {"fastapi": None, "huggingface_hub": None}
        )
        self.optional.start()
        self.addCleanup(self.optional.stop)
        self.network = mock.patch(
            "urllib.request.urlopen", side_effect=AssertionError("network forbidden")
        )
        self.network.start()
        self.addCleanup(self.network.stop)
        self.worker = mock.patch(
            "threading.Thread.start", side_effect=AssertionError("worker forbidden")
        )
        self.worker.start()
        self.addCleanup(self.worker.stop)
        self.app = self.load_app()

    def load_app(self):
        spec = importlib.util.spec_from_file_location("auditor_history_fixture", APP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_preserves_every_receipt_field(self):
        receipt = fixture()
        self.app.remember(self.app.state, receipt)
        self.assertEqual(self.app.state["history"], [receipt])
        self.assertEqual(json.loads(self.state_path.read_text())["history"], [receipt])

    def test_stored_body_can_be_rehashed_without_reconstructed_fields(self):
        self.app.remember(self.app.state, fixture(prev="a" * 64))
        stored = self.app.state["history"][0]
        self.assertEqual(body_digest(stored), stored["body_sha256"])

    def test_full_signature_remains_exact(self):
        receipt = fixture()
        self.app.remember(self.app.state, receipt)
        stored = self.app.state["history"][0]
        self.assertEqual(stored["signature"], receipt["signature"])
        self.assertEqual(len(stored["signature"]), 64)
        expected = hmac.new(
            TEST_KEY, stored["body_sha256"].encode(), hashlib.sha256
        ).hexdigest()
        self.assertTrue(hmac.compare_digest(stored["signature"], expected))

    def test_reload_retains_complete_predecessor_and_envelope(self):
        receipt = fixture(cycle=2, prev="b" * 64)
        self.app.remember(self.app.state, receipt)
        restarted = self.load_app()
        self.assertEqual(restarted.state["history"][0], receipt)

    def test_retained_history_does_not_alias_caller_nested_findings(self):
        receipt = fixture(details={"nested": ["before"]})
        expected = copy.deepcopy(receipt)
        self.app.remember(self.app.state, receipt)
        receipt["findings"][0]["verdict"] = "FAIL"
        receipt["details"]["nested"].append("after")
        self.assertEqual(self.app.state["history"][0], expected)
        self.assertEqual(
            json.loads(self.state_path.read_text())["history"][0], expected
        )

    def test_failed_save_does_not_advance_memory_or_trim_existing_history(self):
        self.app.state["history"] = [fixture(cycle=n) for n in range(1, 25)]
        self.app.save_state(self.app.state)
        before = copy.deepcopy(self.app.state)
        original_history = self.app.state["history"]
        disk = self.state_path.read_bytes()
        with mock.patch.object(self.app, "save_state", side_effect=OSError("fixture")):
            with self.assertRaises(OSError):
                self.app.remember(self.app.state, fixture(cycle=25))
        self.assertEqual(self.app.state, before)
        self.assertIs(self.app.state["history"], original_history)
        self.assertEqual(self.state_path.read_bytes(), disk)

    def test_first_failed_save_does_not_create_history_in_memory(self):
        before = copy.deepcopy(self.app.state)
        with mock.patch.object(self.app, "save_state", side_effect=OSError("fixture")):
            with self.assertRaises(OSError):
                self.app.remember(self.app.state, fixture())
        self.assertEqual(self.app.state, before)
        self.assertFalse(self.state_path.exists())

    def test_actual_replace_failure_preserves_disk_and_memory(self):
        self.app.remember(self.app.state, fixture())
        before = copy.deepcopy(self.app.state)
        disk = self.state_path.read_bytes()
        with mock.patch.object(Path, "replace", side_effect=OSError("fixture")):
            with self.assertRaises(OSError):
                self.app.remember(self.app.state, fixture(cycle=2))
        self.assertEqual(self.app.state, before)
        self.assertEqual(self.state_path.read_bytes(), disk)

    def test_last_24_complete_receipts_are_retained_in_order(self):
        receipts = []
        prev = None
        for cycle in range(1, 31):
            receipt = fixture(cycle=cycle, prev=prev)
            self.app.remember(self.app.state, receipt)
            receipts.append(receipt)
            prev = receipt["body_sha256"]
        self.assertEqual(self.app.state["history"], receipts[-24:])
        self.assertEqual(
            json.loads(self.state_path.read_text())["history"], receipts[-24:]
        )

    def test_legacy_rows_are_preserved_not_reconstructed(self):
        legacy = {"cycle": 3, "signature": "legacy-as-recorded", "findings": []}
        self.app.state["history"] = [copy.deepcopy(legacy)]
        receipt = fixture(cycle=4)
        self.app.remember(self.app.state, receipt)
        self.assertEqual(self.app.state["history"], [legacy, receipt])
        self.assertNotIn("kind", self.app.state["history"][0])

    def test_additional_signed_fields_survive_without_schema_guessing(self):
        receipt = fixture(extension={"text": "π / 雪", "optional": None, "values": []})
        self.app.remember(self.app.state, receipt)
        stored = self.app.state["history"][0]
        self.assertEqual(stored, receipt)
        self.assertEqual(body_digest(stored), stored["body_sha256"])

    def test_unserializable_receipt_preserves_existing_state(self):
        self.app.remember(self.app.state, fixture())
        before = copy.deepcopy(self.app.state)
        disk = self.state_path.read_bytes()
        for bad in (object(), float("nan"), float("inf"), float("-inf")):
            with self.subTest(value_type=type(bad).__name__):
                receipt = fixture(cycle=2)
                receipt["findings"][0]["value"] = bad
                with self.assertRaises((TypeError, ValueError)):
                    self.app.remember(self.app.state, receipt)
                self.assertEqual(self.app.state, before)
                self.assertEqual(self.state_path.read_bytes(), disk)

    def test_existing_cycle_helper_keeps_restart_chain_compatible(self):
        first = self.app.run_cycle([], secret=TEST_KEY.decode())
        self.app.remember(self.app.state, first)
        restarted = self.load_app()
        second = restarted.run_cycle([], secret=TEST_KEY.decode())
        restarted.remember(restarted.state, second)
        self.assertEqual(second["cycle"], 2)
        self.assertEqual(second["prev_receipt_sha256"], first["body_sha256"])
        self.assertEqual(restarted.state["history"], [first, second])
        self.assertEqual(body_digest(second), second["body_sha256"])

    def test_real_loop_stores_full_receipt_without_live_side_effects(self):
        self.app.SECRET = TEST_KEY.decode()
        with (
            mock.patch.object(self.app, "audit_cycle", return_value=[]),
            mock.patch.object(self.app.time, "sleep", side_effect=StopIteration),
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(StopIteration):
                self.app.loop()
        persisted = json.loads(self.state_path.read_text())
        self.assertEqual(len(persisted["history"]), 1)
        stored = persisted["history"][0]
        self.assertEqual(stored["kind"], "szl.auditor-receipt/v2")
        self.assertEqual(stored["officer"], "eternal-auditor")
        self.assertIsNone(stored["prev_receipt_sha256"])
        self.assertEqual(stored["signature_length"], 64)
        self.assertEqual(body_digest(stored), stored["body_sha256"])

    def test_malformed_history_is_not_replaced_with_a_fresh_list(self):
        for history in (None, "broken", {"broken": True}):
            with self.subTest(history=history):
                state = {"cycle": 3, "prev": "c" * 64, "history": history}
                before = copy.deepcopy(state)
                with mock.patch.object(self.app, "save_state") as save:
                    with self.assertRaises(ValueError):
                        self.app.remember(state, fixture(cycle=4))
                    save.assert_not_called()
                self.assertEqual(state, before)

    def test_unknown_observation_is_not_promoted(self):
        self.app.remember(self.app.state, fixture())
        finding = self.app.state["history"][0]["findings"][0]
        self.assertEqual(finding["verdict"], "UNKNOWN")
        self.assertIsNone(finding["value"])


if __name__ == "__main__":
    unittest.main()
