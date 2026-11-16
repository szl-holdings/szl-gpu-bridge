"""Credential-free tests with synthetic addresses and mocked JSON-RPC."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "transfer_watch",
    Path(__file__).resolve().parents[1] / "tools/evidence/transfer_watch.py",
)
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)


def bh(n):
    return "0x" + format(n + 1, "064x")


class FakeRpc:
    def __init__(self, head=15, failure=None, events=None):
        self.head, self.failure, self.events = head, failure, events or []
        self.calls = []
        self.chain = "0x2105"
        self.changed_anchor = None

    def __call__(self, method, params):
        self.calls.append((method, params))
        if method == "eth_chainId":
            return self.chain
        if method == "eth_getBlockByNumber":
            n = self.head if params[0] == "finalized" else int(params[0], 16)
            return {
                "number": hex(n),
                "hash": bh(n) if self.changed_anchor != n else bh(n + 1000),
            }
        if method == "eth_getLogs":
            lo, hi = int(params[0]["fromBlock"], 16), int(params[0]["toBlock"], 16)
            if self.failure == lo:
                raise w.Hold("RPC_RESULT_UNAVAILABLE")
            return [e for e in self.events if lo <= int(e["blockNumber"], 16) <= hi]
        raise AssertionError("unexpected RPC method")


class Tests(unittest.TestCase):
    def setUp(self):
        self.c = {
            "rpc_url": "https://example.invalid",
            "chain_id": 8453,
            "token_address": "0x" + "1" * 40,
            "recipient_topic": "0x" + "0" * 24 + "2" * 40,
            "start_block": 10,
            "chunk_size": 2,
            "max_chunks": 2,
            "max_polls": 2880,
            "poll_seconds": 300,
        }
        self.state = w.initial(self.c)
        self.saved = []

    def commit(self, state, receipt):
        self.saved.append((copy.deepcopy(state), copy.deepcopy(receipt)))

    def event(self, n=10, amount=5):
        return {
            "removed": False,
            "address": self.c["token_address"],
            "topics": [w.TRANSFER, "0x" + "3" * 64, self.c["recipient_topic"]],
            "blockNumber": hex(n),
            "blockHash": bh(n),
            "transactionHash": "0x" + "4" * 64,
            "logIndex": "0x0",
            "data": "0x" + format(amount, "064x"),
        }

    def test_chunk_budget_retains_unscanned_cursor(self):
        result = w.scan(self.state, self.c, FakeRpc(20), self.commit)
        self.assertEqual(result["next_block"], 14)
        self.assertEqual(result["status"], "CATCHING_UP")
        self.assertEqual(len(self.saved), 2)

    def test_failed_first_request_never_advances(self):
        self.assertRaises(
            w.Hold, w.scan, self.state, self.c, FakeRpc(failure=10), self.commit
        )
        self.assertEqual(self.saved, [])
        self.assertEqual(self.state["next_block"], 10)

    def test_later_failure_keeps_last_success(self):
        self.assertRaises(
            w.Hold, w.scan, self.state, self.c, FakeRpc(failure=12), self.commit
        )
        self.assertEqual(self.saved[-1][0]["next_block"], 12)

    def test_resume_no_block_skip(self):
        first = w.scan(self.state, self.c, FakeRpc(), self.commit)
        second = w.scan(first, self.c, FakeRpc(), self.commit)
        self.assertEqual(second["next_block"], 16)
        self.assertEqual(
            [(r["from_block"], r["to_block"]) for _, r in self.saved],
            [(10, 11), (12, 13), (14, 15)],
        )

    def test_empty_success_is_scoped_not_revenue(self):
        result = w.scan(self.state, self.c, FakeRpc(11), self.commit)
        self.assertEqual(result["status"], "OBSERVED_NO_POSITIVE_TRANSFERS_IN_RANGE")
        self.assertFalse(result["revenue_verified"])

    def test_positive_transfer_is_not_revenue(self):
        result = w.scan(self.state, self.c, FakeRpc(events=[self.event()]), self.commit)
        self.assertEqual(result["status"], "TRANSFER_CANDIDATE")
        self.assertEqual(result["next_block"], 12)
        self.assertFalse(self.saved[0][1]["revenue_verified"])

    def test_zero_value_does_not_trigger_candidate(self):
        result = w.scan(
            self.state, self.c, FakeRpc(11, events=[self.event(amount=0)]), self.commit
        )
        self.assertEqual(self.saved[0][1]["positive_transfer_count"], 0)
        self.assertNotEqual(result["status"], "TRANSFER_CANDIDATE")

    def test_deduplicate_log_identity(self):
        event = self.event()
        result = w.logs_check([event, event], self.c, 10, 11)
        self.assertEqual(len(result), 1)

    def test_contradictory_duplicate_rejected(self):
        self.assertRaises(
            w.Hold,
            w.logs_check,
            [self.event(amount=1), self.event(amount=2)],
            self.c,
            10,
            11,
        )

    def test_wrong_event_signature_rejected(self):
        e = self.event()
        e["topics"][0] = "0x" + "a" * 64
        self.assertRaises(w.Hold, w.logs_check, [e], self.c, 10, 11)

    def test_wrong_recipient_rejected(self):
        e = self.event()
        e["topics"][2] = "0x" + "b" * 64
        self.assertRaises(w.Hold, w.logs_check, [e], self.c, 10, 11)

    def test_wrong_token_rejected(self):
        e = self.event()
        e["address"] = "0x" + "a" * 40
        self.assertRaises(w.Hold, w.logs_check, [e], self.c, 10, 11)

    def test_removed_log_rejected(self):
        e = self.event()
        e["removed"] = True
        self.assertRaises(w.Hold, w.logs_check, [e], self.c, 10, 11)

    def test_missing_removed_flag_rejected(self):
        e = self.event()
        del e["removed"]
        self.assertRaises(w.Hold, w.logs_check, [e], self.c, 10, 11)

    def test_out_of_range_log_rejected(self):
        self.assertRaises(w.Hold, w.logs_check, [self.event(12)], self.c, 10, 11)

    def test_non_list_rpc_logs_rejected(self):
        self.assertRaises(w.Hold, w.logs_check, None, self.c, 10, 11)

    def test_log_response_cap(self):
        self.assertRaises(w.Hold, w.logs_check, [self.event()] * 1001, self.c, 10, 11)

    def test_wrong_chain_rejected(self):
        rpc = FakeRpc()
        rpc.chain = "0x1"
        self.assertRaises(w.Hold, w.scan, self.state, self.c, rpc, self.commit)
        self.assertEqual(self.saved, [])

    def test_prior_anchor_change_rejected(self):
        first = w.scan(self.state, self.c, FakeRpc(11), self.commit)
        rpc = FakeRpc()
        rpc.changed_anchor = 11
        self.assertRaises(w.Hold, w.scan, first, self.c, rpc, self.commit)

    def test_finalized_regression_rejected(self):
        first = w.scan(self.state, self.c, FakeRpc(11), self.commit)
        self.assertRaises(w.Hold, w.scan, first, self.c, FakeRpc(10), self.commit)

    def test_poll_limit_survives_resume(self):
        s = w.initial(self.c, {"polls": 2880})
        self.assertRaises(w.Hold, w.scan, s, self.c, FakeRpc(), self.commit)

    def test_legacy_cursor_not_trusted(self):
        s = w.initial(self.c, {"polls": 99, "last_block": 999999})
        self.assertEqual(s["next_block"], 10)
        self.assertEqual(s["polls"], 99)
        self.assertEqual(s["earlier_history"], "UNVERIFIED")

    def test_config_change_requires_reconciliation(self):
        self.c["chunk_size"] = 3
        self.assertRaises(w.Hold, w.scan, self.state, self.c, FakeRpc(), self.commit)

    def test_cursor_boolean_rejected(self):
        self.state["next_block"] = True
        self.assertRaises(w.Hold, w.scan, self.state, self.c, FakeRpc(), self.commit)

    def test_bad_config_budget(self):
        self.c["max_chunks"] = 100
        self.assertRaises(w.Hold, w.config_check, self.c)

    def test_url_credentials_rejected(self):
        self.c["rpc_url"] = "https://user:password@example.invalid"
        self.assertRaises(w.Hold, w.config_check, self.c)

    def test_plain_http_rejected(self):
        self.c["rpc_url"] = "http://example.invalid"
        self.assertRaises(w.Hold, w.config_check, self.c)

    def test_nonread_method_rejected_without_network(self):
        self.assertRaises(
            w.Hold, w.Rpc(self.c["rpc_url"]), "eth_sendRawTransaction", []
        )

    def test_request_budget_without_network(self):
        r = w.Rpc(self.c["rpc_url"])
        r.calls = 52
        self.assertRaises(w.Hold, r, "eth_getLogs", [])

    def test_os_lock_rejects_competing_writer(self):
        with tempfile.TemporaryDirectory() as d:
            with w.Writer(Path(d)):
                self.assertRaises(w.Hold, w.Writer(Path(d)).__enter__)
            with w.Writer(Path(d)):
                pass

    def test_atomic_record_precedes_cursor(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            w.persist(root, self.state, {"kind": "test", "n": 1})
            saved = json.loads((root / "status.json").read_bytes())
            receipt = root / saved["last_receipt"]
            self.assertTrue(receipt.is_file())
            self.assertEqual(
                w.digest(receipt.read_bytes()), saved["last_receipt_sha256"]
            )

    def test_source_state_not_mutated(self):
        before = copy.deepcopy(self.state)
        w.scan(self.state, self.c, FakeRpc(), self.commit)
        self.assertEqual(self.state, before)

    def test_legacy_hit_requires_review(self):
        self.assertRaises(w.Hold, w.initial, self.c, {"polls": 99, "fired": True})

    def test_exact_transfer_filter(self):
        rpc = FakeRpc(11)
        w.scan(self.state, self.c, rpc, self.commit)
        params = next(p for method, p in rpc.calls if method == "eth_getLogs")
        self.assertEqual(
            params[0]["topics"], [w.TRANSFER, None, self.c["recipient_topic"]]
        )

    def test_denied_provider_not_retried(self):
        for status in (401, 403, 429):
            error = urllib.error.HTTPError(self.c["rpc_url"], status, "test", {}, None)
            with patch.object(w.urllib.request, "urlopen", side_effect=error) as call:
                with self.assertRaisesRegex(w.Hold, "ACCESS_OR_RATE_LIMIT"):
                    w.Rpc(self.c["rpc_url"])("eth_chainId", [])
                self.assertEqual(call.call_count, 1)

    def test_receipt_durable_before_failed_state_write(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            w.atomic(root / "status.json", self.state)
            real = w.atomic

            def fail_status(path, value):
                if path.name == "status.json":
                    raise OSError("simulated full disk")
                real(path, value)

            with patch.object(w, "atomic", side_effect=fail_status):
                self.assertRaises(
                    OSError,
                    w.persist,
                    root,
                    {**self.state, "next_block": 12},
                    {"kind": "range"},
                )
            self.assertEqual(
                json.loads((root / "status.json").read_bytes())["next_block"], 10
            )
            self.assertEqual(len(list(root.glob("observation-*.json"))), 1)

    def test_no_new_finalized_range_keeps_cursor(self):
        first = w.scan(self.state, self.c, FakeRpc(11), self.commit)
        after = w.scan(first, self.c, FakeRpc(11), self.commit)
        self.assertEqual(after["next_block"], 12)
        self.assertEqual(after["status"], "NO_NEW_FINALIZED_RANGE")


if __name__ == "__main__":
    unittest.main()
