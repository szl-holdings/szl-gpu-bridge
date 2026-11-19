"""Offline tests: 500 receipts become a handful of commits and survive HTTP 429."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tools.salad.batched_hub_uploader import (
    BatchedHubUploader,
    STATUS_FILE,
    UploadFlushError,
    retry_after_seconds,
)


class FakeHttpError(Exception):
    def __init__(self, status, headers=None, body="SECRET-TOKEN-hf_abc123 url"):
        super().__init__(body)
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def op(path, data):
    return (path, data)


def make_api(failures=()):
    """create_commit raises each queued failure once, in order, then succeeds."""
    queue = list(failures)
    api = Mock()
    committed = []

    def create_commit(**kwargs):
        if queue:
            raise queue.pop(0)
        committed.append(kwargs)
        return SimpleNamespace(oid="%040x" % len(committed))

    api.create_commit.side_effect = create_commit
    api.committed = committed
    return api


class BatchedUploaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        self.sleeps = []

        def sleep(seconds):
            self.sleeps.append(seconds)
            self.clock.sleep(seconds)

        self.sleep = sleep

    def uploader(self, api, **kw):
        defaults = dict(
            max_files=100,
            max_interval_s=300,
            operation_factory=op,
            clock=self.clock,
            sleep=self.sleep,
        )
        defaults.update(kw)
        return BatchedHubUploader(
            api, "SZLHOLDINGS/frontier-54-runs", self.tmp.name, **defaults
        )

    def burst(self, up, count=500):
        for i in range(count):
            up.add(f"mesh/salad-{i:04d}.json", json.dumps({"i": i}).encode())

    def test_burst_of_500_makes_a_few_commits_and_never_one_per_receipt(self):
        api = make_api()
        up = self.uploader(api)
        self.burst(up)
        up.close()
        self.assertLessEqual(api.create_commit.call_count, 6)
        uploaded = [p for call in api.committed for p, _ in call["operations"]]
        self.assertEqual(len(uploaded), 500)
        self.assertEqual(len(set(uploaded)), 500)
        self.assertEqual(up.pending_count(), 0)
        self.assertEqual(list(Path(self.tmp.name).glob("*.spool")), [])

    def test_burst_survives_a_429_without_failing_or_losing_receipts(self):
        api = make_api([FakeHttpError(429, {"Retry-After": "90"})])
        up = self.uploader(api)
        self.burst(up)  # first full batch is throttled; add() must not raise
        self.assertEqual(up.throttled_count, 1)
        up.close()  # final flush retries and publishes everything
        uploaded = [p for call in api.committed for p, _ in call["operations"]]
        self.assertEqual(sorted(set(uploaded)), sorted(uploaded))
        self.assertEqual(len(uploaded), 500)
        self.assertLessEqual(api.create_commit.call_count, 7)

    def test_throttled_non_final_flush_keeps_batch_and_respects_cooldown(self):
        api = make_api([FakeHttpError(429, {"Retry-After": "120"})])
        up = self.uploader(api, max_files=2)
        up.add("a.json", b"1")
        up.add("b.json", b"2")  # full -> attempt -> 429
        self.assertEqual(api.create_commit.call_count, 1)
        self.assertEqual(up.pending_count(), 2)
        up.add("c.json", b"3")  # still cooling down: no new request
        self.assertEqual(api.create_commit.call_count, 1)
        self.clock.now += 121
        up.maybe_flush()
        self.assertEqual(api.create_commit.call_count, 2)
        self.assertEqual(up.pending_count(), 0)
        self.assertEqual(len(api.committed[0]["operations"]), 3)

    def test_final_flush_honors_retry_after_then_exponential_backoff(self):
        api = make_api(
            [
                FakeHttpError(429, {"Retry-After": "7"}),
                FakeHttpError(429),
                FakeHttpError(503),
            ]
        )
        up = self.uploader(api, base_delay_s=10, max_delay_s=600)
        up.add("a.json", b"1")
        up.close()
        self.assertEqual(self.sleeps, [7.0, 20, 40])
        self.assertEqual(api.create_commit.call_count, 4)

    def test_retry_after_is_capped_and_garbage_is_ignored(self):
        self.assertEqual(
            retry_after_seconds(FakeHttpError(429, {"Retry-After": "99999"}), 3600),
            3600,
        )
        self.assertIsNone(
            retry_after_seconds(FakeHttpError(429, {"Retry-After": "soon"}), 3600)
        )
        self.assertIsNone(
            retry_after_seconds(FakeHttpError(429, {"Retry-After": "-5"}), 3600)
        )
        self.assertIsNone(retry_after_seconds(FakeHttpError(429), 3600))

    def test_only_an_exhausted_final_flush_fails_and_the_spool_is_kept(self):
        api = make_api([FakeHttpError(429)] * 3)
        up = self.uploader(api, max_attempts=3, base_delay_s=1)
        up.add("a.json", b"1")
        with self.assertRaises(UploadFlushError) as ctx:
            up.close()
        self.assertEqual(api.create_commit.call_count, 3)
        self.assertEqual(up.pending_count(), 1)
        self.assertEqual(len(list(Path(self.tmp.name).glob("*.spool"))), 1)
        self.assertIn("HTTP 429", str(ctx.exception))
        self.assertNotIn("SECRET", str(ctx.exception))
        self.assertNotIn("hf_abc123", Path(self.tmp.name, STATUS_FILE).read_text())

    def test_permanent_4xx_fails_final_flush_immediately(self):
        api = make_api([FakeHttpError(403)] * 5)
        up = self.uploader(api)
        up.add("a.json", b"1")
        with self.assertRaises(UploadFlushError):
            up.close()
        self.assertEqual(api.create_commit.call_count, 1)
        self.assertEqual(self.sleeps, [])

    def test_non_final_failures_of_any_kind_never_raise(self):
        api = make_api([FakeHttpError(403), RuntimeError("boom")])
        up = self.uploader(api, max_files=1)
        up.add("a.json", b"1")
        self.clock.now += 10_000
        up.add("b.json", b"2")
        up.close()
        uploaded = [p for call in api.committed for p, _ in call["operations"]]
        self.assertEqual(sorted(uploaded), ["a.json", "b.json"])

    def test_interval_flush_and_status_file_for_watchers(self):
        api = make_api([FakeHttpError(429, {"Retry-After": "30"})])
        up = self.uploader(api, max_files=1000, max_interval_s=60)
        up.add("a.json", b"1")
        self.assertEqual(api.create_commit.call_count, 0)
        self.clock.now += 61
        up.maybe_flush()
        status = json.loads(Path(self.tmp.name, STATUS_FILE).read_text())
        self.assertEqual(status["state"], "THROTTLED")
        self.assertEqual(status["pending_files"], 1)
        self.assertEqual(status["last_error"], "HTTP 429")
        self.assertGreater(status["retry_in_s"], 0)
        self.clock.now += 31
        up.maybe_flush()
        status = json.loads(Path(self.tmp.name, STATUS_FILE).read_text())
        self.assertEqual(status["state"], "OK")
        self.assertEqual(status["pending_files"], 0)

    def test_same_path_is_last_write_wins_and_bad_paths_are_rejected(self):
        api = make_api()
        up = self.uploader(api)
        up.add("hb.log", b"old")
        up.add("hb.log", b"new")
        up.close()
        self.assertEqual(api.committed[0]["operations"], [("hb.log", b"new")])
        for bad in ("", "/abs.json", "../x.json", "a/../../x.json"):
            with self.assertRaises(ValueError):
                up.add(bad, b"x")
        with self.assertRaises(TypeError):
            up.add("x.json", "not bytes")


if __name__ == "__main__":
    unittest.main()
