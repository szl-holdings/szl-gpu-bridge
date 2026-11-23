import unittest

from lifecycle.idempotent import (CALLBACK_ERROR, DUPLICATE_NOT_REEXECUTED, EXECUTED,
                                  IdempotentRegistry, JobDecision, JobSpec,
                                  TOKEN_EXPIRED_REJECTED, TokenError)

NOW = 10_000.0


def spec(token="job-1", ttl=300.0, submitted_at=NOW, fingerprint="sha256:deadbeef"):
    return JobSpec(token=token, payload_fingerprint=fingerprint, ttl_s=ttl,
                   submitted_at=submitted_at)


class SpecTests(unittest.TestCase):
    def test_valid_spec_expires_at(self):
        self.assertEqual(spec().expires_at(), NOW + 300.0)

    def test_malformed_specs_raise(self):
        for mk in (lambda: spec(token=""), lambda: spec(fingerprint=""),
                   lambda: spec(ttl=-1.0)):
            with self.assertRaises(TokenError):
                mk()


class SubmitTests(unittest.TestCase):
    def setUp(self):
        self.reg = IdempotentRegistry(clock=lambda: NOW)
        self.calls = []

    def test_first_submission_executes(self):
        d = self.reg.submit(spec(), lambda: (self.calls.append(1) or "ok"), cost_usd=1.5)
        self.assertEqual(d.outcome, EXECUTED)
        self.assertEqual(len(self.calls), 1)

    def test_duplicate_never_reexecutes(self):
        run = lambda: (self.calls.append(1) or "ok")
        self.reg.submit(spec(), run, cost_usd=1.5)
        d2 = self.reg.submit(spec(), run, cost_usd=1.5)
        self.assertEqual(d2.outcome, DUPLICATE_NOT_REEXECUTED)
        self.assertFalse(d2.executed)
        self.assertEqual(len(self.calls), 1)  # run called exactly once
        self.assertEqual(d2.result, "ok")

    def test_expired_token_rejected(self):
        run = lambda: (self.calls.append(1) or "ok")
        self.reg.submit(spec(), run, cost_usd=1.5)
        d2 = self.reg.submit(spec(), run, cost_usd=1.5, now_ts=NOW + 400.0)
        self.assertEqual(d2.outcome, TOKEN_EXPIRED_REJECTED)
        self.assertIn("fresh token", d2.error)
        self.assertEqual(len(self.calls), 1)  # never silently re-executed

    def test_error_callback_stored_and_replayed(self):
        def boom():
            raise RuntimeError("gpu node vanished")
        d1 = self.reg.submit(spec(), boom, cost_usd=2.0)
        self.assertEqual(d1.outcome, CALLBACK_ERROR)
        self.assertIn("gpu node vanished", d1.error)
        def boom2():
            raise AssertionError("must not re-run")
        d2 = self.reg.submit(spec(), boom2)
        self.assertEqual(d2.outcome, DUPLICATE_NOT_REEXECUTED)
        self.assertIn("gpu node vanished", d2.error)
        self.assertEqual(d2.error, d1.error)  # prior terminal state replayed

    def test_ledger_accumulates_per_execution(self):
        self.reg.submit(spec("a"), lambda: "x", cost_usd=1.0)
        self.reg.submit(spec("b"), lambda: "y", cost_usd=2.5)
        self.reg.submit(spec("a"), lambda: "z", cost_usd=1.0)  # duplicate, free
        self.assertEqual(self.reg.total_cost_usd(), 3.5)

    def test_negative_cost_rejected(self):
        with self.assertRaises(TokenError):
            self.reg.submit(spec(), lambda: "x", cost_usd=-1.0)


if __name__ == "__main__":
    unittest.main()
