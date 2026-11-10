"""Offline tests for the szl SDK. No network, no GPU."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from szl import kamachikuna, khipu, provenance, router  # noqa: E402


class KhipuTests(unittest.TestCase):
    def test_mint_and_check(self):
        r = khipu.mint({"claim": "training completed", "steps": 2000}, secret="s")
        self.assertTrue(khipu.check(r, secret="s"))
        self.assertFalse(khipu.check(r, secret="wrong"))
        self.assertEqual(len(r["signature"]), 64)

    def test_tamper_detection(self):
        r = khipu.mint({"claim": "x"}, secret="s")
        r2 = dict(r)
        r2["claim"] = {"claim": "y"}  # tamper
        self.assertFalse(khipu.check(r2, secret="s"))

    def test_chain(self):
        secret = "s"
        r1 = khipu.mint({"n": 1}, secret)
        r2 = khipu.mint({"n": 2}, secret, prev_body_sha256=r1["body_sha256"])
        r3 = khipu.mint({"n": 3}, secret, prev_body_sha256=r2["body_sha256"])
        self.assertTrue(khipu.check_chain([r1, r2, r3], secret))
        self.assertFalse(khipu.check_chain([r1, r3, r2], secret))  # reorder
        # a prefix is internally valid; truncation needs the expected head
        self.assertTrue(khipu.check_chain([r1, r2], secret))  # prefix: valid
        self.assertFalse(
            khipu.check_chain([r1, r2], secret, expected_head=r3["body_sha256"])
        )  # truncated vs known head
        self.assertTrue(
            khipu.check_chain([r1, r2, r3], secret, expected_head=r3["body_sha256"])
        )

    def test_truncated_signature_rejected(self):
        r = khipu.mint({"claim": "x"}, secret="s")
        r["signature"] = r["signature"][:48]  # the v1 auditor truncation bug
        self.assertFalse(khipu.check(r, secret="s"))


class FreshnessTests(unittest.TestCase):
    def test_stale_chain_is_not_green(self):
        """A valid-but-old chain must fail a freshness check. Cached values
        sat 57 days untouched in the Vidal-Gadea tracker and were read with
        no error and no warning."""
        import datetime

        secret = "s"
        old_ts = (datetime.datetime.now(datetime.timezone.utc)
                  - datetime.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        r1 = khipu.mint({"n": 1}, secret, ts=old_ts)
        self.assertTrue(khipu.check_chain([r1], secret))  # integrity: valid
        self.assertFalse(khipu.check_chain([r1], secret, max_age_days=7))  # stale
        fresh = khipu.mint({"n": 2}, secret, prev_body_sha256=r1["body_sha256"])
        self.assertTrue(khipu.check_chain([r1, fresh], secret, max_age_days=7))

    def test_unreadable_timestamp_fails_freshness(self):
        r = khipu.mint({"n": 1}, "s")
        r["ts"] = "not-a-date"
        # integrity check re-hashes the body, so we cannot tamper ts post-mint;
        # instead verify that a bad ts inside a properly minted receipt fails
        r2 = khipu.mint({"n": 1}, "s")
        self.assertTrue(khipu.check_chain([r2], "s"))
        # mint with a garbage ts and check freshness rejects it
        r3 = khipu.mint({"n": 1}, "s", ts="garbage")
        self.assertFalse(khipu.check_chain([r3], "s", max_age_days=7))


class KamachikunaTests(unittest.TestCase):
    def test_constants_complete(self):
        self.assertEqual(len(kamachikuna.LAWS), 10)
        self.assertEqual(len(kamachikuna.PRIME_PRINCIPLES), 5)
        self.assertEqual(len(kamachikuna.LEAN_PROVEN), 5)
        self.assertEqual(len(kamachikuna.AMENDMENTS), 3)

    def test_fail_closed_gate(self):
        self.assertEqual(kamachikuna.gate("deploy", verified=True)[1], "ALLOWED")
        self.assertEqual(kamachikuna.gate("deploy", verified=False)[1], "BLOCKED")


class ProvenanceTests(unittest.TestCase):
    def test_honest_states(self):
        untrained = provenance.ProvenanceClaim(model_id="m")
        self.assertEqual(untrained.state(), "UNTRAINED")
        card_only = provenance.ProvenanceClaim(model_id="m", source_sha="a" * 40)
        self.assertEqual(card_only.state(), "CARD_ONLY")
        # measured requires a verifiable receipt + source + datasets
        secret = "s"
        receipt = khipu.mint({"kind": "szl.training-receipt/v1"}, secret)
        measured = provenance.ProvenanceClaim(
            model_id="m", source_sha="a" * 40, training_receipt=receipt,
            datasets={"train": "b" * 64},
        )
        self.assertEqual(measured.state(secret), "MEASURED")
        # unverifiable receipt -> UNAVAILABLE, never MEASURED
        self.assertEqual(measured.state("wrong-secret"), "UNAVAILABLE")

    def test_badge_is_signed(self):
        secret = "s"
        receipt = khipu.mint({"kind": "szl.training-receipt/v1"}, secret)
        c = provenance.ProvenanceClaim(
            model_id="m", source_sha="a" * 40, training_receipt=receipt,
            datasets={"train": "b" * 64},
        )
        badge = c.badge(secret)
        self.assertTrue(khipu.check(badge, secret))
        self.assertEqual(badge["claim"]["state"], "MEASURED")
        # unsigned inspection must NEVER be MEASURED (Law 6)
        unsigned = c.badge(secret=None)
        self.assertEqual(unsigned["state"], "UNAVAILABLE")


class RouterTests(unittest.TestCase):
    def test_honest_cheapest_wins(self):
        models = [
            router.Model("cheap-liar", "FAILED", 0.0, None, None, 0.01),
            router.Model("honest-mid", "MEASURED", 0.9, "PROVE-OR-ABSTAIN-MASTER", 2.0, 1.0),
            router.Model("honest-expensive", "MEASURED", 0.95, None, 1.0, 5.0),
        ]
        pick = router.route("eval", models)
        self.assertEqual(pick.model_id, "honest-mid")

    def test_honesty_critical_fails_closed(self):
        models = [router.Model("cheap-liar", "FAILED", 0.0, None, None, 0.01)]
        self.assertIsNone(router.route("honesty-critical", models))
        # non-critical tasks may still use unmeasured models
        self.assertIsNotNone(router.route("bulk", models))

    def test_unmeasured_is_none_not_zero(self):
        m = router.Model("m", "UNMEASURED", None, None, None, None)
        self.assertIsNone(m.honesty_score())
        self.assertIsNone(m.cost_per_mtok())


class TightenedSecurityTests(unittest.TestCase):
    """Codex review findings: empty receipts, missing timestamps, future timestamps."""

    def test_empty_receipt_never_verifies(self):
        r = {}  # empty receipt — must fail
        self.assertFalse(khipu.check(r, secret="s"))

    def test_unsigned_receipt_never_measured(self):
        c = provenance.ProvenanceClaim(model_id="m", source_sha="a" * 40,
                                        training_receipt={"kind": "szl.training-receipt/v1"},
                                        datasets={"train": "b" * 64})
        # no secret = UNAVAILABLE, never MEASURED
        self.assertEqual(c.state(None), "UNAVAILABLE")

    def test_missing_timestamp_never_verifies(self):
        r = khipu.mint({"claim": "x"}, secret="s")
        del r["ts"]  # remove the timestamp
        self.assertFalse(khipu.check(r, secret="s"))

    def test_future_timestamp_never_verifies(self):
        import datetime
        future = (datetime.datetime.now(datetime.timezone.utc)
                  + datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = khipu.mint({"claim": "x"}, secret="s", ts=future)
        self.assertFalse(khipu.check(r, secret="s"))

    def test_measured_zero_energy_beats_unknown(self):
        models = [
            router.Model("unknown-energy", "MEASURED", 0.9, None, None, 1.0),
            router.Model("measured-zero", "MEASURED", 0.9, None, 0.0, 1.0),
        ]
        pick = router.route("eval", models)
        self.assertEqual(pick.model_id, "measured-zero")  # known zero > unknown

    def test_timezone_free_timestamp_never_verifies(self):
        r = khipu.mint({"claim": "x"}, secret="s")
        r["ts"] = "2026-11-10T07:00:00"  # no timezone
        self.assertFalse(khipu.check(r, secret="s"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
