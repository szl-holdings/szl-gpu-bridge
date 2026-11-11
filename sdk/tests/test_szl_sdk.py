"""Offline tests for the szl SDK. No network, no GPU."""

import unittest

from szl import kamachikuna, khipu, provenance, router


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
        old_ts = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
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
        receipt = khipu.mint(
            {
                "kind": "szl.training-receipt/v1",
                "model_id": "m",
                "source_sha": "a" * 40,
                "datasets": {"train": "b" * 64},
            },
            secret,
        )
        measured = provenance.ProvenanceClaim(
            model_id="m",
            source_sha="a" * 40,
            training_receipt=receipt,
            datasets={"train": "b" * 64},
        )
        self.assertEqual(measured.state(secret), "MEASURED")
        # unverifiable receipt -> UNAVAILABLE, never MEASURED
        self.assertEqual(measured.state("wrong-secret"), "UNAVAILABLE")

    def test_badge_is_signed(self):
        secret = "s"
        receipt = khipu.mint(
            {
                "kind": "szl.training-receipt/v1",
                "model_id": "m",
                "source_sha": "a" * 40,
                "datasets": {"train": "b" * 64},
            },
            secret,
        )
        c = provenance.ProvenanceClaim(
            model_id="m",
            source_sha="a" * 40,
            training_receipt=receipt,
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
            router.Model(
                "honest-mid", "MEASURED", 0.9, "PROVE-OR-ABSTAIN-MASTER", 2.0, 1.0
            ),
            router.Model("honest-expensive", "MEASURED", 0.95, None, 1.0, 5.0),
        ]
        pick = router.route("eval", models)
        self.assertEqual(pick.model_id, "honest-mid")

    def test_honesty_critical_fails_closed(self):
        models = [router.Model("cheap-liar", "FAILED", 0.0, None, None, 0.01)]
        self.assertIsNone(router.route("honesty-critical", models))
        # A known failed model is ineligible even for non-critical tasks.
        self.assertIsNone(router.route("bulk", models))
        unmeasured = router.Model("unknown", "UNMEASURED", None, None, None, 0.1)
        self.assertIs(router.route("bulk", [unmeasured]), unmeasured)

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
        c = provenance.ProvenanceClaim(
            model_id="m",
            source_sha="a" * 40,
            training_receipt={"kind": "szl.training-receipt/v1"},
            datasets={"train": "b" * 64},
        )
        # no secret = UNAVAILABLE, never MEASURED
        self.assertEqual(c.state(None), "UNAVAILABLE")

    def test_missing_timestamp_never_verifies(self):
        r = khipu.mint({"claim": "x"}, secret="s")
        del r["ts"]  # remove the timestamp
        self.assertFalse(khipu.check(r, secret="s"))

    def test_future_timestamp_never_verifies(self):
        import datetime

        future = (
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
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


class OpenShellPolicyTests(unittest.TestCase):
    """Kamachikuna as OpenShell-reviewable policy-as-code."""

    def setUp(self):
        from szl import openshell

        self.os_ = openshell

    def test_policy_document_is_deterministic_and_hashed(self):
        d1 = self.os_.policy_document()
        d2 = self.os_.policy_document()
        self.assertEqual(d1, d2)  # version-controllable: byte-stable
        self.assertEqual(d1["spec"]["default"], "deny")  # Law 6
        self.assertEqual(len(d1["spec"]["rules"]), 10)
        self.assertTrue(all("sha256" in d1["metadata"] for _ in [1]))

    def test_laws_single_sourced(self):
        doc = self.os_.policy_document()
        texts = [r["text"] for r in doc["spec"]["rules"]]
        from szl import kamachikuna

        self.assertEqual(sorted(texts), sorted(kamachikuna.LAWS.values()))

    def test_lean_proven_flags_are_true_only_for_proven_laws(self):
        doc = self.os_.policy_document()
        proven = [r["id"] for r in doc["spec"]["rules"] if r["lean_proven"]]
        # laws 1,4,5,7,9 have Lean-proven theorems
        self.assertEqual(proven, ["kamachikuna.law%d" % n for n in (1, 4, 5, 7, 9)])

    def test_gate_fails_closed(self):
        d = self.os_.admission_gate("train.launch", verified=False)
        self.assertEqual(d["decision"], "BLOCKED")
        self.assertIn(6, d["laws"])
        self.assertIn("fail closed", d["reason"])

    def test_allow_requires_receipt(self):
        # verified but no receipt -> Law 7 blocks the claim
        d = self.os_.admission_gate("train.launch", verified=True, receipt=None)
        self.assertEqual(d["decision"], "BLOCKED")
        self.assertIn(7, d["laws"])
        # with a receipt -> allowed
        d2 = self.os_.admission_gate(
            "train.launch", verified=True, receipt={"sha256": "ab" * 32}
        )
        self.assertEqual(d2["decision"], "ALLOWED")
        self.assertEqual(d2["receiptSha256"], "ab" * 32)

    def test_irreversible_actions_are_human_only(self):
        d = self.os_.admission_gate(
            "payment.execute", verified=True, receipt={"sha256": "cd" * 32}
        )
        self.assertEqual(d["decision"], "BLOCKED")  # Law 7: human-only
        self.assertIn(7, d["laws"])
        self.assertIn("human-only", d["reason"])

    def test_audit_event_is_ocsf_shaped(self):
        d = self.os_.admission_gate("train.launch", verified=False)
        ev = self.os_.audit_event(d)
        self.assertEqual(ev["class"], "application_activity")
        self.assertFalse(ev["authorized"])
        self.assertEqual(ev["metadata"]["doctrine"], "v11")


if __name__ == "__main__":
    unittest.main(verbosity=2)
