"""Regression tests for untrusted receipt and routing inputs; no network."""

import datetime
import hashlib
import hmac
import json
import unittest

from szl import kamachikuna, khipu, openshell, provenance, router


def legacy_receipt(claim):
    """Construct adversarial signed input independently of the mint helper."""
    body = {
        "kind": khipu.KIND,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prev_receipt_sha256": None,
        "claim": claim,
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **body,
        "body_sha256": digest,
        "signature": hmac.new(
            b"test-only", digest.encode(), hashlib.sha256
        ).hexdigest(),
    }


class AdmissionRegressionTests(unittest.TestCase):
    def test_only_boolean_true_can_allow(self):
        for verified in ("false", "true", 1, [], {}, None):
            with self.subTest(verified=verified):
                self.assertEqual(
                    kamachikuna.gate("train.launch", verified)[1], "BLOCKED"
                )
                self.assertEqual(
                    openshell.admission_gate(
                        "train.launch", verified, receipt={"sha256": "a" * 64}
                    )["decision"],
                    "BLOCKED",
                )

    def test_receipt_reference_requires_a_full_digest(self):
        for receipt in (
            {},
            [],
            "receipt",
            {"sha256": ""},
            {"sha256": "a" * 63},
            {"sha256": "g" * 64},
            {"sha256": 1},
        ):
            with self.subTest(receipt=receipt):
                decision = openshell.admission_gate(
                    "train.launch", True, receipt=receipt
                )
                self.assertEqual(decision["decision"], "BLOCKED")
                self.assertIsNone(decision["receiptSha256"])

    def test_request_cannot_downgrade_an_irreversible_action(self):
        for action in openshell.IRREVERSIBLE_ACTIONS:
            with self.subTest(action=action):
                decision = openshell.admission_gate(
                    action, True, receipt={"sha256": "a" * 64}, irreversible=False
                )
                self.assertEqual(decision["decision"], "BLOCKED")
                self.assertIn(7, decision["laws"])

    def test_malformed_irreversible_flag_blocks(self):
        for flag in ("false", 0, [], {}):
            with self.subTest(flag=flag):
                self.assertEqual(
                    openshell.admission_gate(
                        "train.launch",
                        True,
                        receipt={"sha256": "a" * 64},
                        irreversible=flag,
                    )["decision"],
                    "BLOCKED",
                )

    def test_valid_khipu_body_digest_is_preserved(self):
        receipt = khipu.mint({"action": "train.launch"}, "test-only")
        decision = openshell.admission_gate("train.launch", True, receipt=receipt)
        self.assertEqual(decision["decision"], "ALLOWED")
        self.assertEqual(decision["receiptSha256"], receipt["body_sha256"])

    def test_ambiguous_receipt_hashes_block(self):
        decision = openshell.admission_gate(
            "train.launch", True, receipt={"sha256": "a" * 64, "body_sha256": "b" * 64}
        )
        self.assertEqual(decision["decision"], "BLOCKED")


class ReceiptRegressionTests(unittest.TestCase):
    def test_nonfinite_claims_cannot_be_minted_or_verified(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.assertFalse(
                    khipu.check(legacy_receipt({"metric": value}), "test-only")
                )
                with self.assertRaises(ValueError):
                    khipu.mint({"metric": value}, "test-only")

    def test_malformed_signed_claims_do_not_verify(self):
        for claim in (None, [], "claim", 1):
            with self.subTest(claim=claim):
                self.assertFalse(khipu.check(legacy_receipt(claim), "test-only"))

    def test_non_json_input_returns_false(self):
        receipt = legacy_receipt({"valid": True})
        receipt["claim"] = {"bad": object()}
        self.assertFalse(khipu.check(receipt, "test-only"))

    def test_unicode_signature_returns_false(self):
        receipt = legacy_receipt({"valid": True})
        receipt["signature"] = "é" * 64
        self.assertFalse(khipu.check(receipt, "test-only"))

    def test_empty_chain_cannot_prove_freshness(self):
        self.assertFalse(khipu.check_chain([], "test-only", max_age_days=7))

    def test_invalid_freshness_window_fails_closed(self):
        receipt = khipu.mint({"valid": True}, "test-only")
        for value in (float("nan"), float("inf"), -1, True, "7"):
            with self.subTest(value=value):
                self.assertFalse(
                    khipu.check_chain([receipt], "test-only", max_age_days=value)
                )

    def test_empty_signing_secret_is_not_a_verifier(self):
        with self.assertRaises(ValueError):
            khipu.mint({"valid": True}, "")
        self.assertFalse(khipu.check(legacy_receipt({"valid": True}), ""))


class ProvenanceRegressionTests(unittest.TestCase):
    def setUp(self):
        self.claim = {
            "kind": "szl.training-receipt/v1",
            "model_id": "model-a",
            "source_sha": "a" * 40,
            "datasets": {"train": "b" * 64},
        }

    def state(self, signed_claim, **overrides):
        fields = {
            key: self.claim[key] for key in ("model_id", "source_sha", "datasets")
        }
        fields.update(overrides)
        return provenance.ProvenanceClaim(
            **fields, training_receipt=khipu.mint(signed_claim, "test-only")
        ).state("test-only")

    def test_bound_training_receipt_is_measured(self):
        self.assertEqual(self.state(self.claim), "MEASURED")

    def test_unrelated_signed_claim_is_unavailable(self):
        self.assertEqual(self.state({"kind": "szl.training-receipt/v1"}), "UNAVAILABLE")

    def test_signed_evidence_cannot_be_reused_for_different_provenance(self):
        for overrides in (
            {"model_id": "model-b"},
            {"source_sha": "c" * 40},
            {"datasets": {"train": "d" * 64}},
        ):
            with self.subTest(overrides=overrides):
                self.assertEqual(self.state(self.claim, **overrides), "UNAVAILABLE")

    def test_wrong_receipt_kind_is_unavailable(self):
        self.assertEqual(self.state({**self.claim, "kind": "other"}), "UNAVAILABLE")

    def test_malformed_bound_digests_are_unavailable(self):
        for overrides in (
            {"source_sha": "not-a-sha"},
            {"datasets": {"train": "not-a-sha"}},
        ):
            with self.subTest(overrides=overrides):
                self.assertEqual(
                    self.state({**self.claim, **overrides}, **overrides), "UNAVAILABLE"
                )


class RouterRegressionTests(unittest.TestCase):
    def model(self, name="valid", **overrides):
        fields = dict(
            model_id=name,
            honesty_state="MEASURED",
            abstain_rate=0.9,
            prove_badge=None,
            joules_per_token=1.0,
            price_per_mtok=1.0,
        )
        fields.update(overrides)
        return router.Model(**fields)

    def test_invalid_metrics_cannot_win_routing(self):
        valid = self.model()
        for field in ("price_per_mtok", "joules_per_token", "abstain_rate"):
            for value in (-1, float("nan"), float("inf"), "0", True):
                with self.subTest(field=field, value=value):
                    bad = self.model("invalid", **{field: value})
                    self.assertIs(router.route("honesty-critical", [bad, valid]), valid)

    def test_measured_label_requires_a_valid_honesty_measurement(self):
        for value in (None, -0.1, 1.1):
            with self.subTest(value=value):
                model = self.model(abstain_rate=value)
                self.assertIsNone(model.honesty_score())
                self.assertIsNone(router.route("honesty-critical", [model]))

    def test_finite_zero_cost_and_energy_remain_valid(self):
        free = self.model("free", price_per_mtok=0.0, joules_per_token=0.0)
        self.assertIs(router.route("honesty-critical", [self.model(), free]), free)

    def test_failed_and_unknown_states_never_route(self):
        for state in ("FAILED", "UNAVAILABLE", "invented", "", None):
            for task in ("bulk", "honesty-critical"):
                with self.subTest(state=state, task=task):
                    invalid = self.model(honesty_state=state)
                    self.assertIsNone(router.route(task, [invalid]))

    def test_unmeasured_model_beats_failed_fallback(self):
        unknown = self.model("unknown", honesty_state="UNMEASURED", abstain_rate=None)
        failed = self.model("failed", honesty_state="FAILED", price_per_mtok=0.0)
        self.assertIs(router.route("bulk", [failed, unknown]), unknown)


if __name__ == "__main__":
    unittest.main()
