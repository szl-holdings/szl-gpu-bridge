"""Tests for attestation receipts (tee-quote / proof-of-sampling binding)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from szl import attest, khipu  # noqa: E402

SECRET = "s"
BASE = dict(model_sha256="ab" * 32, input_sha256="cd" * 32, output_sha256="ef" * 32)


class AttestTests(unittest.TestCase):
    def test_verified_tee_receipt(self):
        r = attest.mint_attestation(**BASE, attestation_source="tee-quote",
                                    attestation_digest="11" * 32,
                                    tee_quote_verified=True, secret=SECRET)
        self.assertTrue(khipu.check(r, SECRET))
        self.assertEqual(attest.attestation_state(r, SECRET), "VERIFIED")

    def test_unverified_quote_is_unavailable_not_claim(self):
        r = attest.mint_attestation(**BASE, attestation_source="tee-quote",
                                    attestation_digest="11" * 32,
                                    tee_quote_verified=False, secret=SECRET)
        self.assertTrue(khipu.check(r, SECRET))  # record itself intact
        self.assertEqual(attest.attestation_state(r, SECRET), "UNAVAILABLE")

    def test_none_source_is_unattested(self):
        r = attest.mint_attestation(**BASE, attestation_source="none",
                                    attestation_digest=None,
                                    tee_quote_verified=False, secret=SECRET)
        self.assertEqual(attest.attestation_state(r, SECRET), "UNATTESTED")

    def test_tampered_receipt_unavailable(self):
        r = attest.mint_attestation(**BASE, attestation_source="proof-of-sampling",
                                    attestation_digest="22" * 32,
                                    tee_quote_verified=False, secret=SECRET)
        r["claim"]["model_sha256"] = "00" * 32
        self.assertEqual(attest.attestation_state(r, SECRET), "UNAVAILABLE")

    def test_fail_closed_inputs(self):
        with self.assertRaises(ValueError):
            attest.mint_attestation(**BASE, attestation_source="magic",
                                    attestation_digest="x", tee_quote_verified=False,
                                    secret=SECRET)
        with self.assertRaises(ValueError):
            attest.mint_attestation(**BASE, attestation_source="tee-quote",
                                    attestation_digest=None, tee_quote_verified=True,
                                    secret=SECRET)

    def test_wrong_secret_unavailable(self):
        r = attest.mint_attestation(**BASE, attestation_source="tee-quote",
                                    attestation_digest="11" * 32,
                                    tee_quote_verified=True, secret=SECRET)
        self.assertEqual(attest.attestation_state(r, "wrong"), "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
