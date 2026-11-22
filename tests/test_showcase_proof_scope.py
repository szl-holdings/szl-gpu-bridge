"""Reject unproven universal receipt/energy claims in public HF source snapshots.

This is a static source contract, not live HF readback or signature verification.
"""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    "hf-spaces/frontier-showcase/snapshot/index.html",
    "hf-spaces/frontier-showcase/snapshot/static/index.html",
    "hf-spaces/khipu-playground/snapshot/index.html",
    "hf-spaces/szl-energy-hologram/snapshot/index.html",
    "hf-spaces/szl-energy-hologram/snapshot/static/index.html",
    "hf-spaces/estate-gardener/static/index.html",
    "hf-spaces/eternal-auditor/static/index.html",
)
UNPROVEN = (
    "Everything on this page is fetched live from the public hash-chained receipts",
    "THE CHAIN — every receipt, hash-linked, khipu-signed, public",
    "every training run, every measurement, every deployment, every failure",
    "Every joule measured. Every receipt hash-chained.",
    "Every receipt above is HMAC-signed (full 64-hex)",
    "Nothing in this diagram is claimed; every stage is measured or published.",
)
SCOPE = {
    "hf-spaces/frontier-showcase/snapshot/index.html": "independent signer trust",
    "hf-spaces/khipu-playground/snapshot/index.html": "complete estate coverage",
    "hf-spaces/szl-energy-hologram/snapshot/index.html": "missing energy observations remain UNAVAILABLE",
    "hf-spaces/estate-gardener/static/index.html": "independently trusted signatures",
    "hf-spaces/eternal-auditor/static/index.html": "independently trusted signatures",
}


class SourceCopyTruthTests(unittest.TestCase):
    def test_pages_exist_and_disallow_universal_proof_claims(self):
        for path in PAGES:
            with self.subTest(path=path):
                content = (ROOT / path).read_text(encoding="utf-8")
                for phrase in UNPROVEN:
                    self.assertNotIn(phrase, content)

    def test_scope_disclosures_present(self):
        for path, phrase in SCOPE.items():
            with self.subTest(path=path):
                self.assertIn(phrase, (ROOT / path).read_text(encoding="utf-8"))

    def test_khipu_browser_hash_is_not_claim_truth(self):
        """Untrusted free-text input must not become a measured evidence claim."""
        html = (
            ROOT / "hf-spaces/khipu-playground/snapshot/index.html"
        ).read_text(encoding="utf-8")
        self.assertIn('tier:"SELF_ASSERTED"', html)
        self.assertIn("SELF_ASSERTED (bytes hashed; claim unverified)", html)
        self.assertIn("claim truth, signer authority", html)
        self.assertNotIn('tier:"MEASURED"', html)
        self.assertNotIn("MEASURED (you just measured it)", html)
        self.assertNotIn("3/6 on the abstention", html)

    def test_khipu_source_candidate_not_claimed_deployed(self):
        """A reviewed GitHub snapshot edit is not an immutable Hub revision."""
        readme = (
            ROOT / "hf-spaces/khipu-playground/README.md"
        ).read_text(encoding="utf-8")
        self.assertIn("source-only correction candidate", readme)
        self.assertIn("not deployment evidence", readme)

    def test_static_snapshot_mirrors_identical(self):
        for slug in ("frontier-showcase", "szl-energy-hologram"):
            with self.subTest(slug=slug):
                base = ROOT / "hf-spaces" / slug / "snapshot"
                self.assertEqual(
                    (base / "index.html").read_bytes(),
                    (base / "static" / "index.html").read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
