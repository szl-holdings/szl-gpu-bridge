"""The doctrine set in stone: pin the sha256 of the canonical constitution.

Any edit to kamachikuna.py (laws, operational definitions, principles,
amendments, version) changes this hash and fails the suite. Amending the
constitution is therefore a deliberate, review-visible act — Law 5 applied
to the laws themselves.
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from szl import kamachikuna  # noqa: E402

PINNED_DOCTRINE_SHA256 = "280e23d484b3ac01732278b690da934ac9edc058d4033e42b90e855492c62125"


class DoctrineStoneTests(unittest.TestCase):
    def test_doctrine_hash_matches_pinned_value(self):
        doc = {
            "doctrine": kamachikuna.DOCTRINE_VERSION,
            "primePrinciples": kamachikuna.PRIME_PRINCIPLES,
            "laws": kamachikuna.LAWS,
            "operational": kamachikuna.OPERATIONAL,
            "amendments": kamachikuna.AMENDMENTS,
        }
        canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"))
        self.assertEqual(
            hashlib.sha256(canonical.encode()).hexdigest(),
            PINNED_DOCTRINE_SHA256,
        )

    def test_operational_covers_all_laws(self):
        self.assertEqual(sorted(kamachikuna.OPERATIONAL), list(range(1, 11)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
