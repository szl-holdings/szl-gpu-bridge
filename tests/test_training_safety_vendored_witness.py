"""Drift guard: the training_safety copy of the PR #264 transition witness
must stay byte-identical to the canonical tools/salad copy."""

import hashlib
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VendoredWitnessTest(unittest.TestCase):
    def test_copy_matches_canonical(self):
        canonical = ROOT / "tools" / "salad" / "provider_transition_witness.py"
        vendored = ROOT / "tools" / "training_safety" / "provider_transition_witness.py"
        self.assertEqual(sha(canonical), sha(vendored))


if __name__ == "__main__":
    unittest.main()
