"""Tests for the e-means reference implementation (arXiv:2611.09501, CC BY 4.0)."""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from szl import emeans, khipu  # noqa: E402


def make_mixture(n=1200, k=8, d=8, seed=7):
    rng = random.Random(seed)
    centers = [[rng.gauss(0, 4) for _ in range(d)] for _ in range(k)]
    X = []
    for _ in range(n):
        c = centers[rng.randrange(k)]
        X.append([v + rng.gauss(0, 0.5) for v in c])
    return X, centers


class ScalingLawTests(unittest.TestCase):
    def test_laws_sane_and_deterministic(self):
        B1, a1 = emeans.scaling_laws(1_000_000, 16_384, 5)
        B2, a2 = emeans.scaling_laws(1_000_000, 16_384, 5)
        self.assertEqual((B1, a1), (B2, a2))
        self.assertTrue(1 <= B1 <= 1_000_000)
        self.assertTrue(0 < a1 <= 1.0)
        self.assertEqual(B1 & (B1 - 1), 0)  # power of two

    def test_large_n_batch_correction(self):
        B_small, _ = emeans.scaling_laws(1_000_000, 4096, 5)
        B_large, _ = emeans.scaling_laws(10_000_000, 4096, 5)
        self.assertLess(B_large, B_small * 2)  # /8 correction keeps B from exploding

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            emeans.scaling_laws(100, 200, 5)  # k > n
        with self.assertRaises(ValueError):
            emeans.e_means([[0.0, 0.0]], 2, 3)  # k > len(X)


class EMeansTests(unittest.TestCase):
    def test_beats_lloyd25_within_few_epochs(self):
        X, _ = make_mixture()
        C_lloyd = emeans.lloyd(X, 8, iters=25, seed=1)
        ref = emeans.mse(X, C_lloyd)
        _, stats = emeans.e_means(X, 8, T=5, seed=1)
        self.assertLessEqual(stats["mse"], ref * 1.05)  # within 5% of Lloyd@25

    def test_mse_matches_manual(self):
        X = [[0.0], [2.0], [10.0]]
        C = [[1.0], [10.0]]
        self.assertAlmostEqual(emeans.mse(X, C), (1.0 + 1.0 + 0.0) / 3)

    def test_dead_cluster_revive_runs(self):
        # k larger than natural clusters forces dead-cluster handling
        X, _ = make_mixture(n=600, k=2)
        C, stats = emeans.e_means(X, 12, T=3, seed=3)
        self.assertEqual(len(C), 12)
        self.assertGreater(stats["mse"], 0)

    def test_run_receipt_verifies_and_tamper_fails(self):
        X, _ = make_mixture(n=400, k=4, d=4)
        r = emeans.run_receipt(X, 4, 3, secret="s", seed=1)
        self.assertTrue(khipu.check(r, "s"))
        self.assertFalse(khipu.check(r, "wrong"))
        r["claim"]["config"]["mse"] = 0.0
        self.assertFalse(khipu.check(r, "s"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
