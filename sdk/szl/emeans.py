"""emeans — reference implementation of e-means with SZL receipts.

Adapted from "e-means: outperforming k-means (in one-tenth the epochs)"
Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou — FAIR at Meta /
École Nationale des Ponts et Chaussées, arXiv:2611.09501 (2026),
license CC BY 4.0. Attribution required; this file is a derived work.

What e-means adds over Lloyd: EMA centroid updates with a cosine-annealed
step, dead-cluster revival by donor splitting, last-epoch-mean
finalization, and scaling laws that pick batch size B and peak step α0
from (n, k, T) — no hyperparameter sweep.

SZL sauce: every run can mint a signed khipu receipt binding the exact
configuration and final MSE (Law 5/7: a clustering claim carries a receipt).

Pure stdlib (the SDK has no runtime dependencies); intended for reference,
teaching, and small/medium data — the paper's 10-45x GPU speedups need
their CUDA kernels, not this file.
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

from . import khipu

# Scaling-law coefficients, Table 2 of arXiv:2611.09501 (fitted on DEEP).
_STEP_LAW = {"c": 0.824, "a_k": -0.46, "a_B": 0.90, "a_n": -0.45, "a_T": -0.11}
_BATCH_LAW = {"c": 0.035, "b_k": 0.13, "b_n": 0.88, "b_T": 0.81}
_EPS = 0.1          # cosine floor fraction (Eq. 5)
_TAU = 0.1          # dead-cluster threshold (fraction of mean count)
_REVIVAL_EVERY = 5  # batches between revival passes
_DELTA = 2 ** -8    # donor perturbation (BF16 machine epsilon, Eq. 8)


def scaling_laws(n: int, k: int, T: int) -> tuple[int, float]:
    """Formulaic (B, α0) from the problem configuration — no sweep."""
    if n <= 0 or k <= 0 or T <= 0 or k > n:
        raise ValueError("need 0 < k <= n and T > 0")
    B = _BATCH_LAW["c"] * k ** _BATCH_LAW["b_k"] * n ** _BATCH_LAW["b_n"] * T ** _BATCH_LAW["b_T"]
    if n > 2_000_000:
        B /= 8  # large-scale correction from the paper
    B = max(1, min(n, 2 ** round(math.log2(max(B, 1)))))  # nearest power of two, capped at n
    alpha0 = _STEP_LAW["c"] * k ** _STEP_LAW["a_k"] * B ** _STEP_LAW["a_B"] \
        * n ** _STEP_LAW["a_n"] * T ** _STEP_LAW["a_T"]
    return B, min(alpha0, 1.0)


def _dist2(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def mse(X: Sequence[Sequence[float]], C: Sequence[Sequence[float]]) -> float:
    """Mean squared distance to nearest centroid (Eq. 1)."""
    if not X or not C:
        raise ValueError("empty input")
    total = 0.0
    for x in X:
        total += min(_dist2(x, c) for c in C)
    return total / len(X)


def lloyd(X: Sequence[Sequence[float]], k: int, iters: int = 25,
          seed: int = 0) -> list[list[float]]:
    """Classic Lloyd full-batch k-means, random init (the Faiss default)."""
    rng = random.Random(seed)
    C = [list(x) for x in rng.sample(list(X), k)]
    for _ in range(iters):
        sums = [[0.0] * len(X[0]) for _ in range(k)]
        counts = [0] * k
        for x in X:
            j = min(range(k), key=lambda j: _dist2(x, C[j]))
            counts[j] += 1
            for d, v in enumerate(x):
                sums[j][d] += v
        for j in range(k):
            if counts[j]:
                C[j] = [s / counts[j] for s in sums[j]]
    return C


def e_means(X: Sequence[Sequence[float]], k: int, T: int = 5,
            B: int | None = None, alpha0: float | None = None,
            seed: int = 0) -> tuple[list[list[float]], dict[str, Any]]:
    """e-means (Algorithm 1): EMA centroids, cosine-annealed step,
    dead-cluster revival, last-epoch-mean finalization.

    Returns (centroids, stats). Hyperparameters default to the scaling laws.
    """
    if not X or k <= 0 or k > len(X) or T <= 0:
        raise ValueError("need 0 < k <= len(X) and T > 0")
    if B is None or alpha0 is None:
        B, law_alpha0 = scaling_laws(len(X), k, T)
        alpha0 = law_alpha0 if alpha0 is None else alpha0
    rng = random.Random(seed)
    d = len(X[0])
    C0 = rng.sample(list(X), k)
    N = [1e-10] * k
    S = [[1e-10 * v for v in c] for c in C0]
    M = T * max(1, len(X) // B)
    t = 0
    last_assign: list[int] | None = None
    for _ in range(T):
        order = list(X)
        rng.shuffle(order)
        for start in range(0, len(order), B):
            batch = order[start:start + B]
            alpha = alpha0 * (_EPS + (1 - _EPS) * 0.5 * (1 + math.cos(math.pi * t / M)))
            gamma = 1.0 - alpha
            nj = [0] * k
            sj = [[0.0] * d for _ in range(k)]
            assign = []
            for x in batch:
                j = min(range(k), key=lambda j: _dist2(x, [s / N[j] for s in S[j]]))
                assign.append(j)
                nj[j] += 1
                for dd, v in enumerate(x):
                    sj[j][dd] += v
            for j in range(k):
                N[j] = gamma * N[j] + (1 - gamma) * nj[j]
                S[j] = [gamma * s + (1 - gamma) * v for s, v in zip(S[j], sj[j])]
            t += 1
            if t % _REVIVAL_EVERY == 0:
                _revive(N, S, k, d, rng)
            last_assign = assign
    # finalization: last-epoch mean over the final pass's assignments
    sums = [[0.0] * d for _ in range(k)]
    counts = [0] * k
    order = list(X)
    rng2 = random.Random(seed + 1)
    rng2.shuffle(order)
    for x in order:
        j = min(range(k), key=lambda j: _dist2(x, [s / N[j] for s in S[j]]))
        counts[j] += 1
        for dd, v in enumerate(x):
            sums[j][dd] += v
    C = [[s / counts[j] if counts[j] else S[j][i] / N[j] for i, s in enumerate(sums[j])]
         for j in range(k)]
    stats = {"n": len(X), "k": k, "T": T, "B": B, "alpha0": alpha0, "updates": M,
             "mse": mse(X, C)}
    return C, stats


def _revive(N: list[float], S: list[list[float]], k: int, d: int, rng: random.Random) -> None:
    """Dead-cluster revival: split a count-proportional donor (Eq. 8)."""
    mean_n = sum(N) / k
    dead = [j for j in range(k) if N[j] < _TAU * mean_n]
    for j in dead:
        donors = [jp for jp in range(k) if jp != j and N[jp] > 0]
        if not donors:
            break
        donor = rng.choices(donors, weights=[N[jp] for jp in donors])[0]
        c_donor = [S[donor][i] / N[donor] for i in range(d)]
        child_a = [c * (1 + _DELTA) if i % 2 == 0 else c * (1 - _DELTA)
                   for i, c in enumerate(c_donor)]
        child_b = [c * (1 - _DELTA) if i % 2 == 0 else c * (1 + _DELTA)
                   for i, c in enumerate(c_donor)]
        N[donor] /= 2
        S[donor] = [N[donor] * v for v in child_a]
        N[j] = N[donor]
        S[j] = [N[j] * v for v in child_b]


def run_receipt(X: Sequence[Sequence[float]], k: int, T: int, secret: str,
                seed: int = 0) -> dict[str, Any]:
    """Run e-means and mint a signed khipu receipt for the run (Law 5/7)."""
    C, stats = e_means(X, k, T, seed=seed)
    return khipu.mint({"kind": "szl.emeans-run/v1", "config": stats}, secret)
