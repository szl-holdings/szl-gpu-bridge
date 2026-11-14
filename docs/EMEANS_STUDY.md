# e-means field study + SZL adaptation

Source: Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou (FAIR at Meta,
Ponts et Chaussées), "e-means: outperforming k-means (in one-tenth the
epochs)", arXiv:2611.09501, 2026-11-06, license **CC BY 4.0** — adaptation
permitted with attribution. Implementation: `sdk/szl/emeans.py`.

## What the paper does (verified from the text)
- **EMA centroids** (VQ-VAE rule retargeted to fixed-dataset k-means):
  N_j ← γN_j + (1−γ)n_j, S_j ← γS_j + (1−γ)s_j, centroid = S_j/N_j (Eq. 2–4).
- **Cosine-annealed step** α(t) = α0(ε + (1−ε)·½(1+cos πt/M)), ε=0.1 (Eq. 5) —
  novel in this context; ablation shows it helps train MSE (+0.26% without).
- **Dead-cluster revival** every 5 batches: clusters below τ=0.1 × mean count
  are rebuilt by splitting a count-proportional donor with ±δ (BF16 eps)
  perturbation (Eq. 8). Biggest single contributor: without it, +2.97% train
  MSE and 26% dead clusters on DEEP.
- **Finalization**: last-epoch mean (train MSE) or slow-EMA (test MSE).
- **Scaling laws** (Table 2, fitted on DEEP, ~75k runs, OOD regret ~0.07–0.2%):
  α0 = 0.824·k^−0.46·B^0.90·n^−0.45·T^−0.11 (capped at 1);
  B = 0.035·k^0.13·n^0.88·T^0.81 → nearest power of 2, capped at n, /8 for n>2M.
- **Headline results**: beats Lloyd@25 loss in 1.4–2.6 epochs → 9.5–45.5×
  wall-clock vs Flash-KMeans; DINO10B (10B DINOv3 vectors, released CC BY 4.0)
  into 16.7M centroids in 2.5 epochs; IVF assignment cuts cost 34× at 10B.
- **Limits (honest)**: laws tuned for high-dim embeddings, weak on tiny/low-dim
  data; EMA beats SGD/Adam at large batches because centroids stay in the
  convex hull of assigned points.

## SZL adaptation ("our sauce")
- `sdk/szl/emeans.py`: pure-stdlib reference implementation (the SDK has zero
  runtime deps; the paper's GPU speedups need their CUDA kernels).
- Scaling-law hyperparameters, EMA+cosine, revival, finalization — all in.
- **Every clustering run mints a signed khipu receipt** binding config + final
  MSE (`run_receipt`) — Law 5/7: a clustering claim carries a receipt.
- Tests prove the core claim on synthetic data: e-means@5 epochs within 5%
  of Lloyd@25 MSE, plus law sanity, revival, and receipt tamper-detection.

## Where this plugs into the estate
- Killinchu (evidence/corpus layer): cluster embeddings at 10–45× less GPU
  time; receipt each run.
- KV-cache quantization (ClusterKV-style): e-means is the natural codebook
  trainer; receipt the codebook revision.
- DINO10B is public (CC BY 4.0) — a clean benchmark for the proving ground.
