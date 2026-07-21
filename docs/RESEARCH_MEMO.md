# Research memo — leaders studied → lessons adopted → what SZL does differently

*2026-07-21. Sources: four parallel research passes (unsloth deep-ingest from a cloned tree; field-leader survey; SZL estate audit; bridge transport design). Labels: MEASURED = primary artifact read directly; REPORTED = docs/web with URL; MODELED = our estimate, not run. Trust ceiling 0.97 honored throughout.*

## 1. Leaders studied

| Who | Signature pattern | What we verified |
|---|---|---|
| Tri Dao / Dao-AILab | FlashAttention-3: warp-specialization, async TMA/WGMMA, FP8 | MEASURED (abstract read) |
| Tim Dettmers / bitsandbytes | NF4 4-bit, double-quant, paged optimizers (QLoRA) | REPORTED |
| Answer.AI | FSDP+QLoRA — 70B on 2×24 GB consumer cards | REPORTED |
| Hugging Face | PEFT/TRL/accelerate + **Kernels Hub** (Hub-loaded compiled kernels, `build.toml`, versioned `vN` branches, provenance block in `metadata.json`) | MEASURED (docs + repo format read) |
| LinkedIn Liger-Kernel | fused Triton RMSNorm/RoPE/SwiGLU/CE as one-line patch | REPORTED |
| Unsloth | monkeypatch-at-load fused kernels; dynamic 4-bit skip-lists; offloaded sqrt-spaced gradient checkpointing; chunked-logsumexp cross-entropy; padding-free packing | MEASURED (v2026.7.4 tree cloned and read) |
| Axolotl / LLaMA-Factory / torchtune | config-driven finetune orchestration | REPORTED |
| Keller Jordan | Muon (orthogonalized momentum); nanoGPT speedrun culture | REPORTED |
| PEFT frontier | LoRA+, rsLoRA, DoRA, GaLore, Spectrum layer-freezing | REPORTED |

## 2. Lessons adopted (pattern → SZL shape)

1. **QLoRA/NF4 floor** → job-spec `kind: unsloth-qlora-sft-v1`; 4-bit base pinned by revision; receipt records quant config. *(biggest VRAM lever for a 1.5B on one RTX)*
2. **Offloaded gradient checkpointing** → recipe flag `gradientCheckpointing: "unsloth"`; measured step-time cost lands in the training receipt, not marketing.
3. **8-bit paged optimizer** → `optimizer: adamw_8bit` default; optimizer choice is part of the signed spec.
4. **VRAM-aware auto-sizing (Unsloth autotune)** → inverted into a **fail-closed gate**: probe free VRAM *before* load; below `minFreeVramGb` → signed BLOCKED receipt, never a swap-thrash attempt.
5. **Padding-free packing** → optional `recipe.packing`; off for v1 (correctness first), adoption gated on a MEASURED no-contamination check.
6. **Config-driven orchestration (Axolotl et al.)** → but our config is a **DSSE-signed contract**, not a mutable YAML: the laptop refuses unsigned/expired/tampered specs.
7. **Kernels-Hub provenance block + dirty flag** → adopted as the model for szl-kernels compliance; a `dirty` build flag maps naturally onto an honest BLOCKED gate.
8. **Speedrun culture (measure, publish, iterate)** → every run publishes MEASURED wallclock/VRAM/loss in receipts; no number ships without a receipt.
9. **LoRA+ ratio** → optional `loraPlusLrRatio` field reserved in schema v1; off until a MEASURED A/B on our metal justifies it.
10. **Service-survives-reboot (GH runner docs)** → scheduled task AtStartup + StartWhenAvailable + single-flight lock + idempotent job ledger; the runner *mechanism* itself was rejected (no workflow-write scope, no laptop GitHub auth).

**License boundaries honored (MEASURED from cloned LICENSE files):** unsloth core Apache-2.0 (usable), unsloth-zoo LGPL-3.0 (**depend, never vendor**), `kernels/moe/grouped_gemm` AGPL-3.0 (**not used at all** — network copyleft would reach szl services). Adapted patterns carry attribution in `NOTICE`; no upstream file is copied.

## 3. What SZL does differently (the honest gap)

Prior art we checked and respect: **OpenSSF Model Signing** signs artifact bytes (no training→eval chain, no recomputable evals); **ZK proof-of-training** (Verifiable Fine-Tuning, ZKPROV, SUMMER) proves computation cryptographically but at prover costs far beyond a laptop; **C2PA** targets media manifests; classic **Proof-of-Learning is known broken** (spoofing literature). Nobody mainstream ships the cheap middle: **a signature-chained receipt pair (training → eval), ed25519 over canonical JSON, published beside the weights, re-verifiable by anyone, with honesty labels and fail-closed BLOCKED verdicts** — deployable on one Windows laptop with an `hf` login and no prover farm.

That chain is exactly what this bridge extends end-to-end:

```
market data (REPORTED) → szl-quant backtest receipts (MEASURED, DSSE)
  → content-addressed dataset archives → SFT rows (deterministic replay, every row cites receipt sha + archive sha)
  → signed dataset manifest → DSSE-signed job spec (this repo)
  → laptop training receipt (MEASURED wallclock/VRAM/loss, signed on-metal)
  → eval receipt pinning the training receipt sha (chain)
  → cloud verifier — unverified claim = no claim
```

**What we do NOT claim:** receipts are attestations by a keyholder, not cryptographic proof the computation ran (that is ZK-PoT's territory; we cite it rather than imitate it). Model quality claims are limited to the eval receipt's held-out MEASURED numbers; market skill is claimed **nowhere** — the upstream engine is paper-only and its own backtest receipts state their negative/limited results honestly.

## 4. Estate actions this research triggered

- **szl-quant-sft-v1 dataset** (HF): receipt-derived training rows with per-row provenance — shipped with signed manifest.
- **szl-kernels**: format-valid but not proven-loadable (estate audit #1 gap); adopt Kernels-Hub `get_kernel()` round-trip proof + provenance block — queued as follow-up, stated honestly on its card until then.
- **Model cards**: add HF `model-index` structured eval metadata so receipt numbers become machine-readable (audit gap #2).
- **szl-nemo**: unsigned-receipt outlier + incoherent Modelfile — flagged for re-signing or honest demotion.

*Full source briefs preserved under `.agents/deliverables/` in the workspace: unsloth ingest, leader survey, estate audit, bridge design.*
