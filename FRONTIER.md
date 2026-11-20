# Frontier compute program — Salad-native

SZL's frontier training program runs on Salad GPU containers with governed receipts on the
Hugging Face Hub. This document is the standing program: doctrine, the measured fix ladder,
the post-train bench, and the research-derived candidate stack. Every claim below is anchored
to a receipted run from 2026-11-15.

## Doctrine

- All new frontier compute runs on Salad. No new Massed spend.
- Fail-closed everywhere: the worker exits non-zero on any unprovable state and never
  fabricates counts, timestamps, or losses. Receipts record read clocks only.
- Energy is MEASURED, never estimated: NVML cumulative counter deltas, cross-checked
  against Zeus; both readings ship in the receipt.
- Harvest standard: verify Hub finals FIRST (checkpoint ladder + `final/adapter_model.safetensors`
  + hbfinal), then patch the container group to replicas=0 and re-verify with a GET.
  Definitions are retained for provenance.

## Worker fix ladder (measured on live lanes)

| Fix | Version | What it closes | Evidence |
|-----|---------|-----------------|----------|
| F1–F6 | v3.2 | boot/heartbeat/receipt/exit honesty | 19 lanes reaped + verified this session |
| F7 | v3.3 | object-JSON datasets in the F3 fallback (JSONL, whole-file lists, row-list keys) | fpg lane: crash-loop → 200/200 in 2328.6 s, loss 0.5289, dual-energy 122,476 J / 122,277 J |
| F8 | v3.4 | (parallel lane) | merged #187 |
| F9 | v3.5 | honest resume-on-start: latest checkpoint verified before continuation, partial local ckpts deleted + re-downloaded, download retries 3×60 s then fail-closed exit 7 | merged #188; boot receipts state "checkpoint-N (verified)" vs "GENESIS" |

## Post-train bench (`tools/salad/bench_eval.py`, v2)

The v2 harness computes shifted-target-token-weighted causal-LM loss (`L-1` targets),
records offered/scored/skipped rows and input/predicted/correct token counts, and
converts NVML millijoule counters to joules. Its meter scope is GPU-0 device-total:
this benchmark does not independently establish process/whole-node energy or Zeus
cross-verification. A completed measurement is not a passed acceptance criterion;
held-out quality remains explicitly false until separate gates are measured.

Base, adapter and dataset revisions are resolved before hydration. A local original
result/source/hash bundle is preserved before one parent-bound Hub publication,
followed by exact-commit byte readback. Publication-only recovery never remeasures.
Historical v1 receipts retain their v1 identity and are not silently relabelled v2.

The plan-first [one-shot launcher](docs/runbooks/salad-benchmark-launch.md) binds
attempt, declared source revision, exact harness bytes and typed receipt destination.
It does not download mutable scripts, decode source at boot, hide a child failure
behind a shell success tail, start a watcher, or confer source/compute admission.
Deployment and generation-bound resource reclamation remain controller-owned gates.
Historical snapshots elsewhere in this program are not fresh runtime qualification.

## Active A/B: frontier-v5-r1 (kernel build at boot)

frontier-v4-r1 completed 200/200 in 25,831.9 s (~130 s/it) on the slow reference mamba
path; energy dual-readings agreed to 0.01% (NVML 4,877,372.7 J / Zeus 4,876,903.9 J, RTX 3090).
v5 clones the recipe exactly (base, dataset, steps, save cadence, receipts) but boots the
devel image and builds `causal-conv1d==1.7.0` + `mamba-ssm==2.3.2.post1` from source
before training, with an import self-test that logs KERNELS_OK or KERNEL-BUILD-FAILED
(graceful fallback to the reference path — the A/B never silently degrades). Decision rule:
adopt the kernel path only if the measured s/it and final receipts beat v4 under the same
harvest standard.

## Resolved A/B: frontier-v5 kernel verdict (measured 2026-11-20)

Measured receipts: SZLHOLDINGS/salad-frontier-v5-r1 (Hub finals: 200/200 steps, checkpoint + final + benchmark + held-out evaluation + integrity receipts); speed and energy cross-read from the v5-r7 and v4-r1 completion receipts in the shared receipts dataset (the v5-r1 shared-receipts folder was absent at read time - same kernel-path recipe, stated openly).

- **Kernel A/B resolved in favor of the kernel path** under the decision rule above: same final train loss (1.079), measured 82.35 s/step vs 129.16 s/step (**1.57x faster**), final-receipt energy 3,537,901 J vs 4,877,373 J (**27.5% less**, dual NVML + Zeus meters agree). The compiled-kernel path becomes the default for every Salad frontier run.
- **Quality is explicitly NOT accepted.** The held-out evaluation (2026-11-17) FAILED: the adapter refuses 75% of probes (base: 10%) and answers neutral 25% of the time (base: 95%). The kernel path is adopted for compute; the recipe is rejected for behavior. Held-out quality remains explicitly false until separate gates are measured.
- **Root-cause direction: data, not kernels.** v5 cloned v4's recipe exactly, including the szl-frontier-sft-mix-v4 SFT mix (26.7K rows, source labels), so the over-refusal lives in the mix. The next run (v6) must rebalance that mix (fewer refusal examples, more ordinary helpful answers) and re-run the same held-out check before any acceptance claim.
- **v6 is blocked until a valid Salad API key exists:** both known keys now return 401 (invalidated portal-side, 2026-11-20). No run starts on an unauthenticated control plane; fail closed.

Honesty markers: KERNEL_PATH_ADOPTED_1.57X_27PCT_LESS_ENERGY; RECIPE_HELDOUT_FAIL_OVERREFUSAL_75_VS_10; V6_DATA_REBALANCE_GATED_ON_NEW_SALAD_KEY. No decipherment claims. Fail closed.

## Research-derived candidate stack (each gated on a receipted A/B before adoption)

1. **DoRA / rsLoRA** — peft flags on the next training lane (LoRA SOTA 2025 family);
   adopt only if bench_eval criterion improves.
2. **Liger kernel** — fused CE/cross-entropy throughput candidate (~20% class);
   measure on a 0.8–4B lane before estate-wide use.
3. **Unsloth** — 0.8–1.5B lanes (estate has a receipted-unsloth precedent); 2×-class
   claims must be re-measured on Salad hardware.
4. **Contamination notes** — k-verify-benchmark / governed-receipts-bench datasets carry
   explicit contamination notes (2025 contamination-aware eval surveys) before any
   leaderboard claim.
5. **szl-salad-worker on PyPI** — publish the worker as a versioned package via Trusted
   Publishing (pattern proven by szl-kernels v0.2.0, live 2026-11-14); the container
   boot script then pins a release instead of a branch head.

## Leader benchmark (measured 2026-11-15)

Spot-checkpointing (NVIDIA Run:ai pattern) is what our F9 implements. Dual energy metering
(NVML + Zeus) already matches leader-class practice (ML.ENERGY). The open lane leaders do
NOT verify on their model cards — receipts remain our differentiator.

## Audit posture

Org audit (10-loop) and HF org audit run continuously across parallel lanes; archived repos
are excluded from actionable-red counts by policy; every red must be attributed to a lane
before action (never race a parallel lane). HF hygiene: 134 public models / 71 datasets /
21 spaces audited 2026-11-15; card/license gaps closed (4/4).
