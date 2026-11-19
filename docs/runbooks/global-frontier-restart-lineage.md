# China–Korea–Ireland frontier study → SZL restart-lineage gate

**Status: source-only research, not a deployed trainer, watcher, credential reader, scheduler, GPU approval, or production qualification.** Studied November 19, 2026. No external code, paper text, model weights, or dataset records were copied.

## Research-to-implementation matrix

| Region | Primary papers / code | SZL transfer and limits |
| --- | --- | --- |
| China — DeepSeek | [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437), [DeepSeek-V3 code](https://github.com/deepseek-ai/DeepSeek-V3), [DeepEP](https://github.com/deepseek-ai/DeepEP) | MoE load balance, multi-token prediction, FP8 and expert communication. Benchmark only after GPU architecture, network topology and compatible MoE workload are measured; do not install expert-parallel kernels on unqualified laptop or A6000 devices |
| China — Alibaba | [Qwen3 paper](https://arxiv.org/abs/2505.09388), [Qwen3 repository](https://github.com/QwenLM/Qwen3) | Compare dense/MoE, reasoning/non-reasoning modes and multilingual training with licensed pinned small-model baselines |
| South Korea — NAVER | [HyperCLOVA X THINK report](https://clova.ai/cdn/media/2025/06/HyperCLOVA_X_THINK_Technical_Report.pdf), [transformers integration](https://huggingface.co/docs/transformers/model_doc/hyperclovax) | Stage SFT → RLVR and consider μP proxy experiments only on suitable model families; do not transfer reported benchmark numbers to SZL weights |
| South Korea — KAIST | [KORMo paper](https://arxiv.org/abs/2510.09426), [KORMo tutorial](https://github.com/MLP-Lab/KORMo-tutorial) | Synthetic-data provenance, proxy ablations, published recipes/logs; independent non-synthetic holdouts and legal source admission required |
| Ireland — Trinity/UCC/QUB | [Qomhrá paper](https://aclanthology.org/2026.loreslm-1.18/), [UCCIX study](https://cora.ucc.ie/items/c068fc00-45c8-4e23-b0e5-5407683dbc06), [ADAPT Centre](https://www.adaptcentre.ie/about/) | Low-resource language adaptation and human evaluation: Qomhrá reports divergence between human Irish judgments and model-as-judge. For SZL Khipu/codices, require human adjudication, rights, and no archaeological decipherment overclaim |
| NVIDIA | [Megatron Bridge checkpointing](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html) | Preserve full optimizer/RNG/scheduler/dataloader state on resume; weight-only initialization is NOT equivalent |
| PyTorch / Ray | [TorchTitan](https://github.com/pytorch/torchtitan), [Ray Train recovery](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html) | Reuse existing protected worker and owner/generation locks; do not create another scheduler |

The above are third-party reported methods, not reproduced SZL benchmarks. Data and licenses require individual verification before ingestion.

## SZL's original, conservative mathematics

For previous recorded trainer step \`s\`, full-state checkpoint step \`c\` and target \`T\` under an identical immutable run:

\`0 ≤ c ≤ s ≤ T\`

\`Δ_replay = s − c\` and \`replay_fraction = (s − c) / T\`.

These are exact **conditional arithmetic** only. They do not authenticate the reported trainer step, checkpoint bytes, source/command pin, or an independent backup. If the checkpoint does not exist and \`s > 0\`, a claimed full resume must HOLD. A genesis attempt with previous work also HOLDs pending separate owner cost/authority review.

The new \`tools/salad/restart_lineage_witness.py\` consumes a strict small JSON evidence proposal and returns \`HOLD\` or \`REVIEW_CANDIDATE_ONLY\`. It rejects duplicate keys, NaN, stale clocks, bad step arithmetic, mismatched base/dataset/source/command hashes, and missing adapter/trainer/optimizer/RNG/scheduler/dataloader/tokenizer components. Input hashes and booleans are **not independently verified** by the checker. \`gpu_training_authorized\` is always false.

### SZL formula and proof boundaries

- Hard, noncompensatory pre-admission: \`G = G_owner ∧ G_budget ∧ G_source ∧ G_data_rights ∧ G_checkpoint ∧ G_backup ∧ G_eval\`. Missing facts are HOLD, not an optimistic default.
- [SZL prior art](https://github.com/szl-holdings/szl-papers/blob/main/prior-art/README.md) describes the PURIQ score, Yuyay-13 conjunctive gate and Khipu receipt DAG. Use these as **governance research**, never as a substitute for signed action authority.
- Geometric mean \`Λ_k(x) = (∏ x_i)^(1/k)\` may rank **already admitted** candidates with disclosed factorization assumptions. The [SZL v23 outline](https://github.com/szl-holdings/szl-papers/blob/main/thesis/ouroboros/papers/v23/outline.md) explicitly states unconditional Λ uniqueness under A1–A5 is false; do not claim it proved or let a high score override a veto.
- Khipu can content-address the exact reviewed receipt chain, but a SHA-256 digest does not prove signer identity, data rights, training progress, or physical GPU activity.
- ORO/Ouroboros research loop: propose → source rights → short proxy → disjoint heldout/human review → measured improvement and cost → human admission. Lean/Lake may verify finite event-order invariants at a pinned kernel; they cannot prove external checkpoints exist.

## Owner-only deployment gate

The active controller must independently establish current Salad group and all actual instances, exclusive owner/generation, signed budget, source/base/data/image, current trainer progress, immutable full-state checkpoint and independently verified backup. Preserve opaque optimizer/RNG pickles as bytes; never deserialize unknown code. Do not rearm from a \`final/\` path, liveness ping, HTTP 200, or green CI. This PR does not alter any existing workers, Docker images, paid replicas, public model cards or network endpoints.

Offline tests: \`python -m unittest discover -s tests -p test_restart_lineage_witness.py -v\`.

## 2026-11-19: data-parallel topology-safe restart witness v2

**Problem discovered by source review:** The original `szl.restart-lineage-input/v1` checked the existence of a declared `dataloader: true` flag but not whether a purported full-state resume had the same data-parallel rank topology or the exact selected loader checkpoint step. It could therefore report `REVIEW_CANDIDATE_ONLY` on a structurally incomplete self-report. The check was always non-authorizing, but the wording risked a false positive for owner review.

**External technical basis:** NVIDIA's [Megatron Bridge checkpointing guidance](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html) explicitly distinguishes model/optimizer resharding from Energon rank-local data-loader restoration. The latter needs the same data-parallel size. A missing rank-local loader state is fatal, not permission to silently resume at the beginning. Energon may replay deterministic samples internally from an earlier snapshot; the `checkpoint_step` below identifies the *selected full checkpoint*, not proof of exact sample-position reconstruction.

**V2 shape:** The strict JSON input uses `"schema": "szl.restart-lineage-input/v2"` and all the existing v1 fields plus required `resume_topology`. For a `resume` intent, `resume_topology` must be an object with exactly:

- `saved_dp_size` and `requested_dp_size`: genuine integers 1–128, identical;
- `dataloader_checkpoint_step`: an integer equal to the selected full-state `checkpoint.step`;
- `rank_digests`: a list of exactly `saved_dp_size` objects, each with the exact keys `rank`, `checkpoint_step`, `state_sha256`. Each integer `rank` is unique in `[0, saved_dp_size)`; every `checkpoint_step` equals the selected full checkpoint step; each digest is a lower-case 64-hex-character string.

The structure is intentionally strict, finite and independent of input ordering. Unknown keys, missing rank manifests, duplicate or boolean ranks, unbounded/changed DP size, inconsistent steps, stale observation clocks, changed source/base/dataset/command pins and non-finite JSON all HOLD. Historical v1 JSON remains parseable, but **v1 `resume` always HOLDS for `TOPOLOGY_EVIDENCE_REQUIRED`**. An existing v1 new/genesis declaration may remain an unauthenticated owner-review candidate when there are zero prior steps, but never gains GPU authority.

**Mathematical acceptance condition:** Let `c` be selected checkpoint step and `d` saved DP size. The conditional topology declaration predicate is

`G_topology = (d = requested_dp_size) AND (d >= 1) AND (len(rank_digests) = d) AND (set(ranks) = {0,...,d-1}) AND (dataloader_checkpoint_step = c) AND (forall r, rank_checkpoint_step[r] = c)`.

This is a **conjunctive veto**, not a component that high aggregate scores can compensate. SZL's [PURIQ/Yuyay-13 prior-art note](https://github.com/szl-holdings/szl-papers/blob/main/prior-art/README.md) describes a conjunctive policy gate; its Lambda aggregator and Ouroboros research ranking cannot override a missing topology, owner authority, or independent checkpoint proof. The public [v23 outline](https://github.com/szl-holdings/szl-papers/blob/main/thesis/ouroboros/papers/v23/outline.md) preserves the conditional nature of Lambda uniqueness.

**Crucial nonclaims:** Caller-supplied SHA strings are **not** file byte checks, signatures, source authenticity, physical rank observations, GPU/process liveness, optimizer compatibility, sample-cursor proof, or independent backups. The fields `topology_bytes_verified_here`, `dataloader_resume_verified_here` and `gpu_training_authorized` stay `false` on all outcomes, including `REVIEW_CANDIDATE_ONLY`. Kimi proposed independent state-hash recomputation, but the pure witness has no access to those bytes; we intentionally did **not** claim to implement it. A separately reviewed owner must check rank-state bytes, signed work authority, exact trainer/iterator recovery and independent backup before any charged restart.

**Research comparison, not code import:** [DeepSeek's DeepEP](https://github.com/deepseek-ai/DeepEP) shows expert-parallel throughput techniques but current V2.5 requirements include Hopper-or-newer GPUs and high-speed interconnect, so it is not assumed compatible with SZL's laptop RTX 5050 or the earlier A6000 footprint. [KAIST KORMo](https://kaist-mlplab.github.io/kormo/) is a reproducibility model (published code, recipes, data and checkpoints); take its source-bound evaluations and explicit proxy ablations, not its benchmark scores. Ireland's [Qomhrá paper](https://aclanthology.org/2026.loreslm-1.18/) illustrates how L1-speaker evaluations can disagree with model-as-judge; apply human/heldout review before any Khipu translation or low-resource cultural claims. [SLSA provenance v1.2](https://slsa.dev/spec/v1.2/build-provenance) reinforces that an artifact's declared source is not the same as independently verified provenance.

**Safe local validation:** `python -m unittest discover -s tests -p test_restart_lineage_witness.py -v`. On an isolated Windows source copy, 31 tests passed, with 300 deterministic malformed-JSON mutations; the source and tests also passed `ruff format --check` and the repository's explicit E4/E7/E9/F lint subset. Passing tests is source-level verification only. No scheduler, controller, process, paid job, publication or operator policy was modified.
