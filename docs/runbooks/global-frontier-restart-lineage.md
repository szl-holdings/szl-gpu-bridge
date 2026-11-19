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
