# SZL Unified Model Portfolio — 54-hour request, source-only admission plan

**Status:** `SOURCE_REVIEW_ONLY_ALL_HELD` · 2026-11-20 UTC · no funded or owner-signed run.

This is a proposed **single logical control plane** for the SZL model estate.
It does not put 401 model weights in GPU memory or launch 401 trainers. The
source of truth remains protected `szl-holdings/szl-gpu-bridge` alongside
existing Hatun policies, Khipu receipts, Lean invariants and the
single-writer Salad admission controller. There must not be a second bridge,
dispatcher, lease signer, or cross-repository publishing agent.

## Fresh public inventory — not a trainability certificate

One anonymous public Hugging Face author-only API readback at
2026-11-20 08:44 UTC enumerated **401 model repositories and 172 dataset
repositories**, each bound to a public immutable Hub commit SHA. The metadata
snapshot has local SHA-256
`fb19204aff273323a37bb43b5492b42ed256910ee3f44f7150e24597180715d7`.
This is a *metadata receipt*, unsigned and not training evidence.

Model file signals, from API `siblings[].rfilename` suffixes only:

| Signal | Repositories | What it proves |
| --- | ---: | --- |
| Adapter filename listed | 189 | A possible adapter; base identity and bytes **unverified** |
| Other recognized weight filename listed | 37 | File listing only; architecture/weights **unverified** |
| No recognized weight filename listed | 175 | Not eligible to claim local training from this listing |
| **Total** | **401** | Repository count is not trained-model count |

Dataset *declared* licenses from public cards: Apache-2.0 **98**, MIT **9**,
CC BY 4.0 **8**, CC BY-SA 4.0 **1**, `other` **13**, undeclared **43**.
**56** therefore have no specific declared license in this coarse review.
Even a recognized license string is **not** authorization to use every row,
mix it with other corpora, send it to external providers, or contaminate a
held-out benchmark. No raw dataset rows were opened and no rights clearance
was conducted.

The tested source-only compiler produced **401/401 HOLD** review entries,
zero trainer launches and zero reward grants. Output is an *unsigned*
create-only JSON receipt, SHA-256
`30ca4ddfb439c7c6fd419d23edacbbb4b45a2f737de18645cdb506d9e97a919c`.
The dataset inventory and this report remain on BETTERWITHAGE, not uploaded
to Hugging Face or any other model provider.

## Nominal capacity mathematics — not a budget or GPU booking

An unspecialized request to devote 54 **single-GPU-hours per listed model**
would nominally require `401 × 54 = 21,654 single-GPU-hours`. This is **not**
an estimate of actual trainable model count or real GPU demand: the 401
entries include adapters, inference weights and nonrecognized formats.
Training architecture, rights, memory, multiple-GPU needs, evaluation and
availability must be independently qualified per candidate.

For an illustrative **20 single-GPU-slot** scenario over a 54-hour wall-clock
window, aggregate theoretical capacity is only `20 × 54 = 1,080 GPU-hours`,
leaving `20,574 nominal GPU-hours` uncovered. The fractional/preemptible
**aggregate lower bound** at 20 slots is `ceil(21,654 / 20) = 1,083 h`,
but **indivisible complete 54-hour jobs cannot be arbitrarily split**.
If each model instead occupies one slot for its entire 54-hour job,
the real mathematical lower bound becomes `ceil(401 / 20) × 54 = 1,134 h`
across **21 waves**. A 54-hour window can fit only 20 such complete jobs,
leaving 381 unserved, even before memory/quality/rights/overhead gates.
To finish within 54 hours would hypothetically require 401 simultaneous
single-GPU slots (not 401 verified trainable models). A counterexample to
aggregate-only admission: three 54-hour jobs need 162 GPU-hours; two slots
over an 81-hour window provide that aggregate, but can finish only two
complete jobs. The third requires a second 54-hour wave (108h total).
This is **not** evidence that SaladCloud offers 20 slots, that they have the
right VRAM, or that any billable capacity, quotas, rates, permissions, or
execution rights have been granted.

The compiler now supports optional **offline-only arithmetic** using
`--hypothetical-gpu-slots 20`; it produces no trainer run, API request,
lease, payment or estimate in USD. All returned `provider_capacity_verified`,
`spend_limit_verified`, `training_authorized` and
`training_quality_verified` fields remain `false`. It is a transparency
aid, not a scheduler. Existing owner-controlled Salad admission remains
the only possible future execution path.

## The one-stack architecture

1. **Registry / lazy residency.** A versioned graph indexed by
   `(kind, owner/repo, immutable_sha)` stores 401 model metadata entries,
   172 dataset references, lineage, and explicit `UNKNOWN` states.
   Workers load only separately admitted, hardware-compatible checkpoints.
   `ADAPTER_FILES_LISTED` requires its verified base model and license;
   GGUF/quantized inference assets are not assumed trainable checkpoints.
2. **Rights-aware data plane.** A per-model allowlist of permitted datasets,
   exact revisions, consent/source records, safe data transformations and
   train/validation/heldout splits. Detect near-duplicate holdout overlap,
   PII and cross-dataset license incompatibility. **Never** combine all
   dataset bytes by default. Persist only permitted source references; keep
   private data local to its approved execution boundary.
3. **Single admitted execution plane.** Reuse the existing
   `tools/salad/admission.py` and source-owner lease/dispatcher contract.
   A request for **54 hours per model** is a proposed budget horizon, not a
   mandatory/automatic training run. Select a qualified model/base pair and
   available GPU by *measured* VRAM, compute, runtime and energy receipts.
   The workstation has an observed GeForce RTX 5050 Laptop GPU with
   **8,151 MiB total and ~2,124 MiB free**; this does not establish 400-way
   local training. Remote A6000/Salad availability, sole owner identity,
   lease, signer, and budget remain **UNVERIFIED**.
4. **Hatun → Khipu → Lean validation.** Hatun enforces permitted actions,
   budgets and tenant/data boundaries; Khipu binds exact source, dataset,
   checkpoint and receipt digests in append-only history; Lean checks
   declared governance invariants. Lean proofs of gate logic do **not**
   prove model accuracy or trained competence. DSSE signature authority
   and independent verifier root must be established before any promotion.
5. **Independent reward/evaluation plane.** Gate rewards on rights-cleared
   data, an untouched independently bound holdout, replayable model outputs,
   a signed evaluator receipt, and known-denominator safety/abstention
   metrics. A possible research utility is
   `verified_improvement - lambda * verified_risk - mu * measured_cost`;
   **no numeric reward is defined** when any constituent is UNKNOWN.
   Neither time elapsed, API calls, self-reported loss nor downloads count
   as success or reward eligibility.

## Training gate and execution order

**DENY if any required field is not established.** Per candidate:
immutable model and base hashes, complete architecture/weight evidence,
rights for model and EACH dataset, PII/consent clearance, disjoint
heldout/test evaluation, expected VRAM and fixed GPU-hour budget, owner-pinned
single-writer lease, independent signer, checkpoint/resume anti-replay,
pre-training canary, and protected-source independent review.

Only after those gates pass in their respective authorized systems:
one owner-approved canary, inspect boot/step/checkpoint and OOM evidence,
fixed-sha reproducible heldout eval, then propose a bounded 54-hour
trial. Preserve immutable checkpoints and refund/stop rules. On uncertain
provider state, STOP/UNKNOWN does **not** permit auto-restart or resubmission.
A nonterminal receipt must never be counted as a completed model.

The spend planning equation for each independently approved job is
`GPU_hours_cap * provider_GPU_hour_rate + storage + egress + evaluation`,
with currency, rate revision and maximum total explicitly owner-ratified.
No provider quote/current rate is assumed. Each job needs its own
capacity/lease/reward decision; an unqualified repo gets **no job**.

## Data provenance and model council

OpenAI GPT-6, Mistral and Kimi can compare public architectures and critique
read-only, sanitized evidence. **Their actual proprietary pretraining
datasets, internal model weights, private instructions and training recipes
are not available for redistribution** into SZL or Hugging Face. If they
generate *new* synthetic examples, first establish provider terms, ownership,
licensed reuse, privacy controls and independent data-quality review;
otherwise hold. None of the three reviewers can confer signing, budget,
data-rights or independent human-merge authority merely by agreeing.

## Exact source commands (no provider I/O)

```sh
python -m unittest discover -s tests -p test_estate_54h_portfolio.py -v
python tools/estate_54h_portfolio.py --public-snapshot <public-metadata.json> --out <new-create-only-receipt.json>
```

No model weights, dataset rows, HF Jobs, GPU reservation, Cloudflare billing,
Snowflake warehouse, protected branch, private key or live service is touched
by the compiler. All positive admissions remain owner- and verifier-gated.
