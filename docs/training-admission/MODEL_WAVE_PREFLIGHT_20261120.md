# SZL model portfolio — zero-GPU wave preflight (v0)

**Status: CPU-only source candidate; NOT a new Salad runner. No production authorization.**

This companion builds on the merged single-writer Salad admission
controller and dry-run runner in tools/salad/admission.py and
tools/salad/admission_runner.py. The owner-controlled Salad mutation
seam remains unique. This preflight never calls that seam.

Mistral Large 4 and Kimi K3.1 Pro were separately consulted over Ollama on
2026-11-20 UTC with PUBLIC and sanitized facts only. No credentials, private
source, browser cookies, payment authority, or customer data were delegated.
Their advice is not an approval or a signature.

## Why this lane

A Hugging Face model repository is an asset pointer, not proof that a
distinct training run occurred or that its resulting model meets an
independent held-out evaluation gate. Do not queue 400 paid jobs from a
Hub repo count. Distinguish LoRA, merged, GGUF and evaluation artifacts.
Compare immutable model base/dataset/recipe identity before requesting compute.

The new tools/salad/training_wave_preflight.py accepts ONE operator-reported
local JSON plan with 1–500 candidate rows:

- Schema szl.training-wave-plan/v0; mode DRY_RUN; catalog SHA256.
- Auth witness state MISSING, DENIED_401, READ_ONLY, UNKNOWN or
  EXTERNALLY_REPORTED_VALID; these are metadata, NOT credential verification.
- Owner budget approval reported as a Boolean, owner budget in micro-USD,
  and the in-flight lane-ID set. None confer signing or payment authority.
- Per model: model ID, lane ID, train/evaluate intent, expected artifact,
  immutable base and dataset revisions, recipe and trainer commits, reported
  license rights, GPU SKU and VRAM, replicas, time and price ceilings,
  checkpoint cadence/target, bounded attempt count (1–3), canary/eval
  receipt digests and dependencies.

Every input read is bounded to 2 MiB. Duplicate JSON keys, unknown fields,
unhashable types, malformed immutable revisions, 0 or negative prices, and
oversized waves are rejected. Identical training recipe identities, duplicate target-model repositories,
and in-flight lanes receive explicit denial labels, not silent deduplication.

Price calculation uses integer micro-USD per GPU-hour:
CEIL(hourly_micro_usd * replicas * max_runtime_minutes * max_attempts / 60)
per row,
with an upper bound on the total. This is a hypothetical maximum based on
operator-supplied prices, NOT a live Salad quote or a verified authorization.

Even a report claiming owner budget and external auth stays HOLD_NO_COMPUTE,
training_authorized=false, provider_writes_authorized=false. The CLI always
returns nonzero, never starts Salad, never refreshes tokens, never schedules
training or changes protected branches. Report rows are indexed and do not
echo model names, credentials or local source paths.

## CLI

Run from the reviewed szl-gpu-bridge checkout:

    python tools/salad/training_wave_preflight.py local_dry_run_manifest.json held_report.json

Keep any input manifest in a private workstation directory. The report is
an offline redacted planning artifact. A structural candidate is not a signed
receipt, evaluation, authorized lease, entitlement or deployment proof.

## Separate future admission gates

1. Timestamped public + permitted HF catalog and immutable revision for each
   model, source code and artifact contract. Unreadable private assets UNKNOWN.
2. Independently established model/dataset license and derivative rights.
3. Exact base/dataset/recipe SHA, signed canary and evaluation receipts
   verified against pinned keys; schema field presence alone never qualifies.
4. A single bounded 20-step GPU smoke under a finite owner budget after
   Salad 401 repair, using a tested pinned torch/CUDA image and live restore.
5. Signed owner/contract-author admission by the EXISTING single-writer Salad
   controller, not this preflight. Explicit abort, outbox, no duplicate runners.
6. Post-run immutable artifact digest, held-out evaluation and rollback witness
   before changing model card or production product readiness.

The szl-khipu-scholar Unsloth next-lane spec is NOT SCHEDULED until prior
4B signed completion, scholar evaluations and the 20-step smoke are proved.

## References and boundaries

- Unsloth GPU/VRAM minimum reference:
  https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
  (not SZL-measured throughput).
- SkyPilot checkpoint/recovery:
  https://docs.skypilot.ai/en/latest/examples/managed-jobs.html
  (automatic restart is not equivalent to verified checkpoint restore).
- Existing local dispatch hardening: docs/DISPATCH_HARDENING_20261115.md
  (prebuilt image, pinned package lock, boot canary, bounded retries).

This task did not establish trained status of ~401 Hub models. A CDP readback
of the existing Edge Cloudflare tab displayed sign-in, not monetization
analytics. Snowflake tab readback was unavailable. Do not report either
provider fully wired from those observations.

## Revision identity versus artifact checksum ? 2026-11-20 follow-up

**Source-only compatibility correction:** GitHub and Hugging Face repository
revisions read through the current Git SHA-1 APIs are immutable **40-character
lowercase hexadecimal commit IDs**. A SHA-256 of recipe bytes, catalog bytes,
or a separate canary/evaluation receipt is a **64-character hexadecimal
checksum**. These are different identity domains, and neither establishes a
signature or independent training/evaluation success.

The original preflight used a 64-hex \`SHA\` validator for
\`base_revision\`, \`dataset_revision\` and \`trainer_commit_sha\`. It rejected
actual pinned Git commit IDs and syntactically accepted arbitrary SHA-256
strings as if they named a repository revision. The distinct
\`_git_revision\` validator now requires 40-hex syntax for those fields while
\`_sha\` still requires 64-hex for content and receipt fingerprints.
The exact source commit must be independently resolved at the correct
repository before a real execution gate could be considered.

This is **not an authorization improvement** by itself; all preflight
reports remain \`HOLD_NO_COMPUTE\`, including externally reported valid
credentials or budget. Input rights, dataset license, canary evaluation
and receipt digest fields remain caller-reported and unsigned.

Kimi K3.1 Pro and Mistral Large 4 separately identified this mismatch in
public-source advisory reviews via local Ollama. The third-party
\`treyleo16/gpt-6-astra:latest\` alias also noted it; its suggested
single \`40|64\` regex was **not adopted** because it conflates identity
types. Neither alias proves access to proprietary OpenAI/Mistral/Moonshot
training internals. Several extra peer claims about missing global gates
were rejected after inspecting the *full* \`evaluate\` body rather than
the bounded excerpt the models saw.

Published techniques worth independently evaluating rather than copying:
[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) for LoRA,
assistant-only loss and validated TRL version compatibility;
[NVIDIA NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/get-started/about)
for configuration-bound training recipes and distributed checkpointing;
and [PyTorch torchtune](https://github.com/meta-pytorch/torchtune)
for source-visible LoRA/DPO recipes. Before a live experiment, require a
fixed held-out evaluation, legal data rights, exact image/dependency
resolution, checkpoint/resume proof, provider auth and owner-controlled
cost cap. Do not launch 401 paid model jobs from a namespace census.
