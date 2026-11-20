# SZL Holdings — USD $2,000 maximum training budget (November 20, 2026)

**Owner-stated envelope:** up to USD $2,000 TOTAL may be spent on
the SZLHOLDINGS training campaign, inclusive of GPU instances, retries,
evaluation, checkpoint storage, networking/egress and other billable services.
This does not authorize a separate $2,000 per model, container, or cloud.

**Status: proposed budget envelope, source-only; NO paid dispatch.**
The existing `tools/salad/training_wave_preflight.py` accepts
`owner_budget_micro_usd`. The owner-reported ceiling is
`2_000_000_000` micro-USD. The offline preflight compares estimated
prices × replicas × minutes × attempts against the declared budget,
but its inputs are self-reported and **not** live quoted prices,
real spend meter, GPU access, licensed data, or verified authorization.
Even a syntactically complete plan remains `HOLD_NO_COMPUTE`.

**Do not equate the populations.** The previously observed 400-row
`stack400_manifest.json` contains external base-model candidates, not
400 SZLHOLDINGS-owned train destinations. A separate public HF census
returned 401 SZLHOLDINGS model repositories, also not proof of training.
For this specific program the CPU preflight now requires output `model_id`
and `checkpoint_target` under `SZLHOLDINGS/*`. External pretrained
models may remain named `base_model_id`; each requires separate exact
revision, license and permitted derivatives. Namespace syntax alone
never proves authority to write there.

**Actual next expensive action** can only be an owner-controlled
one-job canary *after* a trusted signer confirms the authorized
single-writer Salad/MassedComputing training lane, existing in-flight
group/instance ownership, its real current billing, image and model
digest, dataset rights, approved recipe, finite time/spend cap and
checkpoint/restore proof. Every additional wave needs separate
evidence and measured cost reserve. Do not spend the budget by creating
401 simultaneous replicas or treating a hypothetical $0.09/GPU-hour
lowest-tier price as universally obtainable.

Upstream reference: HF TRL SFTTrainer
https://huggingface.co/docs/trl/sft_trainer
(streaming `max_steps`, assistant/completion-only loss and PEFT);
Unsloth LoRA/QLoRA guide
https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
(architecture-specific parameter-efficient choices). No upstream
benchmark figures have been reproduced on SZL hardware here.
No model jobs, scheduler, provider credentials, branch rules,
HF/Cloudflare/PyPI publication or financial transactions are changed.
