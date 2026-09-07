# Training candidates — 2026-09-07 (PROPOSAL, unsigned)

This is a proposal document, not a jobspec. Real jobspecs are DSSE-signed by the
owner and enter via `jobspecs/` under the bridge schema. Nothing here executes,
schedules, or provisions hardware.

## Owner request

"Get the training done for all the models that have not have the weights — make
them fully operational." This document sorts the weightless estate into what CAN
honestly be trained, what is doctrine-locked against it, and what each honest
training run would require.

## Trainable candidates (curriculum exists, no doctrine bar)

1. **SZLHOLDINGS/SZL-Khipu-1.5B-abstain** — curriculum-only abstain artifact.
   The curriculum is committed; the abstain label says "research-only, abstain-retrain".
   An honest retrain = QLoRA on the declared base (Qwen2.5-1.5B-Instruct) over the
   pinned curriculum, with training + held-out eval receipts in the ReceiptAgent
   pattern (owner-signed ed25519, sha256-pinned datasets, hash-chained).
   If trained, the abstain page remains as lineage; the new weights get a NEW repo
   or a clearly versioned successor — never overwrite the abstain receipt.

## Doctrine-locked (do NOT train under these identities)

- **szl-nemo** — declared SOFTWARE/SURROGATE, "not a checkpoint", joblib
  quarantined (sha256 d3f0cd7b…, "do not invent it"). Its card (#167) now carries
  the NOT-A-LOADABLE plate. Training a checkpoint under the nemo name would
  contradict its own card.
- **TinyKhipu-Nano, ReceiptAgent-Nano, Moons-Nano, MiniEmbed-Nano** — synthetic
  test fixtures (needs-loader). They exist to be small and known, not capable.
- **KILLINCHU-EYE, qantu, waman, chakana, tinku** — roadmap stubs. No curriculum
  exists; "training" them would mean fabricating a purpose first.
- **governed-inference-meter** — deprecated/superseded; szl-energy-attest is canonical.

## Already-trained line (do not retrain without a receipted reason)

SZL-Forge-1.5B-ReceiptAgent (loss 0.1038, 5/5 + 6/6 signed), SZL-Khipu-1.5B,
chaski, KHIPU-R2, WILLAY, khipu-r3, brain-navigator-r2, chaski-r2, chaski-5050,
szl-receiptagent-qwen35-0.8b-v2/v3 — all carry weights or adapters on the Hub.

## Bench side

frontier-bench / szl-engine-bench / szl-retrieval-bench / szl-quant-bench report
BLOCKED without measured hardware — that is the honesty contract working, not a
defect. Any bench numbers must come from a measured run on owner metal via this
bridge, never from this document.

Λ = Conjecture 1 OPEN. Energy UNAVAILABLE until measured on the run itself.
