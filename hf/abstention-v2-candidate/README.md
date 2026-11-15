---
license: apache-2.0
library_name: peft
tags:
- lora
- experimental
- research-only
- szl-estate
---

🜂 **SZL Holdings estate artifact** — *the estate that receipts its own thinking.*


# abstention-v2-candidate

ὰ2

![Receipt visualization](receipts/receipt_visual.svg)

Registry artifact of the SZLHOLDINGS estate. This card was rewritten on 2026-11-14 to replace the
wave-4 uniform stamped card: every claim below is bound to a receipt stored in this repository, and
everything unverified is stated as not measured. This is an experimental research artifact, not a
qualified or production model; the prior stamped card remains in revision history.

## What this artifact is
- LoRA adapter candidate for base model Qwen/Qwen3-1.7B.
- Adapter configuration observed in this repository: rank 32, alpha 64, dropout 0.05.

## Measured status (receipt-bound)
- Evaluation battery receipt [receipts/eval_battery.json](./receipts/eval_battery.json) in this repository,
  sha256 `45a24761e66438086b0c362d347fbc8d2712a11af38fc8eacf4f96353782c208`, read at observed revision e2db375b26ac.
- Receipt fields, verbatim:
  - kind: "szl.estate-battery/v2"
  - model: "SZLHOLDINGS/abstention-v2-candidate/final"
  - base: "Qwen/Qwen3-1.7B"
  - score: "3/6"
  - state: "MEASURED"
  - curriculum: "SZLHOLDINGS/abstention-curriculum-v2 (70% fabrication-prone unanswerable + 30% answerable, 2000 rows)"
  - question: "does the v2 curriculum move the 3/6 confabulation floor measured on v1 models?"
  - ts: "2026-11-11T15:27:10.276053+00:00"
  - gpu: "NVIDIA RTX A6000"
- The receipt carries 6 scored rows; read the receipt file for the per-row results, including failures.

## Not measured
- No general capability, benchmark, safety, or production-readiness claim is made for this artifact.
- No weight-equivalence, cross-run, or cross-model result inheritance is claimed.
- Any use outside the receipt's declared scope is unverified by this estate.

## Evidence binding (2026-11-14 rewrite)
- Observed Hub revision before this edit: `e2db375b26aca83c421ef464b04b4acc3cfb18b8`.
- Source receipt bytes sha256 `45a24761e66438086b0c362d347fbc8d2712a11af38fc8eacf4f96353782c208` (in-repo file, unchanged by this edit; this commit touches README.md only).
- No new measurement, evaluation, or weight change was performed by this edit.

Historical rewrite attribution remains in the prior revision; this correction creates no authorship or signature attestation.

## Evaluation scope and revision binding

A historical exploratory six-probe evaluation exists in this repository. The receipt
reports **3 PASS and 3 FAIL out of 6 probes**. The previous appended statement that
no measured evaluation exists was overbroad: a scan limited to another repository's
`receipts/` directory could not establish absence of this in-repository receipt.

- [Pinned original evaluation receipt](https://huggingface.co/SZLHOLDINGS/abstention-v2-candidate/resolve/c950364c41071e43646305974082cf694c9ffc0c/receipts/eval_battery.json).
- File-byte SHA-256: `45a24761e66438086b0c362d347fbc8d2712a11af38fc8eacf4f96353782c208`.
- Receipt timestamp: `2026-11-11T15:27:10.276053+00:00`.
- The receipt names the model path but does not bind an immutable adapter revision.
  Its stored scores must not be inherited by the current weights or another model.

This correction reconciles documentation with preserved evidence; it does not
rerun the battery, regrade its rows, establish weight equivalence, or qualify model
safety, general capability, deployment, or production readiness. The original receipt
and every failure remain unchanged.

## Readiness boundaries

| Gate | Evidence status |
|---|---|
| Training | Recorded checkpoint 200/200; not independent execution proof |
| Artifact integrity | Full weights not independently verified by this card correction |
| Evaluation | Historical six-probe receipt only; current-revision binding unverified |
| Deployed runtime | Not established by this evidence |
| Production qualification | Not established |

Training-state reference: [pinned trainer JSON](https://huggingface.co/SZLHOLDINGS/abstention-v2-candidate/resolve/c950364c41071e43646305974082cf694c9ffc0c/checkpoint-200/trainer_state.json), SHA-256
`e87f0ca9b54e0c6ce4e3003466aec65c73d265f3c6bdf1e4357d4371c1407c4a`.
No weights, receipt payloads, repository visibility, or evaluation outcomes change.
