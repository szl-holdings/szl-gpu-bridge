# MUHU v1 → SZL flagship: next-cycle training admission review (HOLD)

**Observed:** 2026-11-17 UTC. **Stage:** SOURCE-ONLY REVIEW; **admitted rows: 0**.
**Owner:** SZL Holdings; canonical training dispatch remains under the existing
`szl-gpu-bridge` + A11oy signed-job contracts. This file is NOT a job spec,
a queue entry, an approval, a source merge, or evidence that training occurred.

## What was independently checked

The user-owned original synthetic MUHU release was replayed offline with
`python -B -m dataset.verify --root dataset --replay` (48 tests passed,
dataset replay reported 1,220 original synthetic rows). The release is
unpublished on Hugging Face as of the latest attempted read; there is no
immutable Hub revision for training admission. Its source-only local ZIP
SHA-256 is `13788fefd549b06d2fc93eb0901d9a8e61ccd045492c0fb5359d69ffde1f19aa`.
This hash checks a local archive, **not** the identity of a provider object.

| Split | Total | Mini Sudoku | Synthetic wave | Research gate |
| --- | ---: | ---: | ---: | ---: |
| Train | 842 | 156 | 366 | 320 |
| Validation | 176 | 45 | 66 | 65 |
| Test | 202 | 39 | 48 | 115 |

Split file SHA-256 values:

- Train: `6f1e47e974fb574e01a935433e044c28a8d798eb38398410ec90477015f8b817`
- Validation: `fb0a55a5720c1aee5c0d362a64f17ad1c6e41fbb507d504d2db29da473d4a874`
- Test: `f047a8427321ce4805c7e72caf681275906e1ca2c6b8d40c0f9d7599e16679b9`

**Rights:** 1,220 rows are declared original synthetic CC-BY-4.0. The
73 literature references are **citation-only and training-ineligible**.
Sara's source puzzles, Vidal-Gadea lab recordings, private manuscripts,
patient genomes, genomic sequences and therapy-design data are not present.
No collaborator endorsement or genetic/clinical ability is inferred.

## New blocker revealed by the independent candidate audit

Group splitting is not enough here. For `research_gate`, the training
partition contains **320 rows but only 40 distinct model-facing input
patterns**. Its validation partition has 33 distinct input patterns,
**all 33 already in training**; its test partition has 38 distinct
input patterns, **all 38 already in training**. Every validation/test
research-gate row repeats an exact train input. The 65 validation and
115 test rows thus do not provide an unseen-input generalization test
for that task, despite different simulated group IDs.

The wave task's `input` descriptor is constant across all sessions;
the model consumes numerical `features`, which vary. It cannot be
treated as genuine new-experiment generalization without a separately
generated and independently reviewed distribution-shift evaluation.

The local small classifier's harder synthetic tests report:
`mini_sudoku_status` 60/120 (50%), and `wave_phase_structure`
32/50 (64%), with abstention counted as nonpass. These are **synthetic
stress results**, not evidence for human genetics, wet-lab behavior,
therapy efficacy, or flagship improvement.

## Recommended research progression (not an executable directive)

1. **Do not append the 1,220 JSON rows to an active Flagship54 trainer.**
   That run is governed by a frozen command/dataset/target; changing the
   curriculum would break its immutable training and shutdown evidence.
2. Prepare a new separate **research-methods instruction SFT**
   curriculum with a genuinely independent evaluation source. Reserve
   all validation/test rows; require example-level, semantic and
   source-group leakage checks; do not merely relabel existing rows.
3. Keep the original certified reasoning-puzzle data a separate,
   explicitly attributed math lane. Test newly generated puzzle banks
   with solver-grounded **exact-answer** scoring; do not borrow existing
   exposed benchmark scores as new holdout evidence.
4. Keep mathematical simulation and biological conclusions separate.
   A wave model must beat independent baselines under camera, noise,
   frame rate and individual/experiment holdouts before claiming a
   biological measurement improvement.
5. Once a rights-reviewed dataset has an immutable provider revision,
   propose a separately signed, budget-approved **next-cycle** Khipu
   research-adapter run. No new paid replicas, modifications to active
   commands, resume-state changes, model promotion or data publication
   are admitted by this document.
6. Pre-register an independently blinded evaluation including the
   existing abstention/confabulation battery, exact-solution checks,
   research-identifier invention, provenance staleness, empty
   denominators, performance regressions and cross-domain forgetting.
   Use a fresh baseline from the identical pinned base revision, report
   denominators and uncertainty, and keep qualification separate
   from model completion.
7. Require accepted source, immutable dataset+base SHA, data-rights
   manifest, training/runtime preflight, one dispatcher owner,
   independent checkpoint backups, review, and explicit human
   authorization before creating any executable signed job.

## Non-actions and next gate

No current SaladCloud project/group, GPU process, training target, job
queue, model, Hub dataset, signer, credential, public card, or runtime
was changed. BETTERWITHAGE was **OFFLINE** at this check, so live
Flagship54 training and its guard were **UNOBSERVED**, not healthy or
completed. The authenticated Hub connector has only read scopes.
The latest independently observed live state must be rechecked
through its approved owner and Gate-1 observer before future execution.

**Reopen this HOLD only** after a new source- and rights-bound curriculum
has a genuinely unseen, preregistered evaluation and the owner admits a
distinct next-cycle training generation. Do not interpret CI-green,
hashes, model filenames, or this documentation as training completion.
