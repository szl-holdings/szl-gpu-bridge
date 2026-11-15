# The Self-Proving Estate Loop — full-stack wiring (2026-11-15)

**Owner directive:** "wire it all up make it cutting edge we're building the
future things no one has dreamed of use all math and evolve it push the
frontier then leverage and harness it through our stack and anatomy and
brain."

This is the wiring map: how the estate's math core, frontier training lane,
battery discipline, brain, and anatomy connect into ONE loop that no one else
has — a governed organization whose formal corpus generates its own
proof-discharge curriculum, whose frontier models attempt that curriculum
against the corpus's own pinned kernel, whose verification is mechanical and
fail-closed, whose promotion is human-gated, and whose every step is
receipted. The corpus evolves the models; the models evolve the corpus.

## The loop (what connects to what)

```
        +---------------------+   export_sorry_challenges.py    +--------------------------------+
        | lutar-lean (math)   | ------------------------------> | sorry-discharge-challenges-v1  |
        | locked-8 kernel     |                                  | (HF, owner-gated first publish |
        | sorry-gate 67 -> 0  |                                  |  after export review)          |
        +---------^-----------+                                  +----------------+---------------+
                  |                                                               |
     promotion    |                                        attempts (PROOF/ABSTAIN)  |
     (PhD gate,   |                                                               |
      receipts)   |                     +-------------------------------------------+
                  |                     |
        +---------+-----------+         v
        | proposals/ mirror   |   +------------------------------------------+
        | + khipu ledger      |   | FRONTIER LANE                            |
        | receipts (append-   |   | nemo-r8-v4 (VM1, training NOW) -> v5 ...  |
        |  only, under the    |   | lean-proofs SFT corpus regen after each   |
        |  szl-lake-ledger    |   | promotion; brain-navigator orchestrates  |
        |  lock)              |   | attempt budget; ABSTAIN is first-class    |
        +---------------------+   +------------------------------------------+
```

## The five rails (existing estate assets, now wired)

1. **Math rail — `szl-holdings/lutar-lean`.** Locked-8 proven set
   {F1,F4,F7,F11,F12,F18,F19,F22} @ kernel c7c0ba17; experimental scope with
   1304 decls; sorry-gate baseline 67, no-increase, target 0. PR #342 (open)
   adds the deterministic challenge exporter + the attempt contract
   (`docs/SORRY_DISCHARGE_LOOP.md`).
2. **Formula rail — `szl-formulas` + `thesis-formula-index`.** The kernel-hub
   publish pattern (torch-cpu/universal builds, source bindings, weekly
   re-verify) is the template for shipping every new verified artifact the loop
   produces. Discharged theorems become kernel carriers exactly the way the
   locked-8 reference ships today.
3. **Frontier rail — the Nemotron lane.** `nemo-r8-v4-20261115` is TRAINING on
   VM1 now (LoRA r32/a64, 26,706-row leader-leverage mix, 5,001 steps,
   receipt-synced to `frontier-54-runs/receipts/v4-fullrun-20261115` every
   10 min — dispatch-hardening doctrine applied). The challenge set is its
   first eval: the model reads a challenge and answers PROOF or ABSTAIN.
4. **Honesty rail — the estate battery.** The abstention discipline the battery
   measures on every estate model applies verbatim to proof attempts: a
   hallucinated proof is a failed attempt (receipted, never hidden); an honest
   ABSTAIN is a valid, scored outcome. A frontier model that cannot abstain
   cannot be trusted with the corpus.
5. **Governance rail — anatomy + brain.** Anatomy (org keeper) surfaces the
   loop's artifacts like every other governed asset. Brain
   (brain-navigator-r2; `BrainBeliefUpdate.lean` already formalizes its belief
   update in the corpus) schedules and budgets attempts; its harvest/alarm
   invariants are already Lean artifacts, so the orchestrator is itself part
   of the corpus it orchestrates. THAT is the "no one has dreamed of" part:
   the scheduler of the loop is a theorem in the loop's own kernel.

## Attempt lifecycle (fail-closed at every edge)

1. Challenge read from `sorry-discharge-challenges-v1` (id-pinned, sha256 manifest).
2. Attempt generated (PROOF with candidate term/tactic block, or ABSTAIN with reason).
3. Verification: rebuild the REAL file at the manifest kernel_head on the pinned
   toolchain; `lake build` green on the touched module; axiom count does not
   increase; gated sorry count strictly decreases; no native_decide/unsafe
   escapes (red tests stay red).
4. Receipt: append to the khipu ledger under the existing
   `szl-lake-ledger/lutar-lean` lock BEFORE review — attempts and failures are
   history, not noise.
5. PhD gate (proposals/ flow): human review decides promotion.
6. Promotion: corpus PR; SORRIES.md scoreboard + sorry-gate baseline ratchet down
   in the same commit; discharged proof regenerates SFT material
   (lean-proofs corpus) for the next model generation.

## Standing rules (this loop inherits)

- Silence = alarm (dispatch-hardening doctrine, PR #177): a stalled attempt
  batch is investigated, never waited out.
- Losses recorded honestly with timestamps; receipts are append-only.
- GPU spend for attempt generation is owner-gated like all dispatches.
- No Hub write for the challenge dataset without an owner-gated dispatch
  (lutar-lean standing rule; PR #342 respects it with card SOURCE only).

## First concrete milestones

1. Merge PR #342 (exporter + contract) — owner-gated.
2. Run `python3 tools/export_sorry_challenges.py`, review output vs SORRIES.md.
3. Owner-gated first publication of `SZLHOLDINGS/sorry-discharge-challenges-v1`.
4. When nemo-r8-v4 finishes (~5001 steps): run the first discharge pass as an
   eval (PROOF/ABSTAIN per challenge, receipted); expect mostly ABSTAIN — the
   point is the honest baseline, not a miracle.
5. Putnam DEMO->REAL + the two Materials-roadmap sorries (PDDInjective,
   PACBayesMaterials) are the highest-value named targets on the same rails.

*Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr.*
