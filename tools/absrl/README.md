# Abs-RL lane (GRPO abstention training) - STAGED, owner-gated launch

The estate play (leader-scan 2026-11-15): honest abstention as a **trained**
behavior, scored on AbstentionBench (facebook/AbstentionBench, CC-BY-NC-4.0,
eval use), priced per-joule via the worker's MEASURED/zeus receipts. This
directory stages the lane. **Launching it on Salad is a GPU-cost decision that
belongs to the owner** - nothing here dispatches, trains, or publishes by
existing.

## Files

- `rewards.py` - pure reward core, importable with no torch/trl/GPU (test surface).
- `grpo_absention.py` - the staged worker (receipts to the chain, fail-closed).
- `tests/test_rewards.py` - reward semantics tests.

## Reward contract

| behavior | reward |
|---|---|
| correct decisive answer | +1.0 |
| honest abstention on an unanswerable item | +0.5 |
| wrong decisive answer | 0.0 |
| over-abstention (abstains when answerable) | 0.0 |

The abstention marker is the literal token `UNANSWERABLE` in the completion.
Rewards are computed only from dataset fields (`prompt`, `answer`,
`answerable`) and completion text; misaligned columns raise - nothing is
invented.

## Dataset contract (fail-closed)

The dataset must expose `prompt`, `answer`, `answerable` columns; otherwise
the worker emits a fail-closed receipt and exits 5. Candidate sources:
AbstentionBench subsets converted to this schema (license: CC-BY-NC-4.0 -
eval use; check before any commercial use).

## Image requirement (owner decision)

The salad image pins trl==0.13.0; GRPOTrainer needs trl>=0.14. The launch
decision includes building a dedicated absrl image. A missing GRPOTrainer
produces an honest `unavailable` receipt - never a fabricated run.

## Honesty notes

- Energy on this lane is UNMEASURED so far; the staged worker reports that
  instead of fabricating joules (the SFT lane carries the zeus window).
- Conjecture 1 remains open.
