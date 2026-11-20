# Kimi × Mistral × GPT-6 — ABS-RL verifier-gap shadow (candidate)

**SOURCE-ONLY / INACTIVE / NO TRAINING AUTHORITY.** This is an offline, opt-in
research gate in the existing szl-gpu-bridge source owner. It does not create a
training controller, change the staged GRPO worker, install a reward callback,
launch GPU compute, publish a model or mint signatures.

The staged GRPO worker still imports tools/absrl/rewards.py (v1).
The reviewed tools/absrl/rewards_v2.py is separately **inactive**.
Do not automatically replace either with this module.

## Why governance belongs in training

A model trained against an imperfect verifier may optimize that verifier's
mistakes rather than the intended task. Existing ABS-RL v1 uses an
over-permissive substring match: the gold "4" can reward "14", "not 4", and
"the answer is 4". An unknown answerability label can be treated as false.
The inactive v2 fixes terminal-answer grammar but still relies on
caller-declared labels. This new shadow targets a **different** problem:
measuring *claimed* oracle/proxy reward disagreement on a frozen case
population before considering any learning run.

## Original discrete rule

A typed frozen plan identifies exact source/dataset/oracle/proxy Git revisions,
SHA-256 case handles, answerable vs unanswerable strata, and distinct declared
oracle/proxy owner digests. A caller also supplies the expected SHA-256 of the
entire canonical plan. The actual observations must contain every frozen case
exactly once, with exactly matching source/dataset/oracle/proxy pins.

For each observed case i:

- The declared oracle outcome is CORRECT, INCORRECT, UNKNOWN, or MISSING.
- The declared proxy reward is an exact integer number of half-units
  r_i in {0, 1, 2}, or null only for UNKNOWN/MISSING.
- If oracle INCORRECT and r_i > 0, optimistic proxy credit is recorded;
  the entire input is HOLD. This is a **hard gate**, not a mean-score penalty.
- If oracle CORRECT in the answerable stratum, the expected reward is 2
  half-units (1.0). In the unanswerable stratum it is 1 half-unit (0.5).
- A missing or unknown oracle outcome **never becomes false, zero or pass**.
  A zero attempted denominator or absent negative-control population is HOLD.

The result contains only bounded aggregate counters: frozen case total,
attempted, missing, unknown, oracle-correct, oracle-incorrect,
optimistic-credit violations, and reward mismatches. No prompts, gold answers,
raw completions or private case IDs appear in the report.

Even a source-consistent result is labeled
SOURCE_SHADOW_CONSISTENT_NOT_ATTESTED, and every authority flag is false:
training, model quality, signing, independent oracle, preregistration,
dataset rights, release and provider writes.

### Limits that must not be hidden

The plan digest is **not a signature or secure preregistration timestamp**.
Caller-declared distinct owner digests do not authenticate separate humans,
signers or evaluators. Two colluding records can forge self-consistency. The
8-case minimum is a basic shape constraint, not a statistically powered
reward-hacking study. A green offline test is not evidence of model faithfulness,
current TRL compatibility or protection against unobserved reward exploits.

## External research, separate from SZL measurements

- [DeepSeek-R1 GRPO, Nature 2025](https://doi.org/10.1038/s41586-025-09422-z):
  verifiable/reward-based RL and documented model-based reward hacking.
- [Optimistic Verifiers Get Hacked, arXiv 2611.21538](https://arxiv.org/abs/2611.21538):
  proxy/oracle gap measurement and the risk of optimistic process checks.
  Its external results are **not** SZL benchmark measurements.
- [AAMAS 2026: Preventing Process Reward Model Hacking](https://doi.org/10.65109/BWFN6920):
  dense process rewards can incentivize plausible but incorrect reasoning.
- [Hugging Face TRL](https://huggingface.co/docs/trl/index):
  evolving, distinct SFT, DPO and GRPO trainer contracts; a real pinned
  callback compatibility fixture remains mandatory.
- [Hugging Face model card guidance](https://huggingface.co/docs/hub/model-cards):
  measured evaluations need pinned dataset, metric, intended use and limits.

This source module is an original SZL **proposed diagnostic**; no author source
code, proprietary weights, training data or artifact were copied into SZL.

## Admitted sequence before a real GRPO trainer could use it

1. **Custody and rights:** Existing owners independently freeze a much larger
   license-cleared set of test cases, both source revisions, test population,
   verifier provenance, and external expected plan digest **before** any rollout.
2. **Negative-control sensitivity:** Independently execute an exact pinned
   oracle/proxy comparison, include wrong answers, abstention traps, varied
   strata, counterexamples, unseen source/data and complete attempted/unknown
   denominators. Repeat on a separate lineage-disjoint test set.
3. **Operational admission:** Pin actual TRL and transformer APIs, training
   source/image, budget, authenticated signer/owner, checkpoint/resume state,
   provider controller and rollback. Require owner PR and independent review.
4. **Training and post-training evidence:** Only the existing authorized
   dispatcher may start a bounded canary; independently verify final weight
   hashes, resume state, held-out results, serving, and model-card provenance.
   A model on the Hub is not necessarily trained by this run or qualified.

The work here implements **only** the offline first-layer source consistency
check; it does not wire the gate into a live dispatcher. It does not claim
that every model in SZLHOLDINGS needs retraining. No live training occurred.

## Source-only tests

Run without network or ML libraries:

    python -B -m unittest discover -s tests -p test_verifier_gap_shadow.py -v

The negative controls cover optimistic credit on oracle-wrong outputs,
half-credit exploits, missing and unknown labels, zero attempted denominator,
stratum reward mismatch, no incorrect-oracle control, omitted/duplicate/swapped
cases, pin drift, owner collision, forged extra authority fields and numeric
type confusion. The staged worker remains unchanged.
