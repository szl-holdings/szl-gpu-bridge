# ABS-RL reward verifier v2 — independent shadow contract

**Status:** research candidate, source-only and **inactive**. This does not
replace tools/absrl/rewards.py and does not change the staged
tools/absrl/grpo_absention.py worker, any prompt, model, container or provider.
This note specifies an adversarially testable scoring boundary, not a claim
that model quality or training has improved.

## Motivation: an existing verifier exploit

The historical v1 reward core used a substring check on the entire generated
response. For gold "4", outputs such as "14", "not 4", and "the answer is 4"
could all receive +1. It also treated an absent or unknown answerability label
as false, potentially rewarding an unwarranted abstention. In GRPO, model
behavior is driven by the reward function; a permissive verifier can amplify
its own errors. These are **source-level demonstrated fixtures**, not an
observation of a completed GRPO training run.

Candidate module: tools/absrl/rewards_v2.py. The reviewed v1 code is untouched.
The only executable entry points are pure Python functions. There are no
network calls, file writes, subprocesses, training dispatches or secrets.

## Reward grammar and exact mathematics

Each candidate has an explicit answerability label l, a gold answer a, and a
model-authored completion c. Only a complete one-line answer or an entire
"FINAL: <answer>" line counts; no substring, trailing explanation, hidden
answer coercion, or multi-line fallback is accepted. "UNANSWERABLE" is a
reserved entire-token abstention, or "FINAL: UNANSWERABLE".

Let F(c) be the extracted terminal text, or missing. Then:

    r(c,a,l) =
        1.0  if l=true  and F(c)=a exactly
        0.5  if l=false and F(c)="UNANSWERABLE"
        0.0  otherwise

For label true, a must be a nonempty text gold. For label false, gold must
be absent or empty. Missing/unknown labels, contradictory rows, malformed
assistant messages, or misaligned columns raise an explicit ValueError
**before any member of that group is scored**. String comparison uses
NFC normalization and trimming only: no case folding, fuzzy matching, or
unreviewed mathematical-equivalence parser.

The candidate accepts plain TRL-style text completions, assistant message
dictionaries, and single-message assistant sequences. This **does not prove**
compatibility with every released TRL revision or every current SZL prompt.
A separate pinned, dependency-isolated, credential-free integration fixture
is mandatory before any owner considers enabling v2.

## Aggregate-only shadow diagnostics

The optional shadow_summary_v2 returns group/sample counts, number of correct
decisive answers, honest abstentions, over-abstentions and all-zero groups.
It also reports an **exact integer half-reward-unit** comparison against a
degenerate policy that always says "UNANSWERABLE":

    U_observed = 2*N_exact_correct + N_honest_abstain
    U_constant = N_unanswerable
    candidate_difference = U_observed - U_constant

This is only a conditional score diagnostic on caller-supplied labels. The
result explicitly sets training_authorized, trusted_signer_verified,
labels_independently_verified, prompt_format_verified,
dataset_rights_verified, and model_quality_verified to false. The result
does not expose raw completions, prompt contents or gold labels.

A better conditional score than the constant policy does **not** establish
generalization, formal proof correctness, independence, dataset rights,
source-of-label trust, model quality, deployment or compute admission.

## Negative controls and boundedness

The top-level test module tests substring/negation reward hacking, embedded
abstention markers, unknown labels, contradictory gold, forged message roles,
oversized text, Unicode confusables, mixed-validity batches, nonfinite labels,
zero-size samples, all-zero groups, group misalignment, and no-effectful source
imports. The historical staged worker remains pinned to v1; the test asserts
that this candidate has not accidentally been imported by it.

    python -B -m unittest discover -s tests -p 'test_absrl_reward_v2.py' -v

Limits: at most 2,048 completions per batch; at most 8,192 characters per
completion; at most 512 characters per gold answer. No arbitrary JSON,
external dataset read or hierarchical proof parsing is performed.

## Required steps before any training integration

1. Pin the exact TRL release/API and test real reward callback shapes
   against the current model tokenizer and actual prompt template.
2. Use owner-verified rights-cleared training labels and a separately
   source-bound held-out set with contamination checks.
3. Show a nonzero, nondegenerate, reproducible shadow score distribution and
   compare a constant-abstain control and incorrect-answer adversarial cases.
4. Obtain the independently authenticated dispatch owner, budget, signed
   job, image digest, checkpoint/restore proof and protection-gated code review.
5. Admit a separate prompt/worker integration only after these controls pass.

**Do not** launch an unchanged v1 reward trainer to generate a passing
baseline. Do not merge this candidate merely because its synthetic tests are
green. Its source is an original SZL adaptation informed by, not copied from:
[DeepSeek-R1 GRPO](https://doi.org/10.1038/s41586-025-09422-z),
[LG EXAONE Deep](https://github.com/LG-AI-EXAONE/EXAONE-Deep),
[TRL GRPOTrainer callback contract](https://huggingface.co/docs/trl/grpo_trainer),
[NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL), and
[EleutherAI evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).
No external model weights, papers, code files or datasets are vendored.
