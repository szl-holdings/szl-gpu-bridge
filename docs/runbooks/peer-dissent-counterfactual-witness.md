# SZL peer-dissent counterfactual witness — Kimi × Mistral

**Source-only research. No runtime authority. No model execution, network, tokens, GPU, workflows or deployment.** Reviewed November 19–20, 2026. The tool is a companion to the existing frozen-family paired-evolution evaluator, not another training controller or reviewer approval process.

## Why two agent opinions are not a safety gate

Kimi and Mistral may disagree on code quality. A textual consensus is **not independent verification**, a cryptographic signature, proof of source identity, or authorization to execute. This verifier treats both model identities and all five counterfactual outcomes as **caller-supplied untrusted declarations**.

Research references:
- [Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents](https://arxiv.org/abs/2509.23045): structured code repair training and agent adaptation; published benchmark results are not SZL reproductions.
- [Mistral Devstral 2 and Vibe CLI](https://mistral.ai/news/devstral-2-vibe-cli/): coding agents and tool-supported workflows; no copying of provider code.
- [SecureAgentBench](https://arxiv.org/abs/2509.22097): functional correctness alone is not sufficient for secure agent coding.
- [Mistral Small 4](https://mistral.ai/news/mistral-small-4/) Apache-2.0 and [Mistral model life cycle](https://docs.mistral.ai/models/devstral-small-2-25-12): older Devstral Small 2 is deprecated; version names are not interchangeable.
- [SLSA provenance](https://slsa.dev/spec/v1.2/build-provenance) and [in-toto](https://in-toto.io/): proper authenticated provenance belongs outside this unsigned JSON classifier.

## Machine contract

Programmatic import: tools/proving-grounds/peer_dissent_contract.py exports assess(raw_bytes, now=aware_datetime).

Input schema: szl.peer-dissent-request/v1. Accept only szl-holdings/szl-gpu-bridge and syntactically pinned source, patch, evaluation suite and counterexample suite identities.

Exactly two reviewers, with proposer and adversary roles and distinct Kimi and Mistral declared model families. Each declares the exact same source and patch, different prompt and response hashes, review/hold verdict and highest risk. The hash inequalities test syntax, **not genuine independence**. No model prompts, responses or secrets are included.

Exactly five required counterfactual fixtures, each with distinct hashes, expected=HOLD and observed=HOLD:
1. Missing signer
2. Source revision mismatch
3. Unauthorized action
4. Stale evidence
5. Unknown provider response

Missing, duplicated or passing broken controls, stale times, mixed source, unknown reviewer identities, malformed JSON, injected extra authority fields, reviewer disagreement, or medium-or-higher reported risk all HOLD.

Even when all declarations satisfy the contract, the maximum state is REVIEW_CANDIDATE_ONLY. Source authenticity, physical model identity, reviewer independence, test execution, file bytes, counterexample execution, signer trust, code quality, model novelty, merge authority, cloud authority and provider effects all remain explicitly **false**. A SHA-256 digest is not an attestation.

The diagnostic dissent_count is the number of reviewers not declaring REVIEW; the fixed negative-control predicate is the conjunction of five declared HOLD outcomes. This is not a statistically powered improvement estimate. The existing [paired frozen-family evaluator](https://github.com/szl-holdings/szl-gpu-bridge/blob/main/tools/proving-grounds/paired_evolution_gate.py) handles research-only paired Hoeffding-Bonferroni analysis and its own broken-control check. Their noncompensating holds must not be averaged into an optimistic score.

SZL's [PURIQ and Yuyay prior art](https://github.com/szl-holdings/szl-papers/blob/main/prior-art/README.md) describes conjunctive admission. The [Λ v23 outline](https://github.com/szl-holdings/szl-papers/blob/main/thesis/ouroboros/papers/v23/outline.md) preserves conditional, not unconditional, uniqueness.

## Current model availability and test scope

During this pass, Kimi K3.1 Pro was reachable through Ollama. The local Ministral 3B model was listed but an inference request timed out during workstation GPU/RAM contention. Devstral Small 2 cloud returned HTTP 410. **This PR does not claim that a real Mistral peer review ran or that two models agreed.** Unit-test reviewers are synthetic.

Run offline: python -m unittest discover -s tests -p test_peer_dissent_contract.py -v

The stdlib tests cover all five controls, impostor reviewer families, source mismatches, stale and duplicate JSON, injected authority fields, digest replay, risk overrides, and deterministic malformed-input fuzzing. The source has no executable model or provider adapter and cannot request an autonomous action.

## Next admission gate

Verify actual response bytes from both independently invoked models, real negative-control test execution with fixed artifacts, distinct prompts, immutable tested source and patch, current owner/security approval, signed provenance, protected GitHub CI, and later deployment readback before attempting to integrate this research into a working agent. Two models' unverified votes can never authorize a GitHub merge or GPU transaction.
