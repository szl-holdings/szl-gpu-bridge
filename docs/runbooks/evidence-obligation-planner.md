# Bounded evidence-obligation advisor (offline research preview)

This is a deterministic, read-only information-acquisition advisor in the
existing training evidence tooling. It is not an action executor, workflow
engine, signer, model evaluator, clinical recommender or source-of-truth database.
It creates no network connection and changes no provider object. It does not
verify a caller-supplied authority-state declaration or a purported evidence
digest. Execution and promotion must continue through existing authoritative
runtime controls and the Lutar non-compensation gates.

The input contract szl.evidence-obligation-request/v1 declares an operation,
ALLOW/DENY/UNKNOWN operation state, integer budget, a finite set of obligations,
and a finite list of candidate observations. Every candidate says which missing
obligations it MAY INFORM, a nonnegative integer caller-supplied relative cost,
and its independently admitted READ_ONLY status. Informed is not satisfied.
Only independently eligible read candidates are considered; denied writes and
unknown permissions never enter the optimizer.

The algorithm solves a finite minimum-cost set cover by exact dynamic
programming, breaking ties by fewer reads and stable identifiers. It is
optimal ONLY over the submitted read candidates and modeled costs, provided
the finite state bound was not exhausted. Above 4,096 states it returns
ABSTAIN_SEARCH_BOUND with no asserted optimum. A zero cost is an input
assumption, not evidence of zero actual computational or energy cost.

All outcomes carry authority_granted=false, remote_effects_performed=0,
requirements_satisfied_by_advice=0, quality_or_promotion_qualified=false,
and signature=UNSIGNED. An empty obligation list says only that this input
does not request new observations, never that a model or workflow is ready.
Neither the script nor its CLI reads credentials or uses an external API.
Inputs and outputs are local JSON. The CLI refuses to overwrite its result.

Adversarial tests cover denial and missing authority, stale evidence,
duplicate/malformed JSON, missing proof identifiers, unknown target IDs, cost
type confusion, forbidden writes, insufficient budget, tie breaking, search
bounds and refusal to replace existing output.

This source preview may inform a future Hatun / Anatomy read projection, but
connecting it is a separate reviewed contract change. No newly created route,
agent, scheduler or display is claimed. Candidate lists must be populated by
an admitted source owner before any deployed recommendation is displayed.
