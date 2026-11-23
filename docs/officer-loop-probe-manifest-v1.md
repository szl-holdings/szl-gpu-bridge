# Shared officer-loop probe manifest (proposed)

**Status: PROPOSED DRAFT (PROPOSE_ONLY; authorizes nothing until owner/council ratification).**

Landed as an additive DRAFT document under the W26L landing-triage blocker queue conversion (row D5, landing_priority 3).

## Provenance and landing truth
- Queue record: mesh/2026-11-23/w26l-landing-triage/w26l-blocker-queue.json
- Draft source (PROPOSE_ONLY): mesh/2026-11-23/w26g-cohort6-drafts/w26g-proposal-drafts.json
- Blocker clearance: SIGNATURE_MINT cleared 2026-11-23: owner signature mint landed (W26C LEG A complete + W28X corrected-head verified commits; key id 1356467, fp SHA256:/w0Rx9...; going-forward SSH signing live on the owner device)
- Fresh Kimi GO at landing: nonce W26M-D5-044634 (receipt published in wave-logs mesh/2026-11-23/w26m-queue-conversion/)
- Landed 2026-11-23T04:46:37Z. Landed by delegated estate automation under the owner standing green-light directive; NOT independent human review.
- Target adjudication: szl-gpu-bridge is the card-declared canonical source for szl-chief-engineer and szl-cyber-shield (hf-spaces paths); szl-infra-warden card declares only the org root, disclosed honestly; the manifest is shared so one additive doc covers the trio.

## Source citation
W26F D5; sleeping-tier card reads

## Proposed change
shared officer-loop probe manifest: per-space cadence, target, auth mode, timeout, stale threshold, UNAVAILABLE and not-RUNNING semantics, fail-closed outcomes

## Proposed manifest (additive)
Applies to the SLEEPING-tier officer spaces: szl-chief-engineer, szl-infra-warden, szl-cyber-shield.
- cadence: 30 min (chief-engineer), 15 min (infra-warden), 60 min (cyber-shield)
- targets: tier-0 latest CI conclusions via public API (chief-engineer); runtime-stage and DNS checks (infra-warden); public posture surfaces (cyber-shield)
- auth_mode: unauthenticated public reads only; UNAUTHENTICATED NEVER EQUALS AUTHORIZATION
- timeout: PROPOSED 10 s per probe
- stale_threshold: PROPOSED 2 x cadence without a recorded outcome = STALE
- semantics: UNAVAILABLE (reachable, no answer); not-RUNNING (objectively not running); FAIL (probe failed)

## Fail-closed outcomes
- every honest outcome is recorded visibly; no outcome converts a SLEEPING space to RUNNING
- probe failures are visible, never swallowed, never retried to green

## Acceptance gate
no sleeping-to-RUNNING conversion; unauthenticated never equals authorization; probe failures visible

## Rollback
doc removal only

## Baseline
sleeping-tier baselines unchanged; this proposal does not modify it.
