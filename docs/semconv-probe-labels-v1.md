# Semconv probe labels (proposed)

**Status: PROPOSED DRAFT (PROPOSE_ONLY; authorizes nothing until owner/council ratification).**

Landed as an additive DRAFT document under the W26L landing-triage blocker queue conversion (row D3, landing_priority 4).

## Provenance and landing truth
- Queue record: mesh/2026-11-23/w26l-landing-triage/w26l-blocker-queue.json
- Draft source (PROPOSE_ONLY): mesh/2026-11-23/w26g-cohort6-drafts/w26g-proposal-drafts.json
- Blocker clearance: SIGNATURE_MINT cleared 2026-11-23: owner signature mint landed (W26C LEG A complete + W28X corrected-head verified commits; key id 1356467, fp SHA256:/w0Rx9...; going-forward SSH signing live on the owner device)
- Fresh Kimi GO at landing: nonce W26M-D3-144832 (receipt published in wave-logs mesh/2026-11-23/w26m-queue-conversion/)
- Landed 2026-11-23T14:48:35Z. Landed by delegated estate automation under the owner standing green-light directive; NOT independent human review.
- Target adjudication: szl-gpu-bridge is the card-declared canonical source for szl-dream-architect (hf-spaces path).

## Source citation
OTel GenAI semconv, W26F D3

## Proposed change
versioned semconv probe labels, namespaced estate attributes where no standard fits, mappings for GH issues/arXiv abstentions/compute-plane probes

## Proposed labels (additive)
- versioned OpenTelemetry GenAI semantic-convention attributes where standards exist; the exact convention version is recorded with each label
- namespaced estate attributes (szl.* prefix) where no standard fits
- probe classes: gh_issue_probe (public GitHub API), arxiv_abstention_probe (titles only, no fetch implied), compute_plane_probe

## Integrity rule
- probe metadata never implies successful retrieval, computation, or a passing security result
- an UNAVAILABLE probe remains UNAVAILABLE in every derivative view

## Acceptance gate
probe metadata never implies successful retrieval, computation, or passing security result

## Rollback
doc removal only

## Baseline
szl-dream-architect baseline unchanged; this proposal does not modify it.
