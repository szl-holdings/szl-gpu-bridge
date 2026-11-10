# The Autonomy Ladder — 99.7% automated, 0.3% human, by law

**Directive:** the estate runs fully automated. The human-in-the-loop share
is 0.3% — and that number is not a hope, it is a *partition*: every action
is either reversible (automate, receipt, alarm) or irreversible (human gate,
by the machine-checked Kamachikuna invariant `law7_irreversible_needs_human`).

The 0.3% is not overhead. It is the exact set of decisions the constitution
refuses to delegate, and it is small *because* everything else was made safe
to automate first.

## Tier 0 — fully automated (the ~99.7%)

| Loop | Cadence | Automation | Alarm for silence |
|---|---|---|---|
| Receipt verification (khipu chains) | every action | SDK + CI | wiring witness (daily) |
| Red-main detection & issue filing | daily | main-health watch | workflow itself is watched by the other watch |
| Space liveness + receipts freshness | daily | wiring witness | expected_head + max_age_days |
| CI test → fix-branch → PR | per failure | chief-engineer pattern; PRs wait for the 0.3% |
| Fleet sweep: receipt-less leg = stopped = restart on A6000 | 15 min | Infra Warden | boot-receipt-first (runbook) |
| Secrets/chain integrity scan | hourly | cyber-shield | drift issue |
| License/README care | daily | gardener (append-only, content-preserving) | before/after sha256 |
| Energy measurement | every run | energy_meter (UNAVAILABLE, never zero) | sample counters |
| Dependency bumps with green CI | on PR | auto-merge when all checks green and scope is lockfile-only | main-health watch |
| Readback of every write | every write | byte-count + hash verify (roadmap lesson 8) | verify step fails closed |

**The two rules that make Tier 0 safe:** (1) every automated action is
reversible or fail-closed; (2) every automated thing has an alarm for its
own silence. A rule nobody checks is a rule that has stopped.

## Tier 1 — automated with human ratification (part of the 0.3%)

| Action | Why human | Ratification form |
|---|---|---|
| Merge to protected main | release = public claim | one click; CI must already be green |
| Production deploy / HF overwrite | source authority is GitHub; deploys are claims | one click after parity check |
| Credential creation/rotation | custody cannot be delegated | you in the dashboard; never in chat |
| Payment / x402 settlement | money is irreversible | canary sequence, then auto below a bound |
| New third-party dependency admission | policy decision (the openai 3.x case) | one decision, recorded |
| Model promotion (NOT_PROMOTABLE → PROMOTED) | capability claim about the world | receipt + your GO WEIGHT |

## Tier 2 — human only (the rest of the 0.3%)

- Kamachikuna amendments (the constitution is yours)
- Anything that touches a person's data, attribution, or their name (Law 8)
- Partnerships, sponsorships, legal commitments
- The Cornell-class life decisions

## Measuring the 0.3%

The number is checkable, not aspirational: over any 30-day window,
`human-gated actions / total receipted actions ≤ 0.003`. The receipts chain
makes the numerator and denominator machine-countable — the auditor counts
them hourly. When the ratio drifts above 0.3%, the automation has a gap;
file it like any other drift.

## The failure modes this design forbids

- **Confident paraphrase hardening into record** — anything in your name is
  quoted verbatim; anything adopted is labeled by how it was adopted.
- **A stale record that still verifies** — freshness is a separate axis
  (`max_age_days`): stale is not green.
- **Silent empty denominators** — a scan that read nothing and returned
  clean is the most common way automation misleads; assert and fail loudly.
- **Self-verification** — never verify with code that shares the suspect
  logic; the proving ground cross-checks.

*Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr. ·
Informed by the Vidal-Gadea Lab roadmap (Nov 2026), attributed.*
