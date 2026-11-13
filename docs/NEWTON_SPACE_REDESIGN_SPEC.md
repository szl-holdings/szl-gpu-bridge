# Newton Trust Upgrade Space — redesign spec (dev-ready)

Scope: restyle the live HF Space landing surface (demo-first, receipts-first).
No backend changes; no new claims. Deploy requires an authorized HF token —
this spec is the implementation handoff, not a deployment.

## Page structure (top to bottom)
1. **Hero** — above the fold, no ambiguity.
   - Title: `Newton Trust Upgrade`
   - Subline: `Receipts-first trust infrastructure for the SZL ecosystem`
   - CTAs: `Launch Demo` · `View Receipts` · `Explore Ecosystem`
   - Trust chips: live Space count, latest receipt digest (short form), A11oy revision.
2. **What it is** — three pillars: trust · proofs · ecosystem execution.
3. **Live demo** — embedded panel or interactive mock terminal. Minimum flows:
   run a trust check · generate/inspect a receipt · open linked ecosystem module ·
   see proof trail · inspect result state. Never text-and-screenshots only.
4. **Receipts** — dedicated section, operational not decorative. Cards show:
   title, timestamp, status, short hash, scope, `View receipt` button.
   Types: latest verification receipt, system integrity receipt, demo execution
   receipt, artifact provenance receipt.
5. **Ecosystem** — holo card grid: A11oy (execution fabric), Killinchu
   (evidence/corpus layer), Hatun (second-brain/anatomy surface), Newton Trust
   Upgrade (trust/receipts layer).
6. **Why it matters** — developer / operator / investor copy, short.
7. **Footer trust rail** — provenance links, org identity, contact.

## Visual direction
Matte dark base, thin glass panels, restrained holo gradients (cyan/violet/blue),
fine borders, generous spacing, sharp typography, motion used sparingly.
Avoid: noisy gradients, crypto-bro neon, overstuffed pages, giant paragraphs.

## Responsive
- Desktop: split hero (message/CTAs left, live holo panel right).
- Mobile: stacked, CTAs first, Demo and Receipts buttons always visible early,
  ecosystem cards vertical stack. No tiny text.

## Copy tone
Confident, precise, technical, proof-oriented, non-hypey.

## Acceptance checks (must pass before deploy)
- All CTAs visible above the fold on desktop AND mobile viewport.
- Receipts section renders real signed receipts (verify signature client-side;
  never display an unverified signature as valid — Law 6/7).
- Stale-data handling: refresh failure must show explicit STALE/UNKNOWN state,
  never silently keep old status (Law 9).
- No revenue or paid-production claims (Nov 8 hold remains in force).
- 0 console errors; Lighthouse accessibility pass on mobile.
