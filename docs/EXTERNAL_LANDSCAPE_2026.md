# External landscape 2026 — Korea, EU/Ireland, NVIDIA — and where SZL is ahead

Field study of what the reference jurisdictions/vendors are doing, mapped
against the Kamachikuna. Sources are primary docs and regulator pages, not
marketing. Facts as of 2026-11.

## South Korea — AI Basic Act (in force 2026-01-22, MSIT)
- High-impact AI (employment, healthcare, finance, public safety, education):
  pre-deployment fundamental-rights impact assessments, risk management,
  explainability, human oversight, recordkeeping.
- Generative AI: mandatory Korean-language notices and AI-content labeling.
- "High-performance" AI at ≥10^26 cumulative training FLOPs: extra safety
  obligations. (~10x the EU GPAI threshold.)
- Foreign operators above thresholds must appoint a Korea-based local rep.
- 2026 grace period on fines (except serious harm); Help Desk via KOSA.
- Naver/HyperCLOVA X: HF model cards + technical reports; public "from
  scratch" dispute shows disclosure standards are the contested ground.

**SZL position:** ahead on receipts — Korea requires recordkeeping; khipu
already mints signed, hash-chained evidence per training leg, and the
10^26-FLOPs class maps directly onto compute-receipted training. Gap: no
formal impact-assessment artifact per high-impact deployment.

## EU / Ireland — AI Act transparency wave (Art. 50 applies 2026-08-02)
- Art. 50: machine-readable marking of synthetic content, deepfake
  disclosure; Code of Practice published June 2026 (~190 signatories).
- GPAI Code of Practice (July 2025) gives presumption of conformity.
- Digital Omnibus (agreed May 2026, in force 2026-07-27) deferred parts of
  the high-risk regime to 2027-12-02.
- Ireland implements via national competent authorities alongside the
  AI Office; product-liability Directive (EU) 2024/2853 transposition due
  2026-12-09.

**SZL position:** Amendment 1 was corrected to cite **Article 50** (not
Art. 11, which is technical documentation). Khipu provenance badges are
machine-readable and signed — they can serve as Art. 50(2)-style marks.
Gap: no watermark/detection interoperability commitment yet (due 2027-02-02).

## NVIDIA — OpenShell (open-source agent runtime, GTC 2026)
- Kernel-level sandbox per agent (Landlock/seccomp/default-deny egress),
  YAML policy-as-code, gateway-brokered credentials (agents never see real
  keys), formal verification of policy changes before apply, full
  allow/deny audit logs, fleet governance via Gateway.
- Scope limit (acknowledged by third parties): agent-to-agent and
  cross-fleet governance sit outside it.

**SZL position:** complementary, not competitive. OpenShell governs what an
agent may TOUCH; the Kamachikuna governs what an agent may CLAIM.
`sdk/szl/openshell.py` already emits OpenShell-reviewable policy + OCSF audit
events so both planes land in one audit trail. Next step: run estate agents
inside OpenShell sandboxes and route kamachikuna decisions into the same log.

## Recommendations (do these, in order)
1. Ship the corrected Art. 50 amendment (done in this branch; stone hash
   re-pinned — the change is deliberate and visible).
2. Add a per-deployment impact-assessment receipt type (Korea high-impact
   alignment) to khipu.
3. Vendor the Lean 4 proofs into the repo or drop the LEAN_PROVEN claim —
   currently external and unverifiable here (Law 2 applies to us).
4. Register the khipu badge as the estate's machine-readable synthetic-
   content mark; add detection-interoperability before 2027-02-02.
5. Keep training on A6000 nodes per the runbook; Salad stays burst-only.
