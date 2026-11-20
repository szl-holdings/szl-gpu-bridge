# Grok-Bot Training Handoff — the SZL training estate

**Status: point-in-time, NON-AUTHORIZING context transfer.** This document transfers
context to successor agents ("Grok bots"). It transfers **no** owner, merge, dispatch,
signing, promotion, or credential authority. Alliance with any of it as evidence, not
permission.

- **Snapshot observed at:** 2026-11-20T03:18Z, via a GET-only Salad container-group census
  (no writes; key kept in a device-local env/file, never in chat).
- **Canonical sources of truth:** this repo (`szl-holdings/szl-gpu-bridge` `main`),
  `docs/NONSTOP_TRAINING_RUNBOOK.md`, `docs/NEMO_V3_EXECUTION.md`, `README.md`,
  and the receipts chain `SZLHOLDINGS/frontier-54-runs` on Hugging Face.
- A status text is a claim. Only a signed receipt (or a fresh GET-only observation made
  by the bot itself) counts as proof. No receipt + no live process = STOPPED, never
  "running."

## 0. Doctrine you must follow (locked)

Honest labels apply **per claim / per table row**, not per section:

- `MEASURED`: backed by a fresh exporter delta, a signed receipt, or a GET-only
  observation the bot just made itself (with UTC timestamp and sanitized evidence source).
- `SAMPLE`: plausible, not freshly measured. `MODELED`: design only, no hardware.
- `ROADMAP`: not built. `UNAVAILABLE`: cannot be read — unknown, not failure and not success.
  Never report UNAVAILABLE as 0.
- Locked-proven formulas = exactly 8: {F1, F4, F7, F11, F12, F18, F19, F22}. Lambda is
  Conjecture 1 and Khipu BFT is Conjecture 2. Neither is a theorem.
- Energy joules are MEASURED only from an NVML cumulative-counter delta; otherwise UNAVAILABLE, never 0 J.
- Model provenance stays MODELED until a signed run receipt fills it.
- A failed gate produces a signed BLOCKED receipt — that is a valid result. Never "fix"
  an eval to improve a number; never weaken a CI gate.
- Never paste keys/tokens in chat, never sign on a read path, never call anything
  "live"/"done"/"ALLOW" from an HTTP 200 alone.

## 1. Fleet truth — MEASURED at 2026-11-20T03:18Z (GET-only census)

`146 container groups = 129 stopped / 16 running / 1 deploying`
(evidence: device receipt `salad_census_vibe_out.json`; GET-only; sanitized).

Named RUNNING groups (verbatim from census):

| Group | Role (label) |
|---|---|
| szl-train-khipu-scholar-4b-v2 | training lane — MEASURED (running state only) |
| szl-frontier-op1-4b-r3 | training lane — MEASURED (running state only) |
| szl-frontier-op1-9b-r3 | training lane — MEASURED (running state only) |
| szl-frontier-vspc-01 | training lane — MEASURED (running state only) |
| szl-train-distill-27b-r1-r2 | training lane — MEASURED (running state only) |
| szl-serve-flagship54-v2-v1, szl-ring-v1, szl-ring-v1b, szl-ring-serve-llama1, szl-ring-ouro-l16, szl-ring-ouro-l17, szl-inference-v3 | serve/eval lanes — role UNAVAILABLE unless separately identified |
| szl-absteval-chaski-lb-ho1, szl-absteval-chaski-r4r2-ho1, szl-eval-vgl-fresh-r1, szl-gate-repair-b-v1 | eval/repair lanes — role UNAVAILABLE unless separately identified |

**Read this honestly:** `running` is a container state, not proof of training progress,
health, or success. Do not infer that all 16 running groups are training lanes; unobserved
roles are `UNAVAILABLE`. `deploying` is not `running`.

## 2. The training lanes (what exists, with per-row label)

| Lane | Where | What | State (honest label) |
|---|---|---|---|
| A. Sovereign doctrine fine-tune | a11oy/training/ + training/box/ | Qwen2.5-3B: QLoRA -> SFT -> ORPO -> GGUF q4_k_m -> Ollama szl-sovereign-qwen; run by hand on the OMEN laptop | Corpora deterministic: SFT 1,556 rows (szl_seed_full.jsonl) + ORPO preference set + eval split — MEASURED (files/hashable). No training-run receipt seen: weights MODELED. Standing ruling: laptop = control room only, all GPU work on Salad |
| B. Nemotron v3 governed job | gpu-bridge jobspecs/ + queue/ | DSSE-signed jobs (active key b8041281c81c4caa) -> owner GPU -> signed receipts in private HF SZLHOLDINGS/szl-training-receipts | Attempts 1-16 terminal, each with its actual per-attempt quarantined/NEVER_DISPATCH disposition — MEASURED (hash-pinned in queue/quarantine/ + queue/evidence/). Attempt 17: EXPIRED_AWAITING_ENGINE_SIGNATURE — see section 4 |
| C. Salad burst worker | tools/salad/salad_worker.py v3.10 | Resumable spot training: HF checkpoints, SIGTERM checkpointing, boot/step/complete receipts, NVML/zeus energy, F1-F10 failure guards | Code ACTIVE in main (PRs #236-#265 through 2026-11-19) — MEASURED (repo history). Live footprint at 03:18Z: 5 training-named groups running (section 1) — MEASURED (state only) |
| D. Frontier-54 + the brains (Brain v3.5, Brain-2c mathlib, Nano-2, The Collective) | Salad; on any restart move to A6000s (runbook section 4) | 54-hour fine-tune + brain legs | The `frontier-op1-4b-r3` / `frontier-op1-9b-r3` / `frontier-vspc-01` groups are running — MEASURED (state only); their mapping to the 2026-11-09 leg names is UNVERIFIED, do not claim it. The 2026-11-09 legs themselves: UNKNOWN from the audit seat |
| E. Mint station | tools/salad/salad_mint_worker.py (#255) | Merges adapters behind a receipt gate only | Code merged — MEASURED (repo). No promotion claimed anywhere |
| F. MUHU v1 -> flagship | docs/training-admission/ | Admission review | HOLD: admitted_to_flagship_training = false — MEASURED (document) |

Hardware (Massed A6000 mesh): .145 brain/curator node (training-safe); .118 farm 1;
.191 farm 2; .31 production inference (vLLM nemotron3-nano-4b) — **never touch .31's GPU.**
Shared receipt bus: HF dataset SZLHOLDINGS/frontier-54-runs (public; last update
2026-11-19 by prior audit; verify freshness yourself before citing).

## 3. Per-cycle procedure (each bot cycle)

1. Read canonical docs (runbook, NEMO_V3_EXECUTION.md, README) first.
2. GET-only observation: Salad census (key in env var on a networked operator machine,
   never in chat; CF 1010 is intermittent — retry or read the portal), then
   `tools/salad/fleet_watch_readonly.py`. Node truth via SSH: `screen -ls`,
   `nvidia-smi`, `tail ~/szl/*.log` on .145/.118/.191; never .31.
3. Timestamp and diff against the previous snapshot; reconcile discrepancies.
4. Verdict per leg: receipt in sweep window -> MOVING; no receipt + live process + fresh
   logs -> MOVING (receipt lag, note it); no receipt + no process -> STOPPED (restart
   per runbook section 4); Salad container error/exited -> STOPPED.
5. Report blockers; escalate owner-only items and stop there.
6. **No autonomous remediation, no status promotion from stale snapshots.** If your
   observation is older than the cycle, re-measure before claiming.

## 4. Nemotron v3 — exact state (do not soften)

- Attempts 1-16 are terminal, each with its actual per-attempt `quarantined` /
  `NEVER_DISPATCH` disposition preserved byte-for-byte in `queue/quarantine/` and
  `queue/evidence/`. Never resend, re-sign, edit, or retry any of them.
- Attempt 17 (`jobspecs/nemo-v3-20260801-attempt-17-reviewed.json`) is
  **EXPIRED_AWAITING_ENGINE_SIGNATURE** (non-terminal; tracked in issue #93): the
  reviewed expiresAt (2026-08-15T17:05:00Z) passed with no engine-signed queue envelope.
- **NOT SETTLED; no terminal receipt; no automated retry, dispatch, or quarantine
  override.** Any next attempt needs a fresh engine signature from the owner — bots
  cannot sign. Canonical references: `docs/NEMO_V3_EXECUTION.md`, `README.md`.
- Do not present this history as failure to hide: it is a fail-closed pipeline that
  refused 16 times rather than shipping a bad model ("16 signed or quarantined
  refusals, 0 fabricated runs").

## 5. Restart rule (and the new gate)

- Boot receipt is the FIRST line of every leg; one screen session per leg; checkpoints
  every 2 hours; receipts on every publish. Stopped legs restart on A6000s, not Salad.
- **New (merged in main via PR #270):** restarts must pass the restart-lineage witness
  (`tools/salad/restart_lineage_witness.py`, fail-closed; it NEVER admits training;
  REVIEW_CANDIDATE_ONLY always means owner review). Restarts of training are
  owner-gated admissions.
- Fleet doctors are currently DOWN by standing ruling (they restarted groups without
  resumable-state verification; two doctors = overlapping ownership). At most ONE doctor
  may return, re-armed wired to the PR #270 gate, via a new PR. Do not revive one.
- Watchers (never disable, only extend): hourly fleet watch :23, daily estate witness
  05:17, daily main-health watch 05:43, hourly Nemo status. A watcher opening an issue
  is the watcher WORKING.

## 6. Showcase rules (public claims)

Product: https://a-11-oy.com · Proof: https://a11oy.net (never a11oy.com) ·
Receipts: the frontier-54-runs dataset. Show only claims linked to a receipt commit, a
live endpoint, a named Lean theorem, or a signed artifact; everything else carries
ROADMAP/MODELED/SAMPLE labels. Only sanitized MEASURED artifacts may be presented as
real: no fabricated logs, no implied training progress, no Nemotron completion claims.
Doctrine smoke prompts for model demos: "Is Lambda a theorem?" (expect: no, Conjecture
1), "How many locked formulas?" (expect 8), "Joules for the last inference?" (expect
SAMPLE or UNAVAILABLE). Show only output from a model that actually ran. Banned without
a receipt hash: "trained", "live", "SOTA", "production model", "54 hours completed",
any accuracy or energy number.

## 7. Report template (paste into each update)

```
SZL TRAINING REPORT — <UTC timestamp> — reporter: <bot id>
as-of/source/access mode: <timestamp; Salad API | portal | UNAVAILABLE (reason)>
Census totals: <n stopped / n running / n deploying | UNAVAILABLE>
Chain head: frontier-54-runs @ <commit sha> (<age>)

LEG | HW | VERDICT | LAST RECEIPT (sha, age) | LABEL
scholar-4b-v2 | Salad | MOVING/STOPPED/UNKNOWN | | MEASURED/SAMPLE/...
frontier-op1-4b-r3 | | | |
frontier-op1-9b-r3 | | | |
frontier-vspc-01 | | | |
distill-27b-r1-r2 | | | |
2026-11-09 legs (Frontier-54 v2/Brain v3.5/Brain-2c/Nano-2/Collective) | | UNKNOWN (mapping unverified) | | UNAVAILABLE
Nemotron v3 | bridge | EXPIRED_AWAITING_ENGINE_SIGNATURE (a17, issue #93) | — | MEASURED (canonical docs)
Laptop sovereign | OMEN | NOT RUN / <loss, VRAM, eval pass %> | | provenance MODELED|signed

Measured deltas this cycle: <...>
Nemotron/blockers: <...>
Actions taken (GET-only unless owner-ticked): <...>
UNAVAILABLE/discrepancies: <...>
Owner decisions needed: <list>
Sanitized evidence links: <...>
Labels check: no MEASURED without receipt/fresh GET ✔  locked=8 ✔  Lambda=Conjecture 1 ✔
```

## 8. Owner-only decisions (bots must not decide)

- Engine-key signing of any job; Nemotron attempt 17 or later.
- Admitting any training restart through the PR #270 witness (REVIEW_CANDIDATE_ONLY -> owner).
- Re-arming a fleet doctor (at most one, gated).
- Credential rotation (HF, Cloudflare, Salad, Massed, node passwords — all transcript-exposed).
- Promoting/publishing any model, model card, or dataset; MUHU admission to flagship.
- Clearing the a11oy release freeze (A11OY_AUTOMATIC_RELEASE_FREEZE repo variable — owned
  by an in-flight execution review; owner clears after verify_current_deployment).
- Anything touching .31.
- Gateway start path on the laptop (blocked on four deleted credentials; rotation needs
  the owner's named backup-vs-rotate order).

*Point-in-time snapshot; re-measure before acting. Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr.*
