# OPERATOR HANDOFF — the complete manual for the next session with hands

**For:** any agent session with Desktop Commander / SSH / HF connector / Massed MCP.
**State as of:** 2026-11-10 ~08:30 UTC. Everything below is merged on `szl-gpu-bridge/main`.

## The doctrine (read first, it is the law of the estate)
Kamachikuna: fail closed; every claim carries a receipt (no receipt = UNKNOWN, never a
claim); abstain when you cannot prove; no silent failure; tamper-evident everything.
Honest states: MEASURED / CONJECTURE / UNAVAILABLE — never zero for unmeasured, never
"running" without a receipt. Full text: `docs/AUTONOMY_LADDER.md` (the 0.3% partition).

## The immediate queue, in order

1. **The five `-v2` Salad legs** — they ALREADY EXIST (name_conflict proves it).
   Check each in the portal: running+old-start = training (then verify chain commits);
   stopped = read its log, fix env, delete + recreate with the proven body
   (base64 command, map env, cpu/memory/gpu_classes resources — see PR #142 body).
2. **frontier54-v2 logs** — 27h "running", ZERO receipts ever. Portal → Logs.
   Real training = rescue checkpoints to HF NOW (ephemeral storage!); stall = stop it.
3. **Node bootstrap** (.118/.191): clone szl-gpu-bridge, `tools/a6000-migration/launch_leg.sh`
   per leg (brain-v35+hatun24 on .118; a11oy-code+bench on .191). HF token via `read -s`.
   Boot receipts land at `~/szl-legs/<leg>/boot-receipt.json`.
4. **The 54 hours** — per `docs/OUROBOROS_CURRICULUM.md`: soul (constitution/prime/lean)
   → skills (code/formulas/papers/verticals) → proof (proving drills). Checkpoint pushes
   every 2h; receipts to the chain every publish.
5. **The mesh** — `tools/mesh/node_sync.sh` hourly cron on every node (push receipts,
   pull curriculum manifest, heartbeat). Brain node = .145 until the stack instance lands.
6. **The proving ground** — `.145`: `tools/aeroff_proving_ground.sh` (24h, CPU-first).
   The numerics stage already ran (receipt fc38532d; TOF sign-convention finding).
7. **Serve the models** — after the 54h, `tools/a6000-migration/serve_model.sh <model>`
   per `docs/MODELS_ON_MASSED.md` (serve where you trained).

## The watchers (already live — never disable, only extend)
- **Fleet watch** hourly :23 (SALAD_API_KEY secret is SET) — fleet truth + chain freshness
- **Estate wiring witness** daily 05:17 — 12 Spaces + dataset freshness
- **Main health watch** daily 05:43 — 16 repos' latest workflows
- **Nemo governed status** hourly — signed jobspec + terminal receipt
Drift = labeled issue with evidence. A watcher that fails loudly is working.

## The known-open items (owner-gated — do not decide these)
Credential rotation (HF/CF/Salad/Massed/node passwords — many transcript-exposed) ·
CF canary sequence (API-host → deploy → 402 → paid settle) · a11oy khipu counters
(adopt `szl.khipu` from the SDK) · OpenAI 3.x admission (#3279) · Snowflake warehouse
auto-resume · Cornell Dec 15 + Nov 18 · Sara (wait for her return) · Andrés (send the
acknowledgment note) · Chad (the engagement note + proving ground results).

## The receipts chain (the shared state)
`SZLHOLDINGS/frontier-54-runs` — 28 commits as of this writing. Every push
content-addressed; signatures full-length; freshness is a separate axis from integrity
(`szl.khipu.check_chain(max_age_days=...)`). A leg without chain commits inside the
sweep window is STOPPED, whatever its status text says.

## The one rule that summarizes everything
**Status text is a claim; a commit is a receipt.** If you remember nothing else,
remember that — it cost the estate 24 silent hours to learn, and it is why every
command in this manual starts with a boot receipt.
