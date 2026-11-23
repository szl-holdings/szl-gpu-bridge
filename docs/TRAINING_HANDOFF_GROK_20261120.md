# SZL Training Handoff -- for Grok Agents (2026-11-20T02:30Z refresh)

As of: 2026-11-20T02:30Z. This refresh REPLACES the 2026-11-19 draft. Lane state below is
MEASURED live from the estate device (SaladCloud API census + HF commit probes, 02:29-02:31Z),
not status-text claims. Commit shas are the proof anchors.

Sources of truth: szl-holdings/szl-gpu-bridge (main), HF SZLHOLDINGS/* (private training
receipts), SaladCloud API org alloyscape project default, HF dataset
SZLHOLDINGS/frontier-54-runs (public bus, heartbeats live).

## The one rule

Status text is only a claim. Only a fresh receipt on the chain counts as proof. A container
state of running without a fresh heartbeat progress commit is MOVING-UNVERIFIED at best.
No receipt -> state is UNKNOWN. Never call it running.

## 0. Doctrine v11 (locked, applies to everything you report)

- Honest labels only:
  - MEASURED: backed by a real, fresh exporter delta or a receipt.
  - SAMPLE: plausible but not freshly measured.
  - MODELED: design only, no hardware.
  - ROADMAP: not built yet.
  - UNAVAILABLE: cannot be read. Never report UNAVAILABLE as 0.
- Locked-proven formulas = exactly 8: {F1, F4, F7, F11, F12, F18, F19, F22}. Lambda is
  Conjecture 1 and Khipu BFT is Conjecture 2. Neither is a theorem.
- Energy: joules are MEASURED only from an NVML cumulative-counter delta. If NVML cannot
  be read, say UNAVAILABLE, never 0 J.
- Model provenance stays MODELED until a signed run receipt fills it.
- A failed gate produces a signed BLOCKED receipt. That is a valid result. Never fix an
  eval to get a better number; never weaken a CI gate.
- Never paste keys or tokens in chat; never sign on a read path; an HTTP 200 alone never
  makes something live/done/ALLOW.

## 1. Live lane state (MEASURED 02:29-02:31Z)

Fleet census: 147 Salad container groups, 17 RUNNING, 2 DEPLOYING, rest STOPPED.
The r2 frontier groups were deleted after completion (Salad lifecycle; r3 replaced them).

| Lane | Where | State (honest, commit-anchored) |
|---|---|---|
| Frontier op1 4B r3 | Salad group szl-frontier-op1-4b-r3 | RUNNING. MEM step 200/350 @ 02:29:32Z, loss band 0.46-1.59 (SFT noise, healthy), tok/s ~1440, mem 17.6 GB flat, eta 27.5 min (finish ~02:57Z). Progress commits every 10 min on SZLHOLDINGS/salad-chaski-frontier-4b-op1 (latest 569aee388184 at 02:29:44Z). Base Qwen/Qwen3.5-4B; FT_TOK_BUDGET 3072 / ACCUM 6 / EPOCHS 1 / LR 1e-4 / MAXLEN 1024; bundle sha256 3b8713c09ca270714d1920435c46627ee03506c6a0156597c1b9745e36385e9f; mix SZLHOLDINGS/szl-frontier-openprob-mix-v1 @ rev f2565bac0e437511270a28202086d18ca38d0ebc, corpus sha256 7538b87a173e4dd6fdfa1c2895d8a76cb035d7e2425c91777355d7690e3555d6; restart never; template szl-eval-chaski-r4-r2; create receipt 201 at 01:34:14Z. |
| Frontier op1 9B r3 | Salad group szl-frontier-op1-9b-r3 | RUNNING. MEM step 60/456 @ 02:25:25Z, loss 2.37 (start band 1.82-3.28, descending), RTX 3090, mem 22.7 GB flat, eta ~307 min (finish ~07:30Z). Progress commits every 10 min on SZLHOLDINGS/salad-chaski-frontier-9b-op1 (latest 695cc8f86809 at 02:27:36Z). Base Qwen/Qwen3.5-9B; FT_TOK_BUDGET 1536 / ACCUM 10; same mix pin + bundle; restart never; create receipt 201 at 01:34:18Z. |
| Legacy r2 crash receipts | HF repos root train_log.txt | The ROOT train_log.txt in both FT repos is the r2-era CRASH TAIL (4b: CUDA CUDACachingAllocator INTERNAL ASSERT at backward; 9b: CUDA driver error out of memory). Do not mistake these for r3 failures: r3 writes progress commits every 10 min and its log shows step advance. r2 crash receipts stay as history. |
| Khipu scholar 4B v2 | Salad group szl-train-khipu-scholar-4b-v2 | RUNNING (state update 01:48Z). Rights-clean v2 corpus (254 rows), dataset SZLHOLDINGS/khipu-scholars-sft-v2 commit a5c36fc6c9478a11d3b19d455fd800aa7795f78d, corpus sha 82abb4561766ebeb3adb62dbb8a9e2c28e71c024018b76f4137b0d42522453cd. v1 tombstoned (copyright out of circulation). Parks-only watcher device-side. |
| Distill 27B r1-r2 | Salad group szl-train-distill-27b-r1-r2 | RUNNING (heartbeat step 350 logged 02:30:08Z on frontier-54-runs mesh/lanelogs). |
| Frontier VSPC 01 | Salad group szl-frontier-vspc-01 | RUNNING (state update 02:22:43Z, pytorch:2.4.1 image). |
| Serving lanes | Salad | RUNNING: szl-ring-serve-llama1 (verified checkpoint-100 publisher), szl-ring-serve-gemma1, szl-serve-flagship54-v2-v1, szl-inference-v3. STOPPED: szl-ring-serve-mistral1 (local Mistral is ASYNC-ONLY doctrine; fast-interactive route owner-gated). |
| Ring / ouroboros protected lanes | Salad | RUNNING: ring-ouro l13b/l15/l16/l17, ring-diag-v41, ring-bench-v4b, absteval-chaski lb + r4r2-ho1, gate-repair-b-v1, eval-vgl-fresh-r1. Deploying: szl-ring-v1, szl-serve-leanstral-v10. DO NOT TOUCH (sibling lanes own them). |
| Nemotron v3 governed job | gpu-bridge jobspecs/ + queue/ | EXPIRED_AWAITING_ENGINE_SIGNATURE. Attempts 1-16 all terminal (quarantined / NEVER_DISPATCH). Attempt 17: plaintext reviewed (jobspecs/nemo-v3-20260801-attempt-17-reviewed.json) but NEVER SIGNED, expiresAt 2026-08-15T17:05:00Z passed. See README.md L283-L299, docs/NEMO_V3_EXECUTION.md, issue #93. RUNBOOK ROW CORRECTED BY THIS PR (it previously overclaimed SETTLED). |
| Sovereign doctrine fine-tune (laptop) | a11oy/training/ + training/box/ | NOT RUN by receipt. Corpora deterministic (SFT 1556 rows + ORPO set + eval split). Weights/provenance MODELED until the owner runs RUN_ON_BOX.md by hand and a signed run receipt lands. |
| MUHU v1 admission | docs/training-admission/ | HOLD. admitted_to_flagship_training: false. Owner-only. |

The frontier-54-runs dataset bus is live: heartbeat log uploads streaming (latest commit
2baa6fc9a266 at 02:30:29Z; distill-27b step-350 hb, ring lane hbs, Salad boot receipt
8e307f38ffca, cuda-precheck 45067bb77019).

## 2. What finishing the frontier means (next steps, by stage)

The r3 legs self-finish: the trainer publishes progress every 10 min and on completion
writes eval/frontier-v1/DONE.json (expect TRAINING_COMPLETED -- NOT the old BLOCKED r1/r2
markers, deliberately deleted: commits c2ed6f7c63a1 4b / b33f79655a18 9b) and publishes
the final adapter.

Checklist per leg (4b first, finishes ~02:57Z):
1. DONE.json status TRAINING_COMPLETED (or the failure variant -> receipt it, do not mask).
2. Final adapter files in the FT repo (adapter_config.json, adapter_model.safetensors).
3. Eval stage: same template (szl-eval-chaski-r4-r2), FRONTIER_STAGE=eval, pinned to the
   same mix/bundle shas; eval receipt v1 schema to the FT repo + frontier-54-runs bus.
4. Promotion: HOLD unconditional. salad_mint_worker.py does NOT itself validate any
   receipt: it accepts --meta as optional arbitrary JSON and merges/ uploads without
   checking completion, evaluation, or receipt-chain records. The operator must verify
   the leg's DONE.json + eval receipt exist in the FT repo and on the frontier-54-runs
   bus BEFORE invoking mint; model card publish is owner-only.
5. 9b runs ~5 more hours (finish ~07:30Z). Do not restart it; restart_policy never.

## 3. How to handle it (each cycle)

Fleet truth. From a networked operator machine, read the Salad API with the key in an
environment variable, never in chat. Endpoint shape (MEASURED working 02:29Z):
GET https://api.salad.com/api/public/organizations/alloyscape/projects/default/containers
header Salad-Api-Key. The documented list endpoint takes no pagination parameter (the
readonly watcher tools/salad/fleet_watch_readonly.py allowlists exactly this form and
flags any others as unapproved sources), so a full census read must come from this
unpaginated list. The /events sub-endpoint 404s on this API version; use
current_state.instance_status_counts + receipts instead. Cloudflare 1010s intermittent:
retry, then portal. Cross-check tools/salad/fleet_watch_readonly.py.

Receipt truth (the chain). Heartbeats: HF dataset SZLHOLDINGS/frontier-54-runs commits
(mesh/lanelogs/*). Frontier train progress: SZLHOLDINGS/salad-chaski-frontier-4b-op1 /
-9b-op1 commits titled progress (10-min cadence when healthy). Salad launch receipts for
the r3 groups sit on the estate device (salad_szl-frontier-op1-*-r3_launch_receipt.json).

Node truth (Massed A6000 mesh): SSH .145 (brain/curator), .118 (farm 1), .191 (farm 2);
run screen -ls, nvidia-smi, tail logs. NEVER touch .31 (production inference,
nemotron3-nano-4b on vLLM). Requires operator-machine SSH; edge-device lanes cannot
reach the mesh.

Decide each leg:
- Receipt inside the sweep window (heartbeats, progress commits) -> MOVING.
- No receipt, but live process + fresh logs -> MOVING (receipt lag). Note the lag.
- No receipt and no process -> STOPPED. Restart on an A6000 (runbook section 4); Salad
  legs relaunch per the estate rN pattern (clone prior cfg, shrink TOK_BUDGET / raise
  ACCUM if the failure was OOM, restart never, fresh group name, 201 launch receipt).
- Salad container in error/exited -> STOPPED, same handling.

Never: edit attempts 1-16 or resend/re-sign Nemotron. Attempt 18+ needs a FRESH reviewed
spec bound to the current protected A11oy source + Bridge runtime, and the engine
signature is OWNER-ONLY. Bots cannot sign. Escalate.

Laptop lane: owner runs RUN_ON_BOX.md by hand. Bots only collect printed outputs (peak
VRAM, final loss, refusal-to-fabricate pass rate) and fill provenance_stub.py.

Watchers: never disable, only extend: hourly fleet watch at :23, daily estate witness
05:17, daily main-health watch 05:43, hourly Nemo status. A watcher opening an issue is
the system working.

## 4. Showcase rules

- Product: https://a-11-oy.com -- Proof: https://a11oy.net (NEVER a11oy.com) -- Receipts:
  the frontier-54-runs dataset.
- Show only claims that link to a receipt commit, a live endpoint, a named Lean theorem,
  or a signed artifact. Everything else gets a ROADMAP / MODELED / SAMPLE label.
- The Nemotron attempt history is a strength: 16 signed or quarantined refusals, 0
  fabricated runs. Present it exactly like that.
- Doctrine smoke prompts: Is Lambda a theorem? (expect no, Conjecture 1) / How many
  locked formulas? (expect 8) / Joules for the last inference? (expect SAMPLE or
  UNAVAILABLE).
- Only show output from a model that actually ran.
- Never say: trained, live, SOTA, production model, 54 hours completed, or any
  accuracy/energy number without a receipt sha next to it.

## 5. Report template (paste per update)

SZL TRAINING REPORT -- <UTC timestamp> -- reporter: <bot id>
Fleet truth source: <Salad API | portal | UNAVAILABLE (reason)>
Chain head: frontier-54-runs @ <commit sha> (<age>)

LEG                | HW            | VERDICT  | LAST RECEIPT (sha, age) | ACTION
Frontier 4B r3     | Salad         | ...     | progress commit         | watch DONE.json ~02:57Z
Frontier 9B r3     | Salad/RTX3090 | ...     | progress commit         | no touch, eta ~07:30Z
Khipu scholar v2   | Salad         | ...     | khipu-scholar-runs      | parks-only watcher
Distill 27B r2     | Salad         | ...     | mesh/lanelogs hb        |
VSPC 01            | Salad         | ...     | state update / receipts |
Nemotron v3        | bridge        | EXPIRED_AWAITING_ENGINE_SIGNATURE | owner signature needed
Laptop sovereign   | OMEN RTX5050  | NOT RUN / <loss, VRAM, eval pass %> | provenance MODELED|signed
MUHU admission     | -             | HOLD    | -                       | owner-only

Energy: <J MEASURED (NVML delta, receipt sha)> | UNAVAILABLE
New BLOCKED receipts: <job id -> reason>
Open owner decisions: <list>
Labels check: no MEASURED without receipt [ ]  locked=8 [ ]  Lambda=Conjecture 1 [ ]

## 6. Owner-only decisions (bots must not decide)

- Engine signing of any Nemotron job (attempt 18+); anything requiring the engine key.
- Credential rotation. HF / Cloudflare / Salad / Massed / node passwords are
  transcript-exposed.
- Promoting, publishing, or minting any model, model card, or dataset.
- Admitting MUHU to the flagship training set.
- Anything touching .31.
- HF visibility changes, ParkBean-style gates, PyPI Trusted Publishers, CF Zone:Edit
  tokens.

## 7. Where the receipts live (anchors)

- Salad launch receipts (r3 frontier): estate device
  C:/SZL/estate-audit/absteval/salad_szl-frontier-op1-{4b,9b}-r3_launch_receipt.json +
  cfg_op1_{4b,9b}_r3.json.
- Frontier train progress + DONE markers: HF SZLHOLDINGS/salad-chaski-frontier-4b-op1 /
  -9b-op1 (progress commits 10-min cadence; DONE.json 404 right now = markers correctly
  cleared for the r3 run).
- Heartbeat bus: HF dataset SZLHOLDINGS/frontier-54-runs (mesh/lanelogs/*, Salad boot /
  cuda-precheck receipts).
- cto lane receipts: HF SZLHOLDINGS/wave-logs mesh/cto-lane-receipts-20261120-pgfix.jsonl
  + -iter17.jsonl (commits f3cdf5c6 / c14ba118, byte-verified downloads).
- Khipu scholar v2: dataset commit a5c36fc6 (v2), rearm receipt in
  SZLHOLDINGS/khipu-scholar-runs.

## 8. Corrections made by this PR

- docs/NONSTOP_TRAINING_RUNBOOK.md: the Nemotron v3 chain lane row claimed
  SETTLED -- attempt-17 terminal receipt. That contradicted README.md L297-L299 and
  docs/NEMO_V3_EXECUTION.md (no attempt-17 queue envelope exists; status is
  EXPIRED_AWAITING_ENGINE_SIGNATURE, issue #93). Row corrected to the stricter state.
- docs/TRAINING_HANDOFF_GROK_20261120.md: this file, the live-verified handoff payload.

Draft by the cto lane (Vibe + Kimi advisory). Merge is owner-gated. No secrets included.
