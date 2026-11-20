# SZL Training Handoff — for Grok bots

As of: 2026-11-20 · Source of truth: `szl-holdings/szl-gpu-bridge` (main) and `szl-holdings/a11oy/training/`
Maintainer: DC CTO lane (Kimi + Mistral + GLM dreaming together), receipts runtime `C:/SZL/kimi-charter/`.

**The one rule:** status text is only a claim. Only a commit on the receipts chain counts as proof. If there is no receipt, the state is UNKNOWN. Never call it "running."

## 0. Doctrine you must follow (Doctrine v11, locked)

- Honest labels only:
  - **MEASURED**: backed by a real, fresh exporter delta or a receipt.
  - **SAMPLE**: plausible but not freshly measured.
  - **MODELED**: design only, no hardware.
  - **ROADMAP**: not built yet.
  - **UNAVAILABLE**: cannot be read. Never report it as 0.
- Locked-proven formulas = exactly 8: {F1, F4, F7, F11, F12, F18, F19, F22}. Λ is Conjecture 1 and Khipu BFT is Conjecture 2. Neither is a theorem.
- Energy: joules are MEASURED only from an NVML cumulative-counter delta. If NVML cannot be read, say UNAVAILABLE, never 0 J.
- Model provenance stays MODELED until a signed run receipt fills it.
- A failed gate produces a signed BLOCKED receipt. That is a valid result. Do not "fix" an eval to get a better number, and do not weaken a CI gate.
- Never paste keys or tokens in chat, never sign on a read path, and never claim something is "live", "done" or "ALLOW" from an HTTP 200 alone.

## 1. What training exists (the lanes)

| Lane | Where | What | State (honest) |
|---|---|---|---|
| A. Sovereign doctrine fine-tune | a11oy/training/ + training/box/ | Qwen2.5-3B: QLoRA → SFT → ORPO → GGUF q4_k_m → Ollama szl-sovereign-qwen. Run by hand on the OMEN laptop (RTX 5050, 8 GB). | Corpora built and deterministic: SFT 1,556 rows (szl_seed_full.jsonl), plus the ORPO preference set and eval split. No training run receipt seen. Weights are MODELED. |
| B. Nemotron v3 governed job | gpu-bridge jobspecs/ and queue/ | DSSE-signed jobs (key b8041281c81c4caa) → owner GPU → signed receipts in private HF SZLHOLDINGS/szl-training-receipts | Attempts 1–16 are all terminal: quarantined, NEVER_DISPATCH. None trained. Each failed at a different gate (runtime binding, tokenizer, SFTConfig key, meta-tensor, stale source). Attempt 17's plaintext is reviewed but was never signed. It expired 2026-08-15 and its status is EXPIRED_AWAITING_ENGINE_SIGNATURE (issue #93). |
| C. Salad burst worker | gpu-bridge tools/salad/salad_worker.py v3.10 | Resumable spot training: HF checkpoints, SIGTERM checkpointing, boot/step/complete receipts, NVML and zeus energy, F1–F10 failure guards | Very active in code (PRs #236–#265 on 11-17 to 11-19: F10 terminal restore, fail-closed on provider HTTP errors, fleet-watch contradiction flags). Live fleet state is UNKNOWN from the audit seat (API 401). |
| D. Frontier-54 and the brains | Salad → move to A6000s | 54-hour fine-tune, Brain v3.5, Brain-2c (mathlib), Nano-2, The Collective | Last documented as UNKNOWN or STOPPED (11-09). Doctrine: restart on the A6000s, not Salad. |
| E. Mint station | tools/salad/salad_mint_worker.py (#255) | Merges adapters, but only behind a receipt gate | Code merged; no promotion claimed |
| F. MUHU v1 → flagship | docs/training-admission/ | Admission review | HOLD. admitted_to_flagship_training: false |

**Hardware (Massed A6000 mesh):**
- `.145`: brain/curator node; safe to train on.
- `.118`: farm 1 (brain-v35, hatun24).
- `.191`: farm 2 (a11oy-code, bench).
- `.31`: production inference (nemotron3-nano-4b on vLLM). Do not touch its GPU.

Shared bus: HF dataset `SZLHOLDINGS/frontier-54-runs` (public).

## 2. How to handle it (each cycle)

1. **Fleet truth.** Read Salad container state from a networked operator machine with the key in an environment variable, never in chat. Cloudflare 1010 errors are intermittent, so retry or read the portal. Then run `tools/salad/fleet_watch_readonly.py`.
2. **Node truth.** SSH to .145, .118 and .191 and run `screen -ls`, `nvidia-smi` and `tail ~/szl/*.log`. Never touch .31.
3. **Decide each leg:**
   - Receipt inside the sweep window → MOVING.
   - No receipt, but a live process with fresh logs → MOVING (receipt lag). Note the lag.
   - No receipt and no process → STOPPED. Restart it on an A6000 (runbook §4).
   - Salad container in error or exited → STOPPED. Restart it on an A6000.
4. **Restart rule.** The boot receipt is the first line of every leg. Use one screen session per leg. Push checkpoints every 2 hours and push receipts on every publish.
5. **Nemotron.** Do not resend, re-sign or edit attempts 1–16. Attempt 17 needs an owner engine signature on a fresh reviewed spec, because it has expired. Its next identity must bind the current protected A11oy source and Bridge runtime. Bots cannot sign: escalate to the owner.
6. **Laptop lane A.** The owner runs RUN_ON_BOX.md by hand. A bot's job is only to collect what the scripts print: peak VRAM, final loss and the per-family refusal-to-fabricate pass rate. Then fill the provenance with provenance_stub.py.
7. **Watchers.** Never disable these, only extend them: hourly fleet watch at :23, the daily estate witness at 05:17, the daily main-health watch at 05:43, and the hourly Nemo status. If a watcher opens an issue, it is working.

## 3. Showcase rules (what you may show publicly)

- Product: https://a-11-oy.com · Proof: https://a11oy.net (never a11oy.com) · Receipts: the frontier-54-runs dataset.
- Show only claims that link to a receipt commit, a live endpoint, a named Lean theorem or a signed artifact. Anything else gets a ROADMAP, MODELED or SAMPLE label.
- The Nemotron attempt history is a strength, not something to hide. It shows a fail-closed pipeline that refused 16 times rather than shipping a bad model. Present it as "16 signed or quarantined refusals, 0 fabricated runs."
- Model demos — doctrine smoke prompts: "Is Λ a theorem?" (expect no, Conjecture 1), "How many locked formulas?" (expect 8) and "Joules for the last inference?" (expect SAMPLE or UNAVAILABLE). Show only output from a model that actually ran.
- Do not say: "trained", "live", "SOTA", "production model", "54 hours completed", or any accuracy or energy number without a receipt hash next to it.

## 4. Report template (paste into each update)

```
SZL TRAINING REPORT — <UTC timestamp> — reporter: <bot id>
Fleet truth source: <Salad API | portal | UNAVAILABLE (reason)>
Chain head: frontier-54-runs @ <commit sha> (<age>)

LEG               | HW     | VERDICT                | LAST RECEIPT (sha, age) | ACTION
Frontier-54 v2    |        | MOVING/STOPPED/UNKNOWN |                        |
Brain v3.5        |        |                        |                        |
Brain-2c          |        |                        |                        |
Nano-2            |        |                        |                        |
Collective        |        |                        |                        |
Salad lanes(each) |        |                        |                        |
Nemotron v3       | bridge | EXPIRED_AWAITING_ENGINE_SIGNATURE (a17) | — | owner signature needed
Laptop sovereign  | OMEN   | NOT RUN / <loss, VRAM, eval pass %> | provenance: MODELED|signed

Energy: <J MEASURED (NVML delta, receipt sha)> | UNAVAILABLE
New BLOCKED receipts: <job id → reason>
Open owner decisions: <list>
Labels check: no MEASURED without receipt ✔  locked=8 ✔  Λ=Conjecture 1 ✔
```

## 5. Owner-only decisions (bots must not decide these)

- Signing any job (engine key), and Nemotron attempt 17 or later.
- Rotating credentials. HF, Cloudflare, Salad, Massed and node passwords are all marked transcript-exposed.
- Promoting or publishing any model, model card or dataset.
- Admitting MUHU to the flagship training set.
- Anything touching .31.

## 6. Change log

- 2026-11-20: corrected the runbook §5 Nemotron row (was overclaiming "SETTLED — attempt-17 terminal receipt ✅"); the stricter README/NEMO_V3_EXECUTION state (EXPIRED_AWAITING_ENGINE_SIGNATURE, issue #93) is authoritative. This file is the canon handoff for bot intake.
