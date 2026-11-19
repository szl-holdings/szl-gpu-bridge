# Golden Lane Recipe v2 — first-pass doctrine (receipt-codified)

Status: MEASURED doctrine. Every rule below exists because a receipt recorded a
lane dying (or passing) without it. Nothing here is aspirational.

Owner: szl-holdings / SZLHOLDINGS estate. Companion gate: `recipe_lint.py` in
this directory. Companion rehearsal tool: `heldout_practice_bank.py`.

## 0. The goal: first-pass success, honestly

A lane "passes the first time" when it boots, trains to its exact target,
publishes receipts, and gets parked, with zero human intervention. The estate
receipts already contain a measured fix for every class that ever killed a
first pass. This document is that ledger, codified.

**The hard line:** receipts teach the RECIPES and the WORKERS to pass first
time. They must never become training data for a MODEL against the thing being
scored. Training on live probe rows converts a pass into a fabrication and
defeats the contamination gate (c2) that the heldout battery exists to run.

## 1. Measured failure classes and their standing fixes

| Class | Receipt evidence | Standing fix (mandatory in v2) |
|---|---|---|
| Boot fetch: CURL_FAIL instant post-pip | diag logs 2026-11-19 (`CURL_FAIL` same-second as attempt; pip had just downloaded fine) | Never `curl` in worker commands on pytorch runtime images. Fetch kits with `hf_hub_download(repo, path, repo_type='dataset', token=os.environ['HF_TOKEN'])` + explicit `KIT_FETCH_OK/FAIL` markers |
| Create 400 `container_group_update_exception` | probe-proven 2026-11-19: auth-header strings in the canned create command | Trivial command, no environment at create; patch command/env after create. Never token-bearing curl/bearer strings in a create body |
| SFTConfig contract unsupported on trl 0.15-0.19 | ring family boot loop (PR #279 fixes the worker) | Pin a known pin-set (see §2). The worker fail-closes with a receipt on unknown contracts — never bundle pins ad hoc |
| pypi read-timeout during boot | cram-sudoku-ext-r1 boot-1, INNER_EXIT 1 | `PIP_DEFAULT_TIMEOUT=45 PIP_RETRIES=2 PIP_BREAK_SYSTEM_PACKAGES=1` + resolv.conf fixup; single-boot fragility self-heals via platform restart, but the block is mandatory |
| HF 320/hour commit quota 429 | stage0 diag 429×3, receipt storms, ring commit churn | Batch uploads (folder commits), relocate hot rails when a repo is saturated, never fire eval-class containers into a hot quota window |
| Instance churn with no run = queue, not recipe | census doctrine (repsum vs quota) | Park deadweight before blaming the spec; one goal = one active group; dedup-park the loser |
| Off-pin un-receipted relaunches | r3 720-char command, zero pin markers (parked per precedent) | recipe_lint gate refuses any command without a full allowlisted pin-set |
| Park discipline | 415 merge-patch lesson; PATCH-alone does not self-start; always-restart respawn of completed lanes | POST stop → poll stopped → merge-patch replicas 0 → READBACK; completed lanes MUST be parked promptly or they burn a slot per ~10-min cycle |
| Resume lies | F6/F9/F10 worker guards | Never claim already-complete without reading trainer_state.json; terminal checkpoints republish final/ |
| Fabricated quality | LoRA-no-op quarantine; `heldout_quality` gates | base==adapter measures get quarantined, never adjudicated PASS; no receipt = no claim |

## 2. Golden create-spec (mandatory markers)

`recipe_lint.py` checks all of this mechanically. Do not fire a Salad group
whose spec does not lint PASS.

1. **Pin-set (choose one allowlisted set, every member `==`-pinned in command):**
   - MODERN: `transformers==5.22.2 peft==0.21.2 huggingface_hub==0.30.2 pynvml==12.0.0`
   - LEGACY: `transformers==4.51.3 datasets==3.5.0 peft==0.15.2 trl==0.17.0 accelerate==1.6.0 huggingface_hub==0.30.2 safetensors==0.5.3 hf_transfer==0.1.9 pynvml==12.0.0`
2. **Resilience block:** resolv.conf rewrite (`1.1.1.1` + `8.8.8.8`), `PIP_DEFAULT_TIMEOUT`, `PIP_RETRIES`, `HF_HUB_ENABLE_HF_TRANSFER=0`.
3. **Fetch:** `hf_hub_download` (token from env) with `KIT_FETCH_OK/FAIL` markers. No `curl` anywhere.
4. **Auth:** `HF_TOKEN` via environment, never a literal in the command; no `Authorization`/`Bearer`/`Salad-Api-Key` strings.
5. **Evidence:** worker fetched main-first from szl-gpu-bridge (`tools/salad/salad_worker.py`); stage0/boot receipt; heartbeats (`--heartbeat-seconds` or an HB loop); final receipt is mandatory; `INNER_EXIT`/`EVAL_EXIT` markers before exit.
6. **Park watcher:** every lane ships with (a) churn watcher (machines≥3-4, no evidence ⇒ park, receipted), (b) completion watcher that $parks$ the group after the final receipt (stop → merge-patch 0 → readback).
7. **DisplayName** must match `^[ ,-.0-9A-Za-z]+$`.

## 3. Rehearsal: pass the battery the first time, without contamination

Before a lane's live heldout battery, run a **dress rehearsal** against a
fresh-seed practice bank (see `heldout_practice_bank.py`):

- Practice rows are format-matched to each criterion (refusal markers,
  neutral MCQ answer pattern, invented-ID abstention, detector calibration
  rows) but **content-disjoint** from the live embedded probe sets.
- A new seed is required per rehearsal; the manifest records every row's
  sha256 and the disjointness check against the live probe file
  (`--disjoint-from`), fail-closed.
- The live probes are generated at eval time and never used for training.
  A pass is only claimed on the unseen live battery.

## 4. First-pass KPI (measure it from receipts)

first_pass_success = lanes that reached (final receipt + park) with zero
manual intervention / lanes fired. Current honest scorecard from the estate
receipts: pg54 mesh 5/5 first-pass, abst-curriculum first-pass,
flagship54-v1 heldout passed first time (2026-11-19), cram needed one
self-heal boot, ring family never first-passed (fixed by PR #279 pending
merge). Report the number, never the narrative.
