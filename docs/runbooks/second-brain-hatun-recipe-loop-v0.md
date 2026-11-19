# SZL TufaZip × CRAM × Second Brain × Hatun — bounded, source-only model loop

**Status:** RESEARCH / OPERATOR REVIEW; not live training or a deployed Hatun integration.
**Owner:** `szl-holdings/szl-gpu-bridge` (dispatch/evidence), exact source baseline
`23b45e1aa628b273108faeca59c02f6ef031b88c`.
No current fleet controller, image, workflow, model weights, branch rules, route, or provider was changed.

## What this adds

`tools/proving/model_recipe_loop_v0.py` consumes (1) this public 11-model
revision-bound roster, (2) a **private** captured unauthenticated Hugging Face
metadata observation, (3) a private read-only Second Brain/Hatun dependency
observation and (4) optionally the exact user-generated synthetic **train-only**
ZIP from the SZL TufaZip–CRAM proving ground.

The ZIP is admitted only at these exact identities:

- ZIP SHA-256: `4216b3e1130d3cba49f05fe9456d9e2ee936a9eb87ab80639698f0b58caa3c4d`
- Manifest SHA-256: `d8cb77a517e69ded7314747df5e68db3481aaba3639a8529ed1e3eb47c448904`
- `train_sft.jsonl` SHA-256: `36048599efb5abc1b4ca0ef21ac1eae072c7412728f1dacd9cbe437b623df502`
- 1,320 train examples only; 310 dev and 290 test stay outside the ZIP.

**Even exact bytes do not grant training admission.** The `szl-cram` PR #11
verifier is the source owner for independent gold-answer checks.
The original ZIP is a conversation attachment, not a GitHub/HF artifact.
No file or hash proves that an independent model evaluation ran.

11 candidate model identities are drawn from the source-owned Hatun model
registry plus Khipu Scholar/ReceiptAgent cards. Their pinned Hub commits
are **metadata observations**, not proof that model weights, base adapters,
licenses, endpoints, tokens, latency or GPU training are qualified.

## Read-only run

```bash
python -m tools.proving.model_recipe_loop_v0 \
  --plan recipes/reasoning_compression_v0/models.json \
  --hf-observation /private/hf-public-metadata.json \
  --dependency-observation /private/dependency-public-readback.json \
  --train-only-zip /private/SZL_TufaZip_CRAM_TrainingOnly_v0.zip \
  --outdir /private/receipts/model-recipe-new-unique-name \
  --cycles 2 --max-steps 12
```

No automation or perpetual process is installed. Maximum 3 cycles, at most 12
models/steps per cycle. Each cycle **replays the same snapshot** and emits an
append-only local JSON receipt with SHA-256 previous-hash link and a status SVG.
`COMPLETED_BOUNDED_SCAN` means the scan stopped; it is not model convergence.
Successful scanning returns CLI exit **2** because admission remains HOLD.

Do not pass secret tokens, private source texts, Hatun credentials or signed
receipt payloads to this public configuration. Do not copy private manifests
to public branches; the example roster contains public model names and SHAs only.

## Observed service gates on 2026-11-19

- Second Brain Space provider stage: `RUNNING`, provider Space revision
  `8baec29ec65260a0438ca4f57c140e4209ad73e4`.
  Unauthenticated `/health` and `/readyz` both returned **HTTP 404**.
  This is **not ready** despite the Space stage.
- Hatun's `https://a-11-oy.com/wires` product route returned **HTTP 403**
  to an unauthenticated observation. It is NOT the canonical MCP runtime.
  The separately admitted `/mcp/` origin, authenticated `tools/list`,
  `/readyz`, signer/key, Khipu DSSE envelope and tenant gates were **not verified**.
- No independent runtime model inference, paired heldout evaluation, training
  checkpoint or signed receipt was observed in this task.
- Do not hydrate Second Brain's private 9,464-node graph; public 575
  chunk handles remain review-only. Future model-facing retrieval must
  preserve exact handle/digest and the controller-only hydration boundary.
- Hatun's canonical loop is `hatun_mcp.loop.run_bounded_loop`; this
  observer is a source-separated evidence scan, not a second MCP service.

## Admission that remains

1. Source-owner review and signed/DCO branch protection, exact upstream
   source pins, and restored source→HF runtime alignment.
2. Second Brain `/readyz` and reviewed public corpus/checks **on an admitted
   source revision**, not just provider `RUNNING`.
3. Hatun authenticated `/mcp/` tool list, Yuyay policy, valid DSSE signing,
   scoped operator and receipt readback. Never activate on product-route 403/200.
4. The **actual** model weights + adapter/base identity, per-model task and
   rights, heldout contamination exclusion, finite quality metrics, token/compute,
   energy, checkpoint durability, budget, parallel owner leases and exact image.
5. Only then consider wiring this roster into the **existing**
   `tools.salad.training_observer` / `fleet_controller`—without overriding
   another operator, speculative replica restarts, or global training enable.

**Safety:** zero GPU jobs, zero new automations, zero model requests,
zero paid provider writes, zero publications, zero brain private-row reads.
