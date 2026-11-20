# Stack400 — actual public Salad canary versus 54-hour training admission

**Readback:** 2026-11-20 UTC. **Verdict:** SOURCE RECEIPT CONSISTENT / PROD HOLD.
This report does **not** authorize a new Salad group, a second operator or
paid GPU usage. A repository named `SZLHOLDINGS/stack400-runs` already exists.
The canonical owner of that worker's **executed GitHub source** has NOT been
proven; creating another fleet from the same account would risk conflict.

## Actual pinned public observations

* Public HF metadata census: **401 model repositories, 173 dataset
  repositories** at 2026-11-20 14:07Z. The added dataset since the previous
  172 count is `SZLHOLDINGS/stack400-runs`; it is an **execution/evidence
  dataset**, NOT 401 new training examples. Fresh public metadata receipt
  SHA-256: `fb27f5d0518fc9f8a9eef40fdf83bb22101fed704a08ab08c2d09fefc41de2e6`.
  Previously observed model filename classes: 189 possible adapter listings,
  37 other recognized weight listings, 175 without a recognized weight file.
  Such suffixes do not prove trainability.
* Immutable HF dataset revision:
  `SZLHOLDINGS/stack400-runs@86db9b0257fe137f0525b7d5573b74f432f49871`.
  Its public `receipts/stack400-canary-r6/receipt.json` has SHA-256
  `7fa0b9698c9f07f0c855cce0ec27a416103db21ed5de441372f0833d594ac931`.
  Its `markers.txt` has SHA-256
  `498453af8da24605650cd4aabaefb0b837392e635e3487c0e1d30684a822eb4a`.
  Both were fetched via exact Hub revision response headers; no publishing.
* The canary's **self-reported** worker markers describe an RTX 3090,
  `Qwen/Qwen2.5-0.5B`, 12 LoRA steps, loss 4.7365 -> 0.0759,
  nonzero base/adapted logits, 3 sample generations and synthetic
  evaluation score 1.0. The public receipt's `recorded_hash` equals
  SHA-256 of compact sorted JSON excluding `recorded_hash`. This is a
  self-hash, **NOT** a verified digital signature or trusted GPU attestation.
* Public dataset `workers/canary_worker_v7.py` later appeared at the
  same pinned HEAD, byte SHA-256
  `744d673bcffd4258e5fa7fe9e440b63669f4822e0ad145783e0993b5e0795a7e`.
  **Do not claim v7 ran the older r6 receipt.** The receipt does not bind
  an immutable GitHub worker source SHA or an independently verified
  base-model commit. It labels the base `@snapshot`, not a SHA40.
  The adapter artifact bytes and an independent held-out set are unavailable
  from the inspected public receipt path. A public marker that says
  `UPLOAD_OK` establishes neither artifact availability nor publication
  authority.

## Actual executable verification, no provider mutation

`tools/salad/stack400_receipt_verifier.py` is a **network-free, source-only**
independent consistency checker for the public r6 canary format. It validates
strict/duplicate-free JSON, the canonical receipt self-hash, declared file
paths, worker step/loss/probe/eval marker consistency, nonce/run identity,
and no-op quarantine. It requires a completed marker path but still returns
`REVIEW_ONLY_HOLD` with
`signed_receipt_verified=false`,
`gpu_instance_verified=false`,
`adapter_bytes_verified=false`,
`independent_holdout_verified=false`,
`54h_training_complete_verified=false`, and `reward_eligible=false`.

On BETTERWITHAGE the exact public r6 bytes passed consistency checks.
The resulting **unsigned create-only LOCAL** review artifact SHA-256 is
`f99259c40d62ee9e9f4c95c3d5d240df42706e1b7d3dd4856ace9bbd55bca61b`.
It states **12 worker-claimed steps, not 54 hours completed**. Live provider
fleet instance authenticity and cost were NOT verified by this code.

Offline reproduction with user-supplied, previously pinned public files:

```sh
python -B -m unittest discover -s tests -p test_stack400_receipt_verifier.py -v
python tools/salad/stack400_receipt_verifier.py \
  --receipt PATH_TO_IMMUTABLE_RECEIPT \
  --markers PATH_TO_IMMUTABLE_MARKERS \
  --output NEW_CREATE_ONLY_REVIEW_REPORT
```

The script never imports Salad/HF SDKs or uses credentials. Never equate
a matching content hash with signature trust, or a worker's logged GPU model
with an authenticated provider instance.

## Existing owned stack, not a duplicated paid dispatcher

Existing `szl-gpu-bridge` is the **only** candidate source for a Salad
credential-holding executor. `tools/salad/admission.py` and
`tools/salad/admission_runner.py` already encode the owner lease, intent
ledger, HOLD/UNKNOWN outcomes, and single-writer constraints. Draft
[GPU Bridge PR #315](https://github.com/szl-holdings/szl-gpu-bridge/pull/315)
already covers the 401-model CPU-only wave preflight. Draft
[PR #313](https://github.com/szl-holdings/szl-gpu-bridge/pull/313) covers
the rights/identity portfolio. Do **not** create another orchestration loop.

A read-only public Salad GET to `alloyscape/default` without credentials
returned HTTP **401**, and no Salad key was in the observed BETTERWITHAGE
environment. A TinyFish browser profile for Salad is not signed in; the
connected Edge browser session is not accessible from this chat. This is
not current fleet inventory. The last source-backed provider readback
in [GPU Bridge issue #251](https://github.com/szl-holdings/szl-gpu-bridge/issues/251)
was **2026-11-18**: 140 listed groups, 39 desired replicas and 16
provider-reported running instances; none of those counts alone proves
a trainer running. Do not launch a duplicate `stack400` group or assume
that an existing container has free capacity.

## The owner-admitted 54-hour program

1. **Name the source owner and sole Salad executor.** Read back the
   existing group, exact instance state, currently billed replicas,
   available quota, image digest, worker GitHub source SHA, immutable
   base-model commit, and operator-controlled GPU lease. If the instance
   is running, reconcile it before any new create/reallocate/restart.
2. **Verify rights separately for each model+dataset pair.** Public card
   labels are not a universal training license; keep private data out of
   third-party providers absent purpose/consent clearance. 401 model repos
   are not 401 independently trainable checkpoints. Hatun should enforce
   permitted actions, Khipu preserve signed/append-only receipts, and
   SZL Lean check source governance invariants, not substitute for model
   evaluation.
3. **Use exact immutable run inputs.** Pin source image by OCI digest,
   repo+revision, model/base architecture and files, tokenization/masking,
   training/eval splits and separate held-out controls. Verify adapter
   output bytes by independent digest, DSSE/Sigstore signer and trust root,
   and a real heldout evaluator distinct from the synthetic prompt.
4. **Bound budget before start.** 401 x 54 hours equals **21,654
   GPU-hours** for one GPU per model, before checkpoint/eval/retries;
   a single 54-hour campaign is not that many model-hours unless 401 GPUs
   really run concurrently. As an **illustration only**, Salad's
   November 2026 public RTX 3090 Lowest price is about **$0.09/GPU-hour**:
   21,654 hours at that *tier* is about **$1,948.86** base GPU time.
   Actual models may require more memory, different GPUs or priorities;
   supply, preemption, storage, egress, retries and maximum liability
   are UNVERIFIED. Do not convert this public price example into an
   owner-approved spend cap.
5. **First authorized trainer action is a short verified canary, not 401
   replicas.** Restore checkpoint under a genuine provider-owned group,
   independently validate adapter bytes, reward verifier and heldout,
   then request **one** owner-signed 54-hour job with a firm deadline,
   maximum spend, abort switch and post-run readback. Advance waves only
   after separate explicit admission.

Neither a local model named "Astra" nor remote Kimi/Mistral evaluations
can serve as human approval, provide private foundation-model corpora,
or authorize Salad billing. One source-only verification improvement
was executed; no 54-hour provider job was created in this pass.
