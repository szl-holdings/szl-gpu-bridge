# Salad terminal checkpoint recovery — F10 clean-staging contract

**Scope:** Existing `tools/salad/salad_worker.py` v3.10 F10 recovery of an
already-trained terminal checkpoint after preemption. This is not a new
trainer, scheduler, checkpoint controller, budget, replica, remote endpoint,
model promotion, or proof of model quality.

## Incident class and source-level gap

On 2026-11-18, a Salad lane finished its admitted maximum steps but
preemption prevented its final adapter from appearing under Hub `final/`.
F10 already fixes the retry wedge by publishing non-pickle terminal
checkpoint files without retraining. However, its previous source copied
those files into an existing local `output_dir/final/` directory and then
uploaded the **whole directory**. A stale file from a previous run could
therefore be sent alongside the current checkpoint. Also, after its first
visibility check/listing, the destination repository could change from
private to public before uploading.

## Corrected source contract

- A checkpoint must be a real directory, not a symlink, with a regular
  `adapter_model.safetensors` and only individually validated regular
  members; a symlink member is a hard refusal, not an upload candidate.
- The worker stages only the current checkpoint's non-pickle files in a
  uniquely created temporary directory under its already admitted output
  tree. **No pre-existing `final/` file is read, overwritten, deleted, or
  uploaded.** Python removes the worker-created staging directory after
  the synchronous upload returns or fails.
- Destination repo visibility is reverified immediately before preparing
  files and again immediately before upload. If private visibility is
  lost, the worker fails closed without transferring a checkpoint
  (`VisibilityError`, exit 3). Missing/invalid checkpoint material is
  still a terminal recovery failure (`ResumeError`, exit 7); upload
  failure remains an artifact publication failure (exit 9).
- The restored receipt retains exact terminal step count, zero new steps,
  selected checkpoint filename and exact Hub commit returned by the
  upload. It labels its verification honestly:
  `HUB_COMMIT_ACKNOWLEDGED_NOT_FULL_BYTE_READBACK`.
- Existing pickle exclusions and receipt-channel publishing semantics
  remain unchanged. The module does not open or deserialize optimizer,
  scheduler or RNG pickle files.

## Validation

The offline, socket-denied `tests.test_salad_worker_terminal` suite
includes synthetic cases for stale-file isolation, existing-directory
preservation, symlink refusal, visibility changes immediately before
publication, failed uploads, missing weights, and true zero-step
terminal recovery. The test harness snapshots staged bytes *during*
the upload call and verifies temporary cleanup afterward; no real
Hugging Face, GPU, Salad or patient/model data is accessed.

## Separate release/production gate

A Hub commit acknowledgement is **not** independently read-back
safetensors bytes, license clearance, completed science evaluation,
signature/DSSE validation, actual live Salad service version, or
production-grade durability. A separate authorized owner must:

1. Approve the exact source deployment under existing versioned worker
   image and protected CI gates, without restarting running jobs.
2. Read back the final adapter files at the immutable returned Hub
   revision, hash their actual bytes, and compare to the terminal
   checkpoint byte manifest.
3. Confirm dataset/base/model rights, independent held-out evaluation,
   actual trainer target and receipt chain before publishing a
   qualified model card or winding down paid replicas.
4. Preserve worker guard and operator ownership; do not launch duplicate
   training or mutate active replica counts from this source PR.

Rollback is a normal reviewed GitHub PR; reverting reopens the stale
directory and destination-visibility publication gap.
