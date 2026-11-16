# Salad worker terminal-evidence contract

This contract concerns `tools/salad/salad_worker.py` receipt and terminal behavior, not model
quality, fleet authorization, checkpoint durability or public release admission.
Adopt a new worker through the existing source-owner/controller process. Do not
restart active training merely to adopt this change.

## State and publication

After `trainer.train()` returns, `TrainerState.global_step` must be an integer
(not a boolean, string or float), nonnegative, and exactly equal to `max_steps`
before final-adapter publication. The target is not a measurement of completed work.

An interrupted, early-stopped, overshot or unknown-step run returns 7, emits
`interrupted` or `incomplete`, records `actual_steps` (null when unusable),
`target_steps` and `completion_qualified: false`, and closes the energy window.
It does not save or upload a new final adapter. Checkpoint byte durability is
not independently certified by this terminal guard.

An exact-target run retains private-visibility checks and final upload. Only
after that path succeeds does it emit `complete`, with `steps` and `actual_steps`
from Trainer state, plus requested `target_steps`. Its completion scope excludes
model quality. A signal at the exact target does not erase already-observed work.
Upload failure and lost visibility must not produce successful completion.

A checkpoint already at/above target does not start more training. An unavailable
or missing final adapter returns 7; a verified-present final permits the existing
no-training exit. This is metadata evidence, not a new training result or an
independent verification of artifact bytes. Nonpositive step/save/example bounds
and a cap below `min_examples` are rejected before provider operations.

## Receipt envelope

The existing `kind: szl.salad-worker/v1` marker stays. A separate `event` records
the lifecycle event instead of discarding it. Extra fields cannot replace `kind`,
`event`, `detail`, `ts` or inject an old `receipt_sha256`. The hash covers the
actual envelope including the event. A hash is not a signature. Historical
receipts remain unchanged and are not retrospectively relabelled.

Receipt publication is required. The worker retains the exact candidate bytes,
uploads them, requires the returned 40-character Hub commit, and downloads the
same path at that commit with `force_download=True`. It compares the entire file
before acknowledging publication or printing the successful event. A failed
upload, missing commit, mismatched bytes, failed private-channel visibility check,
or unavailable readback returns exit **8**. Raw provider exception strings are not
logged because they may contain credentials or signed URLs.

The worker emits `szl.salad-receipt-publication/v1` acknowledgement metadata with
repository, type, path, commit, file SHA-256 and the envelope's `receipt_sha256`.
The full digest appears in the remote filename to avoid the old truncated-path
collision risk. The two hashes have distinct scopes: the envelope hash excludes
its own field; the file hash covers the exact uploaded bytes. Neither is a
signature, freshness certificate, quality result, or proof of checkpoint bytes.

Boot publication is an admission gate before model loading/training. The same
token passed by `--hf-token` or `HF_TOKEN` authenticates artifact and receipt I/O;
passing only the CLI token no longer silently disables receipts. Upload and
readback each allow at most three attempts, with 5s and 10s delays. Authorization,
not-found and other non-transient 4xx responses stop immediately. Readback retries
do not repeat a successful upload. The existing Hub client's request timeouts
still apply; the retry delays are not an overall wall-clock deadline.

### Explicit receipt destination

The default remains the existing `SZLHOLDINGS/frontier-54-runs` dataset. An owner
may explicitly select an existing private model repository already writable by
the admitted token, including the lane's own adapter repository:

```text
--receipts-repo SZLHOLDINGS/<existing-private-lane-repo>
--receipts-repo-type model
```

The equivalent container environment variables are `SZL_RECEIPTS_REPO` and
`SZL_RECEIPTS_REPO_TYPE`. Pass credentials through the controller's existing
secret environment mechanism, never inline in a command, contract or receipt.
The Docker entrypoint consumes these worker arguments directly; the separate
benchmark launcher has its own receipt contract and is unchanged.

There is no automatic fallback, repository creation for the receipt channel,
visibility mutation or credential-scope widening. Model channels are verified
private before upload and after readback; an observed visibility change fails
closed. This cannot prevent a separate actor from changing visibility between
checks. Dataset destinations retain their explicitly selected visibility.

### Heartbeats, interruption and terminal recovery

`--heartbeat-seconds` defaults to 300 and must be positive. A heartbeat is emitted
at the first Trainer step boundary after that interval; a long or hung step is
not covered by a background timer. It records observed step count and explicitly
claims liveness, not checkpoint durability. A failed heartbeat or checkpoint
receipt asks the Trainer to stop and save. The worker then refuses final
publication, including when the exact target happened to be reached. The
controller must reconcile and must not blindly retry a process exiting 8.

Checkpoint upload failures also request stop rather than allowing unbounded
unreceipted progress. Checkpoint receipts record the Hub-acknowledged checkpoint
commit. They do not claim full checkpoint byte readback. SIGTERM/SIGINT handlers
only set the stop flag; blocking receipt I/O runs outside the signal handler. The
next Trainer step boundary requests save and stop, subject to provider grace time.

An exact-target final receipt records the final adapter's upload commit and its
`final/` directory. A terminal receipt failure after a successful adapter upload
returns 8 and leaves the real uploaded artifact intact, with no printed `complete`
event. It does not mean the artifact was rolled back or the training did not run.
The controller must inspect that commit and the current run's pending evidence;
retraining or fabricating a retrospective completion receipt is not recovery.

The default local outbox is `~/salad-receipt-outbox/<run-id>/`. Overrides resolving
inside the `~/ckpt` artifact tree are refused, so recovery records cannot be swept
into checkpoint or final adapter folder uploads. Files ending in
`.pending.json` retain the exact bytes attempted, including on ambiguous transport
failures; `.publication.json` records successful readback. A pending file remains
even after acknowledgement. Local files are not guaranteed to survive spot-node
loss, are never automatically replayed, and never certify publication on their own.
This guard grants no restart, resource-reclamation or shutdown authority.

Final or optional adapter upload failures return exit **9**, without printing raw
provider exception bodies. They never emit successful completion and must be
reconciled as potentially ambiguous writes, not automatically retried training.

## Separate image and kernel gate

Validate Python against exact package metadata and the PyTorch CUDA build against
the extension-building toolkit before admitting a fast-kernel image. A failed
kernel import is not a performance result. A shell ending in `true` cannot make a
failed training command successful. Require a reviewed pinned image and its own
smoke-test receipt before separately authorized compute; do not select versions
or rerun training merely to obtain green status.

## Software tests

`tests/test_salad_worker_terminal.py` is discovered by existing root unittest CI.
It exercises real worker branches with synthetic provider/model/dataset/Trainer
objects, denied socket connections, and no credentials, GPU, weights, corpus or
live Hub writes. Tests cover denied boot publication, missing/mismatched commit
readback, bounded retries, token propagation, explicit private channels, visibility
loss, heartbeat/checkpoint stop-and-save, final-receipt failure after real artifact
publication, and successful commit bindings. Positive controls retain exact-target
completion, no-retrain behavior and private-upload checks. This is not
hosted-runtime qualification.
