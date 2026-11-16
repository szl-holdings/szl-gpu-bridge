# Salad worker terminal-evidence contract

This contract concerns `tools/salad/salad_worker.py` terminal behavior, not model
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
It does not save or upload a new final adapter. Existing checkpoint callbacks
remain unchanged; their durability is not certified by this terminal guard.

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

The existing best-effort receipt-upload policy is unchanged. Stdout, local
hashing, adapter upload and durable receipt publication are distinct observations.
This guard grants no retry, restart, resource-reclamation or shutdown authority.

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
live Hub writes. Positive controls retain exact-target completion, no-retrain
behavior and private-upload checks. This is not hosted-runtime qualification.
