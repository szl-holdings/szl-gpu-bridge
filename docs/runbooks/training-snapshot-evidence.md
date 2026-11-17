# Training snapshot and historical-evaluation contracts

The snapshot helper is read-only. It is not a trainer, scheduler, archive verifier,
cloud controller, publisher or production qualifier.

## One immutable observation

`model_snapshot` obtains one model metadata response with file metadata. File
presence, checkpoint names and the returned revision are derived from that same
response, not a later listing of a moving branch. Fetch trainer JSON using the
returned exact revision. `verified_trainer_state` checks byte size and Git/LFS
object identity before parsing strict JSON, then checks the directory step and
integer target. Duplicate keys, non-finite values, overflow floats and booleans
in integer fields fail closed. No pickle or remote model code is executed.

The helper reports filenames as metadata only. A target reached in trainer state
does not prove complete final exports, independent backups, model quality,
process liveness, deployment or safe shutdown. A partial latest checkpoint must
remain partial instead of silently falling back to an older checkpoint.

## Coverage

Command-mapped destinations take priority over name-selected research candidates.
Selection returns the exact eligible/selected/omitted counts and names. The caller
must separately record attempts, denied reads, unmapped jobs and inaccessible
providers; do not present a capped catalog as exhaustive. Stop the provider lane
on authentication denial or rate limiting. Training and inference are distinct.

Catalog disappearance means not returned under the observed scope. It does not
prove deletion, lost artifacts or successful retirement. Reconcile a retirement
receipt and its archival scope before making stronger claims. Partial listings
leave prior identifiers unassessed.

## Stop on denied provider access

The source-controlled bounded `tools/salad/training_observer.py` observer
treats Hub metadata/download 401, 403 and 429 and Salad instance-GET
401, 403 and 429 as **provider-lane stops** for all remaining candidates
and subsequent cycles in that same process. It records the HTTP status
without exception contents, and reports later reads as `UNAVAILABLE`
with an explicit stopped reason. A denial from one provider does not
authorize an alternate credential, endpoint or account; the other
independent provider lane may remain observable.

HTTP 200 or Salad's group status does not establish trainer-process
liveness, completion, artifact integrity, quality, or production
qualification. A non-denial 404 is not falsely interpreted as a
successful zero-instance result. The observer's periodic output
remains bounded and observer-owned; this source repair neither creates
another scheduler nor deploys, modifies or supplants the protected
owner-device one-shot release. Verify any release change with its own
hashes, integration checks and admission process.

## Historical evaluation correction

The source card under `hf/abstention-v2-candidate/` preserves an existing six-probe
receipt's 3 PASS / 3 FAIL record, while correcting an overbroad absence statement
from a different repository's receipt-directory scan. The receipt has no immutable
adapter revision; the historical result is not inherited by current weights.
The adjacent evidence JSON binds the observed prior card and receipt.

Publish the card only through an ordinary review and expected-parent write.
Preserve every sibling file and the original receipt; verify the returned commit
and exact README bytes. Do not relabel a card proposal as publication.

## Verification

Run `python -m unittest discover -s tests -p 'test_training_snapshot.py' -v`.
Thirty credential-free tests cover the moving-main false-final regression, pin
validation, partial/unknown metadata, hashes, strict JSON, typed counts, bounded
selection and catalog comparisons. These are software contract tests, not model
quality measurements. An owner-device one-shot observer integration was separately
compiled and exercised live; this source addition alone does not deploy an observer.

Rollback through the normal source-review process. Preserve historical observations
and do not restore a filename-only completion or shutdown rule.
