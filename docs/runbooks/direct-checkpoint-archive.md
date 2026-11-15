# Direct pinned-checkpoint archive

Use this one-shot recovery path when Gate-1 transport is unavailable. It reads the
existing Hugging Face login and writes only to an explicitly selected local archive.
It never controls Salad, restarts training, loads model code or deserializes pickle.

## Invocation

```sh
python tools/salad/direct_checkpoint_archive.py --revision FULL_COMMIT_SHA --step STEP --root ABSOLUTE_PRIVATE_ARCHIVE_DIRECTORY
```

The repo is deliberately fixed to the existing Flagship54 lane. The target remains
37,000 steps. Passing a different target, public/unknown repository, moving branch,
incomplete checkpoint, foreign output path, corrupt existing bytes or writer lock
holds the operation. Exactly eleven allowlisted checkpoint files are required.

A byte-verified trainer state must match the requested step and target. The complete
checkpoint is downloaded at one immutable revision, each file checked against its
Git/LFS object identity, copied atomically, and rehashed. Existing conflicting data
is not overwritten. A manifest is committed only after every file verifies.

The utility is not a scheduler. Do not start another copy while its lock exists.
Use one writer per destination and inspect the prior process/outcome after timeout;
do not delete an unfamiliar lock or infer failure from a client timeout alone.

## Privacy and boundaries

Choose an owner-controlled directory. The tool rejects symlink/junction ancestors;
it does not change Windows ACLs or claim to audit all inherited permissions.
The Hub cache is separate from the evidence copy and must not be edited.
Artifacts remain private. Hashes are integrity digests, not signatures or outside
witness attestations. No cloud mutation, scale-down, paid job or public release occurs.

Success proves the pinned Hub bytes and owner-device copy. Gate-1 is reported
UNOBSERVED by this independent tool; verify that copy separately. It never claims
completed training, model quality, final-export integrity or production readiness.
Do not reuse this one-lane utility for unrelated jobs without a reviewed contract.

The download budget is 1 GiB plus disk reserve. The elapsed budget is checked
between files; individual provider requests have their own timeout/retry behavior,
so it is not a hard wall-clock deadline. No checkpoint-retention deletion is done.

## Tests

```sh
python -m unittest discover -s tests -p 'test_direct_checkpoint_archive.py' -v
```

These are credential-free contract tests. They cover pinning, privacy, exact file
sets, target binding, byte corruption, incomplete downloads, conflict preservation,
lock ownership, destination safety and repeat readback without redownload.
Live archive observations are separate from mocked tests.

Provider reference: https://huggingface.co/docs/huggingface_hub/guides/download
