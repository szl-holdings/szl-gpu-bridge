# SZL-Nemo v3 governed GPU job

This bridge path executes exactly one DSSE-signed, immutable SZL-Nemo v3 attempt on the owner GPU host.

## Fixed boundaries

- `szl-holdings/a11oy` source is pinned to an immutable 40-character commit.
- Base model identity, revision, license, and explicit license acknowledgement are pinned.
- Training accepts only independently project-authored `TRAIN` records.
- Original v2, shadow v2, and preregistered v3 challenge suites are downloaded as separate immutable files and never passed to the trainer.
- Holdout file hashes, byte counts, record order, and record-ID digests are exact gates.
- VRAM, disk, utilization, temperature, wall-clock, row count, and non-finite-loss gates fail closed.
- Every holdout must pass and no generation may be degenerate.
- Candidate upload, publication, deployment, signing, and promotion are disabled.
- The terminal result is either `EVALUATION_FAILED_NOT_PROMOTED_NOT_SIGNED` or `QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW`.
- A failed attempt is not retried automatically.

The model remains an NVIDIA-based downstream adapter candidate. NVIDIA attribution and the upstream model license remain explicit and unchanged.

Signed-off-by: Stephen Lutar <stephenlutar2@gmail.com>
