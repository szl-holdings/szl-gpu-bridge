# Canonical training image: source → signed build → observed runtime

**Status:** source-only design and static checks. **No cloud deployment, training, public publication, independently authenticated runtime, or approval is claimed.** Existing operator, maintenance, quota, signer and Flagship54 guard holds remain active. No new watcher or paid replica is created by this change.

## Verified gap and constrained repair

The former `training-image/salad_worker.py` was a separate, stale snapshot. It lacked the source-owned immutable checkpoint-resume verifier subsequently added to `tools/salad/salad_worker.py`. The image Dockerfile copied that snapshot, which meant a GitHub green state could still build an image **without the canonical resume guard**. It is retired from the working tree (still recoverable through Git history).

The root-context Dockerfile now copies **exactly one canonical worker**, plus `resume_integrity.py` and `training_snapshot.py`. The local progress heartbeat is an opt-in hook in the canonical worker, activated only when the approved pinned image sets `SZL_HEARTBEAT_FILE`; other existing training environments have no new local heartbeat requirement. The source-runtime contract records byte identities for worker/resume/snapshot/heartbeat/probe/uploader at image build and verifies them again at image entry before model imports.

The source manifest is **unsigned local integrity evidence**. It cannot authenticate the container builder, trusted source authority, private data rights or the operator. A malicious actor who can rewrite both the manifest and files could still evade an in-container checksum. External OCI digest and Sigstore identity/provenance verification are required at admission.

## Independently verified convergence gates

| Gate | Required identity/evidence | This PR proves |
| --- | --- | --- |
| 0. Authorized source | Protected GitHub `main` SHA, reviews, clean checkout, exact worker/helper digests | Source contract and tests only |
| 1. Image provenance | GHCR `@sha256` digest with Sigstore signer identity and SLSA provenance subject matching gate 0 | **NOT OBSERVED for a new image** |
| 2. Image contents | Immutable image contains canonical worker+resume modules; build and boot manifest bytes match | Source Dockerfile + synthetic offline checks; **no image build** |
| 3. Launch binding | Owner-approved recipe/base/dataset/target and lease; actual Salad group image digest equals gate 1; no competing writer | **UNOBSERVED** |
| 4. Process and progress | GPU process identity, UTC heartbeat age, actual advancing trainer steps and consistent desired/actual instance counts | **UNOBSERVED** |
| 5. Resumable checkpoint | Pinned full checkpoint with trainer/optimizer/RNG/scheduler/sampler bytes, correct step count, no unsafe deserialization | Source resume guard present, **live bytes unverified** |
| 6. Preservation | Independent storage readback of exact checkpoint/final artifact SHA and documented restore exercise | **UNVERIFIED** |
| 7. Evaluation | Correctly bound weight revision, held-out rights, corpus/denominator/baseline, fresh measured accuracy and abstention, contamination controls | **UNVERIFIED** |
| 8. Deployment and publication | Independent served source/witness, auth, rollback, signed owner approval; public readback with correct proof tiers | **UNVERIFIED / HOLD** |

**Convergence predicate:** `QUALIFIED = source && signed_image && actual_deployed_digest && trainer_progress && immutable_checkpoint && independent_backup && revision_bound_eval && authorized_serving && approved_public_readback`. All are separate evidence-producing gates, not one string badge. A false/unknown mandatory factor means `HOLD`. A successful GitHub Actions run, 200 HTTP, running container or static manifest is **not** equivalent to proof of training.

## What still needs its owner

1. Verify exact image digest, signer and SLSA builder on a reviewed protected-main build. The existing `training-image-ghcr.yml` is **build/sign infrastructure, not a deployment controller**; its current branch-only push and manual dispatch rules are not expanded here.
2. Before any Salad create/update, use the existing authorized owner and configured authentication to compare expected image digest with actual provider group/instance data. This PR changes no running group or published image. Preexisting mutable image references remain unqualified.
3. Preserve and compare full resume-state bytes across an owner-approved preemption/restart canary. Do not infer successful restart from a file named `checkpoint-*` or a heartbeat from an inference server.
4. Restore the separate, approved SHA-pinned one-shot observer and Gate-1 host-key-verified helper through their original owner, not an ad-hoc clone, unsigned script or second scheduled task.
5. Keep the Flagship54 single-lane guard as the only Flagship54 shutdown owner. The standalone stop-preflight draft PR #274 does not replace it. Reconcile historical ambiguous PATCH outcomes by fresh GET before any retry.
6. Admit model cards and public proof only after immutable commits, independent readbacks, baseline and denominator checks, dataset rights and trusted receipt roots. Never copy private checkpoints, credentials or incident details to a public repository.

## Why this architecture follows the leaders

- [Argo CD's desired/live/sync/health distinction](https://argo-cd.readthedocs.io/en/stable/core_concepts/) informs separating source equality from observed health and operational readiness; this does **not** install Argo CD.
- [SLSA build provenance](https://slsa.dev/spec/v1.2/build-provenance) provides builder/subject/source dependency semantics. [Sigstore Cosign verification](https://docs.sigstore.dev/cosign/verifying/verify/) shows how to check image identity and signer, rather than treating a checksum as a signature.
- [NVIDIA Megatron Bridge checkpointing](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html) distinguishes full optimizer/RNG resume from initializing weights only. [Ray Train checkpoint guidance](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html) separates process/node/driver fault recovery from durable data and explains rank ownership.
- [verl/HybridFlow](https://github.com/verl-project/verl) motivates a later, *separately approved* verifiable-reward, source-bound evaluation path. [AIware Lab at Lero](https://aiware-lero.github.io/) motivates adversarial testing against flattering but invalid benchmarks. [KAIST KORMo](https://kaist-mlplab.github.io/kormo/) demonstrates releasing reproducibility ingredients beyond weights.

The distinctive SZL research opportunity is a typed **proof-carrying convergence graph**: every claimed source-to-runtime edge requires its own immutable, authenticated evidence and expiry, with Lean proof status, governance receipts and the experimental Λ advisory kept distinct. That is a falsifiable research direction, **not** a new proved theorem or deployed system. The next step is an operator-owned canary and independent runtime/readback receipts, not more optimistic automation.
