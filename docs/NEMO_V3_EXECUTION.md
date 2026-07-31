# SZL-Nemo v3 — secure single-attempt execution

This runbook completes the human-controlled boundaries that cannot be delegated to a public repository: enrolling the owner GPU host receipt identity and authorizing the reviewed job with the offline engine private key.

The repository already contains the reviewed plaintext job at:

```text
jobspecs/nemo-v3-20260722-reviewed.json
```

A plaintext jobspec is **not executable**. The bridge accepts only a DSSE
envelope signed by a private key corresponding to the sole `ACTIVE` entry in
the reviewed engine keyring. Historical public pins remain available only to
verify immutable evidence.

## 1. Install or refresh the owner GPU bridge

On the intended Windows/NVIDIA host, from an elevated PowerShell session:

```powershell
irm https://raw.githubusercontent.com/szl-holdings/szl-gpu-bridge/main/laptop/bootstrap.ps1 | iex
```

The installer opens no inbound port. It pins the engine public key, generates a host-local Ed25519 receipt key, verifies the training environment, installs the dispatcher and Nemo v3 runner, and starts a single-flight polling task.

Record the exact 16-hex value printed as:

```text
>>> ANNOUNCE THIS KEYID TO THE CLOUD SESSION: <KEYID> <<<
```

Verify it locally before enrollment:

```powershell
Get-Content C:\szl-bridge\keys\laptop_pubkey.json
```

Set the GitHub Actions repository variable `SZL_LAPTOP_RECEIPT_KEY_ID` in `szl-holdings/szl-gpu-bridge` to that exact lowercase 16-hex key ID. The value is a public fingerprint, not a private key. Do not commit `laptop_key.pem`, copy it off the host, or paste it into an issue or chat.

Until that variable is enrolled, a mathematically valid receipt is reported as `AWAITING_LAPTOP_RECEIPT_KEY_ENROLLMENT` and is not trusted as an owner-host result.

Configure the repository Actions secret `HF_TOKEN` with read access to the
private `SZLHOLDINGS/szl-training-receipts` dataset. The status controller
fails closed as `RECEIPT_DISCOVERY_ERROR` when the token is missing, invalid,
or unable to see the authoritative receipt repository. A controller that
cannot inspect the receipt store must never report that no receipt exists.

The isolated launcher creates `C:\szl-bridge\control\attempt-claims\<jobId>.json`
atomically immediately before Docker starts. That durable claim binds the exact
signed job envelope, bridge revision, SHA-256 of the launcher from that clean
revision, image digest, and claim time. The running PowerShell script must hash
to the exact launcher in the reviewed bridge source; the digest is reproduced
inside the signed receipt stack. The claim's presence is the authoritative
one-attempt replay barrier: automatic dispatches fail closed and cannot start
the GPU job again, even if final receipt upload or terminal-ledger publication
is interrupted.

## 2. Review the exact attempt

From a clean checkout at protected `main`:

```powershell
python -m unittest tests.test_reviewed_nemo_v3_spec tests.test_nemo_v3_status -v
node --check cloud/sign-nemo-v3-job.mjs
```

Confirm these non-negotiable boundaries in the reviewed JSON:

- source: `szl-holdings/a11oy@a5351c8e37a7cfe54e0c3cf53c8bbd460a16c11c`;
- base: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16@dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`;
- project-authored training rows only;
- original v2, shadow v2, and v3 challenge suites excluded from gradients;
- exact hashes, byte counts, record order, and record-ID digests;
- 100% pass rate and zero degeneration required;
- no automatic retry;
- candidate publication, deployment, and promotion disabled.

Do not edit the jobspec after review. Any change requires a new job ID and a new review.

## 3. Authorize with the offline engine key

On the controlled signing machine:

```powershell
$env:SZL_QUANT_KEY = "C:\secure\engine_key.pem"
node cloud/sign-nemo-v3-job.mjs jobspecs/nemo-v3-20260722-reviewed.json
Remove-Item Env:SZL_QUANT_KEY
```

The command verifies that the private key matches the pinned public key before signing. It writes exactly one DSSE envelope:

```text
queue/pending/job-2026-nemo-v3-governed-attempt-1.json
```

Review the generated diff. It must contain only the signed queue envelope. Open a protected pull request; do not push directly to protected `main`, weaken checks, or expose the private key.

The Nemo v3 base sets `trustRemoteCode=true`. For this job, the ordinary scheduled
host process is intentionally insufficient: the runner refuses execution unless it
is inside the credentialless, networkless container lane described below. A normal
daemon poll leaves the job unconsumed and returns an honest local-policy failure.

An approved external dispatch lane must select this attempt explicitly. For jobs
that do not execute remote repository code, targeted daemon mode is:

```powershell
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass `
  -File C:\szl-bridge\daemon.ps1 `
  -OnlyJobId job-2026-nemo-v3-governed-attempt-1
```

Targeted mode refuses a missing job and never dispatches a different pending
queue entry. The scheduled task keeps its default all-pending polling behavior.

### Credential-separated container lane

Nemo v3 uses three separate trust zones:

1. **Authenticated prefetch** verifies the engine-signed envelope, downloads the
   exact model revision and the five hash-pinned project inputs, and records a
   prefetch receipt. It does not import or execute model repository code.
2. **GPU execution** runs the exact bridge revision in a digest-pinned container
   with `--network none`, a read-only root filesystem, all Linux capabilities
   dropped, and no Hugging Face token, GitHub token, or laptop signing key. The
   container can emit only an unsigned receipt intent.
3. **Trusted finalization** validates the fresh intent against the exact signed
   job and candidate artifact manifest, signs it outside the sandbox, uploads it,
   reads it back from the immutable Hub commit, and only then writes the local
   one-attempt ledger.

The isolated launcher is:

```text
laptop/run_nemo_v3_isolated.ps1
```

It requires both the bridge Git commit and the exact local Docker image ID
(`sha256:...`) emitted by `laptop/build_nemo_v3_image.ps1`. Registry manifest
references are refused because no signed job or repository allowlist authorizes
a registry image for this one-attempt lane. The local ID keeps the reviewed
training environment private on the owner GPU host while remaining
content-addressed.
The launcher inspects the image locally and records both the supplied reference
and observed image ID in the unsigned receipt intent. For an exact local image
ID, it also requires the build receipt emitted below, checks the image revision
label and Dockerfile hash against the exact bridge revision, and binds those
digests into the durable pre-execution claim. Trusted finalization requires the
receipt stack identity to equal that claim before signing. It refuses a dirty
bridge checkout, a stale outbox, an unverified prefetch receipt, a mutable image
tag, an unavailable or mismatched image ID, an absent or mismatched local build
receipt, or any host shell containing HF/GitHub tokens.

Build and CUDA-smoke the training image from the exact clean bridge revision:

```powershell
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass `
  -File .\laptop\build_nemo_v3_image.ps1 `
  -BridgeRevision (git rev-parse HEAD) `
  -BridgeSource (Get-Location).Path
```

The build uses a digest-pinned PyTorch CUDA 12.8 base, exact direct training
package versions, and no bridge source or credentials. It writes a non-secret
receipt under `C:\szl-bridge\image-receipts` only after the immutable image ID,
source-revision label, offline import smoke, and CUDA device probe all pass.

The trusted helper programs are:

```text
laptop/prefetch_nemo_v3.py
laptop/finalize_nemo_v3_receipt.py
```

Do not mount `laptop_key.pem`, a Hugging Face credential, a GitHub credential, the
Docker socket, or a host profile directory into the training container. A green
container exit is not terminal evidence: code 7 means a fresh intent is waiting
for trusted signing and immutable readback.

## 4. Observe the measured result

The workflow `Nemo v3 Governed Attempt Status` runs every 15 minutes and updates the deterministic issue:

```text
[nemo-v3-attempt] governed single attempt
```

A green workflow run means the status controller executed and the evidence it found was internally valid. It does **not** mean training ran or passed; the issue status remains the source of truth for the attempt lifecycle.

Possible honest states:

| State | Meaning |
|---|---|
| `AWAITING_ENGINE_SIGNATURE` | Reviewed plaintext spec exists; no executable queue envelope exists. |
| `RECEIPT_DISCOVERY_ERROR` | The controller could not inspect the authoritative private receipt store; it made no receipt-absence claim and failed closed. |
| `QUEUED_AWAITING_GPU_RECEIPT` | Engine-signed queue is valid; no terminal owner-host receipt is present. |
| `AWAITING_LAPTOP_RECEIPT_KEY_ENROLLMENT` | Receipt signature is valid but the owner-host key ID has not been pinned. |
| `TERMINAL_FAILURE` | A pinned owner host signed a blocked or failed terminal result. No automatic retry is permitted. |
| `QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW` | Every frozen holdout passed with zero degeneration; no candidate was published or deployed. |
| `INVALID_*` | Queue or receipt evidence violated the contract and the workflow failed closed. |

Receipts are uploaded to the private dataset `SZLHOLDINGS/szl-training-receipts` under the exact job ID. The monitor verifies the laptop signature, enrolled key ID, job ID, exact queue payload hash for Nemo result receipts, all-pass evaluation fields, and the no-publication effects boundary.

## 5. Promotion remains separate

`QUALIFIED_FOR_SEPARATE_PROMOTION_REVIEW` is not a model release. A separate reviewed process must still verify:

- SafeTensors-only artifacts and exact manifest hashes;
- NVIDIA base license and attribution;
- security scan and dependency/runtime evidence;
- live inference parity against the qualified adapter;
- serving resource requirements and rollback;
- a distinct approval and publication receipt.

A terminal failure remains quarantined. It may inform a new preregistered v4 experiment, but it must not be silently retried, signed as a release, uploaded as a candidate, or promoted.

The attempt becomes durably claimed, and therefore unavailable for automatic
retry, when the launcher atomically creates the pre-execution claim immediately
before Docker starts. Trusted finalization later records terminal receipt
publication and immutable readback; that later record is not the replay barrier.
Validation failures before the claim exists do not consume the attempt. An
unsigned intent, container exit, or interrupted upload after the claim exists
remains quarantined and still does not authorize a retry.

## 6. Protected successor generation

The consumed predecessor remains immutable and quarantined. Its claim is not
deleted, renamed, or reused. The separately reviewed successor plaintext is:

```text
jobspecs/nemo-v3-20260729-successor-2-reviewed.json
```

Its machine-validated `lineage` binds the predecessor job ID, exact claim and
queue-envelope digests, bridge revision, image ID, claim time, and incident
record. It also records that the failure happened during pre-training runtime
source parsing, before training, model-repository import, holdout access,
candidate creation, receipt-intent creation, or terminal-ledger publication.
The scientific inputs remain frozen; the job and candidate identities are new.

This protected plaintext is still not executable. It requires a fresh signature
from the offline engine key `5c6cf59741ade920`, a new protected queue envelope,
an exact-main image build and CUDA smoke receipt, authenticated no-code prefetch,
and the pre-claim exact-container compatibility gate. Absence of the offline
engine private key is `AWAITING_ENGINE_SIGNATURE`, not permission to replace the
key, reuse the predecessor envelope, or run the predecessor again.

## 7. Protected owner-dispatch attempt 2

The A11oy owner-dispatch v2 contract admits a distinct new attempt identity:

```text
jobspecs/nemo-v3-20260729-attempt-2-reviewed.json
```

That reviewed plaintext binds protected A11oy source
`b21b8fb65400e7eb39595365c5f54c80ed78aa67`, workflow blob
`7e08ffc8aa87b78d0fa1618d7d3c3e68cb81ca33`, the immutable training image,
the exact receipt repository, and disabled candidate, model-card, and dataset
uploads. It preserves the frozen science inputs and the quarantined predecessor
lineage.

Its exact queue envelope exists as immutable historical evidence, but its
source did not become the settled A11oy release and its signing key is now
verification-only. The record
`queue/quarantine/job-2026-nemo-v3-governed-attempt-2.json` binds the exact
envelope and payload digests and marks it
`STALE_SOURCE + RETIRED_KEY + NEVER_DISPATCH`. It must never be consumed,
deleted, rewritten, re-signed, or retried.

## 8. Quarantined provisional recovery generation

Issue `https://github.com/szl-holdings/szl-gpu-bridge/issues/25` records that
the private material for historical engine key `5c6cf59741ade920` is
unavailable. The public key is retained as `VERIFY_ONLY`; its historical
envelopes remain verifiable and its jobs are not retried, reset, or relabeled.

The separately identified generation is:

```text
jobspecs/nemo-v3-20260730-successor-3-reviewed.json
queue/pending/job-2026-nemo-v3-governed-successor-3.json
```

Its exact queue envelope is preserved as historical evidence, but key
`815714c8d4ae3e4d` was provisional and its source revision did not settle on
protected A11oy main. The record
`queue/quarantine/job-2026-nemo-v3-governed-successor-3.json` binds the exact
envelope and payload digests and marks it
`UNAUTHORIZED_PROVISIONAL_KEY + STALE_SOURCE + NEVER_DISPATCH`.

The dispatcher, isolated launcher, prefetch, runner, finalizer, and status
publisher all enforce every quarantine boundary before any execution or claim.
The original reviewed specs and envelopes must never be consumed, deleted,
rewritten, re-signed, or retried.

The coordinated key `b8041281c81c4caa` is enrolled as a distinct
administrative recovery trust root after the protected A11oy relock. It is the
sole active key for future reviewed jobs; no cryptographic continuity with the
two verification-only predecessors is claimed.

## 9. Coordinated reviewed attempt 4

The distinct reviewed plaintext is:

```text
jobspecs/nemo-v3-20260730-attempt-4-reviewed.json
```

It binds settled A11oy source
`5f98d90a42e021cf29948457a2404a159f236487`, immutable owner-workflow blob
`7e08ffc8aa87b78d0fa1618d7d3c3e68cb81ca33`, corrected bridge revision
`2237bb3f36663343ace29d98cda6c32e165450a0`, coordinated active key
`b8041281c81c4caa`, and full public-SPKI SHA-256
`b8041281c81c4caaea18112df5e8c99ea8472f0711fc796fc3072c27398af2cf`.
Its authorization records the terminal exact-main A11oy relock and explicitly
claims no cryptographic continuity with either verification-only predecessor.
The two immutable quarantine records bind the superseded signed envelopes to
this replacement identity.

The separately protected queue envelope now exists at:

```text
queue/pending/job-2026-nemo-v3-governed-attempt-4.json
```

It verifies under active key `b8041281c81c4caa` and binds the exact reviewed
payload. GitHub rejected its flat 14-property `repository_dispatch` transport
before creating an event. The immutable record
`queue/quarantine/job-2026-nemo-v3-governed-attempt-4.json` therefore marks it
`STALE_SOURCE + TRANSPORT_UNREPRESENTABLE + NEVER_DISPATCH`. Its spec and
envelope remain byte-for-byte evidence and must never be resent, consumed,
deleted, rewritten, or re-signed.

A fresh attempt must bind protected A11oy main
`e3d4a46724b222c8a5b2b6f04877bc115a6c82cb`, owner-workflow blob
`2522d3b54eeb7adc37ffc47e7c685a5ce7edf68f`, and the nested-v3 transport.
The b804-signed payload alone selects the executable Bridge revision. The
later protected envelope revision is data-only, is passed as an exact path,
and must be a strict protected descendant. The create-new claim and terminal
receipt evidence bind both revisions and the immutable Unsloth image digest.

## 10. Reviewed attempt 5 plaintext

The fresh transport-v3 recovery plaintext is:

```text
jobspecs/nemo-v3-20260730-attempt-5-reviewed.json
```

It binds final protected A11oy main
`e3d4a46724b222c8a5b2b6f04877bc115a6c82cb`, owner-workflow blob
`2522d3b54eeb7adc37ffc47e7c685a5ce7edf68f`, workflow version
`nemo-v3-owner-dispatch.v4`, and protected executable Bridge revision
`a2015accc0be8060c4084455e829a9373e5c99e2`. The lineage records that attempt
4 failed during pre-event transport validation with no event, workflow run,
claim, training, holdout, candidate, receipt-intent, or ledger side effect.
Its science inputs remain frozen and all candidate, model-card, and dataset
uploads remain disabled.

The separately protected, exclusive-create queue envelope now exists at:

```text
queue/pending/job-2026-nemo-v3-governed-attempt-5.json
```

Its raw SHA-256 is
`30549fc522238193b4985dbf96a690518bad2ae8c399dc3ee78fb9dd7f551009`.
It verifies under active key `b8041281c81c4caa` and binds canonical payload
SHA-256
`374901dec6923e0c28688407e581d374827d76f7567970d8ec481b6bf140c67b`.
Exactly one dispatch created A11oy workflow run
`https://github.com/szl-holdings/a11oy/actions/runs/30591897165`. Windows host
execution policy rejected the generated PowerShell script before validator
admission. Envelope verification, image use, prefetch, claim creation,
training, finalization, and receipt upload were skipped. No claim, job
directory, prefetch receipt, or terminal receipt exists.

The immutable quarantine record
`queue/quarantine/job-2026-nemo-v3-governed-attempt-5.json` therefore marks
attempt 5
`STALE_SOURCE + HOST_EXECUTION_POLICY_BLOCKED + PRE_ADMISSION + NEVER_DISPATCH`.
Its reviewed spec and signed envelope remain byte-for-byte evidence and must
never be retried, consumed, deleted, rewritten, or re-signed. A future attempt
6 must bind protected A11oy main
`78b35d244b89c7663063372ff459894bab2977b6`, owner-workflow blob
`d29d937b2d398e9c207777a9a819aadd050ac231`, active key
`b8041281c81c4caa`, and a separately reviewed protected Bridge runtime
revision. The protected envelope revision remains data-only and must be a
strict descendant of that signed executable Bridge revision.

## 11. Reviewed attempt 6 and signed envelope

The fresh host-policy recovery plaintext is:

```text
jobspecs/nemo-v3-20260730-attempt-6-reviewed.json
```

It binds terminal protected A11oy main
`78b35d244b89c7663063372ff459894bab2977b6`, owner-workflow blob
`d29d937b2d398e9c207777a9a819aadd050ac231`, workflow version
`nemo-v3-owner-dispatch.v4`, protected Bridge correction
`69a097d2eb0619506d673464353f1aea7174cf05`, active engine key
`b8041281c81c4caa`, and the immutable Unsloth image. Its lineage binds the
exact attempt-5 envelope and payload hashes, envelope-publication revision
`d127d7bcd734235fba83e786de923787ab90c51b`, executable Bridge revision
`a2015accc0be8060c4084455e829a9373e5c99e2`, and failed A11oy run
`30591897165`. It records that the event and workflow run existed while
validator admission, claim, image use, training, holdouts, candidate,
receipt-intent, and terminal ledger effects did not.

The separately protected, exclusive-create queue envelope now exists at:

```text
queue/pending/job-2026-nemo-v3-governed-attempt-6.json
```

Its raw SHA-256 is
`c68e1ecf380d7023c27439e9988ca182ebd9b2446dc769269d4de1c48d507d70`.
It verifies under active key `b8041281c81c4caa` and binds canonical payload
SHA-256
`d0fa9bd15f8e576411b643858d650470b6f1d5ddd56003cd53eda28d83dd914d`.
Pre-dispatch validation proved the then-pinned A11oy validator rejected this
immediate attempt-5 predecessor, so no repository dispatch was sent. No event,
workflow run, runner, claim, job directory, prefetch, image use, training,
holdout access, finalization, or receipt exists.

The immutable quarantine record
`queue/quarantine/job-2026-nemo-v3-governed-attempt-6.json` therefore marks
attempt 6
`STALE_SOURCE + PRE_DISPATCH_VALIDATOR_REJECTED + PRE_EVENT + NEVER_DISPATCH`.
Its reviewed spec and signed envelope remain byte-for-byte evidence and must
never be retried, consumed, deleted, rewritten, or re-signed. A future attempt
7 must bind protected A11oy main
`2b190b3806a5d2b3faa58f34c2db41c5dc4668fa`, owner-workflow blob
`d29d937b2d398e9c207777a9a819aadd050ac231`, active key
`b8041281c81c4caa`, and a separately reviewed protected Bridge runtime
revision. Candidate, model-card, dataset, deployment, promotion, and all
non-receipt uploads remain disabled.

## 12. Reviewed attempt 7 and signed envelope

The fresh validator-lineage recovery plaintext is:

```text
jobspecs/nemo-v3-20260731-attempt-7-reviewed.json
```

It binds terminal protected A11oy main
`2b190b3806a5d2b3faa58f34c2db41c5dc4668fa`, owner-workflow blob
`d29d937b2d398e9c207777a9a819aadd050ac231`, workflow version
`nemo-v3-owner-dispatch.v4`, protected Bridge correction
`2f33607d8fcbec76fe98290258ec3dfa728fb509`, active engine key
`b8041281c81c4caa`, and the immutable Unsloth image. Its lineage binds the
exact attempt-6 envelope and payload hashes, envelope-publication revision
`72f9bf650b081fec0a016825f2cb7f962c52242d`, executable Bridge revision
`69a097d2eb0619506d673464353f1aea7174cf05`, and terminal issue
`https://github.com/szl-holdings/szl-gpu-bridge/issues/41`. It records zero
event, workflow-run, runner, claim, image, training, holdout, candidate,
receipt-intent, and terminal-ledger effects.

The separately protected, exclusive-create queue envelope now exists at:

```text
queue/pending/job-2026-nemo-v3-governed-attempt-7.json
```

Its raw SHA-256 is
`8c1e333f797a8de634217b19cd140994a1d4f3920afebdf6f658dcc984188a96`.
It verifies under active key `b8041281c81c4caa` and binds canonical payload
SHA-256
`0fa239d3e14f0644d26b76c0e605ea8068b305cd4d96ea41385cad38fbdfbde7`.
Run `30605081533` passed nested transport, b804 envelope verification,
protected Bridge history, and the immutable GPU image, then failed in
authenticated prefetch before producing its prefetch receipt. Executable
Bridge revision `2f33607d8fcbec76fe98290258ec3dfa728fb509` correctly rejected
the later attempt-7 reviewed binding. Neither the workflow-level dispatch
claim nor the runtime execution claim exists; no job directory, training,
candidate, terminal ledger entry, or receipt exists.

The immutable quarantine record
`queue/quarantine/job-2026-nemo-v3-governed-attempt-7.json` therefore marks
attempt 7
`RUNTIME_CONTRACT_BINDING_REJECTED + PRE_CLAIM + NEVER_DISPATCH`. Its spec
and b804 envelope remain byte-for-byte evidence and must never be retried,
consumed, deleted, rewritten, or re-signed. A fresh attempt 8 must bind the
protected runtime-recovery revision, and the prefetch, dispatcher, and
isolated runner each require that exact signed execution revision before any
attempt-8 claim. Candidate, model-card, dataset, deployment, promotion, and
all non-receipt uploads remain disabled.
