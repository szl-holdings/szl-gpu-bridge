# SZL-Nemo v3 — secure single-attempt execution

This runbook completes the human-controlled boundaries that cannot be delegated to a public repository: enrolling the owner GPU host receipt identity and authorizing the reviewed job with the offline engine private key.

The repository already contains the reviewed plaintext job at:

```text
jobspecs/nemo-v3-20260722-reviewed.json
```

A plaintext jobspec is **not executable**. The bridge accepts only a DSSE envelope signed by the private key corresponding to the pinned engine key `5c6cf59741ade920`.

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

The file is not an executable job. No `queue/pending/` envelope exists for this
identity until a separate protected publication review authorizes a fresh
offline engine signature.
