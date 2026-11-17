# Training evidence surface

This surface projects one existing observer's local status into a loopback API
and browser view. It observes selected training evidence; it does not qualify
models, control compute, publish artifacts, sign receipts, or authorize promotion.
A source commit or passing test is not evidence that the surface is deployed.

## Producer and projection

| Boundary | Contract |
| --- | --- |
| Existing producer | `tools.salad.training_observer`, schema `szl.training-observation/v1` |
| Canonical input | `C:/SZL/edge-debug/szlw_training_observer_status.json` |
| Existing task | `SZL-Qualified-Training-Observer`; interactive owner login required |
| Producer source pin | `dabd1e681eacb2687546f2a8d9dd905eec2f3a69`, under `C:/SZL/training-control/<sha>` |
| Producer cadence | Five minutes; declared `stale_after_seconds` is 660 |
| Projection | `tools.dashboard.training_server`, schema `szl.training-dashboard/v1`; rereads on every API GET |
| Browser | `http://127.0.0.1:8767/training` |
| JSON | `http://127.0.0.1:8767/api/training/status` |
| Listener health | `http://127.0.0.1:8767/healthz`; does not establish source freshness |

The intended flow is existing producer → canonical local file → validated API
projection → browser. The producer owns its atomic status replacement and history.
Do not start a second observer or scheduler to make this view appear fresher.
The server binds only `127.0.0.1` and checks Host and Origin; it has no public-bind
option. GET handling makes no upstream network calls, reads no credentials,
performs no signing, and has no state-changing action or artifact publication path.

## Scope and meaning

The current producer selects **six legacy candidate observations**. The two new
held SFT groups are outside this selection. This is not an exhaustive estate,
provider, model, or training catalog. Expand producer scope only through its
reviewed contract; do not fabricate rows from repository names or status prose.

A provider's running instance is not verified trainer liveness. A checkpoint
counter, terminal metadata, or final filename is not complete-artifact validation,
held-out acceptance, a signed receipt, or production qualification. Every
`promotion_authorized`, `training_complete_verified`, and
`trainer_process_liveness_verified` claim carried by this observer remains false.
An absent candidate is unassessed under this scope, not proved stopped or deleted.

The API's source SHA-256 identifies the exact local bytes used for that response.
It enables byte comparison and provenance tracing; it does not authenticate the
author or establish a cryptographic signer identity. This file is an unsigned
observation, not a signed receipt. Artifact metadata and observed Hub revisions
retain their original, narrower evidence meaning.

## Fail-closed behavior

Input is bounded to 256 KiB and parsed as strict JSON. Invalid schema, malformed
values, unexpected authority claims, missing/unreadable input, and future
timestamps cannot produce a qualified or healthy training claim. Stale input is
explicitly stale. No fallback may silently turn a prior snapshot into fresh data.

The server caps source age at 900 seconds by default. `--max-age-seconds` accepts
60–3600 seconds; the effective threshold also respects the producer's declared
TTL. With the current producer, 660 seconds is the relevant source freshness bound.
Candidate timestamps retain their own age; completing a later report does not
make an earlier candidate observation newer. The browser must expire a displayed
observation between fetches and surface fetch errors rather than retain green
freshness indefinitely. Listener health and observation freshness are separate.

## Pinned artifact observations

The optional artifact view consumes an existing qualification bundle from
`tools.salad.artifact_qualification`. Configure both
`--qualification-bundle <directory>` and
`--qualification-sha256 <reviewed-qualification-json-sha256>`.
The SHA-256 must be the independently selected digest of the exact
`qualification.json` bytes. Selecting a directory alone cannot select new
evidence implicitly.

On every status request, the backend validates the pinned report, its manifest,
and the two archived verifier source hashes using
`tools.salad.evaluation_binding`. It does not execute those archived sources or
download models. The qualification report and its file members have bounded reads;
symlinks and nonregular paths are rejected. Raw diagnostic messages and local
paths are not exposed to the browser.

A row's qualification is independent of the checkpoint and provider observation:

| Qualification state | Meaning |
| --- | --- |
| `NOT_CONFIGURED` | This view has no selected qualification bundle. |
| `UNAVAILABLE` | The selected local bundle or a required member cannot be read. |
| `INVALID` | Content, hashes, timestamp order, scope, or target contract failed validation. |
| `NOT_ASSESSED` | This candidate is absent from the selected qualification report. |
| `REVISION_MISMATCH` | The report describes a different immutable revision. |
| `ARTIFACT_HOLD` | The report records an artifact hold; no match is asserted. |
| `ARTIFACT_OBSERVATION_MATCHED` | The report's exact repo, revision, and terminal target match this observer row. |

A matched row exposes its recorded revision, adapter SHA-256, report SHA-256,
base repository, target, and observation time. The match establishes consistency
with that unsigned local record. It does not re-inspect the artifact, authenticate
the observer, verify the base revision, establish independent held-out quality,
or authorize compute or promotion. All five remaining release blockers are
retained in the expandable details.

An immutable artifact observation does not become invalid merely because time
has elapsed. Live observer freshness remains a separate requirement: when the
observer is stale, the UI labels the match as attached to the *recorded* revision
and says that the current revision is unestablished. A later revision cannot
inherit the prior match. Missing, invalid, or changed qualification data cannot
fall back to the last successful match.

This additive dashboard contract is `szl.training-dashboard/v1`. An older
response that omits qualification is displayed as `NOT_CONFIGURED`, never as an
artifact pass. Paired evaluation execution uses the same verifier and archives
the binding in a version 3 evaluation bundle; see
[paired held-out measurements](paired-heldout-measurements.md).

## Start and verify

From the reviewed repository root, with Python available:

```powershell
python -m tools.dashboard.training_server --status "C:/SZL/edge-debug/szlw_training_observer_status.json" --port 8767
```

This starts the read-only surface in the foreground. It does not install a task
or start the producer. Use the approved runtime checkout when installing under
an existing supervisor; record the exact source revision, command, and owner.

Before calling a deployment live:

1. Run the surface's focused contract tests and check the reviewed source diff.
2. Confirm the producer task is the intended single producer and its canonical
   file is actually advancing. An installed task alone is not a successful run.
3. Identify the listener process and verify it runs the intended checkout and
   exact command. Confirm loopback binding on port 8767.
4. Fetch `/healthz` and `/api/training/status` separately. Check observed times,
   scope, source hash, explicit false authority flags, and failure reasons.
5. Open `/training`. Verify that its six selected rows and freshness match the
   same API observation, including unavailable or partial evidence.
6. Exercise missing, invalid, and expired inputs using isolated test fixtures,
   never by altering the canonical producer file. Check client-side expiry too.
7. Record deployment time, process or supervisor identity, source revision,
   canonical path, and live checks. Keep source validation and runtime evidence
   distinct; neither establishes that any model is trained or promoted.

A healthy listener with stale or unavailable input is an operational HTTP service
with unavailable training evidence. Repair or reconcile the existing producer;
do not mint a new timestamp, remove a denial, or weaken the freshness bound.

## Restart and rollback

Identify the process listening on 8767 before stopping anything:

```powershell
Get-NetTCPConnection -LocalPort 8767 -State Listen |
  Select-Object LocalAddress, LocalPort, OwningProcess
```

For a foreground server, use Ctrl+C in its own terminal. For a supervised server,
restart only its identified supervisor/process using its recorded configuration.
Do not terminate all Python processes or disturb training or observer tasks.
Relaunch the exact command above from the intended reviewed runtime and repeat
the live checks. Do not assume a source update hot-reloads an existing process.

Rollback selects the previous reviewed server/UI revision and restarts only this
surface. Preserve canonical status/history and existing signed evidence. A
rollback must retain explicit stale/unavailable states and false authority flags.
The existing governed release path remains the only artifact publication path;
this surface introduces no alternate Hugging Face writer.

## Doctrine and prior art

SZL's [doctrine mirror](../KAMACHIKUNA_DOCTRINE_v11.md) and
[machine-usable doctrine](../../sdk/szl/kamachikuna.py) distinguish operational
checks from broader principles. This surface applies truthfulness, abstention,
explicit failure, and freshness limits without claiming universal enforcement.
See [snapshot evidence](training-snapshot-evidence.md),
[published receipt observation](published-receipt-observation.md), and
[provenance limits](../../PROVENANCE.md) for adjacent contracts.

- [Palantir Ontology](https://www.palantir.com/docs/foundry/object-backend/overview):
  shared objects and governed actions inform consistent backend/frontend meaning.
- [Palantir Evals](https://www.palantir.com/docs/foundry/aip-evals/metrics-dashboard):
  metric-to-test-to-trace navigation informs evidence drilldown.
- [Palantir feedback walkthrough](https://www.youtube.com/watch?v=hXSx9uG0R4k):
  feedback-to-test workflows are future adaptation ideas, not features delivered here.
- [NeMo checkpointing](https://docs.nvidia.com/nemo/automodel/latest/development/checkpointing)
  and [Evaluator](https://github.com/NVIDIA-NeMo/Evaluator): complete publication
  boundaries and reproducible evaluation inform later qualification work.
- [LangGraph fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance):
  durable state and idempotency distinguish retries from exactly-once side effects.
- [Model Cards for Model Reporting](https://arxiv.org/html/1810.03993v2):
  intended use, evaluation context, and limitations guide honest evidence claims.

These are attributed architectural patterns, not claims of unique invention or
a requirement to replace the existing SZL implementation with another framework.
