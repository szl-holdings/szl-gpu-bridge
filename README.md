# szl-gpu-bridge

**Doctrine-governed training bridge: cloud-signed job specs → owner GPU metal → signed receipts back. No inbound access, no secrets on the wire, fail-closed in both directions.**

This repository is the controlled channel between SZL cloud sessions and the owner’s Windows/NVIDIA training host. The cloud cannot reach the laptop; the laptop polls this repository. Every executable job is DSSE-signed by the pinned SZL engine key. The laptop verifies the exact envelope bytes before reading job fields, then either executes an allowlisted local runner or records a permanent refusal.

## Architecture

```text
cloud                                                owner GPU host
────────────────────────────────────                 ──────────────────────────────────
cloud/materialize-frontier-spec.mjs
  ├─ resolve exact Hub commits
  ├─ verify model + dataset license metadata
  ├─ hash exact dataset file
  └─ pin tokenizer chat-template hash
cloud/sign-job.mjs
  └─ DSSE-sign v1/v2 spec ─────────────► queue/pending/
                                                     laptop/daemon.ps1
                                                       └─ laptop/dispatcher.py
                                                          1 verify DSSE + engine pin
                                                          2 validate allowlisted contract
                                                          3 run v1 or frontier-v2 runner
                                                          4 train/evaluate/export
                                                          5 reload-smoke released formats
                                                          6 sign exact-byte receipts
                                                          7 publish Hub artifacts + receipts
cloud/verify-receipt.mjs ◄──────────────────────────── independently verifies receipt chain
```

- **Inbound jobs:** public HTTPS polling only; no laptop GitHub credential and no inbound port.
- **Outbound results:** the host uses its local Hugging Face authentication. Tokens and private keys are never committed or sent through the queue.
- **Trust roots:** the host trusts only public keys admitted by the reviewed
  `keys/engine_keyring.json`. Historical key `5c6cf59741ade920` is
  verification-only; provisional key `815714c8d4ae3e4d` is also
  verification-only; coordinated administrative-recovery key
  `b8041281c81c4caa` is the sole active execution authority. No cryptographic
  continuity with either predecessor key is claimed. Attempt 2, successor
  generation 3, attempts 4, 5, and 6 remain byte-preserved under explicit
  `NEVER_DISPATCH` quarantine. Attempt 4's flat 14-property dispatch transport
  was rejected before event creation. Attempt 5 created exactly one workflow
  run, but Windows host execution policy rejected the generated PowerShell
  script before validator admission, claim, image use, training, or receipt.
  Its signed envelope is evidence only under
  `HOST_EXECUTION_POLICY_BLOCKED + PRE_ADMISSION + NEVER_DISPATCH`. Attempt 6
  was never submitted: pre-dispatch validation proved its then-pinned A11oy
  validator rejected the immediate attempt-5 predecessor. Its reviewed spec
  and b804-signed envelope remain immutable evidence under
  `STALE_SOURCE + PRE_DISPATCH_VALIDATOR_REJECTED + PRE_EVENT +
  NEVER_DISPATCH`; no event, workflow run, claim, training, or receipt exists.
  A future attempt 7 must bind protected A11oy main
  `2b190b3806a5d2b3faa58f34c2db41c5dc4668fa`, owner-workflow blob
  `d29d937b2d398e9c207777a9a819aadd050ac231`, and a separately reviewed
  protected Bridge runtime. That distinct attempt-7 contract binds
  protected Bridge revision `2f33607d8fcbec76fe98290258ec3dfa728fb509`,
  preserves the frozen science inputs, and keeps all candidate, model-card,
  and dataset uploads disabled. Its separately protected b804-signed queue
  envelope created one exact workflow run, but the older signed execution
  runtime rejected the new reviewed binding before prefetch output or claim.
  Attempt 7 is immutable evidence under
  `RUNTIME_CONTRACT_BINDING_REJECTED + PRE_CLAIM + NEVER_DISPATCH`; no
  training, candidate, or receipt exists. Reviewed plaintext attempt 8 binds
  protected runtime `dc36af2b264bbdb4cc101593c54c5b2c24c1d9cf`; that exact
  signed revision must be supplied again at dispatch. Its one-time b804
  envelope created one exact workflow run and authenticated prefetch, but
  Python bytecode from that trusted prefetch dirtied the protected execution
  checkout. The strict gate rejected attempt 8 before its claim. Attempt 8 is
  immutable evidence under
  `TRUSTED_PREFETCH_DIRTIED_EXECUTION_CHECKOUT + PRE_CLAIM +
  NEVER_DISPATCH`; no training, candidate, or receipt exists. A fresh attempt
  9 must bind protected A11oy main
  `c6aa4f08f752a22bbae35cf5a618a81811494a43`, owner-workflow blob
  `f0ab364e1db9c48a0d8f49c7f0c17b5e44cad99d`, canonical relock run
  `30607399378`, and protected Bridge runtime
  `eeabd1b52380d2b24439e53d5e4ad38f8114556c`. Its reviewed plaintext
  contract preserves the frozen science inputs and disables candidate,
  model-card, dataset, deployment, and promotion effects. Its exclusive-create
  b804 envelope at
  `queue/pending/job-2026-nemo-v3-governed-attempt-9.json` has raw SHA-256
  `a7b67f1245137b3422d6e2ce5cf379aa9adb193e1f1d9db0dec8abf92bf5fa49`
  and binds canonical payload SHA-256
  `f8ec93b0a2967e548ba2222cbf8a69abbe89987c98e695688c39c0e0d3827c5b`.
  Its one exact workflow run `30609977388` created a durable attempt claim, but
  isolated base-license verification could not traverse the cache mounted below
  root-only `/root`, and trusted finalization then rejected the runtime-bound
  spec without the claim's execution-revision argument. Attempt 9 is immutable
  evidence under `ISOLATED_HF_CACHE_ROOT_PERMISSION_BLOCKED +
  TRUSTED_FINALIZER_RUNTIME_BINDING_REJECTED + POST_CLAIM + NEVER_DISPATCH`.
  No signed receipt, terminal ledger entry, candidate, model card, dataset,
  deployment, promotion, or Hugging Face artifact publication exists. A future
  attempt 10 must use a separately reviewed protected runtime that mounts the
  credentialless cache at `/hf-cache`, binds finalization to the exact durable
  claim, and declares the immutable card's exact custom license ID
  `nvidia-nemotron-open-model-license`. Reviewed plaintext attempt 10 now binds
  that protected runtime at
  `37479c23af3228a57ad6018b3f9134186e6d7fa7`, preserves the exact attempt-9
  post-claim evidence, and remains `AWAITING_ENGINE_SIGNATURE`. No attempt-10
  queue envelope, runner, dispatch, claim, receipt, or publication exists.
  The cloud trusts the separately announced laptop receipt key. An
  unverifiable claim is treated as no claim.
- **Remote-code isolation:** a signed job with `trustRemoteCode=true` cannot use the
  ordinary host lane. Authenticated prefetch, networkless GPU execution, and trusted
  signing/upload are separate processes; the execution sandbox receives neither a
  credential nor a signing key.

## Frontier training contract v2

`schema/jobspec.v2.json` and the dependency-free enforcement in `laptop/frontier_contract.py` define the production training contract. V1 remains supported for already signed jobs; the verify-first dispatcher preserves backward compatibility without letting v1 bypass v2 gates.

V2 requires:

- exact model and dataset revisions;
- exact dataset-file sha256;
- exact-revision model and dataset license metadata;
- an auditable dataset-lineage statement;
- a signed chat-template sha256;
- explicit Unsloth/TRL/PEFT recipe fields, including rsLoRA, packing, and assistant-only-loss behavior;
- resource and wall-clock ceilings;
- signed minimum evaluation rates and a maximum degeneration rate;
- artifact sha256 manifests;
- real reload smoke for merged and GGUF outputs;
- signed training/evaluation receipts chained over exact bytes;
- immutable Hub release commit evidence.

A successful `trainer.train()` call is **not** a release. Any failed input, license, resource, evaluation, export, reload, or upload gate becomes a signed `BLOCKED` receipt.

## Materialize, sign, and enqueue

Create a human-reviewed draft containing the desired model, dataset file, recipe, gates, output repositories, and evaluation thresholds. The materializer replaces moving references with exact evidence:

```powershell
$env:HF_TOKEN = "<read token when private inputs are used>"
node cloud/materialize-frontier-spec.mjs draft.json spec.json
```

It also writes `spec.json.evidence.json`. Review both files, then sign with the existing engine key:

```powershell
$env:SZL_QUANT_KEY = "C:\secure\engine_key.pem"
node cloud/sign-job.mjs spec.json
```

Only the signed envelope enters `queue/pending/`.

## Training and release path

The v2 runner uses Unsloth core as an installed Apache-2.0 dependency, TRL for SFT, PEFT for adapters, Hugging Face repositories for versioned releases, optional Hugging Face Storage Buckets for mutable checkpoints, and llama.cpp for GGUF reload smoke when GGUF is requested. The SZL implementation adds the signed contract, exact input and template pins, license verification, receipt chain, deterministic structural evaluation, export manifests, and fail-closed promotion law.

Released artifacts remain in versioned model repositories. Buckets are used only for mutable checkpoints/working state. The model card records base/dataset lineage, measured run fields, limitations, and artifact anchors.

## One-paste bootstrap

On the owner’s Windows host, in elevated PowerShell:

```powershell
irm https://raw.githubusercontent.com/szl-holdings/szl-gpu-bridge/main/laptop/bootstrap.ps1 | iex
```

The installer:

1. self-checks and writes the pinned engine public key;
2. creates the Python 3.12 environment;
3. installs the current governed interface pins and hardware-specific GPU dependencies;
4. records `stack-freeze.txt` and its sha256 for receipts;
5. creates the host’s Ed25519 receipt key if absent;
6. downloads and compiles the dispatcher, v1/v2 runners, helpers, and schemas;
7. registers the single-flight scheduled polling task.

If Hub authentication or `llama-cli` is unavailable, affected jobs do not silently downgrade; they produce an honest blocked result.

## Honest limits

- No training begins until the owner installs and starts the bridge.
- Cloud-side liveness is inferred only from signed receipts; absence of a receipt is not success.
- Receipts are Ed25519 attestations, not cryptographic proof of computation.
- The v2 code and contract CI do not substitute for a measured GPU run. A model is operational only after a real job passes every signed gate and the resulting release passes the serving plane.
- Existing v1 historical claims remain labeled according to their original receipt contract.
- Advisory research infrastructure; not financial advice.

## Repository map

| path | responsibility |
|---|---|
| `schema/jobspec.v1.json` | legacy signed SFT contract |
| `schema/jobspec.v2.json` | governed frontier training/release contract |
| `cloud/materialize-frontier-spec.mjs` | resolve exact Hub inputs, licenses, template, and dataset digest |
| `cloud/sign-job.mjs` | validate and DSSE-sign v1/v2 specs |
| `cloud/verify-receipt.mjs` | independently verify signed receipt chains |
| `laptop/bootstrap.ps1` | install, pin, compile, and register the polling service |
| `laptop/daemon.ps1` | single-flight poll and ledger loop |
| `laptop/dispatcher.py` | verify envelope, validate contract, select allowlisted runner |
| `laptop/runjob.py` | legacy v1 runner |
| `laptop/runjob_frontier.py` | v2 train/evaluate/export/reload/publish runner |
| `laptop/prefetch_nemo_v3.py` | verify and cache exact Nemo inputs without executing repository code |
| `laptop/run_nemo_v3_isolated.ps1` | launch the digest-pinned, networkless, keyless GPU sandbox |
| `laptop/finalize_nemo_v3_receipt.py` | validate, sign, upload, and immutably read back one fresh receipt intent |
| `laptop/frontier_contract.py` | pure verify-first contract enforcement |
| `laptop/frontier_runtime.py` | evidence, dataset, artifact, and model-card helpers |
| `docs/FRONTIER_TRAINING_V2.md` | v2 architecture and release law |
| `docs/SECURITY_MODEL.md` | threat model |
| `docs/RESEARCH_MEMO.md` | leader study and adaptation record |

Apache-2.0. Adapted public patterns are attributed in `NOTICE`; copyleft subtrees are not vendored into this repository.
