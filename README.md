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
  post-claim evidence, and has one exclusive-create b804 envelope at
  `queue/pending/job-2026-nemo-v3-governed-attempt-10.json`. Its raw SHA-256 is
  `b354d34dcc6487e311b2d40413de4920ef8646d3f40e9d7442d366152aac901b`
  and its canonical payload SHA-256 is
  `2287b1be69239ec0f577ee6e712e0093345e46640485dc6fefa88e8104d727c9`.
  Its single dispatch created run `30612658302`, but the older immutable
  execution runtime rejected the later attempt-10 job binding during trusted
  prefetch. The failure was before claim, training, receipt intent, or upload.
  Attempt 10 is preserved under `IMMUTABLE_RUNTIME_JOB_BINDING_REJECTED +
  PRE_CLAIM + NEVER_DISPATCH`; its spec and envelope are immutable and will not
  be retried. A future attempt 11 must use the separately reviewed A11oy helper
  invocation contract at `434d653eaf100b9b3e5484687db1e6e6ca7116c9`
  and a protected runtime-bound Bridge revision that accepts only its exact
  signed job/source/workflow/execution identity. Reviewed plaintext attempt 11
  now binds protected A11oy source `434d653eaf100b9b3e5484687db1e6e6ca7116c9`,
  workflow blob `7cf0c877399471a084d3e70638ef50ec28d7f646`, and protected
  Bridge runtime `f07263bc37ef6e90b313ba5576ef425d845cf287`. Its one
  exclusive-create b804 envelope is at
  `queue/pending/job-2026-nemo-v3-governed-attempt-11.json`, with raw SHA-256
  `7b9af824b529fa80ec51e060cd0fa14f1af8acc8ded5fff5b10f159acb861918`
  and signer-canonical payload SHA-256
  `85f08bc171370b25606915008d1b96ff50f670d09e20eb631b4c1ebeb108d994`.
  Its one dispatch created run `30620232291` and exact claims, but immutable
  offline execution stopped before training because the Bridge did not pass the
  pinned local tokenizer snapshot to Unsloth. The signed BLOCKED receipt is
  preserved at revision `1a74ad3f5fc2682e6bbdd034a68399dee7e79525` with
  file SHA-256
  `f6f1c5af7c8a47c4c4a4ce35ccb9d2859cf3177c06c439bd529c901308aeb9e3`.
  Attempt 11 is `TOKENIZER_LOAD_BLOCKED + POST_CLAIM +
  SIGNED_BLOCKED_RECEIPT + NEVER_DISPATCH`; candidate, adapter, model-card,
  dataset, deployment, and promotion effects are all false.
  Reviewed plaintext attempt 12 is the distinct successor at
  `jobspecs/nemo-v3-20260731-attempt-12-reviewed.json`. It preserves attempt
  11's signed spec, envelope, and receipt bytes, freezes the same base license
  and science inputs, and binds corrected Bridge runtime
  `d110abb8ea48c9382a70c3eead22dddf555f292b`, where the exact local tokenizer
  artifacts are verified before Unsloth receives their snapshot path. Attempt
  11's structured terminal evidence is preserved separately at
  `queue/evidence/job-2026-nemo-v3-governed-attempt-11.json` so its quarantine
  retains the exact A11oy-admitted dispatch-denial schema. Its one
  exclusive-create b804 envelope is at
  `queue/pending/job-2026-nemo-v3-governed-attempt-12.json`, with raw SHA-256
  `a1c9f3d909b120d3675efe2cee0ba06b1c92c950f3a9ed4cc4e5b242971ed70f`
  and signer-canonical payload SHA-256
  `a5e04951412bb0c4d085e567e4e869d52bdf6987546b16ffcd6d2bcb72768ce8`.
  Its one dispatch created A11oy run `30626533443`; source, envelope, history,
  and image gates passed, then authenticated prefetch rejected the execution
  runtime's missing attempt-12 reviewed-job binding before claim. Attempt 12 is
  preserved as `RUNTIME_JOB_BINDING_REJECTED + PRE_CLAIM + NEVER_DISPATCH`.
  Its exact zero-effect run evidence is hash-pinned at
  `queue/evidence/job-2026-nemo-v3-governed-attempt-12.json`. No claim,
  prefetch receipt, training, terminal receipt, candidate, adapter, model card,
  dataset, deployment, promotion, or publication exists. The next reviewed
  runtime-bound identity must use attempt 13 and must still match the explicit
  execution revision supplied by the protected A11oy workflow; unknown job IDs
  and mismatched runtime revisions remain fail-closed.
  Reviewed plaintext attempt 13 is at
  `jobspecs/nemo-v3-20260731-attempt-13-reviewed.json`. It binds the settled
  A11oy source/workflow, the same immutable science and offline tokenizer
  inputs, and protected Bridge runtime
  `2783b3518abcec9f38d3f6504c06e305a4723801`. Its lineage records attempt
  12's single run `30626533443` as exact pre-claim
  `RUNTIME_JOB_BINDING_REJECTED + NEVER_DISPATCH` evidence. Its
  exclusive-create b804 DSSE queue envelope is preserved with raw SHA-256
  `de31cbb574cdeeaaf611a25fe1e40616b7fe8d4f6e2e138b66697474f5d800b0`.
  Its canonical payload SHA-256 is
  `82f619eb1fff6a7617b5761358d2f5c1d8ca62a306eb7cb1bf2570e096b2b9fc`.
  Its one dispatch created A11oy run `30629929196` and claim SHA-256
  `bb1fd12fb73289864503d5f8d65aacb4b34d0db0d0ba2fcce73a975c71364293`.
  Credentialless, networkless execution then stopped before trainer construction
  because the pinned TRL `0.23.1` `SFTConfig` accepts `eval_strategy`, while the
  old compatibility helper also forwarded `evaluation_strategy` through the
  Unsloth wrapper's `**kwargs`. Trusted finalization published only the signed
  BLOCKED receipt at revision `ac219fe87da9acf57141ff24ffbd330216584f7c`;
  its file SHA-256 is
  `384e64b0ebd43fcfd2f52a3b1139cf1bca04f23c43ccfd9738af3a1fdfe46d02`.
  Attempt 13 is now `SFTCONFIG_STRATEGY_KEY_BLOCKED + POST_CLAIM + PRE_TRAINING +
  SIGNED_BLOCKED_RECEIPT + NEVER_DISPATCH`; the exact run truth is hash-pinned
  under `queue/evidence/`. Candidate, adapter, model-card, dataset, deployment,
  promotion, and all other publication effects remain false. The next reviewed
  identity is attempt 14, bound to a separately protected corrected runtime.
  Reviewed plaintext attempt 14 is at
  `jobspecs/nemo-v3-20260731-attempt-14-reviewed.json`. It binds protected
  Bridge runtime `e150711a6ba6a0c29109a00da7fc82af2967f588`, whose
  fail-closed compatibility boundary maps the one logical evaluation strategy
  only to the explicit field exposed by the installed `SFTConfig` signature.
  Its raw reviewed JSON SHA-256 is
  `99e293ab4c2dd4282bd39a5f741b8359652792c68215c0e7100114a77bbacdf6`
  and its signer-canonical payload SHA-256 is
  `162354602784e8a1cbcecbbfc8a5d7cc9af6be2dd58c66fae442d4f5a292f1da`.
  Its exclusive-create b804 DSSE queue envelope now exists at
  `queue/pending/job-2026-nemo-v3-governed-attempt-14.json`, with raw SHA-256
  `207f0c58525f042d31a748404d0acb678f5fd83722d2a3eacf8399e4e34c9f82`.
  The signature verifies under keyId `b8041281c81c4caa` and binds the same
  canonical payload. Its only run (`30634484969`) reached trainer construction
  but stopped before training when Unsloth tried to copy the frozen,
  CPU-offloaded `lm_head.weight` meta placeholder. Trusted finalization uploaded
  only the signed BLOCKED receipt at revision
  `8c504d466d6b1b3fb0a755768341a34e58b82c11`. Attempt 14 is now
  `META_TENSOR_MATERIALIZATION_BLOCKED + POST_CLAIM + PRE_TRAINING +
  SIGNED_BLOCKED_RECEIPT + NEVER_DISPATCH`; its spec and envelope remain
  immutable. All publication flags are false. The next reviewed identity is
  attempt 15, bound to a separately protected corrected runtime.
  Reviewed plaintext attempt 15 is at
  `jobspecs/nemo-v3-20260731-attempt-15-reviewed.json`. It binds verified
  Bridge runtime `60b9894efe9e0e782999aaa4ee5b0d668e7a9b63`, including the
  fail-closed hook-backed CPU materialization check and assistant-only label
  preparation. Its raw reviewed JSON SHA-256 is
  `6fd61348cb0cba5fdf338935574deaec827da9ee1f827d8a43e6382993519198`
  and its signer-canonical payload SHA-256 is
  `9c55b95627b93e522eaebec5cb9e837b46d8e368065470aa45f55f488aeff873`.
  Its separate exclusive-create b804 DSSE queue envelope exists at
  `queue/pending/job-2026-nemo-v3-governed-attempt-15.json`, with raw SHA-256
  `93d5effe94740af9135c3ffa379c85df1aa88e6ad5717bc6421266d21bb9dbe7`.
  The signature verifies under keyId `b8041281c81c4caa` and binds the same
  canonical payload. Its single dispatch created A11oy run `30641766033`,
  which stopped before claim because runtime `60b9894e` did not contain the
  reviewed attempt-15 binding. Attempt 15 is now
  `RUNTIME_JOB_BINDING_REJECTED + PRE_CLAIM + NEVER_DISPATCH`; claim, job,
  prefetch receipt, training, receipt, and every publication effect are absent.
  Reviewed plaintext attempt 16 is at
  `jobspecs/nemo-v3-20260731-attempt-16-reviewed.json`. It binds protected
  generic runtime `b99f37260bcabf7f5c98cddbc5988a3ba87b766e`, derives authority
  from the exact attempt-15 quarantine/evidence boundary, and preserves the
  immutable execution revision through prefetch, claim, runner, and finalizer.
  Its raw JSON SHA-256 is
  `1daa8ea3a30a1d497f60431f9f4a33a9edd5d286236f3e8bf44240ef8630c5da`
  and its signer-canonical payload SHA-256 is
  `0b80bc0e42edd75de9e63f9f74f53df1d10c328d89b84c8481834a27fa4111f8`.
  Its separate exclusive-create b804 DSSE queue envelope exists at
  `queue/pending/job-2026-nemo-v3-governed-attempt-16.json`, with raw SHA-256
  `5f657aebb650c6a9c19b4b52e710236220fe7ab89e6a50488ee270017a78f756`.
  The signature verifies under keyId `b8041281c81c4caa` and its decoded payload
  matches the reviewed canonical payload byte-for-byte. The pre-dispatch
  validator rejected the envelope before event creation; A11oy PR #1217 then
  advanced the protected source. Attempt 16 is therefore immutable
  `STALE_SOURCE + PRE_DISPATCH_VALIDATOR_REJECTED + PRE_EVENT + NEVER_DISPATCH
  + NEVER_RESEND + NEVER_RESIGN` evidence. No runner, dispatch event, workflow
  run, claim, training, receipt, or release effect exists.
  A separate protected admission binds only the next plaintext identity,
  attempt 17, to exact A11oy source
  `cad529a2cef4cb43024bf4974ae155d89f33fa5b`, immutable owner-workflow blob
  `7cf0c877399471a084d3e70638ef50ec28d7f646`, and terminal relock run
  `30706177629`. It admits the preserved zero-event evidence without modifying
  the attempt-16 spec, envelope, evidence, or `NEVER_*` dispositions. It does
  not modify the attempt-16 evidence or grant runner, dispatch, receipt, or
  upload authority. The next separately reviewed plaintext is now
  `jobspecs/nemo-v3-20260801-attempt-17-reviewed.json`. It binds exact protected
  Bridge runtime `120a49206354ad98779ac46a65ca1fae45131e1c`, preserves the
  attempt-16 zero-event lineage, and keeps every science, license, receipt-only,
  and publication boundary closed. Its raw JSON SHA-256 is
  `de8e70374257f6df4baeeb7d7ce629cc9d8a8adeb0236be638e5eb239eb3c7b8`;
  its signer-canonical payload SHA-256 is
  `3aa118904933b0c5020cd21da1fc42531545a96f758d08ece3151e246255503c`.
  No attempt-17 queue envelope exists in this phase: the status is
  `AWAITING_ENGINE_SIGNATURE`, not dispatched or operational.
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
