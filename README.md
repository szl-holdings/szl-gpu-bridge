# szl-gpu-bridge

**Doctrine-governed training bridge: cloud-signed job specs → owner's GPU metal → signed receipts back. No inbound access, no secrets on the wire, fail-closed in both directions.**

This repo is the *only* channel between SZL's cloud sessions and the owner's Windows/RTX training laptop. The cloud cannot reach the laptop; the laptop polls this repo. Every job spec is DSSE-signed by the szl-quant engine key before it enters the queue, and the laptop refuses — with an honest, signed **BLOCKED** receipt — anything whose signature, schema, or resource gates fail.

## How it works

```
cloud (this repo)                          owner's laptop (Windows, RTX)
─────────────────                          ─────────────────────────────
cloud/sign-job.mjs                          laptop/daemon.ps1 (scheduled task)
  └─ DSSE-sign jobspec ──► queue/pending/ ◄── poll raw.githubusercontent.com
                                              └─ laptop/runjob.py
                                                 1 verify DSSE vs PINNED engine pubkey
                                                 2 fail-closed gates (VRAM, disk, wallclock)
                                                 3 Unsloth QLoRA train (pinned deps)
                                                 4 eval suite
                                                 5 sign training/eval receipts (laptop key)
                                                 6 hf upload weights + receipts ──► HF Hub
cloud/verify-receipt.mjs ◄── pull receipts from HF, verify, then (and only then) claim anything
```

- **Inbound** (jobs): the laptop polls `queue/pending/*.json` over public raw HTTPS — no laptop GitHub auth needed. Signature verification against the **pinned** engine pubkey (`keys/engine_pubkey.json`, keyId `5c6cf59741ade920`, baked into `bootstrap.ps1`) happens **before** any field of a spec is acted on.
- **Outbound** (results): the laptop pushes weights + signed receipts to Hugging Face with its already-authenticated `hf` CLI — the same proven path that shipped 3 GB of khipu weights. The cloud verifies receipts independently; **an unverified claim is treated as no claim**.
- **Trust roots**: the laptop trusts exactly one engine pubkey (baked at bootstrap). The cloud trusts the laptop key that the owner announces after bootstrap prints its keyId. Neither side ever transmits a private key or token.

## Job specs

`schema/jobspec.v1.json` defines the contract. Every reference is pinned: base model by revision, dataset by revision + sha256, output repos named explicitly, VRAM/wallclock gates stated. Specs carry `expiresAt` and a unique `jobId`; the daemon keeps a seen-ledger so a replayed spec is a no-op.

Current queue: `queue/pending/` — first job trains **SZL-Quant-1.5B** on the receipt-derived
[`SZLHOLDINGS/szl-quant-sft-v1`](https://huggingface.co/datasets/SZLHOLDINGS/szl-quant-sft-v1) dataset (every training row traceable to a DSSE-signed backtest receipt and recomputable from content-addressed archives).

## One-paste bootstrap (owner)

On the laptop, in an **elevated** PowerShell:

```powershell
irm https://raw.githubusercontent.com/szl-holdings/szl-gpu-bridge/main/laptop/bootstrap.ps1 | iex
```

It pins a Miniconda py3.12 env (torch cu124-class, triton-windows, bitsandbytes with the two known-bad Windows versions excluded, xformers, unsloth, huggingface_hub, pynacl), generates the laptop signing key, prints its keyId for you to announce, registers the daemon as a scheduled task (runs at startup, survives lock screen), and starts polling. Idempotent — safe to re-run.

## Honest limits (LAW)

- Nothing trains until the owner pastes the bootstrap once — there is no remote-start path, by design.
- From the cloud, laptop liveness is **inferred only from pushed signed receipts**; every cloud-side claim about a run is REPORTED-from-receipt or it does not exist.
- Receipts are **attestations** (ed25519 over canonical JSON), not cryptographic proofs of computation — we say "receipt-verified", never "proof of training".
- Throughput/VRAM numbers for the owner's GPU are **UNAVAILABLE** until a run reports them; nothing here quotes vendor marketing as measurement.
- Advisory research infrastructure. Paper-only lineage (szl-quant). Not financial advice.

## Repo map

| path | what |
|---|---|
| `schema/jobspec.v1.json` | job-spec contract v1 |
| `cloud/sign-job.mjs` | DSSE-sign a spec into the queue (engine key) |
| `cloud/verify-receipt.mjs` | independently verify laptop receipts pulled from HF |
| `laptop/bootstrap.ps1` | one-paste pinned installer + scheduled task |
| `laptop/daemon.ps1` | poll → verify → run → idle loop |
| `laptop/runjob.py` | verify → gates → Unsloth train → eval → sign → upload |
| `queue/pending/` | DSSE-signed job specs awaiting the laptop |
| `queue/done/` | specs the cloud has confirmed via verified receipts |
| `keys/engine_pubkey.json` | pinned engine identity (`5c6cf59741ade920`) |
| `docs/RESEARCH_MEMO.md` | leaders studied → lessons adopted → what SZL does differently |
| `docs/SECURITY_MODEL.md` | threat model, both directions |

Apache-2.0. Adapted patterns are attributed in `NOTICE`; LGPL/AGPL upstream code is depended on or avoided, never vendored.
