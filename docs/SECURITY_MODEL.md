# Security model — szl-gpu-bridge

*Direction-asymmetric by construction: the cloud can only append signed specs to a public queue; the laptop can only push artifacts to HF repos it already controls. Neither side holds the other's credentials.*

## Trust roots

| Party | Trusts | Established by |
|---|---|---|
| Laptop | engine pubkey `5c6cf59741ade920` | baked into `bootstrap.ps1` at paste time (TOFU — see limits) |
| Cloud | laptop keyId announced by owner after bootstrap prints it | out-of-band (owner tells the session; cloud pins it for `verify-receipt.mjs --expect-keyid`) |

Private keys never move: engine key stays in the cloud workspace; laptop seed is generated on-metal and never uploaded.

## Inbound path (job specs) — threats & mitigations

- **Forged/tampered spec in queue** (repo compromise, MITM, CDN poisoning): daemon hands the file to `runjob.py`, which verifies the DSSE envelope (PAE, ed25519) against the *pinned* SPKI **before reading any field**. Bad sig → refusal. Neither branch protection nor GitHub auth is load-bearing for laptop safety — only the signature is.
- **KeyId collision games**: verification checks both the derived keyId *and* byte-equality of the embedded SPKI with the pin.
- **Replay of an old spec**: `expiresAt` window + daemon seen-ledger keyed by idempotent `jobId`.
- **Malicious-but-signed spec** (engine key compromise): blast radius = one QLoRA run on pinned public HF assets. Specs carry no shell commands — the runner executes a *fixed* pipeline with schema-bounded numeric knobs (`additionalProperties: false`, epoch/seq caps). Exfiltration surface: none carried in specs (no secrets present).
- **Resource exhaustion**: fail-closed gates — VRAM probe, disk floor, hard wallclock kill, NaN-loss abort — each emitting a signed BLOCKED receipt.

## Outbound path (receipts/weights) — threats & mitigations

- **Fabricated success**: the cloud verifies every receipt (`verify-receipt.mjs`); an unverifiable claim is treated as no claim. Eval receipts must pin the exact training-receipt sha (chain check).
- **Wrong-key receipts** (stolen HF token elsewhere): `--expect-keyid` pin against the owner-announced laptop key.
- **Dataset substitution on the laptop**: `runjob.py` re-hashes the downloaded dataset file against the spec's sha256 pin; mismatch → BLOCKED.
- **Base-model drift**: base pinned by commit revision, never a branch.

## Honest limits (stated, not hidden)

1. **TOFU bootstrap**: the first paste is the trust ceremony. A pre-compromised laptop or a MITM'd first fetch defeats everything after it. Mitigation: owner can read `bootstrap.ps1` in this public repo before pasting; the pinned pubkey is visible in plain text.
2. **Receipts are attestations**, not proofs of computation. A malicious keyholder could sign false measurements; the chain makes lying *consistent and auditable*, not impossible. We say "receipt-verified", never "proven".
3. **No liveness guarantee**: cloud sees the laptop only through pushed receipts. Silence is indistinguishable from power-off — and is reported as exactly that (UNAVAILABLE).
4. **Public queue metadata**: job specs are world-readable by design (no secrets inside). Anyone can see *what* we train; only the laptop will act on it, and only when signed.
5. **GGUF/export pins**: any future GGUF export inherits the known khipu gotcha — artifact-form hashes verify only against the artifact actually hashed; receipts must name the exact file form they pin.
