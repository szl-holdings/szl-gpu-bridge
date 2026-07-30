# Security model — szl-gpu-bridge

*Direction-asymmetric by construction: the cloud can only append signed specs to a public queue; the laptop can only push artifacts to HF repos it already controls. Neither side holds the other's credentials.*

## Trust roots

| Party | Trusts | Established by |
|---|---|---|
| Laptop | reviewed engine keyring: `5c6cf59741ade920` and provisional `815714c8d4ae3e4d` verification-only; coordinated administrative-recovery key `b8041281c81c4caa` active | baked into `bootstrap.ps1` and independently bound by the signed job authorization |
| Cloud | laptop keyId announced by owner after bootstrap prints it | out-of-band (owner tells the session; cloud pins it for `verify-receipt.mjs --expect-keyid`) |

Private keys never move: an engine signing key stays ACL-locked on the
controlled signing host; the laptop receipt seed is generated on-metal and
never uploaded. Losing an engine private key does not authorize rewriting its
public pin. Recovery requires a recorded incident, a new public keyring entry,
a distinct job generation, and a protected change. Historical envelopes remain
verifiable through the verification-only public pin.

The coordinated active key is an administrative recovery trust root. No
cryptographic continuity with either verification-only predecessor is claimed.
Attempt 2, successor generation 3, and transport-unrepresentable attempt 4 are
immutable historical envelopes, not execution authority. Their quarantine
records bind the original exact envelope and payload digests and mark them
`NEVER_DISPATCH`. Quarantine must never be implemented by deleting, rewriting,
re-signing, or retrying those bytes.

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

## Receipt verification is over exact signed bytes

Laptop receipts carry `bodyBase64` — the exact canonical bytes that were
signed. `cloud/verify-receipt.mjs` verifies the ed25519 signature over those
bytes and then requires the human-readable `receipt` display copy to match
them; re-serialization is display-only, never the verification path.
**Why:** Python `json.dumps` and JS `JSON.stringify` disagree on
integer-valued floats (`2.0` vs `2`) and exponent forms (`1e-07` vs `1e-7`);
a re-canonicalizing verifier would reject honest receipts intermittently —
a fail-closed system must never manufacture false negatives and call them
honesty. The eval→training chain pin (`trainingReceiptSha256`) is likewise
sha256 of the training receipt's decoded `bodyBase64`, recomputable
identically in any language.
