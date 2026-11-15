# hf-spaces — canonical sources for the always-on officers

**Policy (doctrine): GitHub `szl-holdings/szl-gpu-bridge` is CANONICAL for officer
source code. The Hugging Face Spaces (`SZLHOLDINGS/estate-gardener`,
`SZLHOLDINGS/eternal-auditor`, and the five officers) are deploy mirrors.**
Until now the officers existed only on HF with no GitHub source of record —
a provenance gap that made review impossible. This directory closes it.

## Fixes in this revision (advisory 2026-11-09)

| Officer | Defect | Fix |
|---|---|---|
| `estate-gardener` | Could REPLACE an existing dataset README with a generic Apache-2.0 card | APPEND-only license block; byte-prefix preservation check; refuses the write unless the new content is a strict superset; before/after sha256 in every receipt |
| `eternal-auditor` | Truncated signatures to 96 hex chars; cycle state reset on restart | Full 64-hex HMAC signatures stored whole; state persisted to `/data/audit-state.json`; cycle receipts chain to the previous receipt's sha256; restarts are marked `BROKEN-RESTARTED`, never silent |

## Deploying a mirror

1. Review/merge the canonical change here.
2. Copy `hf-spaces/<officer>/app.py` into the Space repo
   (`git clone https://huggingface.co/spaces/SZLHOLDINGS/<officer>`),
   commit with the canonical commit SHA in the message, push.
3. The Space rebuilds; its first cycle receipt after deploy must cite the
   canonical SHA (Law 7: every claim carries a receipt).

## Tests

`python3 tests/test_energy_and_officers.py` — 26 offline tests, no GPU
required, covering: dead-meter honesty (UNAVAILABLE, never zero), both NVML
bindings, power-integral fallback, gardener append-only preservation, and
auditor chain persistence across restarts.
