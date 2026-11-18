---
title: The Receipts Atlas
sdk: docker
pinned: false
license: apache-2.0
---

# Receipts Atlas — evidence before authority

This is an unadmitted source candidate for the existing SZL Receipts Chain
Space, not proof that the new frontend is deployed. It replaces the old
misleading "every claim verified" and "live tamper-evident chain" copy.

It indexes the public Hugging Face dataset's **file paths** at an exact
40-hex Git revision obtained from a bounded metadata GET. A user can search
the enumerated paths and request an immutable path through that SHA.
The app computes SHA-256 of at most 256,000 downloaded bytes per inspection,
parses UTF-8 JSON/JSONL, and presents sanitized excerpts. It does NOT compare
that digest with a trusted prior manifest or independently verify a signature,
signer trust root, Merkle inclusion, publication event, trainer step, model
evaluation or runtime readiness.

The old frontend silently loaded **only the first 120 files** through mutable
main, omitted failed fetches and drew invented chain-like visual edges. This
replacement puts exact denominators, unknown states, and immutable references
in view.

## Definitions

- N_index: public dataset file paths in the one bounded metadata response.
- N_candidates: paths matching a filename heuristic for receipt-like JSON/JSONL.
  This is not a complete receipt-type census.
- N_inspected: distinct candidate files subjected to bounded read/parse.
- N_parsed: UTF-8 JSON/JSONL objects parsed in that bounded session.
- N_unavailable = N_inspected - N_parsed.
- C_parse = N_parsed / N_candidates when denominator is positive, otherwise
  unavailable. This is NOT cryptographic evidence quality or release readiness.
- Signed receipts independently verified: ZERO in this UI because no trusted
  signer certificate, checkpoint or subject-binding verifier is configured.

Public metadata source:
https://huggingface.co/api/datasets/SZLHOLDINGS/frontier-54-runs

Immutable content source (example path):
https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/<sha>/receipts/example.json

Other datasets, private SZL data, additional repositories and deployed runtime
attestations are outside the current one-dataset scope, not proven complete.

## Reproduce

From the repo root, run:
  node --test tests/test_receipts_exhibit.mjs

Serve the UI with:
  cd hf-spaces/szl-receipts-chain/snapshot
  python -m http.server 7860 --bind 127.0.0.1

Open http://127.0.0.1:7860 and use the search/inspection controls.
The Dockerfile copies the two new modules along with index.html.

Browser limits: metadata <=8 MB, file bytes <=256 KB, inspected candidate
batch <=12 and JSONL preview <=12 parsed records; secret-shaped field names
are redacted from the preview. Constellation dots are **indexed filenames**,
not a verified lineage, causal chain or certificate proof.

## Publication and research

Protected-main CI, public metadata and immutable file links do not establish
deployed HF Space/source parity. Publication needs owner maintenance approval,
exact revision and independent runtime readback. Rights, signature and
per-lane training evidence are independent gates, retained as HOLD.

See docs/RECEIPT_ATLAS_RESEARCH_20261117.md for first-party comparisons,
mathematical integrity formulas and specifically deferred implementation.