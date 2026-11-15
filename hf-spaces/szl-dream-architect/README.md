---
title: SZL Dream Architect
emoji: "🌟"
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# SZL Dream Architect

A background loop that runs one vision cycle every 24 hours while this Space is awake, plus a live 3D officer console (drag, zoom, hover — every node is a real probe result).

## What each cycle does

- Reads the estate's open frontier questions (issues 135-138 of szl-holdings/szl-gpu-bridge) through the public GitHub API. If none are reachable, the cycle records UNAVAILABLE.
- Fetches the titles of the 3 newest arXiv papers matching "abstention AND LLM".
- Probes the estate compute plane - the governed inference gateway fronting the Salad GPU fleet - and drafts on it over TLS. If drafting is x402-paywalled, the cycle records PAYWALLED honestly: the paywall is the monetization surface and is never bypassed.
- Emits LABELED HYPOTHESES: nearest prior art, a preregistered test, and a rejection criterion. Fail-closed to an honestly-labelled template when the plane is unreachable.
- Writes one hash-chained, Ed25519-signed JSON receipt to the dataset SZLHOLDINGS/frontier-54-runs under officers/ (content-addressed, chain persisted under /data).

## What it does not do

Its output is labelled HYPOTHESIS. It does not change code, models or data, and a record is not a research result or a reviewed proposal (Law 6/7).

## Security review - resolved 2026-11-14 (v3)

The 2026-11-12 pause is lifted by the v3 fixes: the signing key and counter now persist under /data (restarts marked CHAIN_RESTARTED, never silent); the unencrypted fixed-IP drafting endpoint is replaced by the TLS-only estate gateway; the console ships with a fail-closed CSP (default-src 'self'), no external scripts, and no secrets client-side.

## Status - measured live 2026-11-14

**RUNNING** on shared cpu-basic hardware - Live: <https://szlholdings-szl-dream-architect.hf.space> - canonical source: <https://github.com/szl-holdings/szl-gpu-bridge/tree/main/hf-spaces/szl-dream-architect>

## Estate provenance

🞂 **SZL Holdings estate artifact** - *the estate that receipts its own thinking.*

- Front door: <https://a-11-oy.com> - org: <https://huggingface.co/SZLHOLDINGS> - source: <https://github.com/szl-holdings>
- Doctrine v11 - honesty bus: <https://a-11-oy.com/api/a11oy/healthz>

## Honest limitations

- Shared cpu-basic hardware. No GPU or performance claims.
- This card describes function and measured status only - it is not itself a receipt. Verified estate receipts live on the tamper-evident chain and public datasets, not on this page.
