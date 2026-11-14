---
title: SZL Chief Engineer
emoji: "🛠"
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# SZL Chief Engineer

A background loop that runs every 30 minutes while this Space is awake, plus a live 3D officer console (drag, zoom, hover - every node is a real probe result).

## What each cycle does

- Reads the most recent GitHub Actions run for 4 tier-0 repositories (szl-holdings/a11oy, szl-forge, .github, szl-frontier) through the unauthenticated public API, and records each whose latest run concluded failure.
- Probes the estate honesty bus (https://a-11-oy.com/api/a11oy/healthz) and the estate compute plane (the governed inference gateway fronting the Salad GPU fleet).
- Writes one hash-chained, Ed25519-signed JSON receipt to SZLHOLDINGS/frontier-54-runs under officers/ (content-addressed).

## What it does not do

It does not watch the other repositories, fix CI, review pull requests, or find bugs.

## Records and limits - v3 fixes (2026-11-14)

- The signing key and chain now PERSIST under /data: the v1 defect (a new key at every restart, chain reset to 64 zeros, pre-restart records unverifiable) is fixed. Restarts are marked CHAIN_RESTARTED, never silent (Law 9).
- This Space's own /api/constellation shows whether the loop is currently healthy.

## Status - measured live 2026-11-14

**RUNNING** on shared cpu-basic hardware - Live: <https://szlholdings-szl-chief-engineer.hf.space> - canonical source: <https://github.com/szl-holdings/szl-gpu-bridge/tree/main/hf-spaces/szl-chief-engineer>

## Estate provenance

🞂 **SZL Holdings estate artifact** - *the estate that receipts its own thinking.*

- Front door: <https://a-11-oy.com> - org: <https://huggingface.co/SZLHOLDINGS> - source: <https://github.com/szl-holdings>
- Doctrine v11 - honesty bus: <https://a-11-oy.com/api/a11oy/healthz>

## Honest limitations

- Shared cpu-basic hardware. No GPU or performance claims.
- This card describes function and measured status only - it is not itself a receipt.
