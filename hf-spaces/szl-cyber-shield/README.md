---
title: SZL Cyber Shield
emoji: "🔒"
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# SZL Cyber Shield

A background loop that runs every 60 minutes while this Space is awake, plus a live 3D officer console (drag, zoom, hover - every node is a real probe result).

## What each cycle does (v3 - real probes, fail-closed)

- Verifies estate surface posture: the front door (a-11-oy.com) and the honesty bus must answer 200; the inference gateway must be healthy; the x402 paywall on /v1/completions must return 402 - the monetization surface must stand; gpu2 must stay gated behind Cloudflare Access (403).
- Scans this officer's recently published records for secret patterns (regex over public JSON - no credentials are ever loaded). Any hit is recorded FAIL.
- Writes one hash-chained, Ed25519-signed JSON receipt to SZLHOLDINGS/frontier-54-runs under officers/ (content-addressed).

## What it does not do

It is not a penetration test, an intrusion-detection system, or a credential scanner for private repositories. A PASS means the probed surfaces answered as expected this cycle - nothing more.

## Records and limits - v3 fixes (2026-11-14)

- v2 counted dataset files and wrote a fixed placeholder each cycle. v3 records only measured probe results, with honest UNAVAILABLE when a surface cannot be reached.
- The signing key and chain now PERSIST under /data: the v1 defect (a new key at every restart) is fixed. Restarts are marked CHAIN_RESTARTED, never silent (Law 9).

## Status - measured live 2026-11-14

**RUNNING** on shared cpu-basic hardware - Live: <https://szlholdings-szl-cyber-shield.hf.space> - canonical source: <https://github.com/szl-holdings/szl-gpu-bridge/tree/main/hf-spaces/szl-cyber-shield>

## Estate provenance

🞂 **SZL Holdings estate artifact** - *the estate that receipts its own thinking.*

- Front door: <https://a-11-oy.com> - org: <https://huggingface.co/SZLHOLDINGS> - source: <https://github.com/szl-holdings>
- Doctrine v11 - honesty bus: <https://a-11-oy.com/api/a11oy/healthz>

## Honest limitations

- Shared cpu-basic hardware. No GPU or performance claims.
- This card describes function and measured status only - it is not itself a receipt.
