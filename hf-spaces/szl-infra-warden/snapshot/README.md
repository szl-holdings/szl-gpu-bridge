---
title: SZL Infra Warden
emoji: 💻
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# SZL Infra Warden

A background loop that runs every 15 minutes while this Space is awake.

## What each cycle does

- Lists public SZLHOLDINGS Spaces from the unauthenticated Hub API (up to 100), checks the runtime stage of the first 20, and records up to 5 that are not RUNNING.
- Resolves DNS for anatomy.a-11-oy.com and router.a-11-oy.com and records whether each name resolved.

## What it does not do

It does not monitor GPUs, A6000 nodes, private Spaces, HTTP health, or costs.

## Records and limits

- Each cycle writes one JSON record to the dataset SZLHOLDINGS/frontier-54-runs under officers/. If that upload fails, the cycle is only printed to the Space log.
- Records are hash-chained (prev_sha256) only within one process lifetime and signed with an Ed25519 key that is generated at process start and never persisted. Every restart or rebuild makes a new key and resets the chain to 64 zeros, so a record from before a restart cannot be checked against the current key. The public key is self-asserted inside each record; there is no external trust root.
- Observed 2026-11-12: the officers/ folder of that dataset held no records from this Space (63 files, all from eternal-auditor, szl-dream-architect and estate-gardener), so uploads from this loop are failing or have not run. Check the dataset before relying on any record.
- This page is static. It does not show whether the loop is currently healthy. GET /healthz returns the cycle count since the last restart.

---

## Status — measured live 2026-11-14

**RUNNING** on `cpu-basic` shared hardware · **Live:** <https://szlholdings-szl-infra-warden.hf.space> · deployed revision [`21adbac695`](https://huggingface.co/spaces/SZLHOLDINGS/szl-infra-warden/tree/main)

## Estate provenance

🜂 **SZL Holdings estate artifact** — *the estate that receipts its own thinking.*

- Front door: <https://a-11-oy.com> · org: <https://huggingface.co/SZLHOLDINGS> · source: <https://github.com/szl-holdings>
- Doctrine v11 · Λ = Conjecture 1 (advisory) · estate honesty bus: <https://a-11-oy.com/api/a11oy/healthz>

## Honest limitations

- Shared `cpu-basic` hardware. No GPU or performance claims.
- This card describes function and measured status only — it is not itself a receipt. Verified estate receipts live on the tamper-evident chain and public datasets, not on this page.