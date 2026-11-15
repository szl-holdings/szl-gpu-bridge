---
title: SZL Research Council
emoji: 🧪
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# SZL Research Council

A background loop that runs once every 24 hours while this Space is awake.

## What each cycle does

- Writes a record that lists the arXiv categories cs.AI, cs.CL and stat.ML.

## What it does not do

It does not fetch or read arXiv, design experiments, or track research. The category list is a constant.

## Records and limits

- Each cycle writes one JSON record to the dataset SZLHOLDINGS/frontier-54-runs under officers/. If that upload fails, the cycle is only printed to the Space log.
- Records are hash-chained (prev_sha256) only within one process lifetime and signed with an Ed25519 key that is generated at process start and never persisted. Every restart or rebuild makes a new key and resets the chain to 64 zeros, so a record from before a restart cannot be checked against the current key. The public key is self-asserted inside each record; there is no external trust root.
- Observed 2026-11-12: the officers/ folder of that dataset held no records from this Space (63 files, all from eternal-auditor, szl-dream-architect and estate-gardener), so uploads from this loop are failing or have not run. Check the dataset before relying on any record.
- This page is static. It does not show whether the loop is currently healthy. GET /healthz returns the cycle count since the last restart.

---

## Status — measured live 2026-11-14

**RUNNING** on `cpu-basic` shared hardware · **Live:** <https://szlholdings-szl-research-council.hf.space> · deployed revision [`b6d0c16300`](https://huggingface.co/spaces/SZLHOLDINGS/szl-research-council/tree/main)

## Estate provenance

🜂 **SZL Holdings estate artifact** — *the estate that receipts its own thinking.*

- Front door: <https://a-11-oy.com> · org: <https://huggingface.co/SZLHOLDINGS> · source: <https://github.com/szl-holdings>
- Doctrine v11 · Λ = Conjecture 1 (advisory) · estate honesty bus: <https://a-11-oy.com/api/a11oy/healthz>

## Honest limitations

- Shared `cpu-basic` hardware. No GPU or performance claims.
- This card describes function and measured status only — it is not itself a receipt. Verified estate receipts live on the tamper-evident chain and public datasets, not on this page.