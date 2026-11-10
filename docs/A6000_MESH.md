# The A6000 Mesh — daisy-chained, brain-fed, receipt-backed

**The vision (owner directive):** all four Massed Compute nodes meshed and
daisy-chained like the Salad fleet was — every node feeds the brain node,
the brain feeds back curriculum, and the receipts chain is the shared state.

## Topology

```
 .118 (farm 1) ──┐                        ┌── curriculum manifest (pull)
 .191 (farm 2) ──┼──> HF receipts chain ──┤
 .31  (infer)  ──┘    (the shared bus)     └──> .145 (BRAIN node: curator)
                                                │  pulls all receipts
                                                │  builds weekly curriculum
                                                └── pushes curriculum manifest
                                                    back to the chain → farms pull it
```

**Why HF as the bus, not direct node-to-node:** every node already holds an
HF token; the chain is append-only, signed, and public-checked; and no
node needs to accept inbound SSH from the others (smaller attack surface —
the security team's rule). Direct SSH daisy-chain is optional below.

## Roles

| Node | Mesh role | Pushes | Pulls |
|---|---|---|---|
| .118 | Farm 1 | brain-v35 + hatun24 receipts, checkpoints | curriculum manifest |
| .191 | Farm 2 | a11oy-code + bench receipts, checkpoints | curriculum manifest |
| .31 | Inference | serving telemetry (read-only; DO NOT modify) | nothing (production) |
| .145 | **Brain** | the curated weekly curriculum, the proving-ground receipts | ALL receipts (it is the curator) |

## The daisy-chain contract (every node, hourly)

1. **Push:** any new receipts/checkpoint manifests in `~/szl-legs/*/out/`
   go to the chain (`frontier-54-runs`) — content-addressed, signed.
2. **Pull:** the brain's `curriculum/manifest.json` (latest SHA) is fetched;
   if newer than the local copy, the next training cycle uses it.
3. **Heartbeat:** each node appends a mesh heartbeat receipt (node id,
   uptime, GPU util via energy_meter semantics — UNAVAILABLE never zero).
4. **The loop closes weekly:** the brain node's curator reads every
   receipt from every node, keeps the gated-pass pairs, and mints the
   next curriculum — the Ouroboros (docs/OUROBOROS_CURRICULUM.md).

## Direct SSH daisy-chain (optional, for large checkpoints)

Install the estate key on all four nodes, then each farm can rsync
checkpoints directly to the brain (faster than HF for multi-GB files):

    # on each farm node, hourly:
    rsync -az --progress ~/szl-legs/*/out/checkpoints/ \
        Ubuntu@216.81.248.145:~/mesh-incoming/<node>/

The brain node never pushes TO the farms over SSH — curriculum flows via
the chain so it is signed and public-checkable (Law 5).

## What "meshed" means, honestly

A mesh is not a picture — it is a contract: every node can reach the shared
state, every node proves its liveness hourly, every artifact moves with a
receipt, and the brain's output is checkable by every farm. When the new
stack instance lands, it takes the brain seat (.145 returns to
hatun+science) and the mesh grows by one node — no re-architecture.

*Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr.*
