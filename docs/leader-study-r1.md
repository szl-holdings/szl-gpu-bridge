# szl-gpu-bridge domain leader study r1 (propose-only)

Date: 2026-11-22. Method: web-verified 2026 GPU-orchestration leaders. Documentation only; no code/CI/config changes; all absorb actions are FUTURE PROPOSALS (non-authorizing); merges remain gated.

## Domain: GPU fleet operations + receipts

## Leaders studied (web-verified 2026-11-22)

1. **SkyPilot v0.13** (July 2026; ~10.5k stars): lifecycle hooks, HF storage integration, batch inference abstractions; SkyPilot Endpoints (June 2026) for production inference on owned clusters.
2. **Runhouse** - distributed compute/cluster status patterns.
3. **Weights & Biases** - experiment receipts and cost attribution conventions.

## Absorb actions (FUTURE PROPOSALS, non-authorizing)

1. (SkyPilot lifecycle) Job lifecycle receipts: submit/start/stop/terminate each persisted as its own receipted state transition (matches open PR #335's persist-restart-attempt-before-POST theme, referenced read-only).
2. (Idempotency discipline) Idempotency tokens for all non-idempotent submission POSTs; duplicate submit refused with receipted reason.
3. (W&B) Per-job cost attribution with estimate vs actual separated and unknowns explicit.

## Do not do (per governing seat)

Do not merge, bypass the signature wall, or claim implementation beyond this card.
