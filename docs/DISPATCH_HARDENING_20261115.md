# Dispatch Hardening Doctrine — 2026-11-15

**Status: MANDATORY for every SaladCloud (and any ephemeral-host) GPU dispatch.**
This document is the codification of the 2026-11-14/15 estate-battery incident
postmortem. It exists so that the failure mode is prevented by process, not by
memory. The runbook (`docs/NONSTOP_TRAINING_RUNBOOK.md`) already requires
boot-receipt-first; this doctrine closes the gap **upstream of Python**: the
battery lost 1h47m of GPU time to a failure that happened entirely inside
`pip install`, before a single line of workload code ran.

---

## 1. The incident (what actually happened)

The `szl-estate-battery-v1` container (39-model abstention battery, 6 probes
per model, receipt per model into `SZLHOLDINGS/frontier-54-runs`) was
dispatched 2026-11-14 18:24Z on the RTX 4090 class. At 20:11Z it had produced
**zero receipts**. Verified facts:

1. The battery script **does** upload a receipt per model plus a final
   manifest; the HF token **can** write to the dataset (live commit test
   succeeded). The pipeline was not the problem.
2. The dispatched command **pinned nothing**: `accelerate`, `trl`,
   `datasets` were installed unpinned. In late 2026 those versions demand a
   torch newer than the image's torch, so pip resolved the conflict by
   **silently re-downloading multi-GB torch + CUDA wheels at every boot**.
3. The command also lacked the DNS fix and pip timeouts that later train
   dispatches carried. A hung pip install and a slow-but-working pip install
   are **indistinguishable from the outside**: same container state, same
   zero-telemetry signature.
4. With no canary, the first possible evidence of life was the first
   per-model receipt — which never came.

Result: 1h47m of paid 4090 time produced nothing, and the failure was only
detectable by absence. The fix went through v4 (pin `transformers` —
receipts flowed, but wrong stack for qwen3.5 models → honest `BATTERY_FAIL`s),
v5 (right stack, wrong base image + resolver conflict), and finally **v6**,
which ran clean end-to-end.

## 2. Root cause, stated once

> **An unpinned dependency on an ephemeral host is a bet against a package
> index you do not control, placed with GPU money, with no telemetry while
> the bet resolves.**

Every element of the checklist below removes one term of that bet.

## 3. Mandatory dispatch checklist

Every GPU dispatch command (Salad container command, Massed cloud-init, any
ephemeral host) MUST include all of the following. A dispatch missing any
item is not ready to ship.

| # | Requirement | Why (incident link) |
|---|---|---|
| 1 | **Prebuilt image whose torch/CUDA matches the model stack.** Never install torch via pip at boot. | Unpinned `accelerate`/`trl` pulled a newer torch → multi-GB silent re-download each boot. |
| 2 | **Pin every pip package** (`==`), including `transformers`, `peft`, `accelerate`, `trl`, `bitsandbytes`, `datasets`. | Same. Also record pins in the run receipt for audit. |
| 3 | **Boot canary receipt BEFORE the main pip block.** The canary only needs `huggingface_hub` (pure-python): it proves token + repo + upload path within ~5 min of instance start. | The 1h47m loss was invisible because the first possible receipt came only after pip + model load. |
| 4 | **DNS fix + pip timeouts:** write a working `resolv.conf`; export `PIP_DEFAULT_TIMEOUT=45 PIP_RETRIES=2`. | Salad resolv.conf breakage + default pip timeouts make a hang the null hypothesis. |
| 5 | **Upload retries:** wrap every receipt upload in ≥4 attempts with backoff. | Transient HF 5xx must not eat a completed unit of work. |
| 6 | **Resume-skip idempotency:** before each unit of work, check for its existing receipt; skip and fold into the manifest if present. | An eviction mid-run costs at most one unit; the run never restarts from zero. |
| 7 | **Live log tail:** upload the tail of the run log every ~5 min to the mesh logs path. | Turns "silence" from a hypothesis into a diagnosable state. |
| 8 | **Silence = alarm (watcher doctrine):** no new canary/receipt inside the expected window triggers investigation, not waiting. Also: watchers must list the receipt folder directly (`/tree/main/<folder>`) — **the `?recursive=true` tree API silently truncates** and has reported `receipts=0` while receipts existed. | Both the original loss and a near-misdiagnosis during verification came from trusting absence of evidence. |

## 4. Template preamble

Drop-in preamble for a hardened dispatch command (adapt paths, repo, names):

```bash
# 4) DNS + pip hardening
printf 'nameserver 1.1.1.1\nnameserver 8.8.8.8\noptions timeout:2 attempts:2\n' > /etc/resolv.conf
export PIP_DEFAULT_TIMEOUT=45 PIP_RETRIES=2

# 1-3) Boot canary BEFORE main pip: proves token+repo+upload within minutes.
python3 - <<'PY'
import json, datetime, platform
from huggingface_hub import HfApi
api = HfApi(token=open('/run/secrets/hf_token').read().strip())
api.upload_file(
    path_or_fileobj=json.dumps({
        'kind': 'szl.<lane>/run-start',
        'ts': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host': platform.node(),
    }).encode(),
    path_in_repo='receipts/<lane>-run-start.json',
    repo_id='SZLHOLDINGS/frontier-54-runs',
    repo_type='dataset',
)
print('CANARY_OK', flush=True)
PY

# 2) Pinned stack — versions chosen to match the image's torch
pip install --quiet 'transformers==X.Y.Z' 'peft==X.Y.Z' 'accelerate==X.Y.Z'

# 7) Background live-log tailer (5-min cadence, best-effort)
(while true; do sleep 300; tail -c 20000 /tmp/<lane>.log > /tmp/tail.bin 2>/dev/null; \
  <upload /tmp/tail.bin to mesh/<lane>-live.log, best-effort>; done) &
```

Every receipt upload goes through a retry helper (≥4 attempts, backoff);
see checklist item 5.

## 5. The v6 battery command (reference implementation)

The live, verified reference is the `szl-estate-battery-v1` container at
command version 6 (2026-11-15): image `docker.io/pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime`,
all checklist items present, resume-skip verified (pre-existing receipts
skipped and folded into the manifest). Its boot canary landed 02:15:19Z and
run-start receipt 02:16:39Z on the first v6 instance — end-to-end proof the
doctrine works. Dispatch specs live with the estate audit records; this doc
is the doctrine, not the spec.

## 6. Standing rule for every operator and session

- New dispatch without the checklist → **do not ship it**. Fix the command first.
- Silence inside the expected window → **investigate, never wait it out**.
- Verify receipts by **direct folder listing or direct fetch**, never the
  recursive tree API.
- Losses are recorded honestly, with timestamps, and codified here. The 1h47m
  is unrecoverable; the doctrine is what we bought with it.

*Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr.*
