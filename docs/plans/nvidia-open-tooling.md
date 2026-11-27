# NVIDIA open-license tooling

Status: draft. This document is a plan and a set of non-dispatchable specs.
It does not train, launch, spend, build an image, push an image, write to
the Hub, or change `laptop/nemo_v3_contract.py`.

Stephen's approval of this planning pass was stated on 2026-11-27. A separate
approval artifact was not found in `szl-gpu-bridge`, `szl-constellation`, or
`szl-retrieval-bench`. That artifact remains UNKNOWN.

Runs, when a later review authorizes one, go through Saywa as the single
launcher, on Massed or Salad only, at $100/run, $500/day, and $5,000/month.
No Saywa launcher implementation was found in those three repositories.
`szl-ops-runbook` names Saywa as an SRE lane. The executable entrypoint is
UNKNOWN, so every spec in this pass sets `launchAuthorized` to false.

## What stays in place

The existing trainer is Hugging Face / PEFT. `tools/salad/salad_worker.py`
builds a TRL `SFTTrainer` with a `LoraConfig`. That path is unchanged.

`laptop/nemo_v3_contract.py` still owns the recipe ranges and the
`licenseId` gate. For coordinated recoveries at successor generation 10 and
above, `base.licenseId` must be `nvidia-nemotron-open-model-license`. The
integer recipe bounds stay `maxSeqLength` 256–4096, `loraR` 1–64,
`loraAlpha` 1–256, `batchSize` 1–1, `gradAccum` 1–64, with the existing
epoch, learning-rate, optimizer, and checkpointing checks. Tests in
`tests/test_nemo_framework_backend.py` read that file as text and fail if
those literals move. This pass does not edit the contract.

## License register

License fields below are the Hugging Face model API `cardData.license` and
`cardData.licenseName` (`license_name`) read on 2026-11-27. The API tag is
`license:other` whenever `cardData.license` is `other`. Amaru's license
check is PENDING for every family. Nothing in this register is admitted for
training or evaluation.

The NVIDIA license URLs live under `/agreements/enterprise-software/`. That
path is where NVIDIA publishes the open-model license text. It is not, by
itself, a finding that the weights are NVIDIA AI Enterprise. Amaru still
has to read the license.

| Family | Hugging Face repo | API `license` | API `license_name` | Revision | Named open set | Amaru |
| --- | --- | --- | --- | --- | --- | --- |
| Nemotron Nano-4B | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | `other` | `nvidia-nemotron-open-model-license` | `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` | yes, and it matches the v3 gate | PENDING |
| Nemotron Nano-9B-v2 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | `other` | `nvidia-open-model-license` | `6533e8de2c68e4536bf7c411d7a3ce5734111476` | yes, and it does not match the v3 gate | PENDING |
| Illapa (Nemotron flagship, internal name) | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | no binding | PENDING |
| NeMo Retriever embedder candidate | `nvidia/llama-nemotron-embed-1b-v2` | `other` | `nvidia-open-model-license` | `113abe4acafa848e77ead9c0623205e511932348` | yes | PENDING |
| NeMo Retriever reranker candidate | `nvidia/llama-nemotron-rerank-1b-v2` | `other` | `openmdw-1.1` | `828765652b05bd439c9789d2a6d093db1caa1443` | no | PENDING |
| PyG library (not an HF model) | none | not an HF model | GitHub SPDX `MIT`; PyPI 2.8.1 `MIT` | library pin UNKNOWN | MIT is in the named set | PENDING |
| Estate graph dataset | `SZLHOLDINGS/szl-estate-graph` | `apache-2.0` | none | dataset sha `8262675ce6550c0bb180f54139c34ccd03ec5e77` | Apache-2.0 is in the named set | PENDING |

Named open set used by the new admission code: `nvidia-nemotron-open-model-license`,
`nvidia-open-model-license`, `apache-2.0`, `mit`, `bsd-2-clause`,
`bsd-3-clause`. `openmdw-1.1` is recorded and not treated as a member.
`cc-by-nc-4.0` on `nvidia/NV-Embed-v2` is excluded.

NeMo Framework source `NVIDIA-NeMo/NeMo` reports GitHub SPDX `Apache-2.0`.
The commit to install is UNKNOWN. NGC and `nvcr.io` containers are forbidden.

PyTorch's GitHub license API returns SPDX `NOASSERTION`. The `LICENSE` file
on `pytorch/pytorch` `main` contains BSD-style redistribution clauses. The
exact SPDX identifier and the commit are UNKNOWN. No PyTorch image is pinned.

### Illapa

No repository in the org search binds the name Illapa to a Hugging Face
repo. Two Nemotron-3 text models were observed and left unbound:

- `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` at
  `77df655d5e9f8362164ed14dd8b48f8bce657498`, API `license_name`
  `openmdw-1.1`.
- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` at
  `2dc98e2afe4face0e4ce40972a915c45368bd34a`, API `license_name`
  `nvidia-nemotron-open-model-license`.

Neither is Illapa until that binding is written down and the license check
is done.

### Retriever models that were looked up and not selected

These public NVIDIA repos also returned API `license_name` `openmdw-1.1`
on 2026-11-27. They are not the baseline:

- `nvidia/Nemotron-3-Embed-1B-BF16` (`c0c9fea93ea424587517f2c59e20db9f1d6bf615`)
- `nvidia/Nemotron-3-Embed-8B-BF16` (`d1f2f25730bbd775b99b29185134bc86653bf2d1`)
- `nvidia/Nemotron-3.5-Rerank-1B-BF16` (`3beced21c39e9b7749480654af0a73fa9b343d4d`)
- `nvidia/Nemotron-3.5-Rerank-8B-BF16` (`a9fe147be7a1114faf3df1f5e700e798420d06aa`)

`nvidia/llama-embed-nemotron-8b` reports `license_name` `customized-nscl-v1`.
That name is outside the named open set. It is not the baseline.

## (a) NeMo Framework trainer

Code: `tools/nemo_framework/backend.py`.
Draft spec: `jobspecs/drafts/nemo-framework-trainer.v0.json`.
The draft is not under `queue/`. CI's nemo v3 schema check only globs
`jobspecs/nemo-v3-*.json`, so this draft is not forced through that schema
and does not relax it.

`admit()` accepts two backend names:

- `hf-peft` records that `tools/salad/salad_worker.py` remains the trainer
  and that this module does not call it.
- `nemo-framework` is the additional backend for Nano-4B, Nano-9B-v2, and
  Illapa.

Both receipts set `launchAuthorized`, `trainingStarted`, and `imagePushed`
to false. A request with `launch: true`, a launcher other than `saywa`, a
provider outside `massed` and `salad`, a cap other than the ceiling below,
an `nvcr.io` / NGC / AI Enterprise image reference, a base image other than
the pinned digest, or a successor digest other than `UNKNOWN` is refused.

The pinned base is the image named for this pass:

`ghcr.io/szl-holdings/szl-train-pinned@sha256:8eb919654e0da406d41a2e307e7705cce8f6b79ad0be9556d74d4c4640e63235`

An unauthenticated manifest request to GHCR for that digest returned HTTP
401. Whether the registry currently serves it is UNKNOWN from this
environment. The digest is still the only base the new backend will name.

The pinned image, as described by `training-image/README.md`, ships
`transformers==5.22.2`, `peft==0.21.2`, and `trl==1.17.1` on the CUDA 12.6
PyTorch 2.14.1 runtime. It does not ship NeMo. Adding NeMo Framework is an
image change, so it needs a new digest. That digest is `UNKNOWN` and
`NOT_BUILT`. This pass does not write a Dockerfile, does not run a build,
and does not push.

When a later review builds that image, the plan is:

1. Start `FROM` the pinned digest above. Do not retag it.
2. Install NeMo Framework from the Apache-2.0 git source at one reviewed
   commit. Do not use an NGC or `nvcr.io` container.
3. Publish a new immutable digest. Record it only after the owner verifies
   the signature and the read-back. Until that digest exists, `admit()`
   keeps returning `NEMO_LAYER_DIGEST_UNKNOWN`.
4. Keep the HF/PEFT worker on the current pin. Do not point reviewed
   nemo v3 attempts at the new backend.

Nano-4B's API `license_name` matches the v3 gate, and attempt 17 already
binds `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` at the same revision through
the HF/PEFT contract. The new backend still does not train it. The blocker
is the missing NeMo layer digest, plus the pending license check.

Nano-9B-v2's API `license_name` is `nvidia-open-model-license`. Putting that
string into a coordinated v3 spec would fail the existing gate. This pass
does not change the gate to accept it. The family stays blocked with
`LICENSE_NAME_DIFFERS_FROM_V3_GATE`.

Illapa stays blocked with `HF_REPO_UNKNOWN`.

A recipe attached to a request must sit inside the current contract bounds,
including `gradientCheckpointing` of `unsloth` or `true` and the 8-bit AdamW
optimizers. The draft recipe copies the in-range attempt-17 values. Whether
NeMo Framework can express `unsloth` checkpointing and `paged_adamw_8bit` is
UNKNOWN. A mismatch is a blocker, not a reason to widen the contract.

Compatibility of current NeMo Framework with `torch 2.14.1+cu126` is UNKNOWN.

## (b) PyG job for szl-estate-graph

The graph lives in `szl-holdings/szl-constellation` at
`datasets/szl-estate-graph/`. `graph.json` is schema `szl.estate-graph/v1`.
Counted from the CSV files on `main` during this pass: 67 nodes (49 estate,
8 vertical, 3 hub, 7 substrate) and 126 edges (56 runs-on, 49 synapse, 10
wired-into, 8 governed-by, 3 trinity). Every edge endpoint is a node id.
`docs/OUROBOROS_CURRICULUM.md` in this repo says 369 nodes and 97 edges.
That count was not recomputed from another artifact here, so the curriculum
figure stays an unverified discrepancy. The job spec binds the CSV counts.

The draft spec and the CPU validator live beside that dataset:

- `datasets/szl-estate-graph/pyg-jobspec.v0.json`
- `tools/estate_graph_pyg.py`

The validator recomputes the counts, refuses an enterprise registry, and
returns `launchAuthorized: false` while `container.digest` is `UNKNOWN` and
`container.status` is `NOT_BUILT`. It does not import PyTorch or PyG.

The container, when a later review builds it, is a public CPython or
PyTorch image plus `torch-geometric` from PyPI. The base digest is UNKNOWN.
`nvcr.io` is forbidden. No image is built or pushed in this pass.

The Hugging Face dataset API reports `license` `apache-2.0` for
`SZLHOLDINGS/szl-estate-graph`. Amaru's check on that dataset is PENDING.

## (c) NeMo Retriever baseline for the Q7 embedder eval

`szl-retrieval-bench` keeps BM25, TF-IDF, and RRF as the measured lanes.
The new module does not register with `harness.py`, so those lanes stay
`MEASURED`.

Code: `src/szl_retrieval_bench/nemo_retriever_baseline.py`.
Spec: `specs/nemo-retriever-q7-baseline.v0.json`.

`rank()` raises and emits no score. The record's state is `BLOCKED`.

The embedder candidate is `nvidia/llama-nemotron-embed-1b-v2` with API
`license_name` `nvidia-open-model-license`. It is not downloaded and not
scored.

The reranker candidate `nvidia/llama-nemotron-rerank-1b-v2` has API
`license_name` `openmdw-1.1`, which is outside the named open set. It is
recorded and not selected. No reranker whose API `license_name` is in the
named open set was verified in this pass.

No Q7 embedder eval corpus, qrels, or metric definition was found in
`szl-retrieval-bench` or `szl-gpu-bridge`. The only `q7` string in
`szl-gpu-bridge` is a Kimi q7 hologram refresh note, which is not an
embedder eval. The Q7 corpus is UNKNOWN. The baseline therefore cannot
return nDCG, recall, or any other figure.

## Launcher and spend

| Cap | Dollars | micro-USD (`uUSD`) |
| --- | --- | --- |
| Per run | 100 | 100,000,000 |
| Per day | 500 | 500,000,000 |
| Per month | 5,000 | 5,000,000,000 |

`tools/massed/admission.py` names its money fields `price_uusd_hr` and
`worst_case_micro_usd`. This plan uses 1 USD = 1,000,000 uUSD on that
naming. The dollar figures are the approved ceiling. Providers are `massed`
and `salad` only. Launcher name is `saywa` only. A matching name still does
not authorize a launch, because the Saywa entrypoint is UNKNOWN and the
image digests above are not built.

These caps are not wired into `tools/massed/admission.py`. That gate has
its own lane pins. Connecting them is later work and is not done here.

## Tests

CPU only. No GPU, no network, no install of NeMo, PyG, or model weights.

- `szl-gpu-bridge`: `python -m unittest tests.test_nemo_framework_backend -v`
- `szl-constellation`: `python -m pytest -q tools/tests/test_estate_graph_pyg.py`
- `szl-retrieval-bench`: `python -m pytest -q tests/test_nemo_retriever_baseline.py`

CI for the new gpu-bridge module is the existing `unittest discover` plus
`ruff` and `py_compile` on `tools/nemo_framework`. No `continue-on-error`
was added. Constellation CI already runs `tools/tests`, and the new module
is on the `py_compile` list. Retrieval CI already runs `pytest tests/`.

## Blockers

1. Successor training-image digest is UNKNOWN. NeMo is not in the pinned
   image, and this pass does not build one.
2. GHCR did not confirm the pinned digest without authentication (HTTP 401).
3. Illapa has no Hugging Face repo binding.
4. Nano-9B-v2's API `license_name` is not the v3 `licenseId` gate value.
   The gate stays as it is.
5. NeMo Framework's ability to carry the current recipe (`unsloth`,
   `paged_adamw_8bit`) on `torch 2.14.1+cu126` is UNKNOWN.
6. Amaru's license check is PENDING for every family, including the ones
   whose API `license_name` is already in the named open set.
7. The reranker API `license_name` is `openmdw-1.1`, outside that set. No
   replacement reranker in the named set was verified.
8. The Q7 embedder corpus is UNKNOWN, so the baseline stays `BLOCKED`.
9. The PyG image digest is UNKNOWN. PyTorch's SPDX id is UNKNOWN
   (`NOASSERTION` from the GitHub API).
10. Saywa's launcher implementation was not found. Specs name the lane and
    refuse to launch.
11. `docs/OUROBOROS_CURRICULUM.md` node/edge counts do not match the
    constellation CSV. The job spec uses the CSV.

## Draft pull requests

Branch names for this pass:

- `szl-gpu-bridge`: `cursor/nvidia-open-tooling-4367` (this plan and the
  trainer backend)
- `szl-constellation`: `cursor/estate-graph-pyg-spec-4367`
- `szl-retrieval-bench`: `cursor/nemo-retriever-q7-baseline-4367`

Pull request URLs are filled in after the drafts are opened.
