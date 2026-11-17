# Observation claim graph — evidence before promotion

This is a **source-only, offline, read-only consumer** of the existing
szl.training-watch-once/v2 private one-shot observation. It does not install
another watchdog, scanner, scheduled task, controller, Cloudflare Worker, Hugging
Face job or cloud writer; it never loads model code or pickle state. It does
not change the protected, hashed owner-device observer release.

## Problem and mathematical invariants

A provider may return 132 synthetic container groups while an observer retains
the first 120. A 120-row dashboard cannot silently report complete coverage,
and the 12 omitted synthetic rows cannot be assumed stopped, failed, or inactive.

For each observed project p:

- R_p = provider-reported number of group rows returned across the observer's
  requested pages, as recorded in coverage[].pagination.returned.
- C_p = number of distinct project/name group rows retained in groups[].
- O_p = R_p - C_p >= 0 = groups whose detailed state was **not retained**.
  If either count is unobserved, O_p is **unavailable**, not zero.
- A project catalog is complete only when pagination is independently marked
  complete and O_p = 0. Organization-wide coverage also requires successful,
  complete project discovery. A known default-project fallback after a 404 does
  not prove the organization has no other accessible projects.
- Aggregated instance status counts are sums over *captured rows only*.
  A missing count or a boolean is unknown, not zero. A group-level RUNNING
  string is not trainer-process liveness.

The graph never trusts the producer's capped delta array as an exhaustive
comparison. Its model edges compare only **pinned trainer JSON object bytes**:
global_step, max_steps, matching checkpoint directory, exact immutable model
revision and trainer_state_sha256. When the recorded target is unchanged, a step-counter delta is reported,
without claiming it belongs to the same trainer run or that a process is alive.
Changing the target makes a counter comparison unavailable.

An explicit command-derived model_repo and configured_max_steps is a
**declared command binding**, not proof of image execution, scheduler ownership,
model identity in GPU memory, or successful training.

## Independent evidence gates

The reducer always leaves training_execution_verified,
artifact_integrity_verified, independent_backup_verified,
evaluated_quality_verified, deployed_verified, and
production_qualified **false**. A downstream owner must supply independent
immutable-source proof for each gate. In particular:

1. A reached target and final/adapter_model.safetensors *filename* do not
   certify a complete artifact or backup.
2. A byte hash is object identity, not a signer or reproducible evaluation.
3. A historical heldout receipt does not qualify new adapter weights unless
   revisions, benchmark source, dataset rights and denominators are bound.
4. An HTTP 200, HF Space RUNNING, Cloudflare 402 or PyPI upload is not
   serving quality, authorized payment, settlement or released-source parity.
5. The Flagship54 guard on Gate-1 remains its **only shutdown controller**.
   This consumer has no service-management, compute or publisher capabilities.

No private group names, checkpoint contents, training data, tokens, public
receipts, wallet signatures, secret-bearing exception text, or raw provider
headers are automatically transmitted to a public surface. The JSON output
contains private group and model identifiers and is therefore **private**.

## Reproduce without provider access

Use exact reviewed source bytes and the most recent complete owner observation.
From the repository root, to create a new file under the owner's existing
private evidence directory, replace the placeholders with actual local paths:

    python -m tools.salad.observation_claim_graph --current <private/latest.json> --previous <private/prior-observation.json> --output <private/new-unique-graph.json>

The output is exclusively created; retrying a used filename fails without
replacing historical evidence. File contents are bounded to 4 MB, parsed as
strict JSON with duplicate-key and non-finite-number rejection, with
deterministic failure codes that do not expose original input paths.

    python -m unittest discover -s tests -p test_observation_claim_graph.py -v
    ruff check tools/salad/observation_claim_graph.py tests/test_observation_claim_graph.py
    ruff format --check tools/salad/observation_claim_graph.py tests/test_observation_claim_graph.py

## Admission and limitations

This code runs after an existing protected observation; it cannot repair the
omitted group rows in an upstream snapshot. Source-controlled
salad_group_page_evidence in training_snapshot.py already supports a larger
default metadata cap, but any protected observer release update must be
**separately reviewed, hashed, tested, owned and admitted**, preserving its
single-writer lock and never reusing old trusted hashes for changed bytes.

The output is an **unsigned and unauthenticated derived graph**, not a
successful remote observation, completed training, verified backup, benchmark,
deployed endpoint, publication, financial settlement or whole-estate audit.
To move toward production, perform a separately admitted producer repair,
fresh complete provider pagination, per-instance state checks, independent
artifact and backup readbacks, prospective evaluation, source-runtime parity,
authorized deployment and human-governed qualification.
