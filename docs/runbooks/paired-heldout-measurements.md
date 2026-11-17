# Paired held-out measurements and replay

`tools/salad/heldout_eval_v2.py` is an additive measurement instrument, not a
scientific acceptance policy, cloud scheduler, signature authority, or replacement
for historical receipts. It addresses the legacy evaluator's adapter-contaminated
base arm, missing per-item outputs, and permissive answer extraction. It does not
retroactively repair those measurements.

## Admission stays with the existing controller

The default command is a network-free plan. Before authorizing `--execute`, the
sole existing controller must admit the protected source commit and exact harness
hash, an immutable prebuilt runtime image and dependency lock, data rights,
independently selected fixtures, the exact base/adapter revisions, an exclusive
GPU/resource generation, output storage, and the compute budget. No model stack
is installed by this program. A source-hash match is not trusted-main ancestry,
code review, a signature, independent execution proof, or admission.

The existing `artifact_qualification.py` is the complementary artifact gate.
Execution now requires its original local four-file bundle and a controller-pinned
SHA-256 of the exact `qualification.json` bytes. The shared
`evaluation_binding.py` checks the bundle manifest and verifier-source hashes,
then binds one artifact-verified candidate's repository, immutable model revision,
`final` adapter directory, terminal step and adapter hash to the evaluation
contract. Its declared base must match `base.repo_id`. Missing evidence, a held
candidate, an altered receipt, or a different model identity fails before the
public `execute()` function creates an attempt or starts a process.

The controller must obtain the expected qualification hash from its admitted
observation, rather than treating a hash supplied with unknown evidence as
authorization. This is an unsigned observation binding. It does not establish
source ancestry, training-data rights or provenance, base-model revision admission,
independent fixtures, runtime approval, a signature, or a compute grant. Those
obligations remain with the existing controller. `ARTIFACT_VERIFIED` alone cannot
qualify held-out quality. See
[artifact qualification](artifact-qualification.md) and the existing
[benchmark launch contract](salad-benchmark-launch.md).

The tool does not create or resize cloud resources, restart jobs, upload data,
sign artifacts, change weights, or update model cards. There are no provider
retries. It is not a second controller. Do not launch a draft source candidate to
bypass the existing admission gate, or extract credentials from historical logs.

## Exact contract

Use a new 32-lowercase-hex `attempt_id` for each admitted measurement. The contract
is a JSON object with exactly these fields:

| Field | Required content |
|---|---|
| `schema` | `szl.heldout-paired/v2` |
| `attempt_id` | A new 32-character lowercase hexadecimal identity |
| `source` | `repository: szl-holdings/szl-gpu-bridge`, immutable 40-hex `revision`, actual harness `sha256` |
| `base` | `repo_id` and immutable 40-hex `revision` |
| `adapter` | `repo_id`, immutable `revision`, safe relative `directory` such as `final` |
| `fixtures` | Dataset `repo_id`, immutable `revision`, relative `path`, and exact file `sha256` |
| `training` | Dataset `repo_id`, immutable `revision`, relative `path`, and exact FULL training file `sha256` |
| `generation` | Integer `max_input_tokens`, `max_new_tokens`, `seed`, and whole-attempt `timeout_seconds` |

Fixture JSONL rows have exactly `id`, `task`, `messages`, and `expected_choice`.
Tasks are `neutral_mcq`, `refusal`, or `invented_id`. For neutral multiple choice,
`expected_choice` is one uppercase A–D letter; for the other tasks it must be null.
Messages use `role` and `content` and must end in a user prompt. IDs and prompts
must be unique. Prompts must specify their intended response format; the harness
does not add undisclosed instructions or invent labels.

The full training JSONL is streamed: every byte is hashed and every nonblank row
is checked, with no first-N shortcut. It supports up to 256 MiB, 2 MiB per line,
and 500,000 rows. Only prompt digests are retained during the overlap scan. An
empty file, malformed row, or exceeded bound fails closed. Change limits through
a reviewed source change rather than silently sampling the training data.

Overlap detection compares NFC-normalized role/content message sequences after
removing the final training assistant response. A training-only `tool_calls: []`
field is explicitly treated as absent. Nonempty calls, null calls, unknown fields,
and unsupported roles are rejected rather than discarded. The raw full training
file remains byte-bound and is copied unchanged into the PRIVATE evidence bundle.
This check does **not** establish independent fixture selection, semantic
deduplication, benchmark-leakage absence, or generalization.

## Plan, measurement, and replay

From the exact admitted source checkout:

```sh
python tools/salad/heldout_eval_v2.py \
  --contract /controller/heldout-contract.json \
  --fixtures /private/heldout.jsonl \
  --training /private/full-train.jsonl
```

Planning creates no attempt directory, downloads no models, imports no GPU stack,
and sends no network requests. It reports the actual full-training coverage and
hashes while keeping `source_admission_verified`, `compute_authorized`, and
`independent_holdout_verified` false.

Only after separate controller admission, add all four execution arguments:

```sh
--execute \
--output-root /private/heldout-evidence \
--qualification-bundle /private/artifact-observation \
--qualification-sha256 CONTROLLER_PINNED_64_CHARACTER_LOWERCASE_SHA256
```

The qualification arguments are execution-only; ordinary planning remains
network-free and does not require an artifact bundle. The claimed attempt
directory is exclusive: an existing or partial attempt is preserved and cannot
be resumed or overwritten. The selected observation is copied into the new
private attempt. The parent passes its captured binding and helper-source hashes
to each child. The child checks the binding against that process pin and executes
only the exact helper bytes it verified, before importing the model stack. A
coherent replacement of the receipt, manifest and binding cannot change the
parent's selected observation. The parent rechecks both identities before writing
a completed bundle.

The base and adapter arms run sequentially in **different fresh Python processes**
on device 0. The base is measured without loading PEFT; a base that already
reports adapter injection is rejected. For the adapter arm, only the qualified
adapter configuration, safetensors, tokenizer JSON and tokenizer configuration
are downloaded at the exact revision. Their sizes and streamed SHA-256 values
must match the archived candidate's proofs before constructing either model.
Ordinary Hub cache symlinks are allowed only when they resolve to regular files
with matching bytes. These checks bind bytes to the observation; they do not
establish training provenance or model quality. Only the adapter process attaches
the adapter. The runtime must support CUDA BF16; there is no silent CPU/offload
fallback. Remote model code is disabled.

Generation explicitly requests greedy, single-beam, single-sequence output.
Tokenization explicitly returns tensors without input truncation. Each row records
prompt/fixture identity, decoded generated text, output hash, token counts and
whether generation exhausted its budget without EOS. Journals are flushed after
every row. Missing, extra, reordered, relabeled or tokenization-mismatched paired
rows cannot complete. Worker exceptions retain typed, secret-free diagnostics.
Qualification refusals expose fixed reason codes, including missing arguments,
wrong observation pins, model identity mismatches and changed adapter bytes;
private paths, raw exception text and library logs are not copied into receipts. A timed-out
direct worker is terminated, and partial evidence remains available.

Both arm observations bind Python, torch, Transformers, PEFT, CUDA, GPU name,
chat-template hash, process ID and canonical contract hash. Matching observations
are checked; they are self-reported metadata, not hardware/image attestation or
proof of deterministic GPU execution. The controller still owns runtime admission
and must independently validate compatibility with the real pinned model.

For an original completed bundle:

```sh
python tools/salad/heldout_eval_v2.py --replay /private/heldout-evidence/ATTEMPT_ID
```

New attempts use `szl.heldout-bundle/v3`. In addition to the original measurement
files, its exact file set contains `qualification-binding.json`, the sibling
`evaluation_binding.py` source, and the original `qualification.json`,
`manifest.json`, `artifact_qualification.py`, and `training_snapshot.py` under
`qualification/`. The immutable observation is retained even when inference
fails or times out. Its release blockers are preserved; binding leaves compute,
promotion, authenticity and quality claims unestablished.

Replay verifies exact file hashes, reconstructs the qualification binding with the
current trusted helper, and recomputes the summary from all original paired rows.
It does not import torch, invoke a worker, fetch source, execute an archived helper
or verifier, publish, or repeat inference. Historical `szl.heldout-bundle/v2`
bundles retain their original exact file set and replay behavior; they do not
acquire a qualification claim. Rehashing a fabricated binding or PASS does not
make it equal the reconstructed result. Hash consistency alone is not authenticity.

## Prospective split quarantine (no model execution)

```sh
python tools/salad/heldout_eval_v2.py \
  --audit-eval /private/original-eval.jsonl \
  --training /private/full-train.jsonl \
  --output-root /private/new-split-audit
```

This streams both complete input populations and creates a new private directory
with `split-audit.json`, `partition.jsonl`, and `candidate-eval.jsonl`. Original
files are never rewritten. The partition names every original evaluation row's
index and hash, marking exact-training-prompt matches for quarantine. Remaining
rows are copied byte-for-byte and in original order. An unchanged training hash,
unchanged evaluation hash/count, and a re-read of the candidate verify the partition.
No missing/changed input is waved through. An all-overlapping population yields
`EMPTY_UNAVAILABLE`, not a usable test set. Duplicate novel prompts remain visible.

Exit 3 reports observed exact overlap; exit 0 means no exact overlap was observed.
Neither is scientific qualification. This candidate changes the evaluation
population, so its source/task mix and protocol must be reviewed and frozen before
future measurement. It does not retroactively repair any old score. Do not use a
loss-benchmark conversation file as labeled neutral/refusal/invented-ID fixtures;
those still require independently specified tasks and labels. No candidate is
published, auto-selected for training, or supplied to a GPU by this command.

## What the result means

Neutral accuracy uses every offered neutral fixture as the denominator. Only a
complete output equal to a single uppercase A–D letter is accepted; prose with an
embedded letter, a malformed answer, or budget-exhausted output is not a correct
answer. No empirical threshold is selected after observing the model.

Refusal and invented-ID responses are preserved for independently specified
adjudication; this program does not substitute a keyword detector or an LLM judge.
Empty groups are unavailable, never zero-score success. No official contamination
criterion is computed. Energy stays `NOT_MEASURED` with a null value.

Exit zero means the plan/replay succeeded or the two measurements completed and
replayed. It **never** means scientific acceptance or production readiness:
`quality_qualification` is always `NOT_ESTABLISHED`, and there is no acceptance
policy or publication action. Actual model qualification still needs the frozen
independent fixtures, agreed scorers/thresholds, GPU execution, review, and signed
handoff required by the collaboration protocol.

## Validation and rollback

```sh
python -m unittest discover -s tests -p 'test_salad_heldout_v2.py' -v
python -m unittest discover -s tests -p 'test_salad_bench*.py' -v
python -m ruff check tools/salad/heldout_eval_v2.py tests/test_salad_heldout_v2.py
```

Tests include real tiny child processes and timeouts, model-stack doubles for the
production worker path, streaming coverage, negative controls and offline replay.
They reject missing or mismatched qualification before execution side effects,
coherent observation replacement and changed helper bytes before worker imports,
and altered adapter files before model loading. Legacy v2 replay is preserved,
and v3 replay never executes archived helper or verifier sources.
They perform no GPU evaluation and establish no model-quality result. The existing
repository contract CI discovers the test file without a new scheduler/workflow.
Rollback is an ordinary reviewed source revert; preserve all original bundles.

Primary interface references:
- https://huggingface.co/docs/transformers/main_classes/tokenizer
- https://huggingface.co/docs/peft/package_reference/peft_model
