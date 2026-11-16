# Revision-bound adapter qualification and read-only observation

## Claim boundaries

`tools/salad/artifact_qualification.py` closes the gap between a filename inventory and inspection of the actual terminal adapter. It reuses `training_snapshot.py`; it does not introduce another source of training-state truth. `ARTIFACT_VERIFIED` means the declared files were fetched at one immutable model revision, their Git/LFS object identities matched, terminal trainer metadata exactly matched the requested step budget, final weights matched the terminal checkpoint's LFS identity, the declared base/configuration matched, and every inspected floating tensor was finite on CPU.

It does **not** establish training execution provenance, ownership of the training data, held-out quality, safety, mathematical correctness, a valid signature, runtime readiness, or permission to promote. All those obligations stay explicit. Neither this verifier nor the observer can sign, upload, start/stop a cloud instance, or promote a model. No pickle-bearing optimizer state is downloaded or deserialized.

A private model stays private. Keep the local receipts in the same protected operational boundary as the source models. Do not copy real private input manifests or receipts into a public repository. A root model card in a private repository is not a public showcase.

## Bounded execution

From the repository root, with an existing Hugging Face login or a scoped `HF_TOKEN` environment variable:

```sh
python -m tools.salad.artifact_qualification --spec /private/candidates.json --output /private/evidence/new-observation
```

The spec contains 1-6 distinct candidates:

```json
{"candidates":[{"repo":"SZLHOLDINGS/example","base":"organization/base-model","target_steps":200,"salad_group":"szl-example"}]}
```

Only the observer uses `salad_group`. Targets are explicit operator contracts, not inferred from a repository title. A latest checkpoint before the target is `TRAINING_IN_PROGRESS`; overshoot and target mismatches are separate holds. There is no fallback from an inaccessible observation to an assumed clean state.

Downloads are allowlisted to trainer JSON, adapter config, final safetensors, and tokenizer JSON/config. Per-file metadata is bounded to 128 MiB and the final bundle to 192 MiB per candidate. The tensor inspector uses one CPU thread, never model generation or CUDA. Use the existing pinned training/evaluation environment with `huggingface_hub`, `safetensors`, and `torch`; the verifier does not install dependencies.

Exit 0 means every candidate passed this artifact-only check. Exit 2 means one or more candidates remain on hold; their reasons and partial observations are still preserved. The output directory is exclusive and cannot replace an existing observation. `manifest.json` binds `qualification.json` and both verifier sources by SHA-256. A hash manifest is not a signature or independent execution witness.

## Operational telemetry

```sh
python -m tools.salad.training_observer --spec /private/candidates.json --status /private/status.json --history /private/history.jsonl
```

One cycle is the default. An existing supervisor may explicitly use `--cycles 300 --interval 300`; it is bounded, not a claim of infinite service availability. Provider GETs use a `SALAD_API_KEY` environment variable, never a command-line secret, and refuse redirects. No credential values are included in reports.

The observer writes atomic current status and append-only history. Consumers must apply `stale_after_seconds` to `observed_at_utc`; stale data must not be rendered as current. State meanings:

| State | Supported observation |
| --- | --- |
| `NO_CHECKPOINT_OBSERVED` | Successfully listed model, no checkpoint paths observed. |
| `CHECKPOINT_PROGRESS_OBSERVED` | Pinned trainer JSON verified below the declared target. Not proof the trainer process is alive. |
| `TERMINAL_METADATA_PRESENT` | Exact terminal step and final filenames observed. Not adapter-byte verification or training completion. |
| `TARGET_CHECKPOINT_WITHOUT_FINAL_BUNDLE` | Target checkpoint observed without a complete final metadata set. |
| `TARGET_CONTRACT_MISMATCH` / `TERMINAL_STEP_OVERSHOOT` | Operator contract and observed trainer metadata disagree. |
| `UNAVAILABLE` | The observation failed; never interpreted as an absent instance or a pass. |

Provider observations separately report returned instance states and readiness. Even `running` is not evidence of optimizer progress. No `FLAGSHIP_TRAINING_COMPLETE` flag is emitted by this observer.

## Promotion and the existing estate loop

This is the artifact-admission layer beneath `bench_eval.py` and the governed launch contract. It does not replace the Lutar Lean kernel, ORO/Ouroboros curriculum, owner approval, existing receipt validators, or the five-rail design in `../SELF_PROVING_ESTATE_LOOP_20261115.md`. The later evaluation must bind the same model revision or the verified adapter byte identity and verify base, data, harness, generation and contamination boundaries independently.

The existing loss evaluator explicitly sets `heldout_quality_measured=false`; it cannot close all release obligations. The older six-probe abstention battery is also not a quality qualification: its answerable verdict is based on not abstaining, not factual answer correctness. Do not turn either result into a production badge.

Anatomy, Hatun, second-brain and inference consumers can ingest these JSON observations only within the authorized visibility boundary. This change supplies the contract; it does not claim those separate services have been integrated or deployed. Release remains on hold until provenance/signature, held-out quality, safety, serving/auth/isolation/latency/rollback, and identity-bound owner approval are independently established.

## Regression controls

```sh
python -m unittest discover -s tests -p 'test_artifact_qualification.py' -v
python -m unittest discover -s tests -p 'test_training_observer.py' -v
python -m unittest discover -s tests -p 'test_training_snapshot.py' -v
```

These are fixture-based contract tests, not model evaluations. The existing contract CI discovers the new tests through `unittest discover -s tests`.

Reference APIs: Hugging Face HfApi immutable revision/file metadata; Safetensors `safe_open` on CPU. The NVIDIA Ada compatibility documentation permits compatible Ampere cubins; a missing exact `sm_89` entry alone is not evidence that an entire runtime lacks usable GPU code. This change does not alter GPU runtime compatibility or restart training.
