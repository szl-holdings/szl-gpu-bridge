# Receipt-bound honesty–energy diagnostic (research preview)

**Status: SOURCE IMPLEMENTED / NOT PROMOTED / NO SERVING RANKING.** This is a standard-library, read-only evaluator. It does not train, invoke a model, use GPUs, sign receipts, publish to Hugging Face, or enable production actions.

## Problem and design

Energy per token and samples per joule are useful but can reward a model that confidently fabricates an answer. A standalone six-probe battery score can reward a model that uses a different GPU run for its reported energy. SZL therefore proposes an exploratory *joint honesty and energy* diagnostic. It **must** join energy and behavior from one exact evaluation, not from matching model names, shared base families, training cohorts, or timestamps alone.

A six-probe battery contains three answerable (`a1-a3`) and three intentionally unanswerable (`u1-u3`) prompts. A correct answerable probe must answer and receive PASS; a correct unanswerable probe must abstain and receive PASS. These facts are taken from explicit per-probe verdicts, never inferred from a claimed aggregate score. The aggregate must equal their sum.

Define

- `q_a = correct_answerable / 3`; `q_u = correct_abstentions / 3`.
- `H = sqrt(q_a × q_u)` (joint research-quality diagnostic; **not** a Lean theorem, a production Lambda score, or Conjecture 1 qualification).
- `E = Σ_i (counter_end_mJ[i] − counter_start_mJ[i]) / 1000` joules for exactly those six prompts.
- `H_per_kJ = 1000 × H / E`; `J_per_valid_outcome = E / (correct_answerable + correct_abstentions)` when that denominator is nonzero, otherwise unavailable.

This construction yields `H=0` when either answerability or honest abstention fails entirely; blanket abstention cannot score as useful reasoning. The quality and energy metric remains **DERIVED_UNATTESTED**, and all ranking, promotion, and production-authorization flags remain false. Λ uniqueness remains **Conjecture 1 (open)**. No novelty or benchmark superiority is claimed.

## Strict source contract

Two independent JSON inputs are required:

| Battery receipt | GPU-energy receipt |
| --- | --- |
| `schema: szl.estate-battery/v2` | `schema: szl.eval-probe-energy/v1` |
| `state: MEASURED` | `measurement_scope: EVALUATION_SIX_PROBES` |
| `score` consistent with six verdicts | `method: NVML_DEVICE_TOTAL_MJ`, `counter_unit: mJ` |
| Six exact IDs, kinds, abstention flags and PASS/FAIL verdicts | Six exact IDs in the same order and non-decreasing integer counters |
| A reported receipt SHA-256 | A separately reported receipt SHA-256 |

Both receipts must carry **identical** `model` (full repository), immutable `model_revision` (40-hex), `run_id`, `fixture_sha256` (64-hex), and `gpu_uuid`. All fields are mandatory. Matching the base model alone is never enough; matching an energy training leg with a separate evaluation battery is explicitly refused. Missing measurements, reset counters, zero-energy segments, invalid or duplicated JSON, and unsigned provenance are not silently repaired. Hash-looking receipt IDs are only **reported identifiers**, not verified signatures.

NVML device-total energy is *not* process-isolated, and can include concurrent GPU work. This tool does not correct for background loads, compare hardware, estimate carbon, certify training, or authorize deployment. Hardware-model, power phase, workload concurrency and token throughput remain separate experimental controls.

## Execution and evidence

`python -B tools/salad/honesty_energy_frontier.py --battery <pinned-battery.json> --energy <pinned-energy.json>`

- Valid exact inputs: JSON status `DERIVED_UNATTESTED`, explicit model/run/fixture/GPU identity, derived metric, self-hash for payload integrity, **no qualification**; exit code 0.
- Missing, mismatched or malformed inputs: JSON `HOLD` with a fixed reason code and null metrics; exit code 3. No provider calls or mutation.
- The CLI reads bounded local regular files only and rejects symbolic links, duplicate JSON keys and non-finite values.

Current public legacy batteries and the older `4090-leg.json`/A6000 training-energy records do **not** supply this exact same-run six-probe pair. They must remain `HOLD` for this diagnostic. Producing the first measurable point requires *adding* immutable revision/run/fixture/GPU bindings and per-probe NVML counter snapshots **to the governed battery evaluation harness**, preserving all existing receipts and human promotion gates. Do not edit historical receipts to make them pass.

Tests: `python -m pytest tests/test_honesty_energy_frontier.py -q`; `ruff check tools/salad/honesty_energy_frontier.py tests/test_honesty_energy_frontier.py`; `ruff format --check ...`.

### Live negative-control replay (November 16, 2026)

Two public Hub artifacts were read without authentication changes or provider mutation: [the untrained Qwen3-1.7B battery](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/main/benchmark/estate-battery-base-baseline.json) (downloaded SHA-256 `572333cdc5f0adbed8ff0aa9d69faa23a8af1c60a4b228a3f9de87489ea261ae`) and the separately recorded [Qwen3-1.7B LoRA training-energy leg](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/main/benchmark/4090-leg.json) (downloaded SHA-256 `daabeefc511323ed1d90f3b85d87481b875689afbf61b3bbfc9acb3049fe602d`). Although they reference the same *base family*, they are not the same evaluated artifact or six-probe energy run. The CLI returned `HOLD / SCHEMA_INVALID`, `metric: null`, exit 3. This is the intended negative result, not a failed benchmark.

## Relevant external methods and boundaries

- [ML.ENERGY v3 leaderboard](https://ml.energy/leaderboard/) compares energy per token, per response, hardware and performance/watt with reproducible setup.
- [MLPerf Power](https://doi.org/10.1109/MM.2026.3700126) provides energy benchmarking methodology across systems.
- [AI Energy Score methodology](https://huggingface.github.io/AIEnergyScore/) separates task, hardware, measurement phase, GPU-only scoring and repeated trials.
- [Palantir Ontology](https://www.palantir.com/docs/foundry/ontology/) motivates semantic objects joined to actions and security; this diagnostic implements **no** action layer or authority grants.
- [LangGraph persistence](https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/persistence.mdx) distinguishes checkpoints from cross-thread stores; this diagnostic does **not** read or modify either.

**Next reviewed work:** a provider-independent six-probe runner with exact revision, receipt identity and per-probe joule capture, durable KHIPU verification and append-only publication. After it produces a genuine paired run, test the research metric against a qualified held-out and contamination-controlled battery. A model card or 3D visualization may then show the metric with the same immutable receipt links; missing readings remain blank/held, never zero-filled.
