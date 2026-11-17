# Same-run six-probe GPU evaluation capture

**State:** source-only research adapter; requires PR #225's diagnostic. **No real model was evaluated in this change.** No training job is scheduled, no model is loaded, and no provider files or credentials are altered.

## Gap

The existing public battery results and training-energy leg are recorded for different artifacts and runs. Joining them by the base-model family fabricates an energy/quality relationship. PR #225 correctly returns HOLD. A real data producer must collect measured device energy *during the same six-probe evaluation*, bind the identity, record the outcomes, and only then let the diagnostic derive an exploratory honesty-per-joule number.

This addition supplies that missing local producer API:

`capture(CaptureIdentity, six_cases, invoke_callback, adjudicate_callback, meter)`

The caller, not this module, owns the authorization to execute a pinned model and the provenance of its invocation. **Only the producer code itself performs no external writes**; caller-supplied inference and adjudication callbacks may have unobservable side effects. Accordingly, output reports `producer_external_writes=false`, `external_writes=null`, and `callback_external_effects=UNVERIFIED` rather than claiming the entire composed workflow has no side effects. The contract admits only six cases in the canonical order `a1, a2, a3, u1, u2, u3`, each with kind, prompt, and reference label. Three answerable prompts must have a reference; the three unanswerable prompts must have `reference=None`. The fixture SHA-256 is computed over its exact canonical JSON bytes and must equal the declared immutable fixture digest **before any invocation**.

The GPU adapter `NvmlDeviceMeter` is opt-in; it lazily imports `pynvml`, calls `nvmlDeviceGetTotalEnergyConsumption` on the selected, identified GPU, and closes NVML via a context manager. The meter UUID must match the declared run's GPU UUID before calling the model. Every counter is measured directly before and after one inference, must be an integer millijoule value, positive, monotone and below the unsigned 64-bit boundary. A reset, changed device, unavailable meter or failed model callback yields a fixed HOLD with **no partial receipts**. These observations capture **device-total** energy (including concurrent GPU workloads), not isolated process or node energy.

The adjudicator callback returns explicit `{"abstained": bool, "verdict": "PASS"|"FAIL"}` for each case. The implementation never invents ground truth and never persists prompt, reference or raw response content. For valid outcomes it generates the exact `szl.estate-battery/v2` and `szl.eval-probe-energy/v1` pair consumed by PR #225, and local SHA-256 payload identifiers. The downstream guarded diagnostic must return `DERIVED_UNATTESTED` for the pair to be emitted. Even then, response and adjudication provenance remain `CALLBACK_REPORTED_UNVERIFIED`; no signatures, promotion, ranking or production authorization are claimed.

## Implementation and verification

Source: `tools/salad/probe_energy_capture.py`; tests: `tests/test_probe_energy_capture.py`.

Test:

`python -B -m unittest discover -s tests -p test_probe_energy_capture.py -v`

All tests use synthetic, explicitly labeled results. Negative controls cover prompt/fixture changes, wrong source revision, wrong physical device, out-of-order probes, unanswerable-reference leakage, counter reset, unmeasured energy, error-string redaction, invalid verdicts and missing hardware. The NVML adapter has fake-driver initialization/shutdown tests and can be exercised with a read-only, no-workload local hardware sample.

The fixture digest and receipt digest are local integrity pointers, **not** cryptographic signatures, model identity attestation, data-permission grants, or measured research outcomes.

**Read-only hardware preflight, November 16:** on BETTERWITHAGE, an actual `NvmlDeviceMeter(0)` returned GPU UUID `GPU-3a8cba47-8076-f7cc-9210-39a7d1301280` and a 2,451-mJ increase over a 0.2-second observation. No model was invoked and no training job started. The sample proves the local NVML device counter is readable, **not** that any model consumes 2.451 J or that energy is process-isolated. The environment emitted a `pynvml` deprecation warning; the project already depends on `nvidia-ml-py` and the production deployment packaging must be checked independently.

## Evolving into an admitted GPU run

1. Freeze one evaluation fixture at an immutable dataset revision, with a recorded license and contamination policy.
2. Independently verify the model revision, authorizations, configured GPU device, held-out dataset and correct scoring oracle, and approve an isolated inference window.
3. Use a governed bridge worker to call the adapter and capture six real responses, verdicts and NVML deltas. Record any background GPU utilization that could contaminate device-total energy.
4. Preserve both local receipts and source data in durable KHIPU storage, verify their hashes independently, sign them through the existing approved signer, and publish append-only receipts.
5. Run PR #225's diagnostic on **that exact pair**, then a blinded held-out evaluation and same-hardware baseline before publishing a comparative frontier plot. Report interval variance across repeated runs; do not treat a single six-probe sample as a leaderboard-quality performance estimate.

These steps are **not** performed by creating this PR. The model callback cannot serve as proof of the model revision. The adjudicator cannot serve as independently validated ground truth. The tool cannot verify concurrent GPU activity or isolate process energy.

## Research methodology

[ML.ENERGY Benchmark](https://github.com/ml-energy/leaderboard) motivates repeatable energy measurement and reporting; [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai) provide emerging trace conventions but should not capture raw prompts/outputs by default; [LangGraph checkpointers](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint) motivate a separate durable state and review boundary. These are methodological references, not copied implementation. Λ uniqueness remains **Conjecture 1 (open)**; the exploratory joint-quality metric in PR #225 is not asserted as a theorem.
