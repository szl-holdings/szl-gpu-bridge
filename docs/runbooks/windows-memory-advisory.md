# Windows memory envelope — source-only review candidate

**Status: NOT DEPLOYED; no dispatch wiring.** This additive research-grade
memory observation classifier is not a production preflight or training
authorization. It accepts sanitized offline input, makes zero system or
provider calls, prints no paths or credentials, never installs tasks,
stops processes, edits checkpoints or changes Flagship54.

## Evidence and research boundary

On 2026-11-19 BETTERWITHAGE, sampled storage free space passed the separate
50-GiB minimum and 2-GiB swing gate while free RAM was ~1.8–2.75 GiB,
committed memory was approximately 88.4% of the system commit limit and
pagefile allocation changed rapidly. These non-atomic snapshots are
not evidence of a process leak or a particular workload at fault.

A bounded Kimi review through Ollama used only public research context.
Kimi did not execute code, read private files, or authenticate the hardware.
Its original threshold bands overlapped; this implementation makes
the categories mutually exclusive and never calls an advisory
observation training-ready.

External inspirations, without copying upstream code:

- [Kueue admission](https://kueue.sigs.k8s.io/docs/concepts/admission/):
  quota reservation is not workload admission
- [Ray Train fault tolerance](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html):
  checkpoint restoration is distinct from a running label
- [NVIDIA NeMo resiliency](https://docs.nvidia.com/nemo-framework/user-guide/25.09/resiliency.html):
  hang detection and bounded recovery
- [SLSA provenance](https://slsa.dev/spec/v1.2/build-provenance):
  independent signer, builder and source verification

## Exact integer algebra and limitations

For time-ordered samples i, let F=min(F_i) be minimum free physical
memory, r=max(ceil(10^6 * C_i/L_i)) be the maximum committed fraction
in parts per million, and P_min=min(P_i), P_max=max(P_i).
Let v=ceil(10^6 * (P_max-P_min)/P_min) ppm for P_min>0.
A zero-to-positive pagefile allocation is assigned 1,000,000 ppm.
A consistently zero pagefile is a caution. No floating-point ratio is used.

- ADVISORY_HOLD if F<2 GiB or r>850,000 ppm or v>250,000 ppm
- Otherwise ADVISORY_CAUTION if F<4 GiB or r>800,000 ppm or
  v>100,000 ppm, or a consistently zero pagefile
- Otherwise OBSERVATION_STABLE_ONLY; this is never a training PASS
- Missing/malformed/inconsistent/insufficient input is UNOBSERVED,
  returns nonzero exit, and is never assumed to be zero risk

These proposed thresholds are conservative **policy assumptions**,
not empirically calibrated OOM probabilities or hardware guarantees.
Observed memory counters do not independently prove the source
of the data, signer/owner identity, future stability, or trainer liveness.

## Reproduction

Provide a private sanitized JSON object containing exactly the root
keys schema (value szl.windows-memory-observation/v1) and samples (5-12).
Each row has exactly observed_utc, physical_total_bytes,
physical_free_bytes, committed_bytes, commit_limit_bytes,
and pagefile_allocated_bytes. UTC samples must be 1-5 seconds apart
on one physical memory denominator. No host identity is required.

    python -B -m unittest discover -s tests -p test_windows_memory_envelope.py -v
    python -B tools/evidence/windows_memory_envelope.py --input YOUR_PRIVATE_SANITIZED_INPUT.json

The CLI does not write receipts. Any exclusive owner-private non-secret
evidence receipt must be handled separately under reviewed controls.
This tool reads no credentials, scans no secrets, creates no directories,
cleans no caches, installs no agents and sends no data to Kimi.

**Independent HOLDs remain:** sole authorized training-dispatch
owner/runner, DSSE signer roots, source-to-runtime image parity,
signed jobs, storage preflight, checkpoint durability, dataset rights,
budget and Flagship54 control qualification. Green CI validates
source-level behavior, never a live training deployment.
