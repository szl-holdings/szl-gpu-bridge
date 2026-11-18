# SZL Capacity Mathematics: Research-to-Software Contract

Status: source-only exploratory mathematics. Reviewed November 17, 2026.
This is neither a GPU controller nor a deployed web service. Original SZL
implementation uses Python standard library only. No third-party code was copied.

## Leaders, primary formulas and directly verifiable references

| Leader / publication / GitHub | Formula or architectural insight | SZL-specific use |
| --- | --- | --- |
| John D. C. Little, Operations Research (1961), DOI https://doi.org/10.1287/opre.9.3.383 | L = lambda * W, in stationary long-run systems with finite means. | Report **modeled offered inflight**, never inferred real occupancy or a release signal. |
| J. F. C. Kingman, Cambridge (1961), DOI https://doi.org/10.1017/S0305004100036094; heavy traffic 1962 DOI https://doi.org/10.1111/j.2517-6161.1962.tb00465.x | G/G/1 Wq approx [rho/(1-rho)] * [(ca² + cs²)/2] * mean_service, with rho=lambda*mean_service for ONE server. | Use only for a single ready replica with rho < 1. Multi-GPU and overload return explicit unavailable states. |
| Netflix, Apache-2.0 https://github.com/Netflix/concurrency-limits | Vegas queue proxy L * (1 - minRTT/sampleRTT); Gradient2 compares two EWMA latency horizons. | Only an analytical proxy. **No automatic Netflix controller, network write or cost claim.** |
| Kubernetes HPA, Apache-2.0 https://github.com/kubernetes/kubernetes ; https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/ | desiredReplicas=ceil(currentReplicas * currentMetric/targetMetric), plus hysteresis, stabilization and missing-pod guards. | Use analogous ceil(offered_work/target_utilization) ONLY as a planning candidate. It is not observed CPU nor a real HPA. |
| Google Borg, EuroSys 2015 https://research.google/pubs/large-scale-cluster-management-at-google-with-borg/ | Cluster admission, packing, overcommitment, isolation, monitoring and simulation. | Retain source-bound architecture preflight, per-GPU admission and snapshot labels with SZL's existing controller. |
| Argo Rollouts, Apache-2.0 https://github.com/argoproj/argo-rollouts ; https://argoproj.github.io/argo-rollouts/features/analysis/ | Canary metrics yield Successful, Failed or Inconclusive; controlled rollout may pause or roll back. | Never count an estimate as a successful canary; existing SZL deploy controller remains the only actor. |
| OpenTelemetry semantic conventions, Apache-2.0 https://github.com/open-telemetry/semantic-conventions ; https://opentelemetry.io/docs/specs/semconv/http/http-metrics/ | HTTP server request duration is a histogram in **seconds** with standard boundaries and dimensions. | Input fields end in _seconds; require source/route/clock normalization before real evaluation. |
| Temporal, MIT Python SDK https://github.com/temporalio/sdk-python ; https://github.com/temporalio/documentation/blob/main/docs/evaluate/understanding-temporal.mdx | Durable workflow event history and deterministic replay separate events and side effects. | Keep action, retries, idempotence and archive acknowledgment with existing controllers; this library has no side effects. |
| Palantir Ontology https://www.palantir.com/docs/foundry/architecture-center/ontology-system ; OSDK https://github.com/palantir/osdk-ts | Integrates data, logic, action and security; a graph view does not confer execution permission. | Include source/declared revision and explicit unqualified gate; never promote based on dashboard metrics. |
| DORA 2026 https://dora.dev/insights/dora-metrics-history/ | Change lead time, deployment frequency, failure recovery, change failure and deployment rework. | Track measured release outcomes separately. This script emits no DORA score from synthetic service samples. |

Research source revisions observed on GitHub:
- kubernetes/kubernetes a6cfaea25883475e9bf01970a3866e9c292ad47e
- Netflix/concurrency-limits 78a74b9878d38c4c048b0304ce12a162ab7b7222
- argoproj/argo-rollouts 1147bc34b4ee737e1a6a40d17fff13a73909da4b
- open-telemetry/semantic-conventions f14668288d9e7a7aa2e7c5896a39c8f852351587
- temporalio/sdk-python dd1d49176346798a09283fa6d8f7d6169ee250f6
- palantir/osdk-ts f1d12b07ca7c679b39eeb2315edc2e5d63ac5fa5
These are reference snapshots, not dependencies or code copied into SZL.

## Implementation and model boundaries

The new Python module tools/salad/capacity_math.py reads one finite,
source-pinned local JSON specimen. Its units, rates and formula assumptions
are explicit and falsifiable:

1. Arrival rate lambda = arrivals / window_seconds; completed throughput =
   completions / window_seconds. Backlog delta = arrivals - completions. A
   negative delta can mean backlog draining and is not an invalid input.
2. Mean service time, coefficient of variation squared of interarrival and
   service samples, observed per-replica workload ratio
   rho = lambda * mean_service / ready_replicas.
   This is a workload indicator, NOT measured GPU utilization.
3. Kingman single-server approximation only for ready_replicas == 1,
   rho < 1 and finite sampled variability. Does not generalize to G/G/c.
4. Little-modeled mean inflight = lambda * mean response time.
   Does not establish stationarity, flow conservation or real occupancy.
5. Netflix Vegas queue proxy uses an inflight limit and comparable min and
   current RTT; input comparability is not independently verified.
6. HPA-like raw replicas = ceil(offered_work / target_utilization), with
   separate min/max bounds; over-cap and potential downscale are explicit.
   No provider scaling request is generated.
7. The empirical p95 response time uses a nearest-rank descriptive sample.
   The Wilson two-sided 95%-style upper failure fraction is computed from
   failures/completions, but independence of trials is unverified and should
   not be assumed for correlated GPU failures.
8. The HTML view is rendered by Python using escaped text. It has no JS,
   external assets, database connection or account credentials.

The included SYNTHETIC specimen has lambda 0.75 requests/s, mean service
time 1 s, rho 0.75 with one replica, interarrival CV² 0.09, service CV²
0.04. Kingman's approximate waiting time is 0.195 seconds. The HPA-like
formula produces 2 suggested replicas but outputs scale_action NONE and
promotion_authorized false. This is a **fixture**, not an SZL GPU benchmark.

## Truth and security boundary

- Only a local non-symlink input file, <=128 KiB, with strict JSON object
  fields, duplicate-key rejection, no non-finite values, no unknown fields,
  timestamps UTC, measured durations in seconds and bounded 20–256 samples.
- GitHub revision and telemetry SHA256 are syntax-validated DECLARATIONS;
  no code here fetches their exact remote bytes or verifies any signature.
  Local input raw-byte SHA256 is computed separately.
- A stale window is analyzable as history, never eligible for current
  operational authority. Future windows are rejected.
- A failure is code-only and contains no raw paths, credentials or samples.
- No Salad API, Hugging Face API, Cloudflare, PyPI, GPU, database, signing
  key, subprocess, mutable workflow, model evaluation or payment is accessed.
- Outputs always state provider_writes=0, promotion_authorized=false,
  production_qualified=false, telemetry_authenticity_verified=false.
  The experiment cannot generate a valid authority receipt.
- Do not infer operating costs, energy consumption or scale-up decisions.
  The purpose is to produce **proof obligations** for the existing controller.

## Run and qualification

From the exact reviewed repository source:

    python -X utf8 tools/salad/capacity_math.py --input tests/capacity_synthetic_sample.json --format json
    python -X utf8 tools/salad/capacity_math.py --input tests/capacity_synthetic_sample.json --format html

Only stdout is written; shell redirection to a local HTML file is optional.
The full contract CI runs unittest discovery and verifies formatting/linting
of the module. Regression tests include overload, multiservers, missing
source binding, staleness, duplicate JSON, type confusion, injected HTML,
mixed sample sizes, zero-failure uncertainty and CLI isolation.

**Next validation:** obtain real, consented OTel request/queue measurements
from an authenticated, exact-source runtime with consistent clock, sample
units, tenant, route, model and hardware identities. Independently pin the
measurement receipt and replay held-out traces; compare predicted average
queue wait, observed p95 and utilization error under load and burst, test
rollback/admission, and only then consider a governed scaling experiment.
Nothing in this implementation executes that sequence.
