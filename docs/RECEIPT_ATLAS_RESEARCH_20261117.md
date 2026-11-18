# Receipts Atlas — engineering research and original SZL adaptation
Date: 17 November 2026 · scope: PUBLIC dataset index + source-only frontend
Status: SOURCE CANDIDATE; no claim of HF deployment or verified signatures.

## Empirical problem statement

The former Receipts Chain UI was not a verified chain viewer. It recursively
walked mutable main, took the first 120 JSON/JSONL paths, discarded individual
read failures without accounting, and drew visible lines between unrelated
rows. It described every node as a verifiable SHA-256 hash even where the
receipt only self-reported a digest or where no digest was compared.

An independent, read-only public Hub metadata lookup on 2026-11-17 observed
revision c2228c402e7695c4e6ed22e3f8e52550721ba603 with 7,929 indexed
files: 5,705 JSON, 7 JSONL, 2,175 logs, and several other types; top-level
mesh contained 7,319 paths. Counts are a point-in-time file listing, not
an exhaustive SZL organizational evidence inventory. The mutable repository
changed during observation (a subsequent metadata listing returned 7,930),
which is exactly why every inspected file must use the returned immutable SHA.

The replacement opens the full bounded file-path index, allows searchable
paging and validates a deliberate, size-limited sample. The observed byte
SHA-256 is explicitly separated from any verified signature or inclusion.

## Leaders: ideas, publications, code and how they inform our build

1) **Sigstore / Rekor / RFC 9162 (cryptographic transparency).**
   Code: https://github.com/sigstore/rekor
   Architecture: https://github.com/sigstore/architecture-docs/blob/main/rekor-spec.md
   Standard: https://www.rfc-editor.org/rfc/rfc9162.html
   Math: H_leaf(x) = SHA-256(0x00 || x),
   H_node(L,R) = SHA-256(0x01 || L || R).
   A Merkle inclusion proof is an O(log N) witness against an independently
   signed checkpoint; a local SHA-256 computed by a browser is not that proof.
   SZL adaptation implemented: immutable source-revision paths, bounded
   independent byte hashing, zero verification claims without trusted roots.
   Not implemented: signed checkpoints, inclusion/consistency verification,
   Rekor integration, or public cryptographic proof graph edges.

2) **SLSA and in-toto (provenance of build and subjects).**
   Source code: https://github.com/in-toto/in-toto
   Spec: https://slsa.dev/spec/v1.1/provenance
   Publication: Torres-Arias et al., 2019, "in-toto: Providing farm-to-table
   guarantees for bits and bytes," USENIX Security.
   https://www.usenix.org/system/files/sec19-torres-arias.pdf
   Operational rule: receipt validity is a conjunction of independently
   checked subject digest, source materials, builder identity, signature,
   verification policy and currently deployed revision. An opaque checksum
   is not a signer. A build receipt is not a model evaluation.
   SZL adaptation implemented: all these authorities remain false/hold;
   no page button grants deployment, training or promotion.
   Not implemented: canonical in-toto attestation schema ingestion or a
   verifiable SLSA build predicate; never use SLSA level branding here.

3) **OpenLineage (run/job/dataset identity, truthful graph semantics).**
   Code/spec: https://github.com/OpenLineage/OpenLineage
   Schema: https://github.com/OpenLineage/OpenLineage/blob/main/spec/OpenLineage.md
   Model: Run Event (START/COMPLETE/FAIL/ABORT), Job, Dataset and facets,
   bound to particular run IDs and actual input/output datasets. Edges
   should represent asserted provenance, not spatial proximity on canvas.
   SZL adaptation implemented: constellation is explicitly a scatter
   visualization of filenames with NO evidence or causal edges; dataset
   revision and path are first-class source identity.
   Not implemented: OpenLineage-compatible run event emitter or evidence-
   bound edges. Do not falsely label the visualization as a proof graph.

4) **MLflow (experiment tracking and evaluated trace lineage).**
   Code: https://github.com/mlflow/mlflow
   Publication: "Accelerating the Machine Learning Lifecycle with MLflow"
   (MLflow project CITATION.cff, https://github.com/mlflow/mlflow/blob/master/CITATION.cff).
   Documentation: https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/
   An evaluation is interpretable only with a specified input set, labels,
   scorer version, denominator, immutable model revision and result bytes.
   SZL adaptation: the atlas never turns a receipt record count into
   completed training, benchmark quality or a measured win-rate. Verified
   step/quality graphing requires independent per-lane candidate binding.
   Not implemented: MLflow tracking server or experiment migration.

5) **NVIDIA DCGM (measured resource telemetry vs provider group status).**
   Code: https://github.com/NVIDIA/dcgm-exporter
   Docs: https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html
   GPU utilization, memory, power and joule counters are device/sampler
   evidence; a Salad group-status string and desired replicas are not
   actual GPU energy, trainer steps or billable runtime.
   SZL adaptation: connect these measurements ONLY via existing private,
   source-bound signed training/energy receipts, never via public fiction.
   Not implemented here: GPU telemetry exporter or hardware attribution.

6) **Palantir Ontology + OpenTelemetry semantic conventions (governed view).**
   Ontology: https://www.palantir.com/docs/foundry/ontology/overview
   GenAI conventions: https://github.com/open-telemetry/semantic-conventions-genai
   Distinguish observed objects/properties from verified relationships and
   authorized actions; avoid leaking prompts, credentials or private logs.
   SZL adaptation implemented: no action controls, no raw secret-shaped
   JSON field previews, no network credentials, bounded public reads.
   The separate szl-command-center ontology remains disabled by default;
   this public exhibit does not bypass its private-evidence access gate.
   Not implemented: cross-repository authenticated lineage joins.

## Mathematics: evidence coverage must have a denominator

Let P be the file paths returned by one metadata response at revision r.
Let R be a documented receipt-candidate PATH heuristic (not a signature).
Let I be the paths whose bounded bytes were attempted and let Q subset I
be the ones whose UTF-8 JSON/JSONL successfully parsed.

  N_index       = |P|
  N_candidates  = |P intersect R|
  N_inspected   = |I|
  N_parsed      = |Q|
  N_unavailable = |I| - |Q|
  C_parse       = |Q| / |P intersect R|, if |P intersect R| > 0.

C_parse is **only syntax-inspection coverage of heuristic candidate files**.
It is not completeness of the provider's estate, not integrity/signature
coverage, and not proof of processing every JSONL record in a large file.

For a consequential claim c, define each gate with three-valued state
VERIFIED / UNVERIFIED / UNAVAILABLE. The final release decision is:

  ADMIT(c) = V(subject) AND V(signature) AND V(source)
             AND V(builder) AND V(runtime) AND V(policy).

Unknown, expired or unavailable operands cannot be replaced by false-zero or
optimistic probabilities. The release state is HOLD unless all required
operands are independently verified at the same relevant source revisions.
A valid checksum is an observed byte identity; it does not satisfy V(signature).

For time relevance, delta = observed_now_utc - source_time_utc. A nonnegative
delta bounded by an owner TTL establishes only *freshness*, not authenticity.
Expired or unknown delta requires downgrade; freshness cannot override a
missing signature or insufficient benchmark coverage.

## Implementation / remaining holds

Implemented in source candidate:
- Complete bounded public metadata PATH listing and search instead of
  first-120 file sampling; SHA-pinned immutable content retrieval.
- 8 MB index, 256 KB individual file cap, 12-row JSONL presentation limit
  and 12-file opt-in inspection batch.
- Deterministic multi-category constellation WITHOUT fake graph links,
  separate parsed/unavailable/uninspected counts, explicit zero signatures.
- Byte SHA-256 of retrieved data without signer or inclusion assertions,
  sanitized preview and safe textContent DOM insertion.
- Node negative tests, hosted CI check and Dockerfile copy of both modules.

Unadmitted/blocked:
- Real HF Space deployment/current SOURCE_AHEAD checks and browser readback.
- End-to-end authenticated visibility/access and public CORS validation.
- Cryptographic signature, trust root and Merkle inclusion/consistency.
- Shared 170-repo estate receipts coverage and private data governance.
- Training-runtime/Salad image/checkpoint/evaluation attestations.
- User-visible proof-link cross-navigation from Command Center after a
  source/identity/authorization and privacy review.

This implementation is original application-level integration of public
standards ideas, not a claim to outperform the referenced projects.