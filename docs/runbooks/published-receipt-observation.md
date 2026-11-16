# Published receipt observation

The one-shot observer resolves explicit repository/type/path contracts. Heldout
v4 and v5 results live in their model repositories; the frontier v5 r7 training
receipt lives in the shared dataset. Searching only the dataset is not evidence
that a model-repository receipt is absent.

Run from the repository root:

```sh
python -m tools.salad.receipt_observer
python -m tools.salad.receipt_observer --target heldout-v4 --skip-heartbeats
python -m unittest discover -s tests -p 'test_receipt_observer.py' -v
```

The observer performs Hub reads and may populate the normal local Hub cache. It
has no provider control, scheduler, uploader, model execution or shutdown path.
A zero exit code means the observations were obtained, not that models passed or
are operational. An unavailable read stops the remaining provider reads and exits
2. Absence is scoped to the declared path at the observed repository revision.

Each receipt and its archived source are downloaded using one exact observed Hub
revision. The result preserves producer status and hashes the receipt bytes.
A published error remains an error; a terminal resume with zero new training
steps remains `NO_NEW_TRAINING`. A successful historical receipt never qualifies
current weights, energy attribution or current process liveness.

For heldout v1 receipts, the existing fingerprint-specific legacy audit is
included. It preserves the original receipt and identifies raw NVML mJ values
mislabeled as J, the unisolated baseline arm, a neutral comparison that cannot fail
at its recorded baseline, and missing execution bindings. Derived unit corrections
are not a GPU rerun or exclusive per-job energy verification.

Flagship heartbeat filenames are ordered by numeric sequence. The highest
numbered record and a terminal record are reported separately; neither is a
freshness timestamp. No status is inferred from the lexical ordering of filenames.

## Regression reproduced on 2026-11-16

The prior device helper searched heldout paths only in
`SZLHOLDINGS/frontier-54-runs` (dataset) and reported them missing. Correct model
contracts returned published v4 and v5 results, both producer status `OK`, and
both classified `AUDITED_LEGACY_NOT_QUALIFIED` by the existing exact-source audit.
The historical adapter-download API bug was already repaired in the live scripts
by using `snapshot_download` for `allow_patterns`; this observer does not rewrite
those artifacts or rerun their evaluations.

The prior lexical heartbeat selector returned `hb99` from a catalog containing
later records. Numeric selection returned `hb919` in the verification observation.
These observations do not establish present hardware health or authorize compute.
