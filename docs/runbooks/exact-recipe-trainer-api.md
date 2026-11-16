# Bind training preflight to the exact recipe API

The runtime gate previously checked one hard-coded SFTConfig field set, even when
the hashed recipe passed to the CLI used other keywords. A recipe containing
max_seq_length could therefore pass against a runtime that only accepted
max_length, then fail before training. The contract's source hash and installed
package version pins do not by themselves establish API compatibility.

The CLI now parses the single supported quoted Python heredoc and extracts the
explicit keyword names from its SFTConfig call. It compares those actual names
with the imported runtime's constructor-enabled dataclass fields. The recipe is
never executed by the parser. Missing/ambiguous Python blocks, malformed code,
config aliases or rebinding, positional arguments and dynamic **kwargs hold
before environment probes. Supported form: python or python3, stdin '-', one
quoted uppercase heredoc delimiter, one direct top-level `from trl import
SFTConfig` import and one explicit SFTConfig call. The canonical A11oy migration
recipe is covered by a positive regression.

This is a bounded static compatibility check, not a general shell interpreter,
security sandbox, whole-program call-graph proof or source-admission authority.
It does not prove control flow, accepted argument values, model compatibility,
data rights, a valid train split, GPU execution, numerical quality, durable
resume or publication permission. The output retains source_admission_verified
false and reports the exact checked fields. Direct helper callers retain the
old default field set for compatibility; the CLI always derives fields from the
verified recipe bytes.

Do not repair a missing train split by silently using validation/test data.
Dataset identity, revision, configuration, split and separation from evaluation
remain distinct admission requirements. Likewise, do not repair import failures
with unbounded pip installation on each live worker boot. Use the existing
immutable prebuilt-runtime contract and independent image/GPU checks.

## Validation and rollout

Run `python -m unittest discover -s tests -p 'test_training*.py' -v`.
The 15 added methods include the original false-positive CLI replay with mocked
runtime responses, exact-field positive controls, unsupported forms and actual
canonical recipe parsing. The test runner does not download models, install
packages, use a GPU, start training or call a provider.

This source repair changes the existing CLI gate only. It neither updates
historical cloud payloads nor deploys a new image or restarts a job. Pin the
admitted validator and recipe together through the existing owner-controlled
launch path. Roll back with a normal source revert; preserve failed attempts and
never turn a compatibility hold into a completion or readiness receipt.
