# Training-wave source identifiers: non-zero sentinels and trust boundaries

Stage: source-only test hardening. No provider dispatch or credential access.

Git commit revision fields must be 40 lowercase hexadecimal characters for
the current Git SHA-1 object format, and cannot be 40 zeroes. SHA-256 fields
(catalog, recipe, canary and evaluation receipt digests) must be 64 lowercase
hexadecimal characters and cannot be 64 zeroes. This prevents empty placeholder
IDs from being treated as syntactically meaningful references.

The CPU-only planner never authenticates Git objects, certificate signatures,
content bytes, model/dataset usage rights, training checkpoints or quality.
A plausible non-zero digest remains an *unverified claim* until a separate
source-owner process authenticates its exact provider revision and reads the
actual bytes. All syntactically valid plans remain HOLD_NO_COMPUTE; there is no
new authority to dispatch paid GPU jobs or to bypass SaladCloud HTTP 401.

Synthetic negative-control tests cover all three revision fields, catalog and
receipt digests, case/length crossovers, input immutability, and preservation
of non-authorizing output. This patch stacks on the existing source owner's
review branch, never touching its protected main, training service, or ledgers.
