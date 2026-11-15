# Flagship54 completion guard

Scope: `szl-train-flagship54-v1` and the private working repository
`SZLHOLDINGS/salad-flagship54-v1`. This is a single-lane safety repair, not a
training-quality evaluation or production qualification.

## Why a filename is insufficient

The worker saves a final adapter after `trainer.train()` returns. An interrupted
training loop can therefore leave a `final/adapter_model.safetensors` file before
the requested step target is reached. The previous external watcher treated that
filename as verification and treated desired replicas zero as completed shutdown.
Neither inference is sufficient.

## Shutdown contract

The guard requires all of the following before changing the cloud replica count:

1. The original Salad group identity, command-array SHA-256 and configuration
   version still match. Scheduled/queue autoscaling must not be active.
2. The working repository is currently verifiably private. All artifact reads are
   pinned to one immutable commit, not a moving branch.
3. The latest checkpoint has a readable, hash-verified trainer state whose step
   matches its directory and whose target matches the frozen configuration. The
   recorded step must reach the configured target (37,000 for this run).
4. Final adapter weights/configuration and a tokenizer are present. The tokenizer
   may come from the same completed checkpoint. Pickle/Python files are not
   accepted in the final export.
5. The selected files have been downloaded, byte-hashed against their Hub object
   identifiers, copied into an independent-node archive, and rehashed. Resume
   pickle files are preserved as opaque bytes and are never deserialized.
6. Final weights and configuration exactly match the completed checkpoint.
   Adapter identity, tokenizer JSON and safetensors structure are checked.
7. Repository visibility/head and Salad identity/configuration are rechecked
   immediately before the mutation. A verification receipt and scale-down intent
   are persisted first.

The only cloud mutation is `PATCH {"replicas": 0}` to the allowlisted lane. DONE
requires desired replicas zero **and** actual allocating, creating, running and
stopping counts all equal to zero. A successful PATCH response alone is not DONE.

Missing evidence, authentication failures, changed identity/target, corrupt files,
or insufficient disk leave the run on HOLD rather than authorizing shutdown.
The guard never starts or restarts training and never changes repository visibility.

## Configuration and operation

Store configuration and credentials under an owner-only directory. Credential
files must be regular, owner-owned files with mode 0600; do not put values in
source, command arguments, GitHub, receipts, or logs. The JSON configuration holds
file paths, not credential values.

Required configuration fields are `lane`, `repo`, `max_steps`, `base_model`,
`group_id`, `group_version`, `command_sha256`, `state_dir`, `archive_dir`,
`hf_token_file`, and `salad_key_file`. The command binding is the SHA-256 of the
command array serialized with `json.dumps(value, sort_keys=True,
separators=(',', ':'), ensure_ascii=True).encode()`.

```sh
# Credential-free regression tests; service interactions are mocked.
python3 -m unittest discover -s tools/salad -p 'test_flagship54_guard.py' -v

# Live read-only observation. Does not scale the lane.
python3 tools/salad/flagship54_guard.py --config /absolute/path/config.json --once

# Independent checkpoint copy. Does not scale the lane.
python3 tools/salad/flagship54_guard.py --config /absolute/path/config.json --backup-latest

# Completion guard; singleton lock; poll interval defaults to 120 seconds.
python3 tools/salad/flagship54_guard.py --config /absolute/path/config.json --apply
```

Apply mode writes `status.json`; read-only checks write `dry-run-status.json`;
backup runs write `backup-latest-status.json`. Archive manifests bind exact file
sizes and SHA-256 values. Receipts include the guard's source hash. Their payload
SHA-256 is an unkeyed integrity digest, not a digital signature or outside witness.

Run the guard under a service manager with restart-on-failure, restrictive umask,
no added privileges, read-only system/home mounts, and write access limited to
its state, archive and dedicated Hub cache directories. Verify a fresh successful
live observation before retiring the old watcher. Keep unrelated jobs untouched.

## Credential incident boundary

Removing an exposed credential from source, deleting a stale compiled copy, and
moving the active value to a protected file do **not** rotate that credential.
Replacement and revocation must be coordinated across all consumers. Do not reset
an account-wide key while other training or inference controllers still depend on
it. Preserve a secret-free rollback record plus the original source digest; never
publish a credential-bearing diagnostic dump.

## Verification scope

The initial implementation passed 37 credential-free tests locally and the same
37 tests on the control node. A live read-only run returned TRAINING without a
cloud mutation. This does not prove a future final export will pass, that the
model is useful, or that actual shutdown has already occurred. The GitHub
workflow runs the same contracts on Python 3.11 and 3.12.

## API contracts consulted

- https://docs.salad.com/reference/saladcloud-api/container-groups/update-container-group
- https://huggingface.co/docs/huggingface_hub/package_reference/file_download
